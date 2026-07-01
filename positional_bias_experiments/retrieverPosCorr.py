

# @gauravk: pos-corrected inference for LLaMA — Method 1 inline '+' anchor
#
# Aligned exactly with modelling_llama.py saved_states contract:
#   - model.config.block_boundaries  : list of (b_start, b_end) int tuples — RAW SPANS
#   - model.config.instr_end         : int
#   - model.config.q_start / q_end   : int   (full question slice incl. '+' tokens)
#   - saved_states["instr"]          : {"q": ..., "k": ...}
#   - saved_states["question"]       : {"q": ..., "k": ...}  <- full q_start:q_end slice
#   - saved_states[idx]              : {"q": ..., "k": ...}  <- idx = position in block_boundaries list
#
# q_content_mask / q_plus_mask are NEVER passed to model.config —
# the model saves the entire question slice; we split it here at score time.

import argparse
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from modelling_llama import LlamaForCausalLM
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="llama_pos_corr_only_dens.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

TOP_CORR_HEADS = [(14, 30), (14, 22), (14, 20), (15, 3), (14, 29), (23, 20), (24, 22), (22, 28), (14, 3), (23, 22), (14, 28), (14, 31), (23, 13), (14, 13), (24, 16), (23, 12), (24, 20), (16, 19), (23, 27), (24, 25)]

def load_model_tokenizer(model_name, device, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=True,
        device_map=device,
    )
    model.eval()
    return tokenizer, model


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_queries_and_items(dataset):
    if dataset in ["spider", "bird"]:
        train_db_path = f"/scratch/gaurav/data/BIRD/formatted/{dataset}db_train_schema_info.json"
        dev_db_path   = f"/scratch/gaurav/data/BIRD/formatted/{dataset}db_dev_schema_info.json"
        queries_path  = f"/scratch/gaurav/data/BIRD/formatted/sft_{dataset}_dev_text2sql.json"
        with open(queries_path, "r") as f: test_queries = json.load(f)
        train_dbs, dev_dbs = {}, {}
        with open(train_db_path, "r") as f: train_dbs.update(json.load(f))
        with open(dev_db_path,   "r") as f: dev_dbs.update(json.load(f))
        queries = [{"text": s["text"], "gold_db_name": s["db_id"], "qid": i}
                   for i, s in enumerate(test_queries)]
        return queries, train_dbs, dev_dbs
    return [], {}, {}


def build_prompt_tokens(tokenizer, question_text, all_db_names, dbs, plus_token_id):
    """
    Build the flat token list and the parallel metadata structures that
    modelling_llama.py expects on model.config.

    block_boundaries : list of (b_start, b_end) int tuples, one per DB,
                       in the same order as all_db_names.
                       The model indexes saved_states with the enumeration
                       index into this list (0, 1, 2, ...).

    The question is interleaved with '+' anchor tokens (Method 1).
    q_content_mask / q_plus_mask are returned for use in score_dbs_method1
    but are NOT written to model.config (the model saves the whole slice).

    Returns
    -------
    tokens           : list[int]
    block_boundaries : list of (int, int)   <- passed to model.config
    db_order         : list[str]            <- db name at each boundary index
    q_start, q_end   : int
    q_content_mask   : bool tensor  (True = real word, False = '+')
    q_plus_mask      : bool tensor  (True = '+' anchor)
    instr_end        : int
    """
    system_block = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert database routing system. Your task is to analyze a user's question "
        "and a list of available database schemas. "
        "You must select the most relevant database name that can answer the question.\n"
        "Do not add any other text, explanation, or formatting.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
    )
    tokens    = tokenizer(system_block, add_special_tokens=True)["input_ids"]
    instr_end = len(tokens)

    block_boundaries = []   # list of (b_start, b_end) — model contract
    db_order         = []   # parallel list: db name at each boundary index

    for db_name in all_db_names:
        db_ids = tokenizer(
            f"Database: {db_name}\nSchema: {dbs.get(db_name, '')}\n",
            add_special_tokens=False
        )["input_ids"]
        b_start = len(tokens)
        tokens.extend(db_ids)
        block_boundaries.append((b_start, len(tokens)))
        db_order.append(db_name)

    q_text = (
        f"Select + the + correct + database + name + for + answering + the + question + that + follows.\n"
        f"Question: + {question_text} + \n + Correct + database + name + : + "
    )
    q_ids   = tokenizer(q_text, add_special_tokens=False)["input_ids"]
    q_start = len(tokens)
    tokens.extend(q_ids)
    q_end = len(tokens)

    q_content_mask = torch.tensor(
        [tid != plus_token_id for tid in q_ids], dtype=torch.bool
    )
    q_plus_mask = ~q_content_mask

    ass_ids = tokenizer(
        "<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False
    )["input_ids"]
    tokens.extend(ass_ids)

    return (tokens, block_boundaries, db_order,
            q_start, q_end, q_content_mask, q_plus_mask, instr_end)


def run_forward(model, tokens, block_boundaries, instr_end, q_start, q_end, device):
    """
    Set the config fields that modelling_llama.py's LlamaAttention reads,
    then do a single forward pass.

    Fields set on model.config:
        instr_end        : int
        q_start / q_end  : int
        block_boundaries : list of (int, int)   <-- raw tuple list, NOT dicts

    q_content_mask / q_plus_mask are intentionally NOT set here —
    the model saves the full question slice and we split it ourselves.
    at_start / at_end are deleted if present so the model skips that branch.
    """
    model.config.instr_end        = instr_end
    model.config.q_start          = q_start
    model.config.q_end            = q_end
    model.config.block_boundaries = block_boundaries   # list of (int, int)

    if hasattr(model.config, "at_start"): del model.config.at_start
    if hasattr(model.config, "at_end"):   del model.config.at_end

    with torch.no_grad():
        model(torch.tensor([tokens], device=device), use_cache=True)


def snapshot_and_clear(model, num_db_blocks, num_layers=32):
    """
    Collect saved_states from every attention layer and reset them.

    modelling_llama.py saves:
        saved_states["instr"]    : present when instr_end is set
        saved_states["question"] : present when q_start / q_end are set
        saved_states[0..N-1]     : one int key per entry in block_boundaries
    """
    saved = {}
    for l in range(num_layers):
        attn = model.model.layers[l].self_attn
        layer_snap = {}

        for key in ("instr", "question"):
            if key in attn.saved_states:
                layer_snap[key] = attn.saved_states[key]

        for idx in range(num_db_blocks):
            if idx in attn.saved_states:
                layer_snap[idx] = attn.saved_states[idx]

        saved[l] = layer_snap
        attn.saved_states = {}

    return saved



def score_dbs_method1(saved_states, db_order, heads_by_layer, scaling,
                      q_content_mask, q_plus_mask, device):
    """
    For each DB block at relative index d_rel:

        q_dens  = mean raw attention density from real question tokens -> DB keys
        p_dens  = mean raw attention density from '+' anchor tokens   -> DB keys
        score   = q_dens - p_dens   (d_rel >= 2)   -- subtract positional bias
                = q_dens            (d_rel <  2 )   -- guard: protect first two blocks

    Scores are accumulated (summed) across all selected heads.

    Parameters
    ----------
    saved_states   : {layer_idx: {"question": {...}, int_db_idx: {...}, ...}}
    db_order       : list[str]  db name at each integer index
    heads_by_layer : {layer_idx: list[int]}
    scaling        : float  (head_dim ** -0.5)
    q_content_mask : bool tensor [q_len]
    q_plus_mask    : bool tensor [q_len]
    device         : str

    Returns
    -------
    db_scores : {db_name: float}
    """
    db_scores = {name: 0.0 for name in db_order}

    q_content_mask = q_content_mask.to(device)
    q_plus_mask    = q_plus_mask.to(device)

    for l_idx, head_list in heads_by_layer.items():
        if l_idx not in saved_states:
            continue
        ls = saved_states[l_idx]

        if "question" not in ls:
            continue

        q_states = ls["question"]["q"].to(device)

        q_states_sel = q_states[:, head_list, :, :]           # [1, H, q_len, D]

        q_content = q_states_sel[:, :, q_content_mask, :]     # [1, H, n_content, D]
        q_plus    = q_states_sel[:, :, q_plus_mask,    :]     # [1, H, n_plus,    D]

        q_len_main = q_content.shape[2]
        q_len_plus = q_plus.shape[2]

        for d_rel, db_name in enumerate(db_order):
            if d_rel not in ls:
                continue
            k_db  = ls[d_rel]["k"].to(device)
            k_sel = k_db[:, head_list, :, :]                  # [1, H, db_len, D]
            db_len = k_sel.shape[2]

            w_raw  = torch.matmul(q_content, k_sel.transpose(-1, -2)) * scaling
            q_dens = w_raw.sum(dim=[-1, -2]).squeeze(0).float() / (q_len_main * db_len)

            if d_rel < 2:
                corrected = q_dens
            else:
                w_plus = torch.matmul(q_plus, k_sel.transpose(-1, -2)) * scaling
                p_dens = w_plus.sum(dim=[-1, -2]).squeeze(0).float() / (q_len_plus * db_len)
                corrected = q_dens #- p_dens

            db_scores[db_name] += corrected.sum().cpu().item()   # sum over selected heads

    return db_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',          type=int,  default=64)
    parser.add_argument('--dataset',       type=str,  default="bird")
    parser.add_argument('--model',         type=str,
                        default="/scratch/models/models--meta-llama--Llama-3.1-8B-Instruct/"
                                "snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    parser.add_argument('--num_queries',   type=int,  default=1000)
    parser.add_argument('--use_top_heads', action='store_true',
                        help="Use TOP_CORR_HEADS only; default = all heads", default=TOP_CORR_HEADS)
    args = parser.parse_args()

    seed_all(args.seed)
    device = "cuda:0"

    queries, train_dbs, dev_dbs = get_queries_and_items(args.dataset)

    dbs = {}
    for k, v in dev_dbs.items():   dbs[k] = v
    for k, v in train_dbs.items(): dbs[k] = v

    random.shuffle(queries)
    inference_samples = queries[:args.num_queries]
    all_db_names  = list(dbs.keys())
    num_db_blocks = len(all_db_names)

    tokenizer, model = load_model_tokenizer(args.model, device)
    num_layers = model.config.num_hidden_layers   # 32 for LLaMA-3.1-8B

    plus_token_id = tokenizer(" +", add_special_tokens=False)["input_ids"][0]

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    scaling  = head_dim ** -0.5
    n_heads  = model.config.num_attention_heads

    if args.use_top_heads:
        heads_by_layer = {}
        for (l, h) in TOP_CORR_HEADS:
            heads_by_layer.setdefault(l, []).append(h)
    else:
        heads_by_layer = {l: list(range(n_heads)) for l in range(num_layers)}

    recalls = {1: 0, 5: 0, 10: 0}
    print(f"Inference on {len(inference_samples)} queries | "
          f"method=Method1_inline_plus | heads={'top20' if args.use_top_heads else 'all'}")

    for sample in tqdm(inference_samples):
        gold_db_name = sample["gold_db_name"]

        (tokens, block_boundaries, db_order,
         q_start, q_end,
         q_content_mask, q_plus_mask,
         instr_end) = build_prompt_tokens(
            tokenizer, sample["text"], all_db_names, dbs, plus_token_id
        )

        run_forward(model, tokens, block_boundaries, instr_end, q_start, q_end, device)

        saved = snapshot_and_clear(model, num_db_blocks, num_layers)

        corrected_scores = score_dbs_method1(
            saved, db_order, heads_by_layer, scaling,
            q_content_mask, q_plus_mask, device
        )

        ranked       = sorted(corrected_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_names = [r[0] for r in ranked]
        # import ipdb; ipdb.set_trace()
        logger.info(
            f"QID={sample['qid']} | gold={gold_db_name} | "
            f"top5={ranked_names[:5]} | scores={[round(r[1],4) for r in ranked[:5]]}"
        )

        if gold_db_name == ranked_names[0]:   recalls[1]  += 1
        if gold_db_name in ranked_names[:5]:  recalls[5]  += 1
        if gold_db_name in ranked_names[:10]: recalls[10] += 1
        else:
            logger.info(f"QID={sample['qid']} | gold '{gold_db_name}' not in top 10")

        torch.cuda.empty_cache()

    n = len(inference_samples)
    print(f"\n--- Results for {args.dataset} (N={n}) ---")
    print(f"Recall@1:  {recalls[1]/n:.4f}")
    print(f"Recall@5:  {recalls[5]/n:.4f}")
    print(f"Recall@10: {recalls[10]/n:.4f}")
    logger.info(
        f"Recall@1={recalls[1]/n:.4f} | Recall@5={recalls[5]/n:.4f} | "
        f"Recall@10={recalls[10]/n:.4f} | dataset={args.dataset}"
    )


if __name__ == "__main__":
    main()