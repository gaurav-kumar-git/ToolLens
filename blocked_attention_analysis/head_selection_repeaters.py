import argparse
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
from transformers import AutoTokenizer
import logging
from modelling_llama import LlamaForCausalLM, repeat_kv

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="head_repeaters.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_model_tokenizer(model_name, device, dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype,  
        local_files_only=True,
        attn_implementation="eager" 
    )
    model.to(device)
    model.eval()
    return tokenizer, model

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_queries_and_items(dataset):
    # train_or_dev = "dev"
    if dataset in ["spider", "bird"]:
        train_db_path = f"/scratch/gaurav/data/BIRD/formatted/{dataset}db_train_schema_info.json"
        dev_db_path = f"/scratch/gaurav/data/BIRD/formatted/{dataset}db_dev_schema_info.json"
        # queries_path = f"/scratch/gaurav/data/BIRD/formatted/sft_{dataset}_{train_or_dev}_text2sql.json"
        queries_path = f"/scratch/gaurav/data/BIRD/formatted/sft_{dataset}_dev_text2sql.json"
        with open(queries_path, "r") as f:
            test_queries = json.load(f)
        train_dbs = {}
        dev_dbs = {}
        with open(train_db_path, "r") as f:
            train_dbs.update(json.load(f))
        with open(dev_db_path, "r") as f:
            dev_dbs.update(json.load(f))
        queries = [{"text": s["text"], "gold_db_name": s["db_id"], "qid": i} for i, s in enumerate(test_queries)]
        # import ipdb; ipdb.set_trace()
        return queries, train_dbs, dev_dbs
    return [], {}, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=64)
    parser.add_argument('--dataset', type=str, default="bird")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--num_samples', type=int, default=200)
    args = parser.parse_args()
    seed_all(args.seed)
    device = "cuda:0"
    
    queries, train_dbs, dev_dbs = get_queries_and_items(args.dataset)
    db2 = {}
    count = 0
    for k, v in dev_dbs.items(): # prioritize dev dbs to avoid missing db, as the queries are using the dev set dbs
        if count >= 30:
            break
        db2[k] = v
        count += 1
    for k, v in train_dbs.items():
        if count >= 30:
            break
        db2[k] = v
        count += 1
    dbs = db2
    
    # import ipdb; ipdb.set_trace()
    random.shuffle(queries)
    eval_set_aside = queries[:1000]
    head_selection_samples = queries[1000:1500]
    tokenizer, model = load_model_tokenizer(args.model, device)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    scaling = head_dim**-0.5
    head_scores = torch.zeros(num_layers, num_heads, device='cpu')
    print(f"Starting sequential processing of {len(head_selection_samples)} queries...")
    CHUNK_SIZE = 5  # repeaters at 10 steps, as i am using 30 dbs total
    
    for idx, sample in enumerate(tqdm(head_selection_samples)):
        gold_db_name = sample["gold_db_name"]
        all_db_names = list(dbs.keys())
        random.shuffle(all_db_names)
        system_block = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert database routing system. Your task is to analyze a user's question and a list of available database schemas. "
        "You must select the most relevant database name that can answer the question.\n"
        "Do not add any other text, explanation, or formatting.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        )
        full_tokens = tokenizer(system_block, add_special_tokens=True)["input_ids"]
        instr_end = len(full_tokens)
        
        # @gauravk: the repeater concept is implemented here
        block_boundaries = []
        query_indices = [] # track the repeaters
        gold_block_idx = -1
        db_chunks = [all_db_names[i:i+CHUNK_SIZE] for i in range(0, len(all_db_names), CHUNK_SIZE)]
        block_boundaries.append({
            "span": (0, instr_end),
            "type": "instr",
            "is_gold": False,
            "db_name": "system_instr"
        })
        for chunk in db_chunks:
            for db_name in chunk:
                db_info = dbs.get(db_name, "")
                db_str = f"Database: {db_name}\nSchema: {db_info}\n"
                db_ids = tokenizer(db_str, add_special_tokens=False)["input_ids"]
                start_idx = len(full_tokens)
                full_tokens.extend(db_ids)
                end_idx = len(full_tokens)
                is_gold = (db_name == gold_db_name)
                if is_gold:
                    gold_block_idx = len(block_boundaries) # save the block of block id to calculate the gold attn score later
                block_boundaries.append({
                    "span": (start_idx, end_idx),
                    "type": "db",
                    "is_gold": is_gold
                })
            # repeater added after each chunk
            q_str = f"\nSelect the correct database name for answering the question.\nQuestion: {sample['text']}\nCorrect Database Name:\n"
            q_ids = tokenizer(q_str, add_special_tokens=False)["input_ids"]
            q_start = len(full_tokens)
            full_tokens.extend(q_ids)
            q_end = len(full_tokens)
            query_indices.append(len(block_boundaries))
            block_boundaries.append({
                "span": (q_start, q_end),
                "type": "query",
                "is_gold": False
            })
        # @gauravk: end of repeater implementation
        
        assistant_ids = tokenizer("<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)["input_ids"]
        start_ast = len(full_tokens)
        full_tokens.extend(assistant_ids)
        end_ast = len(full_tokens)
        block_boundaries.append({
            "span": (start_ast, end_ast),
            "type": "assistant",
            "is_gold": False,
            "db_name": "assistant_header"
        })
        input_ids = torch.tensor([full_tokens], device=device)
        seq_len = input_ids.shape[1]

        # @gauravk: mask start: made as causal only, but the repeaters can't see the earlier repeaters
        # import ipdb; ipdb.set_trace()
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        mask = mask.masked_fill(mask == 0, -float('inf'))  # convert to causal mask
        mask = mask.masked_fill(mask == 1, 0.0)
        for i, q_idx_current in enumerate(query_indices):
            curr_start, curr_end = block_boundaries[q_idx_current]["span"]
            for j in range (i):
                prev_q_idx = query_indices[j]
                prev_start, prev_end = block_boundaries[prev_q_idx]["span"]
                mask[curr_start:curr_end, prev_start:prev_end] = -float('inf') # filled as repeaters can't attend to previous repeaters
        # @gauravk: mask end
        
        model.config.instr_end = instr_end
        model.config.block_boundaries = [b["span"] for b in block_boundaries]
        with torch.no_grad():
            model(input_ids, attention_mask=mask.unsqueeze(0).unsqueeze(0), use_cache=True)

        # @gauravk: now accumulation/scoring logic
        # import ipdb; ipdb.set_trace()
        # try:
        #     gold_block_idx = next(i for i, b in enumerate(block_boundaries) if b["is_gold"])
        # except StopIteration:
        #     logger.warning(f'no gold block found for query {idx} with gold db {gold_db_name}')
        # for l_idx in range(num_layers):
        #     attn_module = model.model.layers[l_idx].self_attn
        #     sample_cumulative_head_score = torch.zeros(num_heads, device='cpu')
        #     gold_states = attn_module.saved_states[gold_block_idx]
        #     k_gold = gold_states["k"].to(device) # shape: [1, num_heads, gold_db_len, head_dim]
        #     num_repeaters = len(query_indices)
        #     for r_idx in range(num_repeaters):
        #         q_states = attn_module.saved_states[query_indices[r_idx]]
        #         q_rep = q_states["q"].to(device) # shape: [1, num_heads, q_len, head_dim]
        #         weights = torch.matmul(q_rep, k_gold.transpose(-1, -2)) * scaling # shape: [1, num_heads, q_len, gold_db_len]
        #         repeater_score = weights.mean(dim=-2).mean(dim=-1).squeeze(0) # avg over q_len, then avg over gold_db_len, shape: [num_heads]
        #         sample_cumulative_head_score += repeater_score.cpu()
        #     head_scores[l_idx] += (sample_cumulative_head_score / num_repeaters)
        #     attn_module.saved_states = {}
        # @gauravk: end of scoring logic, now logging the scores for this query
        
        # @gauravk: scoring logic changed: with ques_anchors and whole k, to to the softmax part too
        try:
            gold_block_idx = next(i for i, b in enumerate(block_boundaries) if b["is_gold"])
        except StopIteration:
            logger.warning(f'no gold block found for query {idx} with gold db {gold_db_name}')
        for l_idx in range(num_layers):
            attn_module = model.model.layers[l_idx].self_attn
            k_full = torch.cat([attn_module.saved_states[i]["k"] for i in range(len(block_boundaries))], dim=2).to(device) # dim=2 to concat over the sequence length dimension
            sample_cumulative_head_score = torch.zeros(num_heads, device='cpu')
            num_repeaters = len(query_indices)
            for r_idx in range(num_repeaters):
                q_idx = query_indices[r_idx]
                q_rep = attn_module.saved_states[q_idx]["q"].to(device)
                logits = torch.matmul(q_rep, k_full.transpose(-1, -2)) * scaling
                q_start, q_end = block_boundaries[q_idx]["span"]
                mask_slice = mask[q_start:q_end, :].unsqueeze(0).unsqueeze(0) # capture only the relevant slice of the mask for this repeater block
                logits += mask_slice
                probs = torch.softmax(logits, dim=-1)
                gold_start, gold_end = block_boundaries[gold_block_idx]["span"]
                gold_attn_probs = probs[:, :, :, gold_start:gold_end]
                repeater_score = gold_attn_probs.mean(dim=-2).mean(dim=-1).squeeze(0)
                sample_cumulative_head_score += repeater_score.cpu()
            head_scores[l_idx] += (sample_cumulative_head_score / num_repeaters)
            # import ipdb; ipdb.set_trace()
            attn_module.saved_states = {}
        # @gauravk: scoring logic end
            
        logger.info(f"Processed query {idx+1}/{len(head_selection_samples)} | Gold DB: {gold_db_name} | Gold Attn Score (Layer-wise): {[head_scores[l_idx].tolist() for l_idx in range(num_layers)]}")
        # import ipdb; ipdb.set_trace()
        del input_ids, mask
        torch.cuda.empty_cache()
    
    head_scores /= len(head_selection_samples)
    with open(f"full_head_scores_repeaters.json", "w") as f:
        json.dump({
            "matrix": head_scores.tolist(),
            "num_samples": len(head_selection_samples),
            "dataset": args.dataset
        }, f)
    logger.info(f"Full head scores saved to full_head_scores_repeaters.json")
    
    top_v, top_i = torch.topk(head_scores.flatten(), 20)
    print("\n--- Top 20 Heads (By Cumulative Gold Attention Strength) ---")
    results = []
    for val, idx in zip(top_v, top_i):
        l, h = divmod(idx.item(), num_heads)
        print(f"Layer {l}, Head {h} | Cumulative Score: {val.item():.4f}")
        results.append({"layer": int(l), "head": int(h), "cumulative_score": float(val.item())})

    with open(f"top_heads_{args.dataset}_{args.seed}_repeaters.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Top heads saved to top_heads_{args.dataset}_{args.seed}_repeaters.json")

if __name__ == "__main__":
    main()