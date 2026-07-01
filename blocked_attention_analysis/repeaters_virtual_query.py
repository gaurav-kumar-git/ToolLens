import argparse
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from modelling_llama import LlamaForCausalLM
from transformers.cache_utils import DynamicCache
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="repeaters_inference_virtual_query.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)   

def load_model_tokenizer(model_name, device, dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, local_files_only=True, attn_implementation="eager"
    )
    model.to(device).eval()
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
    parser.add_argument('--top_heads_path', type=str, default="/scratch/gaurav/SchemaLens/blocked_attention_analysis/top_heads_bird_64_repeaters.json")
    args = parser.parse_args()

    seed_all(args.seed)
    device = "cuda:0"
    
    with open(args.top_heads_path, "r") as f:
        top_heads_data = json.load(f)
    
    heads_by_layer = {}
    for item in top_heads_data:
        l, h = item['layer'], item['head']
        if l not in heads_by_layer: heads_by_layer[l] = []
        heads_by_layer[l].append(h)

    queries, train_dbs, dev_dbs = get_queries_and_items(args.dataset)
    # db2 = {}
    # count = 0
    # for k, v in dev_dbs.items():
    #     if count >= 30: break
    #     db2[k] = v
    #     count += 1
    # for k, v in train_dbs.items():
    #     if count >= 30: break
    #     db2[k] = v
    #     count += 1
    # dbs = db2
    
    queries, train_dbs, dev_dbs = get_queries_and_items(args.dataset)
    
    dbs = {}
    dbs.update(dev_dbs)    
    dbs.update(train_dbs) 
    
    print(f"Loaded a total of {len(dbs)} databases.")
    
    random.shuffle(queries)
    inference_samples = queries[:1000] 
    tokenizer, model = load_model_tokenizer(args.model, device)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    scaling = head_dim**-0.5
    CHUNK_SIZE = 5

    print("Building and caching the Static DB Context...")
    system_block = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert database routing system. Your task is to analyze a user's question and a list of available database schemas. "
        "You must select the most relevant database name that can answer the question.\n"
        "Do not add any other text, explanation, or formatting.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
    )
    static_ids = tokenizer(system_block, add_special_tokens=True)["input_ids"]
    
    db_blocks_static = []
    chunk_end_positions = []
    all_db_names = list(dbs.keys())
    db_chunks = [all_db_names[i:i+CHUNK_SIZE] for i in range(0, len(all_db_names), CHUNK_SIZE)]
    
    # @gaurav k: the static chunk made
    for chunk in db_chunks:
        for db_name in chunk:
            db_info = dbs.get(db_name, "")
            db_str = f"Database: {db_name}\nSchema: {db_info}\n"
            db_ids = tokenizer(db_str, add_special_tokens=False)["input_ids"]
            
            start_idx = len(static_ids)
            static_ids.extend(db_ids)
            end_idx = len(static_ids)
            
            db_blocks_static.append({
                "span": (start_idx, end_idx),
                "db_name": db_name
            })
        chunk_end_positions.append(len(static_ids))
        
    static_tensor = torch.tensor([static_ids], device=device)
    S = len(static_ids)
    # @gaurav k: end
    
    # @gaurav k : making the KV cache static, in one pass it explodes hence done part by part
    PREFILL_CHUNK_SIZE = 4096
    static_pkv_raw = None
    with torch.no_grad():
        for i in tqdm(range(0, S, PREFILL_CHUNK_SIZE), desc="Caching DBs"):
            chunk_tensor = static_tensor[:, i:i+PREFILL_CHUNK_SIZE]
            static_outputs = model(chunk_tensor, past_key_values=static_pkv_raw, use_cache=True)
            static_pkv_raw = static_outputs.past_key_values
            
    static_pkv_tuple = ()
    if hasattr(static_pkv_raw, "key_cache"): 
        for i in range(len(static_pkv_raw.key_cache)):
            static_pkv_tuple += ((static_pkv_raw.key_cache[i], static_pkv_raw.value_cache[i]),)
    else:
        static_pkv_tuple = static_pkv_raw

    k_static_dict = {}
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_rep = num_q_heads // num_kv_heads
    
    for l_idx, head_list in heads_by_layer.items():
        k_layer = static_pkv_tuple[l_idx][0] 
        k_layer_repeated = torch.repeat_interleave(k_layer, num_rep, dim=1)
        k_static_dict[l_idx] = k_layer_repeated[:, head_list, :, :]
    
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "saved_states"):
            layer.self_attn.saved_states = {}
    # @gaurav k: end of static KV cache creation, dict creation based on heads and cleared the saved states
    
    
    recalls = {1: 0, 5: 0, 10: 0}
    print(f"Starting extremely fast inference on {len(inference_samples)} queries...")
    
    for idx, sample in enumerate(tqdm(inference_samples)):
        gold_db_name = sample["gold_db_name"]
        
        q_str = f"\nSelect the correct database name for answering the question.\nQuestion: {sample['text']}\nCorrect Database Name:\n"
        q_ids = tokenizer(q_str, add_special_tokens=False)["input_ids"]
        
        dynamic_ids = []
        dynamic_pos_ids = []
        query_block_boundaries = []
        
        # @gaurav k: virtual repeaters made
        for i, chunk_end in enumerate(chunk_end_positions):
            q_start = len(dynamic_ids)
            dynamic_ids.extend(q_ids)
            q_end = len(dynamic_ids)
            query_block_boundaries.append((q_start, q_end))
            dynamic_pos_ids.extend(list(range(chunk_end, chunk_end + len(q_ids)))) # added dynamic pos with the repeaters at reg intervals
            
        ast_ids = tokenizer("<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)["input_ids"]
        ast_start = len(dynamic_ids)
        dynamic_ids.extend(ast_ids)
        ast_end = len(dynamic_ids)
        ast_pos_start = chunk_end_positions[-1] + len(q_ids)
        dynamic_pos_ids.extend(list(range(ast_pos_start, ast_pos_start + len(ast_ids))))
        D = len(dynamic_ids)
        # @gaurav k: end of virtual repeaters placement
        
        # @gaurav k: I need attn only from these query_vrtls to static and from assistant to static+last_query, and causal-attn within each block (query blocks and assistant block)
        mask = torch.full((1, 1, D, S + D), -float('inf'), device=device)
        
        for i, (q_start, q_end) in enumerate(query_block_boundaries):
            chunk_end = chunk_end_positions[i]
            mask[0, 0, q_start:q_end, 0:chunk_end] = 0.0
            for row in range(q_start, q_end):
                # this helps unblock the attn from all virtual query to the chunks
                # still v_query to past v_query is blocked, we never opened it
                mask[0, 0, row, S + q_start : S + row + 1] = 0.0
        
        last_q_start, last_q_end = query_block_boundaries[-1]
        mask[0, 0, ast_start:ast_end, 0:S] = 0.0
        mask[0, 0, ast_start:ast_end, S + last_q_start : S + last_q_end] = 0.0
        for row in range(ast_start, ast_end):
            mask[0, 0, row, S + ast_start : S + row + 1] = 0.0
            
        # import ipdb; ipdb.set_trace()
        dynamic_tensor = torch.tensor([dynamic_ids], device=device)
        dynamic_pos_tensor = torch.tensor([dynamic_pos_ids], device=device)
        # @gaurav k: attn mask end and synaamic tensor or virtual queries with position ids done
        
        model.config.block_boundaries = [(0, D)] 
        cloned_tuple = tuple(
            (k.clone(), v.clone()) for k, v in static_pkv_tuple
        )
        cloned_pkv = DynamicCache.from_legacy_cache(cloned_tuple)
        with torch.no_grad():
            model(
                input_ids=dynamic_tensor, 
                attention_mask=mask,
                position_ids=dynamic_pos_tensor,
                past_key_values=cloned_pkv, 
                use_cache=True
            )
        
        sample_db_scores = {name: 0.0 for name in all_db_names}
        for l_idx, head_list in heads_by_layer.items():
            attn_module = model.model.layers[l_idx].self_attn
            q_dynamic = attn_module.saved_states[0]["q"][:, head_list, :, :].to(device)
            k_static = k_static_dict[l_idx].to(device)
            logits = torch.matmul(q_dynamic, k_static.transpose(-1, -2)) * scaling
            for i, (q_start, q_end) in enumerate(query_block_boundaries):
                chunk_end = chunk_end_positions[i]
                q_logits = logits[:, :, q_start:q_end, 0:chunk_end]
                probs = torch.softmax(q_logits, dim=-1)
                for d_idx, db_dict in enumerate(db_blocks_static):
                    chunk_id = d_idx // CHUNK_SIZE
                    if chunk_id <= i:
                        db_name = db_dict["db_name"]
                        d_start, d_end = db_dict["span"] 
                        db_probs = probs[:, :, :, d_start:d_end]
                        score = db_probs.mean(dim=-2).mean(dim=-1).sum()
                        sample_db_scores[db_name] += score.item()
            attn_module.saved_states = {} 
            
        del cloned_pkv
            
        ranked_results = sorted(sample_db_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_names = [item[0] for item in ranked_results]
        top_scores = [item[1] for item in ranked_results[:10]]
        logger.info(f"Query ID: {sample['qid']} | Gold: {gold_db_name} | Ranked: {ranked_names[:10]} | Scores: {top_scores}")
        
        if gold_db_name == ranked_names[0]: recalls[1] += 1
        if gold_db_name in ranked_names[:5]: recalls[5] += 1
        if gold_db_name in ranked_names[:10]: recalls[10] += 1
        else:
            logger.info(f"Query ID: {sample['qid']} | Gold DB '{gold_db_name}' not found in top 10 ranked results.")

    n = len(inference_samples)
    print(f"\n--- Results for {args.dataset} (N={n}) ---")
    print(f"Recall@1:  {recalls[1]/n:.4f}")
    print(f"Recall@5:  {recalls[5]/n:.4f}")
    print(f"Recall@10: {recalls[10]/n:.4f}")
    logger.info(f"Final Recall@1: {recalls[1]/n:.4f} | Recall@5: {recalls[5]/n:.4f} | Recall@10: {recalls[10]/n:.4f}")
    
if __name__ == "__main__":
    main()