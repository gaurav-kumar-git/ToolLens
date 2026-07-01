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
from modelling_llama_v2 import LlamaForCausalLM, repeat_kv
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="top_head_max_on_gold_PCW_sum_dbs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# logging.basicConfig(
#     filename="top_head_max_on_gold_PCW_max_len.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

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
    train_or_dev = "dev"
    if dataset in ["spider", "bird"]:
        db_path = f"/scratch/gaurav/data/BIRD/formatted/{dataset}db_{train_or_dev}_schema_info.json"
        queries_path = f"/scratch/gaurav/data/BIRD/formatted/sft_{dataset}_{train_or_dev}_text2sql.json"
        with open(queries_path, "r") as f:
            test_queries = json.load(f)
        with open(db_path, "r") as f:
            dbs = json.load(f)  
        queries = [{"text": s["text"], "gold_db_name": s["db_id"], "qid": i} for i, s in enumerate(test_queries)]
        return queries, dbs
    return [], {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=64)
    parser.add_argument('--dataset', type=str, default="bird")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--num_samples', type=int, default=500)
    args = parser.parse_args()

    seed_all(args.seed)
    device = "cuda:0"
    queries, dbs = get_queries_and_items(args.dataset)
    
    random.seed(args.seed)
    random.shuffle(queries)
    head_selection_samples = queries[1000:1500]
    
    tokenizer, model = load_model_tokenizer(args.model, device)
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    scaling = (model.config.hidden_size // num_heads)**-0.5
    head_scores = torch.zeros(num_layers, num_heads, device='cpu')

    print(f"Extracting top heads (Normal Causal) from {len(head_selection_samples)} samples...")

    for sample in tqdm(head_selection_samples):
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
        
        pos_ids_list = list(range(instr_end)) 
        db_boundaries = []
        max_db_len = 0
        
        total_db_len = 0 # @gk 
        for name in all_db_names:
            db_ids = tokenizer(f"Database: {name}\nSchema: {dbs.get(name, '')}\n", add_special_tokens=False)["input_ids"]
            db_len = len(db_ids)
            s = len(full_tokens)
            full_tokens.extend(db_ids)
            db_boundaries.append((s, len(full_tokens)))
            
            pos_ids_list.extend(range(instr_end, instr_end + db_len))
            max_db_len = max(max_db_len, db_len)
            # total_db_len += db_len  # @gk changed to put query atthe same pos index as in normal prompt
        
        
            
        query_text = (
            f"Select the correct database name for answering the question that follows.\nQuestion: {sample['text']} \nCorrect database name:"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        query_ids = tokenizer(query_text, add_special_tokens=False)["input_ids"]
        query_start = len(full_tokens)
        full_tokens.extend(query_ids)
        
        query_pos_start = instr_end + max_db_len
        # query_pos_start = instr_end + total_db_len # @gk changed to put query at the same pos index as in normal prompt
        pos_ids_list.extend(range(query_pos_start, query_pos_start + len(query_ids)))

        input_ids = torch.tensor([full_tokens], device=device)
        position_ids = torch.tensor([pos_ids_list], device=device)
        seq_len = input_ids.shape[1]

        # pcw mask construction
        # start with full negative infinity (masked)
        pcw_mask = torch.full((seq_len, seq_len), -float('inf'), device=device)
        
        # causal self-attention
        inst_submask = torch.tril(torch.ones((instr_end, instr_end), device=device))
        pcw_mask[:instr_end, :instr_end] = (1.0 - inst_submask) * -float('inf')
        
        # databases: instruction + causal self
        for (s, e) in db_boundaries:
            pcw_mask[s:e, :instr_end] = 0.0 # can see instruction
            db_len = e - s
            db_submask = torch.tril(torch.ones((db_len, db_len), device=device))
            pcw_mask[s:e, s:e] = (1.0 - db_submask) * -float('inf') # causal self-see
            
        # query: see instruction + all dbs + causal self
        pcw_mask[query_start:, :query_start] = 0.0 # can see instruction and all dbs
        q_len = seq_len - query_start
        q_submask = torch.tril(torch.ones((q_len, q_len), device=device))
        pcw_mask[query_start:, query_start:] = (1.0 - q_submask) * -float('inf')

        model.config.instr_end = instr_end
        model.config.db_boundaries = db_boundaries
        model.config.query_start = query_start
        model.config.block_boundaries = True 
        
        with torch.no_grad():
            # position_ids and the custom pcw_mask
            model(
                input_ids, 
                attention_mask=pcw_mask.unsqueeze(0).unsqueeze(0), 
                position_ids=position_ids,
                use_cache=True
            )

        for l_idx in range(num_layers):
            attn_module = model.model.layers[l_idx].self_attn
            
            q_query = attn_module.saved_states["query_q"].to(device) 
            k_instr = attn_module.saved_states["instr_k"].to(device)
            k_dbs = torch.cat([k.to(device) for k in attn_module.saved_states["db_k"]], dim=2)
            k_full = torch.cat([k_instr, k_dbs], dim=2) 
            
            q_len = q_query.shape[2]
            db_len = k_dbs.shape[2]
            
            weights = torch.softmax(torch.matmul(q_query, k_full.transpose(-1, -2)) * scaling, dim=-1)
            
            gold_pos = all_db_names.index(gold_db_name)
            db_lens_before = sum([e-s for (s, e) in db_boundaries[:gold_pos]]) 
            gold_start_in_keys = instr_end + db_lens_before # physical index of gold db in the combined keys
            gold_end_in_keys = gold_start_in_keys + (db_boundaries[gold_pos][1] - db_boundaries[gold_pos][0])
            
            gold_attn_slice = weights[:, :, :, gold_start_in_keys:gold_end_in_keys]
            total_gold_attn = gold_attn_slice.sum(dim=[-1, -2]).squeeze(0) / (q_len * db_len) # shape: [num_heads]
            head_scores[l_idx] += total_gold_attn.cpu()
            
            attn_module.saved_states = {}
            
        del input_ids, pcw_mask
        torch.cuda.empty_cache()
    
    
    with open(f"full_head_scores_{args.dataset}_PCW_max_len.json", "w") as f:
        json.dump({
            "matrix": head_scores.tolist(),
            "num_samples": len(head_selection_samples),
            "dataset": args.dataset
        }, f)
    # with open(f"full_head_scores_{args.dataset}_PCW_sum_dbs.json", "w") as f:
    #     json.dump({
    #         "matrix": head_scores.tolist(),
    #         "num_samples": len(head_selection_samples),
    #         "dataset": args.dataset
    #     }, f)
        
    logger.info(f"Completed head score extraction for {len(head_selection_samples)} samples on {args.dataset} dataset.")

    top_v, top_i = torch.topk(head_scores.flatten(), 20)
    print("\n--- Top 20 Heads (By Cumulative Gold Attention Strength) ---")
    results = []
    for val, idx in zip(top_v, top_i):
        l, h = divmod(idx.item(), num_heads)
        print(f"Layer {l}, Head {h} | Cumulative Score: {val.item():.4f}")
        results.append({"layer": int(l), "head": int(h), "cumulative_score": float(val.item())})

    # with open(f"top_heads_{args.dataset}_{args.seed}_PCW_sum_dbs.json", "w") as f:
    #     json.dump(results, f, indent=4)
        
    with open(f"top_heads_{args.dataset}_{args.seed}_PCW_max_len.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Top heads saved to top_heads_{args.dataset}_{args.seed}_PCW_sum_dbs.json")

if __name__ == "__main__":
    main()