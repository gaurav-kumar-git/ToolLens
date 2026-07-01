# os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import json 
import time
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
import os
from SchemaLens.rankGPT_experiments.utils import load_model_tokenizer
from SchemaLens.rankGPT_experiments.rank_gpt_reranker import RankGPTModel

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--dataset', type=str, default = "spider")
parser.add_argument('--N', type=int, default=200)
parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
parser.add_argument('--limit', type=int, default=1000)
parser.add_argument('--prompt_variation', type=str, default="STRING") # or  "ID"
parser.add_argument('--window_size', type=int, default=80, help="Docs per LLM prompt")
parser.add_argument('--stride', type=int, default=40, help="Sliding window shift size")
parser.add_argument('--output_dir', type=str, default="./constrained/results")
args = parser.parse_args()


def get_queries_and_items(dataset):
    if dataset in ["spider", "bird"]:
        db_paths = [
            f"/scratch/gaurav/rankGPT_experiments/data/{dataset}/create/{dataset}db_train_schema_info.json",
            f"/scratch/gaurav/rankGPT_experiments/data/{dataset}/create/{dataset}db_dev_schema_info.json"
            ]
        queries_path = f"/scratch/gaurav/rankGPT_experiments/data/{dataset}/codes/sft_{dataset}_dev_text2sql.json"
        with open(queries_path) as f:
            test_queries = json.load(f)

        queries_path = f"/scratch/gaurav/rankGPT_experiments/data/{dataset}/codes/sft_{dataset}_train_text2sql.json"
        with open(queries_path) as f:
            train_queries = json.load(f)
        # train_queries = []
        dbs = {}
        for db_path in db_paths:
            with open(db_path) as f:
                dbs |= json.load(f)  
        
        queries = []
        for idx, sample in enumerate(train_queries + test_queries):
            queries.append({
                "text": sample["text"],
                "gold_db_name": sample["db_id"],
                "qid": idx
                }
            )
    if dataset in ["toole"]:
        tool_path = "/scratch/gaurav/rankGPT_experiments/data/toole/all_clean_data.csv"   
        tool_desc_path = "/scratch/gaurav/rankGPT_experiments/data/toole/plugin_des.json"
        df =  pd.read_csv(tool_path)
        with open(tool_desc_path) as f:
            dbs = json.load(f)

        queries = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            queries.append({
                "text": row["Query"],
                "gold_db_name": row["Tool"],
                "qid": idx
                }
            )
    return queries, dbs 


if __name__ == '__main__':
    
    os.makedirs(args.output_dir, exist_ok=True)
    seed_all(seed=args.seed)
    queries, dbs = get_queries_and_items(dataset=args.dataset)
    
    random.shuffle(queries)
    shuffled_db_names = list(dbs.keys())
    random.shuffle(shuffled_db_names)
    
    N_queries = random.sample(queries, args.N)
    N_queries_id = []
    N_queries_id = [example["qid"] for example in N_queries]
    print("N_queries_id: ", N_queries_id)
    # print()
    
    candidate_names = shuffled_db_names
    candidate_texts = []
    for name in candidate_names:
        content = dbs[name]
        doc_str = f"Database: {name}. Schema: {json.dumps(content)}" if args.dataset in ["bird", "spider"] else f"Tool: {name}. Description: {content}"
        candidate_texts.append(doc_str)

    
    # import ipdb; ipdb.set_trace()
    TOTAL_DOCS = len(candidate_texts)
    print(f"Total Documents in Corpus: {TOTAL_DOCS}")
    # import ipdb; ipdb.set_trace()

    # tokenizer, model = load_model_tokenizer(model_name=args.model, device="cuda", dtype=torch.bfloat16)
    reranker = RankGPTModel(base_llm_name=args.model, sliding_window_size=args.window_size, sliding_window_stride=args.stride)
    # Without sliding window
    # reranker = RankGPTModel(base_llm_name=args.model, sliding_window_size=TOTAL_DOCS, sliding_window_stride=TOTAL_DOCS)
    # reranker.model = model
    # reranker.tokenizer = tokenizer

    eval_queries = queries[:args.limit] if args.limit > 0 else queries
    
    exclude_set = set(N_queries_id)
    eval_queries = [q for q in eval_queries if q["qid"] not in exclude_set]

    print(f"Reranking full corpus for {len(eval_queries)} queries...")

    save_path = os.path.join(args.output_dir, f"{args.dataset}_qwen_1M_constrained_sw_rerank_results.json")
    final_logs = []
    recalls = {1: 0, 5: 0, 10: 0}

    start_idx=0
    
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r') as f:
                existing_data = json.load(f)
                
                if isinstance(existing_data, dict):
                    final_logs = existing_data.get("details", [])
                else:
                    final_logs = existing_data
                
                start_idx = len(final_logs)
                
                if start_idx > 0:
                    print(f"Resuming from checkpoint. Found {start_idx} processed queries.")
                    for entry in final_logs:
                        for k in recalls.keys():
                            if entry["gold"] in entry["final_ranked_names"][:k]:
                                recalls[k] += 1
        except Exception as e:
            print(f"Failed to load checkpoint ({e}). Starting from scratch.")
            final_logs = []
            start_idx = 0

    print(f"Reranking for {len(eval_queries)} queries (Skipping first {start_idx})...")

    for i in tqdm(range(start_idx, len(eval_queries))):
        sample = eval_queries[i]
        question = sample["text"]
        gold_name = sample["gold_db_name"]

        try:
            sorted_indices = reranker.rerank(question, candidate_texts)
            final_names = [candidate_names[idx] for idx in sorted_indices]
        except Exception as e:
            print(f"\nError on QID {sample['qid']}: {e}")
            final_names = candidate_names 

        for k in recalls.keys():
            if gold_name in final_names[:k]:
                recalls[k] += 1

        final_logs.append({
            "qid": sample["qid"],
            "query": question,
            "gold": gold_name,
            "rankgpt_top_candidate": final_names[0],
            "final_ranked_names": final_names
        })

        if (i + 1) % 10 == 0:
            with open(save_path, 'w') as f:
                json.dump(final_logs, f, indent=2)
                
    final_metrics = {f"Recall@{k}": v/len(eval_queries) for k, v in recalls.items()}
    
    final_output = {
        "config": vars(args),
        "metrics": final_metrics,
        "details": final_logs
    }

    with open(save_path, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\nProcessing complete. Total processed: {len(final_logs)}")
    print(f"Final Recall@1: {final_metrics['Recall@1']:.4f} | Recall@10: {final_metrics['Recall@10']:.4f}")