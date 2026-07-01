import argparse
import json
import  torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from modelling_llama_v2 import LlamaForCausalLM
import logging

# Set up logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename="blockwise_attention_PCW_top_20_heads_max_len.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="blockwise_attention_PCW_top_20_heads_sum_dbs_2.log",
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
    dev_db_path = f"/scratch/gaurav/data/BIRD/formatted/{dataset}db_dev_schema_info.json"
    # train_db_path = f"/scratch/gaurav/data/BIRD/formatted/{dataset}db_train_schema_info.json"
    queries_path = f"/scratch/gaurav/data/BIRD/formatted/sft_{dataset}_dev_text2sql.json"
    with open(queries_path, "r") as f: test_queries = json.load(f)
    with open(dev_db_path, "r") as f: dbs = json.load(f)
    # with open(train_db_path, "r") as f: dbs.update(json.load(f))
    queries = [{"text": s["text"], "gold_db_name": s["db_id"], "qid": i} for i, s in enumerate(test_queries)]
    return queries, dbs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=64)
    parser.add_argument('--dataset', type=str, default="bird")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--top_heads_path', type=str, default="/scratch/gaurav/SchemaLens/blocked_attention_analysis/top_heads_bird_64_PCW_max_len_2.json")
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

    queries, dbs = get_queries_and_items(args.dataset)
    random.shuffle(queries)
    inference_samples = queries[:1000] 

    tokenizer, model = load_model_tokenizer(args.model, device)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    scaling = head_dim**-0.5

    recalls = {1: 0, 5: 0, 10: 0}
    print(f"Inference on {len(inference_samples)} queries using Top 20 heads and New Mask...")

    for idx, sample in enumerate(tqdm(inference_samples)):
        gold_db_name = sample["gold_db_name"]
        all_db_names = list(dbs.keys())
        
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
        
        # total_db_len = 0 # @gk 
        for name in all_db_names:
            db_ids = tokenizer(f"Database: {name}\nSchema: {dbs.get(name, '')}\n", add_special_tokens=False)["input_ids"]
            db_len = len(db_ids)
            s = len(full_tokens)
            full_tokens.extend(db_ids)
            db_boundaries.append((s, len(full_tokens)))
            
            pos_ids_list.extend(range(instr_end, instr_end + db_len))
            if db_len > max_db_len:
                max_db_len = db_len
            # total_db_len += db_len  # @gk changed to put query at the same pos index as in normal prompt
             
                
            
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

        # pcw attention mask construction
        # init with large negs
        mask = torch.full((seq_len, seq_len), -float('inf'), device=device)
        
        # causal mask on instr
        inst_submask = torch.tril(torch.ones((instr_end, instr_end), device=device))
        mask[:instr_end, :instr_end] = (1.0 - inst_submask) * -float('inf')
        
        # db mask: db can see itself and the instr, but not other dbs
        for (s, e) in db_boundaries:
            # can see instr
            mask[s:e, :instr_end] = 0.0
            # can see itself
            db_len = e - s
            db_submask = torch.tril(torch.ones((db_len, db_len), device=device))
            mask[s:e, s:e] = (1.0 - db_submask) * -float('inf')
            
        # query can see everything 
        mask[query_start:, :] = 0.0 # query can see everything (instr + all dbs) but here causal mask is violated
        # we need to apply causal mask within the query tokens themselves to prevent information leakage
        q_len = seq_len - query_start
        q_submask = torch.tril(torch.ones((q_len, q_len), device=device))
        mask[query_start:, query_start:] = (1.0 - q_submask) * -float('inf')

        # import ipdb; ipdb.set_trace()
        
        # update model config for attention mask and position ids
        model.config.instr_end = instr_end
        model.config.db_boundaries = db_boundaries
        model.config.query_start = query_start
        model.config.block_boundaries = True
        
        with torch.no_grad():
            model(
                input_ids, 
                attention_mask=mask.unsqueeze(0).unsqueeze(0), 
                position_ids=position_ids,
                use_cache=True
            )

        final_db_scores = torch.zeros(len(all_db_names), device=device)
        
        for l_idx, head_list in heads_by_layer.items():
            attn_module = model.model.layers[l_idx].self_attn
            
            q_query = attn_module.saved_states["query_q"][:, head_list, :, :].to(device)
            k_instr = attn_module.saved_states["instr_k"][:, head_list, :, :].to(device)
            
            query_len = q_query.shape[2]
            num_heads = q_query.shape[1]
            
            for b_idx, k_db_raw in enumerate(attn_module.saved_states["db_k"]):
                # import ipdb; ipdb.set_trace()
                k_db = k_db_raw[:, head_list, :, :].to(device)
                db_len = k_db.shape[2]
                
                k_combined = torch.cat([k_instr, k_db], dim=2)
                weights = torch.softmax(torch.matmul(q_query, k_combined.transpose(-1, -2)) * scaling, dim=-1)
    
                # ignore the instr part and focus on db part
                instr_len = k_instr.shape[2]
                db_weights = weights[:, :, :, instr_len:] # get only the db_tokens weight
                
                # sum the attention directed to the db
                # we sum across the key-tokens (dim -1) and average across query-tokens (dim -2)
                db_score = db_weights.sum() / (query_len * num_heads * db_len)
                # import ipdb; ipdb.set_trace()
                final_db_scores[b_idx] += db_score
            
            # import ipdb; ipdb.set_trace()
            attn_module.saved_states = {}

        _, top_indices = torch.topk(final_db_scores, k=min(10, len(all_db_names)))
        ranked_names = [all_db_names[i.item()] for i in top_indices]
        # import ipdb; ipdb.set_trace()
        
        logger.info(f"Query ID: {sample['qid']} | Gold: {gold_db_name} | Ranked: {ranked_names} | Scores: {[final_db_scores[i].item() for idx, i in enumerate(top_indices)]}")
        
        if gold_db_name == ranked_names[0]: recalls[1] += 1
        if gold_db_name in ranked_names[:5]: recalls[5] += 1
        if gold_db_name in ranked_names[:10]: recalls[10] += 1

        torch.cuda.empty_cache()

    n = len(inference_samples)
    print(f"\n--- Results for {args.dataset} (N=1000) ---")
    print(f"Recall@1:  {recalls[1]/n:.4f}")
    print(f"Recall@5:  {recalls[5]/n:.4f}")
    print(f"Recall@10: {recalls[10]/n:.4f}")
    logger.info(f"Final Results for {args.dataset} | Recall@1: {recalls[1]/n:.4f} | Recall@5: {recalls[5]/n:.4f} | Recall@10: {recalls[10]/n:.4f}")

if __name__ == "__main__":
    main()