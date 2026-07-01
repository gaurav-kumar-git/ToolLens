import os
import json
from sympy import Q
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from modeling_qwen3 import Qwen3ForCausalLM

BIRD_DIR = '/scratch/gaurav/data/BIRD'
OUTPUT_DIR = './pos_bias_results_qwen_m2'
os.makedirs(OUTPUT_DIR, exist_ok=True)
npz_dir = os.path.join(OUTPUT_DIR, "head_analysis_npz")
os.makedirs(npz_dir, exist_ok=True)

# QUERIES_TO_AGGREGATE = [345, 532, 785,  123, 654, 987, 234, 698, 1412] 
# QUERIES_TO_PLOT = [532, 785, 367, 567, 745, 1054, 1245, 1234, 1517, 867, 543, 512, 789, 432]
QUERIES_TO_AGGREGATE = [345, 532, 785, 367] 
QUERIES_TO_PLOT = [567, 745, 1054]
NUM_DBS = 30

print("Loading Model & Tokenizer...")

def load_model_tokenizer(model_name, device, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    rope_scaling = {
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "type": "yarn"
    }

    model = Qwen3ForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype, 
        local_files_only=True, 
        rope_scaling=rope_scaling,
        attn_implementation="eager",
        device_map=device 
    )
    model.eval()
    return tokenizer, model

def get_bird_data():
    def load_json(p):
        with open(p, 'r') as f: return json.load(f)
    data = load_json(os.path.join(BIRD_DIR, 'dev', 'dev.json')) + load_json(os.path.join(BIRD_DIR, 'train', 'train.json'))
    tables = load_json(os.path.join(BIRD_DIR, 'dev', 'dev_tables.json')) + load_json(os.path.join(BIRD_DIR, 'train', 'train_tables.json'))
    sql_map = load_json('/scratch/gaurav/data/BIRD/formatted/birddb_dev_schema_info.json')
    sql_map.update(load_json('/scratch/gaurav/data/BIRD/formatted/birddb_train_schema_info.json'))
    return data, {db['db_id']: db for db in tables}, sql_map

all_questions, db_schemas, sql_map = get_bird_data()
device = 'cuda:0'
tokenizer, model = load_model_tokenizer('Qwen/Qwen3-8B', device)
unique_qids = sorted(list(set(QUERIES_TO_AGGREGATE + QUERIES_TO_PLOT)))
head_dim = model.config.hidden_size // model.config.num_attention_heads
scaling = head_dim**-0.5

query_success_matrices = []   
query_corrected_matrices = [] 
test_query_full_data = {} 

for q_id in unique_qids:
    print(f"\nProcessing QID: {q_id}")
    item = next(i for i in all_questions if str(i.get('question_id')) == str(q_id))
    gold_db, nl_query = item['db_id'], item['question']
    distractors = sorted([d for d in db_schemas.keys() if d != gold_db])[:NUM_DBS-1]
    win_count_raw = np.zeros((32, 32)) 
    win_count_corr = np.zeros((32, 32))
    if q_id in QUERIES_TO_PLOT: test_query_full_data[q_id] = {}

    for pos in tqdm(range(NUM_DBS), desc=f"Positional Sweep"):
        ordered_dbs = distractors[:pos] + [gold_db] + distractors[pos:]
        system_block = (
           "<|im_start|>system\n\n"
            "You are an expert database routing system. Your task is to analyze a user's question and a list of available database schemas. "
            "You must select the most relevant database name that can answer the question.\n"
            "Do not add any other text, explanation, or formatting.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        full_tokens = tokenizer(system_block, add_special_tokens=True)["input_ids"]
        instr_end = len(full_tokens)
    
        block_boundaries = []
        block_boundaries.append({
            "span": (0, instr_end),
            "type": "instr",
            "is_gold": False,
            "db_name": "system_instr"
        }) # for making whole 'k' we need it
        
        for db_name in ordered_dbs:
            db_ids = tokenizer(f"Database: {db_name}\nSchema: {sql_map.get(db_name, '')}\n", add_special_tokens=False)["input_ids"]
            start_idx = len(full_tokens)
            full_tokens.extend(db_ids)
            # save block boundaries for q, k states...
            block_boundaries.append({
                    "span": (start_idx,len(full_tokens)),
                    "type": "db",
                    "db_name": db_name
            })

        # save the anchor text q, k states too...
        at_text = "Now, please output ONLY the correct database_id for the query below.\n"
        at_ids = tokenizer(at_text, add_special_tokens=False)["input_ids"]
        at_start = len(full_tokens)
        full_tokens.extend(at_ids)
        at_end = len(full_tokens)
        block_boundaries.append({
                    "span": (at_start, at_end),
                    "type": "at",
                    "db_name": ' '
        }) # to construct whole 'k'
        
        # save question_states too...
        q_text = f"Select the correct database name for answering the question that follows.\nQuestion: {nl_query} \nCorrect database name:"
        q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]
        q_start = len(full_tokens)
        full_tokens.extend(q_ids)
        q_end = len(full_tokens)
        block_boundaries.append({
                    "span": (q_start, q_end),
                    "type": "question",
                    "db_name": ''
        }) # to construct whole 'k'
        
        assistant_ids = tokenizer("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
        ass_start = len(full_tokens)
        full_tokens.extend(assistant_ids)
        ass_end = len(full_tokens)
        block_boundaries.append({
                    "span": (ass_start, ass_end),
                    "type": "assistant",
                    "db_name": ''
        }) # save to construct whole 'k'
        
        input_ids = torch.tensor([full_tokens], device="cuda:0")

        model.config.instr_end = instr_end
        model.config.at_start = at_start 
        model.config.at_end = at_end
        model.config.q_start = q_start   
        model.config.q_end = q_end
        model.config.block_boundaries = block_boundaries
        
        with torch.no_grad():
            model(input_ids, use_cache=True)

        pos_raw_density = np.zeros((30, 32, 32)) # [db_idx, layer, head]
        pos_corr_density = np.zeros((30, 32, 32))
        db_lengths = [
            block["span"][1] - block["span"][0]
            for block in block_boundaries
            if block["type"] == "db"
        ]
        avg_len = sum(db_lengths) / len(db_lengths)
        db_blocks = [
            block["span"] for block in model.config.block_boundaries
            if block["type"] == "db"
        ]
        for l_idx in range(32):
            attn_module = model.model.layers[l_idx].self_attn
            k_full = torch.cat([attn_module.saved_states[block["type"] if block["type"] in ["question", "at", "assistant"] else idx]["k"] for idx, block in enumerate(block_boundaries)], dim=2).to(device)            
            q_question = attn_module.saved_states["question"]["q"].to(device)
            q_anchor = attn_module.saved_states["at"]["q"].to(device) 
            
            q_len_main = q_question.shape[2]
            q_len_anc = q_anchor.shape[2]
            q_mid = (model.config.q_start + model.config.q_end) / 2
            a_mid = (model.config.at_start + model.config.at_end) / 2
            
            logits_q_full = torch.matmul(q_question, k_full.transpose(-1, -2)) * scaling
            logits_a_full = torch.matmul(q_anchor, k_full.transpose(-1, -2)) * scaling
            probs_q = torch.softmax(logits_q_full, dim=-1)
            probs_a = torch.softmax(logits_a_full, dim=-1)
            for b_idx, (b_start, b_end) in enumerate(db_blocks):
                w_prob_block = probs_q[0, :, :, b_start:b_end] 
                w_anc_prob_block = probs_a[0, :, :, b_start:b_end]
                db_len = b_end - b_start
                q_dens = w_prob_block.mean(dim=-2).mean(dim=-1).cpu().float().numpy()
                a_dens = w_anc_prob_block.mean(dim=-2).mean(dim=-1).cpu().float().numpy()
                # manual calc (method 8 from original)
                db_mid = (b_start + b_end) / 2
                dist_q = abs(q_mid - db_mid)
                dist_a = abs(a_mid - db_mid)
                gamma = 0.5
                q_signal = q_dens * (max(dist_q, 1) ** gamma) # just safety net
                a_signal = a_dens * (max(dist_a, 1) ** gamma)
                k_nh, tau_nh = avg_len, 30
                anc_weight = 1 / (1 + np.exp(-(dist_a - k_nh) / tau_nh))
                # k_pr, tau_pr = 5, 0.8
                # shield = 1 / (1 + np.exp(-(b_idx - k_pr) / tau_pr)) if b_idx < 2 else 1.0 # put the shield on first 2 dbs
                shield = 0 if b_idx < 2 else 1.0 # put the shield on first 2 dbs
                
                pos_raw_density[b_idx, l_idx, :] = q_dens
                pos_corr_density[b_idx, l_idx, :] = q_signal - (anc_weight * a_signal * shield)
            #     # end method 8
                # pos_raw_density[b_idx, l_idx, :] = q_dens
                # pos_corr_density[b_idx, l_idx, :] = q_dens - a_dens
            
            attn_module.saved_states = {}
            
        # filling up the win_raw, win_corr matrix    
        gold_i = pos
        for l in range(32):
            for h in range(32):
                if np.argmax(pos_raw_density[:, l, h]) == gold_i: win_count_raw[l, h] += 1
                if np.argmax(pos_corr_density[:, l, h]) == gold_i: win_count_corr[l, h] += 1
        
        # if i need to plot then i need the data: gold attn, max_dist_attn, cortd_attn, max_cortd_attn (all are [32, 32])
        if q_id in QUERIES_TO_PLOT:
            others = [i for i in range(30) if i != gold_i]
            test_query_full_data[q_id][pos] = {
                'rg': pos_raw_density[gold_i], 'rm': np.max(pos_raw_density[others], axis=0),
                'cg': pos_corr_density[gold_i], 'cm': np.max(pos_corr_density[others], axis=0)
            }

    np.savez(os.path.join(npz_dir, f"qid_{q_id}.npz"), raw=win_count_raw, corr=win_count_corr)
    
    # success matrix is made from aggregation samples
    if q_id in QUERIES_TO_AGGREGATE:
        query_success_matrices.append(win_count_raw) # shape: [num_queries, n_layer, n_head]
        query_corrected_matrices.append(win_count_corr) # shape: [num_queries, n_layer, n_head]

final_raw_map = np.mean(query_success_matrices, axis=0) if query_success_matrices else np.zeros((32,32)) # [32, 32]
final_corr_map = np.mean(query_corrected_matrices, axis=0) if query_corrected_matrices else np.zeros((32,32)) # [32, 32]


def get_top_20(matrix):
    idx = np.argsort(matrix.flatten())[-20:][::-1]
    return [divmod(i, 32) for i in idx]

top_raw = get_top_20(final_raw_map)
top_corr = get_top_20(final_corr_map)

np.savez(os.path.join(npz_dir, "aggregated_results.npz"), final_raw=final_raw_map, final_corr=final_corr_map, top_raw=top_raw, top_corr=top_corr)

x_axis = np.arange(NUM_DBS)

# now plot the graphs according to the heads found by aggregation samples
for q_id in QUERIES_TO_PLOT:
    q_dir = os.path.join(OUTPUT_DIR, f"qid_{q_id}")
    os.makedirs(q_dir, exist_ok=True)
    
    def plot_helper(heads, is_corr, title, fname):
        yg, ym = [], []
        for p in range(NUM_DBS):
            d = test_query_full_data[q_id][p]
            g_map, m_map = (d['cg'], d['cm']) if is_corr else (d['rg'], d['rm'])
            if heads is None:
                yg.append(np.sum(g_map)); ym.append(np.sum(m_map))
            else:
                yg.append(sum(g_map[l,h] for l,h in heads))
                ym.append(sum(m_map[l,h] for l,h in heads))
        
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, yg, 'b-o', label='Gold DB')
        plt.plot(x_axis, ym, 'r--x', label='Max Distractor')
        plt.title(f"{title} (QID {q_id})")
        plt.xlabel("Gold Position Index")
        plt.ylabel("Attention Density")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(q_dir, fname)); plt.close()

    plot_helper(None, False, "Case 1: All Heads (Sum)", "01_all.png")
    plot_helper(top_raw, False, "Case 2: Top 20 Raw Heads", "02_raw_top.png")
    plot_helper(top_corr, True, "Case 3: Top 20 Corrected Heads", "03_corr_top.png")

print("Done! Results saved to:", OUTPUT_DIR)