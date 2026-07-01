import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from modeling_qwen3 import Qwen3ForCausalLM

MODEL_PATH = "Qwen/Qwen3-8B"
BIRD_DIR = '/scratch/gaurav/data/BIRD'
OUTPUT_DIR = './pos_corr_question_anchor_results_qwen'
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
npz_dir = os.path.join(OUTPUT_DIR, "head_analysis_npz")
os.makedirs(npz_dir, exist_ok=True)
 
QUERIES_TO_AGGREGATE = [345, 532, 785, 367, 567, 745, 1054, 1245, 543, 512, 789, 432]
QUERIES_TO_PLOT = [123, 654, 987, 234, 698, 1412, 625, 712]
NUM_DBS = 30
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=False)
rope_scaling = {
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "type": "yarn"
    }
model = Qwen3ForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    rope_scaling=rope_scaling,
    device_map="cuda:0",
    local_files_only=True
)
model.eval()
QUESTION_ANCHOR_IDS = tokenizer("Question:", add_special_tokens=False)["input_ids"]

def find_subsequence(seq, subseq):
    """Find the start index of subseq in seq. Returns -1 if not found."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            return i
    return -1

def get_bird_data():
    def load_json(p):
        with open(p, 'r') as f: return json.load(f)
    data = load_json(os.path.join(BIRD_DIR, 'dev', 'dev.json')) + load_json(os.path.join(BIRD_DIR, 'train', 'train.json'))
    tables = load_json(os.path.join(BIRD_DIR, 'dev', 'dev_tables.json')) + load_json(os.path.join(BIRD_DIR, 'train', 'train_tables.json'))
    sql_map = load_json('/scratch/gaurav/data/BIRD/formatted/birddb_dev_schema_info.json')
    sql_map.update(load_json('/scratch/gaurav/data/BIRD/formatted/birddb_train_schema_info.json'))
    return data, {db['db_id']: db for db in tables}, sql_map


all_questions, db_schemas, sql_map = get_bird_data()
unique_qids = sorted(list(set(QUERIES_TO_AGGREGATE + QUERIES_TO_PLOT)))
head_dim = model.config.hidden_size // model.config.num_attention_heads
scaling = head_dim ** -0.5
query_success_matrices = []
query_corrected_matrices = []
test_query_full_data = {}


for q_id in unique_qids:
    print(f"\nProcessing QID: {q_id}")
    item = next(i for i in all_questions if str(i.get('question_id')) == str(q_id))
    gold_db, nl_query = item['db_id'], item['question']
    distractors = sorted([d for d in db_schemas.keys() if d != gold_db])[:NUM_DBS - 1]
    win_count_raw = np.zeros((32, 32))
    win_count_corr = np.zeros((32, 32))

    if q_id in QUERIES_TO_PLOT:
        test_query_full_data[q_id] = {}

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
        for db_name in ordered_dbs:
            db_ids = tokenizer(f"Database: {db_name}\nSchema: {sql_map.get(db_name, '')}\n", add_special_tokens=False)["input_ids"]
            start_idx = len(full_tokens)
            full_tokens.extend(db_ids)
            block_boundaries.append((start_idx, len(full_tokens)))

        q_text = f"Select the correct database name for answering the question that follows.\nQuestion: {nl_query}\nCorrect database name: "
        q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]
        q_start = len(full_tokens)
        full_tokens.extend(q_ids)
        q_end = len(full_tokens)

        # @gauravk: safety check for anchor token
        anchor_start = find_subsequence(q_ids, QUESTION_ANCHOR_IDS)
        if anchor_start == -1:
            alt_ids = tokenizer(" Question:", add_special_tokens=False)["input_ids"]
            anchor_start = find_subsequence(q_ids, alt_ids)
            anchor_len = len(alt_ids)
        else:
            anchor_len = len(QUESTION_ANCHOR_IDS)
 
        if anchor_start == -1:
            raise RuntimeError(f"Could not locate 'Question:' anchor tokens in q_ids for QID {q_id}.\n" f"q_ids: {q_ids}\nAnchor IDs tried: {QUESTION_ANCHOR_IDS}")
        
        anchor_end = anchor_start + anchor_len
        # @gauravk safety net end
        
        ''' @gauravk: build masks (not attention masks): 
            1. for anchor [....<anchor_span=true, content_span=false>....]
            2. for content [....<content_span=true, anchor_span=false>....]
        '''
        q_anchor_mask = torch.zeros(len(q_ids), dtype=torch.bool, device="cuda:0")
        q_anchor_mask[anchor_start:anchor_end] = True
        q_content_mask = ~q_anchor_mask
        
        assistant_ids = tokenizer("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
        full_tokens.extend(assistant_ids)
        input_ids = torch.tensor([full_tokens], device="cuda:0")
        # import ipdb; ipdb.set_trace()
        model.config.instr_end = instr_end
        model.config.q_start = q_start
        model.config.q_end = q_end
        model.config.block_boundaries = block_boundaries
        model.config.q_content_mask = q_content_mask
        model.config.q_plus_mask = q_anchor_mask

        with torch.no_grad():
            model(input_ids, use_cache=True)

        # import ipdb; ipdb.set_trace()
        pos_raw_density  = np.zeros((30, 32, 32))  # [db_idx, layer, head]
        pos_corr_density = np.zeros((30, 32, 32))

        avg_len = np.mean([b_end - b_start for b_start, b_end in block_boundaries])

        for l_idx in range(32):
            attn_module = model.model.layers[l_idx].self_attn

            # full question query states: [1, num_heads, q_len, head_dim]
            q_states = attn_module.saved_states["question"]["q"]
            q_content_mask_dev = model.config.q_content_mask.to(q_states.device)
            q_anchor_mask_dev  = model.config.q_plus_mask.to(q_states.device)
            
            # split into semantic (real words) and positional anchors ('+' tokens)
            q_question = q_states[:, :, q_content_mask_dev, :]   # extarct only real content tokens
            q_anchor = q_states[:, :, q_anchor_mask_dev, :]   # extract only 'Question:' anchor tokens
            # import ipdb; ipdb.set_trace()

            q_len_main = q_question.shape[2]
            q_len_anchor = q_anchor.shape[2]

            for b_idx, (b_start, b_end) in enumerate(model.config.block_boundaries):
                states = attn_module.saved_states[b_idx]
                k_db   = states["k"]
                db_len = k_db.shape[2]

                # attention from question to db
                w_raw  = torch.matmul(q_question, k_db.transpose(-1, -2)) * scaling
                q_dens = w_raw.sum(dim=[-1, -2]).squeeze(0).cpu().float().numpy() / (q_len_main * db_len)

                # attention from pluses to db
                w_anchor = torch.matmul(q_anchor, k_db.transpose(-1, -2)) * scaling
                p_dens = w_anchor.sum(dim=[-1, -2]).squeeze(0).cpu().float().numpy() / (q_len_anchor * db_len)

                # method to remove pos bias
                # @gauravk: changed to 'Question:' as anchors
                pos_raw_density[b_idx, l_idx, :]  = q_dens
                if b_idx < 2:
                    pos_corr_density[b_idx, l_idx, :] = q_dens
                else:
                    pos_corr_density[b_idx, l_idx, :] = q_dens - p_dens
                # @gauravk method 1 end
                    
                # # method 2: boosted as per distance:
                # pos_raw_density[b_idx, l_idx, :] = q_dens
                # corrected_val = q_dens - p_dens
                # db_mid = (b_start + b_end) / 2
                # q_mid = (model.config.q_start + model.config.q_end) / 2
                # dist_q = q_mid - db_mid
                # gamma = 0.22 
                # boost = (max(dist_q, 1) ** gamma)
                # pos_corr_density[b_idx, l_idx, :] = corrected_val * boost
                # @gauravk method 2 end
                
            attn_module.saved_states = {}

        gold_i = pos
        for l in range(32):
            for h in range(32):
                if np.argmax(pos_raw_density[:, l, h])  == gold_i: win_count_raw[l, h]  += 1
                if np.argmax(pos_corr_density[:, l, h]) == gold_i: win_count_corr[l, h] += 1

        if q_id in QUERIES_TO_PLOT:
            others = [i for i in range(30) if i != gold_i]
            test_query_full_data[q_id][pos] = {
                'rg': pos_raw_density[gold_i], 'rm': np.max(pos_raw_density[others], axis=0),
                'cg': pos_corr_density[gold_i], 'cm': np.max(pos_corr_density[others], axis=0)
            }

    np.savez(os.path.join(npz_dir, f"qid_{q_id}.npz"), raw=win_count_raw, corr=win_count_corr)

    if q_id in QUERIES_TO_AGGREGATE:
        query_success_matrices.append(win_count_raw)
        query_corrected_matrices.append(win_count_corr)


final_raw_map = np.mean(query_success_matrices, axis=0) if query_success_matrices  else np.zeros((32, 32))
final_corr_map = np.mean(query_corrected_matrices, axis=0) if query_corrected_matrices else np.zeros((32, 32))


def get_top_20(matrix):
    idx = np.argsort(matrix.flatten())[-20:][::-1]
    return [divmod(i, 32) for i in idx]


top_raw  = get_top_20(final_raw_map)
top_corr = get_top_20(final_corr_map)

np.savez(os.path.join(npz_dir, "aggregated_results.npz"),
         final_raw=final_raw_map, final_corr=final_corr_map,
         top_raw=top_raw, top_corr=top_corr)

x_axis = np.arange(NUM_DBS)
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
                yg.append(sum(g_map[l, h] for l, h in heads))
                ym.append(sum(m_map[l, h] for l, h in heads))

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

























# old method start
# db_mid = (b_start + b_end) / 2
                # q_mid = (model.config.q_start + model.config.q_end) / 2
                # dist_q = q_mid - db_mid

                # gamma = 0.5
                # q_signal = q_dens * (max(dist_q, 1) ** gamma)
                # p_signal = p_dens * (max(dist_q, 1) ** gamma)  # same distance weighting for '+' since they share positions (as i inserted between the question)

                # # Sigmoid gate: trust '+' correction more when db is far from question
                # k_nh, tau_nh = avg_len, 30
                # anc_weight = 1 / (1 + np.exp(-(dist_q - k_nh) / tau_nh))

                # # Shield: protect first 2 dbs from over-correction
                # k_pr, tau_pr = 5, 0.8
                # shield = 1 / (1 + np.exp(-(b_idx - k_pr) / tau_pr)) if b_idx < 2 else 1.0

                # pos_raw_density[b_idx, l_idx, :]  = q_dens
                # pos_corr_density[b_idx, l_idx, :] = q_signal - (anc_weight * p_signal * shield)
# old method end