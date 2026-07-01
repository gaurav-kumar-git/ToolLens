import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from modelling_llama import LlamaForCausalLM


MODEL_PATH = "/scratch/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
BIRD_DIR = '/scratch/gaurav/data/BIRD'
OUTPUT_DIR = './positional_bias_correction_results_plus_token_llama_2'
os.makedirs(OUTPUT_DIR, exist_ok=True)
npz_dir = os.path.join(OUTPUT_DIR, "head_analysis_npz")
os.makedirs(npz_dir, exist_ok=True)

# QUERIES_TO_AGGREGATE = [345, 532, 785,  123, 654, 987, 234, 698, 1412]
# QUERIES_TO_PLOT = [532, 785, 367, 567, 745, 1054, 1245, 1234, 1517, 867, 543, 512, 789, 432]
QUERIES_TO_AGGREGATE = [345]#, 532, 785, 367, 567, 745, 1054, 1245, 543, 512, 789, 432]
QUERIES_TO_PLOT = [123, 654]#, 987, 234, 698, 1412, 625, 712]
NUM_DBS = 30


print("Loading Model & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="eager",
    local_files_only=True
)
model.eval()

# Pre-compute the '+' token id once
PLUS_TOKEN_ID = tokenizer(" +", add_special_tokens=False)["input_ids"][0]


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
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are an expert database routing system. Your task is to analyze a user's question and a list of available database schemas. "
            "You must select the most relevant database name that can answer the question.\n"
            "Do not add any other text, explanation, or formatting.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
        )
        full_tokens = tokenizer(system_block, add_special_tokens=True)["input_ids"]
        instr_end = len(full_tokens)

        block_boundaries = []
        for db_name in ordered_dbs:
            db_ids = tokenizer(f"Database: {db_name}\nSchema: {sql_map.get(db_name, '')}\n", add_special_tokens=False)["input_ids"]
            start_idx = len(full_tokens)
            full_tokens.extend(db_ids)
            block_boundaries.append((start_idx, len(full_tokens)))

        # Question with '+' as positional anchors interleaved between real tokens.
        # Real tokens carry semantic signal; '+' tokens carry pure positional signal.
        # We will split them and subtract '+' attention from real-token attention
        # to cancel positional bias.
        q_text = f"Select + the + correct + database + name + for + answering + the + question + that + follows.\nQuestion: + {nl_query} + \n + Correct + database + name + : + "
        q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]
        q_start = len(full_tokens)
        full_tokens.extend(q_ids)
        q_end = len(full_tokens)

        # Build masks: True = real content token, False = '+' anchor token
        q_content_mask = torch.tensor(
            [tid != PLUS_TOKEN_ID for tid in q_ids], dtype=torch.bool, device="cuda:0"
        )
        q_plus_mask = ~q_content_mask

        assistant_ids = tokenizer("<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)["input_ids"]
        full_tokens.extend(assistant_ids)
        input_ids = torch.tensor([full_tokens], device="cuda:0")

        model.config.instr_end = instr_end
        model.config.q_start = q_start
        model.config.q_end = q_end
        model.config.block_boundaries = block_boundaries
        model.config.q_content_mask = q_content_mask
        model.config.q_plus_mask = q_plus_mask

        with torch.no_grad():
            model(input_ids, use_cache=True)

        # import ipdb; ipdb.set_trace()
        pos_raw_density  = np.zeros((30, 32, 32))  # [db_idx, layer, head]
        pos_corr_density = np.zeros((30, 32, 32))

        avg_len = np.mean([b_end - b_start for b_start, b_end in block_boundaries])

        for l_idx in range(32):
            attn_module = model.model.layers[l_idx].self_attn

            # Full question query states: [1, num_heads, q_len, head_dim]
            q_states = attn_module.saved_states["question"]["q"]
            q_content_mask = model.config.q_content_mask.to(q_states.device)
            q_plus_mask = model.config.q_plus_mask.to(q_states.device)  
            # Split into semantic (real words) and positional anchors ('+' tokens)
            q_question = q_states[:, :, q_content_mask, :]   # real content tokens
            q_plus     = q_states[:, :, q_plus_mask,    :]   # '+' anchor tokens

            q_len_main = q_question.shape[2]
            q_len_plus = q_plus.shape[2]
            all_raw_logits = [] # [num_dbs, num_heads, q_len_main, db_len]
            all_plus_logits = []

            for b_idx, (b_start, b_end) in enumerate(model.config.block_boundaries):
                k_db = attn_module.saved_states[b_idx]["k"]
                all_raw_logits.append(
                    torch.matmul(q_question, k_db.transpose(-1, -2)) * scaling  # [1, H, q_main, db_len]
                )
                all_plus_logits.append(
                    torch.matmul(q_plus, k_db.transpose(-1, -2)) * scaling      # [1, H, q_plus, db_len]
                )
                
            cat_raw  = torch.cat(all_raw_logits,  dim=-1)
            cat_plus = torch.cat(all_plus_logits, dim=-1)
            sm_raw  = torch.softmax(cat_raw,  dim=-1)  # now all values in [0,1], sum=1
            sm_plus = torch.softmax(cat_plus, dim=-1)
            db_lens = [attn_module.saved_states[b]["k"].shape[2] for b in range(len(model.config.block_boundaries))]
            raw_splits  = torch.split(sm_raw,  db_lens, dim=-1)
            plus_splits = torch.split(sm_plus, db_lens, dim=-1)
            for b_idx, (b_start, b_end) in enumerate(model.config.block_boundaries):
                db_len = db_lens[b_idx]

                q_dens = raw_splits[b_idx].sum(dim=[-1, -2]).squeeze(0).cpu().float().numpy() \
                        / (q_len_main * db_len)
                p_dens = plus_splits[b_idx].sum(dim=[-1, -2]).squeeze(0).cpu().float().numpy() \
                        / (q_len_plus * db_len)

                # ... rest of your correction formula unchanged ...
                db_mid  = (b_start + b_end) / 2
                q_mid   = (model.config.q_start + model.config.q_end) / 2
                dist_q  = abs(q_mid - db_mid)

                gamma    = 0.5
                q_signal = q_dens * (max(dist_q, 1) ** gamma)
                p_signal = p_dens * (max(dist_q, 1) ** gamma)

                k_nh, tau_nh = avg_len, 30
                anc_weight = 1 / (1 + np.exp(-(dist_q - k_nh) / tau_nh))

                k_pr, tau_pr = 5, 0.8
                shield = 1 / (1 + np.exp(-(b_idx - k_pr) / tau_pr)) if b_idx < 2 else 1.0

                pos_raw_density[b_idx, l_idx, :]  = q_dens
                pos_corr_density[b_idx, l_idx, :] = q_signal - (anc_weight * p_signal * shield)

            attn_module.saved_states = {}

        gold_i = pos
        for l in range(32):
            for h in range(32):
                if np.argmax(pos_raw_density[:, l, h])  == gold_i: win_count_raw[l, h]  += 1
                if np.argmax(pos_corr_density[:, l, h]) == gold_i: win_count_corr[l, h] += 1

        if q_id in QUERIES_TO_PLOT:
            others = [i for i in range(30) if i != gold_i]
            test_query_full_data[q_id][pos] = {
                'rg': pos_raw_density[gold_i],            'rm': np.max(pos_raw_density[others],  axis=0),
                'cg': pos_corr_density[gold_i],           'cm': np.max(pos_corr_density[others], axis=0)
            }

    np.savez(os.path.join(npz_dir, f"qid_{q_id}.npz"),
             raw=win_count_raw, corr=win_count_corr)

    if q_id in QUERIES_TO_AGGREGATE:
        query_success_matrices.append(win_count_raw  / NUM_DBS)
        query_corrected_matrices.append(win_count_corr / NUM_DBS)


final_raw_map  = np.mean(query_success_matrices,   axis=0) if query_success_matrices  else np.zeros((32, 32))
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

    # Select top-20 heads per query based on softmax corrected scores
    gold_avg = np.zeros((32, 32))
    for p in range(NUM_DBS):
        gold_avg += test_query_full_data[q_id][p]['cg']
    gold_avg /= NUM_DBS
    top_corr_local = get_top_20(gold_avg)

    # Same for raw
    gold_avg_raw = np.zeros((32, 32))
    for p in range(NUM_DBS):
        gold_avg_raw += test_query_full_data[q_id][p]['rg']
    gold_avg_raw /= NUM_DBS
    top_raw_local = get_top_20(gold_avg_raw)

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

    plot_helper(None,           False, "Case 1: All Heads (Sum)",        "01_all.png")
    plot_helper(top_raw_local,  False, "Case 2: Top 20 Raw Heads",       "02_raw_top.png")
    plot_helper(top_corr_local, True,  "Case 3: Top 20 Corrected Heads", "03_corr_top.png")
    
print("Done! Results saved to:", OUTPUT_DIR)