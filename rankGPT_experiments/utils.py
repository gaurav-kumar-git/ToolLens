import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
def load_model_tokenizer(model_name, device, dtype = torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                #  output_attentions = True,
                                                dtype=dtype,  
                                                local_files_only = True
                                                )
    model.to(device)
    model.eval()
    return tokenizer, model


SYSTEM_PROMPT_DB = (
    "You are an expert database routing system. "
    "Your task is to analyze a user's question and a list of available database schemas. "
    "You must select the most relevant database_id that can answer the question.\n"
    "Do not add any other text, explanation, or formatting."
)
SYSTEM_PROMPT_TOOL = (
    "You are an expert tool selection system. "
    "Your task is to analyze a user's question and a list of available tools. "
    "Each tool has an name, and description describing its functionality. "
    "You must select the single most relevant tool_id that can best answer or handle the user's query.\n"
    "Do not include any explanation, reasoning, or extra text — output only the selected tool_id."
)
def create_instruction_prompt(model_name, tokenizer=None, system_prompt="", user_prompt=""):
    if "llama" in model_name.lower():
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if "qwen" in model_name.lower():
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>"
            f"<|im_start|>user\n{user_prompt}<|im_end|>"
            f"<|im_start|>assistant\n"
        )
    if "oss" in model_name.lower():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
            # return_tensors="pt",
            # return_dict=True,
            )#.to(model.device)
        return prompt

class PromptUtils:
    
    
    def __init__(self, dataset, model_name, tokenizer, all_docs_info_string, doc_names_str):
        self.dataset = dataset
        self.model_name =  model_name
        self.tokenizer = tokenizer
        self.all_docs_info_string = all_docs_info_string
        self.at1 = "Now, follow these in-context examples to understand the task and format.\n"
        if dataset in ["spider", "bird"]:
            self.at2 = (
                f"# Task: Examine all the database schemas provided above and output ONLY the correct database_id for the question below.\n"
                f"# Your selection MUST be from the following list of valid database_ids: {doc_names_str}\n"
                )
            self.system_prompt = SYSTEM_PROMPT_DB
            
            if "Llama" in model_name:
                self.doc_identifier = 12494
        
            if "Qwen" in model_name:
                self.doc_identifier = 12216
        
            
        if dataset in ["toole"]:
            self.at2 = (
                f"# Task: Examine the descriptions of all available tools and output ONLY the correct tool_id for the question below.\n"
                f"# Your selection MUST be from the following list of valid tool_ids: {doc_names_str}\n"
                )
            self.system_prompt = SYSTEM_PROMPT_TOOL
            
            if "Llama" in model_name:
                self.doc_identifier = 14506
        
            if "Qwen" in model_name:
                self.doc_identifier = 14172

        self.at1_length = len(tokenizer(self.at1, add_special_tokens=False).input_ids)
        self.at2_length = len(tokenizer(self.at2, add_special_tokens=False).input_ids)

    
    def create_user_prompt(self, query, in_context_exs):
        if self.dataset in ["spider", "bird"]:
            return (
                f"Here are all the available databases:\n{self.all_docs_info_string}"
                f"{self.at1}{in_context_exs}\n"
                f"{self.at2}"
                f"# Question: {query}\n"
                f"# Correct database_id: \n\n"
                )
        if self.dataset in ["toole"]:
            return (
                f"Here are all the available tools:\n{self.all_docs_info_string}"
                f"{self.at1}{in_context_exs}\n"
                f"{self.at2}"
                f"# Question: {query}\n"
                f"# Correct tool_id: "
                # f"# Correct tool_id: \n\n"
            )
        
    def get_item_spans_fixed_pool(self):
        prompt = self.create_prompt(query="N/A", in_context_exs=" ")
        dummy_ids = self.tokenizer(prompt, return_tensors = "pt", add_special_tokens=False).input_ids[0]
        db_indices = np.where(dummy_ids == self.doc_identifier)[0]
        # import pdb; pdb.set_trace()
        return db_indices, dummy_ids
    
    def create_prompt(self, query, in_context_exs):
        user_prompt = self.create_user_prompt(query=query, in_context_exs=in_context_exs)
        prompt = create_instruction_prompt(model_name=self.model_name, system_prompt=self.system_prompt, user_prompt=user_prompt)
        return prompt
        

class IclUtils:
    def __init__(self, tokenizer, dataset, shuffled_keys, dict_all_docs):
        self.tokenizer =  tokenizer
        self.dataset = dataset

        if dataset in ["spider", "bird"]:
            icl_text = lambda index, query, doc_name: f"Example {index}:\n# Question: {query}\n# Correct database_id: {doc_name}\n"
            doc_text = lambda doc_name, doc_info: f"database_id: {doc_name} \n{doc_info}\n"
            icl_text_nogold =  lambda index, query: f"Example {index}:\n# Question: {query}\n# Correct database_id:"
        if dataset in ["toole"]:
            icl_text = lambda index, query, doc_name: f"Example {index}:\n# Question: {query}\n# Correct tool_id: {doc_name}\n"
            doc_text = lambda doc_name, doc_info: f"tool_id: {doc_name}\n{doc_info}\n"
            icl_text_nogold =  lambda index, query: f"Example {index}:\n# Question: {query}\n# Correct tool_id:"

        self.icl_text = icl_text
        self.icl_text_nogold = icl_text_nogold
        self.doc_text = doc_text
    
        (
            self.all_docs_info_string, self.doc_names_str, 
            self.map_docname_id, self.map_id_docname, 
            self.doc_lengths
        ) = self.create_doc_pool_string(shuffled_keys, dict_all_docs)

    def create_doc_pool_string(self, shuffled_keys, all_docs):
        doc_lengths = []
        doc_list_str = []
        map_docname_id, map_id_docname = {}, {}
        all_schemas = ""
        for key in shuffled_keys:
            value = all_docs[key]
            doc_list_str.append(key)
            text = self.doc_text(doc_name=key, doc_info=value)
            text += "---"*30
            text += "\n"
            db_tokens = self.tokenizer(text, add_special_tokens=False).input_ids
            doc_lengths.append(len(db_tokens))
            all_schemas += text
            map_docname_id[key] = len(map_docname_id)
            map_id_docname[map_docname_id[key]] = key
        doc_list_str = ", ".join(doc_list_str)    
        return all_schemas, doc_list_str, map_docname_id, map_id_docname, doc_lengths

    def format_in_context_examples(self, examples):
        formatted = []
        icl_exs_len = []
        icl_exs_without_gold_db_len = []
        for i, (q, db) in enumerate(examples, 1):
            text_nogold = self.icl_text_nogold(index=i, query=q)
            icl_text = self.icl_text(index=i, query=q, doc_name=db)
            icl_exs_without_gold_db_len.append(len(self.tokenizer(text_nogold, add_special_tokens=False).input_ids))
            icl_exs_len.append(len(self.tokenizer(icl_text, add_special_tokens=False).input_ids))
            formatted.append(icl_text)
        return "\n".join(formatted), icl_exs_len, icl_exs_without_gold_db_len

