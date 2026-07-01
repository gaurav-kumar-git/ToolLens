import torch
import transformers
import re
from tqdm import tqdm
import numpy as np

class RankGPTModel():
    def __init__(self,
                 base_llm_name,
                 sliding_window_size=20,
                 sliding_window_stride=10,
                 ) -> None:
        
        print(f"Loading Model: {base_llm_name} using Transformers...")
        self.base_llm_name = base_llm_name
        print(sliding_window_size, sliding_window_stride)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base_llm_name)
        if 'qwen2.5-7b-instruct-1m' in base_llm_name.lower():
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                base_llm_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        elif 'qwen' in base_llm_name.lower():
            rope = {
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
                "type": "yarn"
            }
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                base_llm_name,  
                #  output_attentions = True,
                dtype=torch.float16,  
                local_files_only = True,
                rope_scaling = rope
            )
            
            # self.model = transformers.AutoModelForCausalLM.from_pretrained(
            #     base_llm_name,
            #     torch_dtype=torch.float16,
            #     device_map="auto",
            #     trust_remote_code=True
            # )
            # import ipdb; ipdb.set_trace()
            print(self.model.config.rope_scaling)
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                        base_llm_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                )
        self.model.eval()

        if 'llama-3' in base_llm_name.lower():
            self.prompt_prefix = '<|start_header_id|>user<|end_header_id|>'
            self.prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        if 'qwen' in base_llm_name.lower():
            self.prompt_prefix = "<|im_start|>user\n"
            self.prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        else:
            self.prompt_prefix = '[INST]'
            self.prompt_suffix = '[/INST]'

        self.sliding_window_size = sliding_window_size
        self.sliding_window_stride = sliding_window_stride

    def _create_prompt(self, query, doc_pool):
        """Exact RankGPT Prompt formulation"""
        documents_prompt = '\n'.join([f'[{i + 1}] ' + doc for i, doc in enumerate(doc_pool)])
        
        prompt = (
            f"{self.prompt_prefix} This is an intelligent assistant that can rank passages based on their relevancy to the query.\n\n"
            f"The following are {len(doc_pool)} passages, each indicated by number identifier []. "
            f"I can rank them based on their relevance to query: \"{query}\"\n\n{documents_prompt}\n\n"
            f"The search query is: \"{query}\". I will rank the {len(doc_pool)} passages above based on their relevance to the search query. "
            f"The passages will be listed in descending order using identifiers, the most relevant passages first and the format should be [1] > [2] > etc. "
            f"Be sure to list all {len(doc_pool)} ranked passages and do not explain your ranking until after the list is done. "
            f"{self.prompt_suffix} Ranked Passages: ["
        )
        # import ipdb; ipdb.set_trace()
        return prompt

    def _get_sorted_docs_from_prompts(self, query, doc_pool):
        """Standard HF Generation"""
        prompt = self._create_prompt(query, doc_pool)
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        # import ipdb; ipdb.set_trace()
        max_new_tokens = max(100, len(doc_pool) * 5)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return "[" + generated_text

    def _parse_output(self, output_str, doc_pool_size):
        """Exact RankGPT Parsing Logic"""
        try:
            found_ids = [int(num) - 1 for num in re.findall(r'\[(\d+)\]', output_str)]
            
            final_order = []
            for idx in found_ids:
                if 0 <= idx < doc_pool_size and idx not in final_order:
                    final_order.append(idx)
            
            if len(final_order) != doc_pool_size:
                for i in range(doc_pool_size):
                    if i not in final_order:
                        final_order.append(i)
                return final_order, False 
            
            return final_order, True
        except:
            return list(range(doc_pool_size)), False

    def rerank(self, query, documents):
        """The Algorithmic Sliding Window Flow"""
        N_docs = len(documents)
        sorted_doc_ids = list(range(N_docs))
        sorted_doc_ids.reverse() 
        
        
        _i = 0
        _j = min(self.sliding_window_size, N_docs)
        # import ipdb; ipdb.set_trace()

        while True:
            current_window_indices = sorted_doc_ids[_i : _j]
            current_window_indices.reverse() 
            
            docs = [documents[idx] for idx in current_window_indices]
            
            output_str = self._get_sorted_docs_from_prompts(query, docs)
            new_local_order, is_valid = self._parse_output(output_str, len(docs))
            # import ipdb; ipdb.set_trace()
            
            new_local_order.reverse() 
            updated_ids = [current_window_indices[idx] for idx in new_local_order]
            
            for k in range(_i, _j):
                sorted_doc_ids[k] = updated_ids[k - _i]
            
            if _j == N_docs:
                break
            
            _i += self.sliding_window_stride
            _j = min(_i + self.sliding_window_size, N_docs)
         
        
        sorted_doc_ids.reverse() 
        # ipdb.set_trace()   
        return sorted_doc_ids