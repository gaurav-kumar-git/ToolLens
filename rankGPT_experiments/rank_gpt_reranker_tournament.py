import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from tqdm import tqdm

@dataclass
class OriginalTimingStats:
    window_times: List[float] = field(default_factory=list)
    total_time: float = 0.0
    label: str = "Original (sliding window)"

    @property
    def num_gpu_calls(self) -> int:
        return len(self.window_times)

    @property
    def avg_window_time(self) -> float:
        return float(np.mean(self.window_times)) if self.window_times else 0.0

    def report(self) -> None:
        W = 62
        print(f"\n{'═' * W}")
        print(f"  {self.label}")
        print(f"{'─' * W}")
        print(f"  {'Sequential GPU calls (batch=1)':<36} {self.num_gpu_calls:>8}")
        print(f"  {'Avg time / call':<36} {self.avg_window_time:>8.4f} s")
        print(f"  {'Total GPU time':<36} {sum(self.window_times):>8.4f} s")
        print(f"  {'Total wall-clock time':<36} {self.total_time:>8.4f} s")
        print(f"{'═' * W}\n")


@dataclass
class RoundStats:
    round_num: int
    pool_size_in: int       # documents entering this round
    num_chunks: int         # = batch_size for the GPU call
    chunk_size: int         # documents per chunk (= window_size, last chunk may differ)
    survivors: int          # documents exiting this round
    elapsed: float          # wall-clock seconds for this round's batch call


@dataclass
class TournamentTimingStats:
    rounds: List[RoundStats] = field(default_factory=list)
    final_round: Optional[RoundStats] = None
    total_time: float = 0.0
    label: str = "Tournament (tree sort)"

    @property
    def total_gpu_calls(self) -> int:
        # each round = 1 batched generate() call
        return len(self.rounds) + (1 if self.final_round else 0)

    @property
    def max_batch_size(self) -> int:
        if not self.rounds:
            return 1
        return max(r.num_chunks for r in self.rounds)

    def report(self) -> None:
        W = 62
        print(f"\n{'═' * W}")
        print(f"  {self.label}")
        print(f"{'─' * W}")
        print(f"  {'Batched GPU calls (total)':<36} {self.total_gpu_calls:>8}")
        print(f"  {'Max batch size across rounds':<36} {self.max_batch_size:>8}")
        print()
        print(f"  {'Rnd':<5} {'In':>6} {'Chunks':>8} {'Survived':>10} {'Time':>10}")
        print(f"  {'─'*5} {'─'*6} {'─'*8} {'─'*10} {'─'*10}")
        for r in self.rounds:
            print(
                f"  {r.round_num:<5} {r.pool_size_in:>6} {r.num_chunks:>8}"
                f" {r.survivors:>10} {r.elapsed:>9.4f}s"
                f"  ← batch_size={r.num_chunks}"
            )
        if self.final_round:
            r = self.final_round
            print(
                f"  {'fin':<5} {r.pool_size_in:>6} {r.num_chunks:>8}"
                f" {r.survivors:>10} {r.elapsed:>9.4f}s"
                f"  ← final full rank"
            )
        print(f"{'─' * W}")
        print(f"  {'Total wall-clock time':<36} {self.total_time:>8.4f} s")
        print(f"{'═' * W}\n")


def compare_both(orig: OriginalTimingStats, tourn: TournamentTimingStats) -> None:
    W = 62

    def fmt(a: float, b: float) -> str:
        return f"{a / b:.2f}x faster" if b > 1e-9 else "n/a"

    print(f"\n{'═' * W}")
    print(f"  Head-to-head comparison")
    print(f"{'─' * W}")
    print(f"  {'Metric':<36} {'Original':>10} {'Tournament':>10}")
    print(f"  {'─'*36} {'─'*10} {'─'*10}")
    print(f"  {'Total GPU calls':<36} {orig.num_gpu_calls:>10} {tourn.total_gpu_calls:>10}")
    print(f"  {'Max batch size':<36} {'1':>10} {tourn.max_batch_size:>10}")
    print(f"  {'Total wall-clock (s)':<36} {orig.total_time:>10.4f} {tourn.total_time:>10.4f}")
    print(f"  {'Speedup':<36} {'':>10} {fmt(orig.total_time, tourn.total_time):>10}")
    print(f"{'═' * W}\n")


class _ModelBase:
    """Loads the LLM and sets prompt delimiters.  Shared by both strategies."""

    def __init__(
        self,
        base_llm_name: str,
        sliding_window_size: int = 20,
        sliding_window_stride: int = 5,
    ) -> None:
        print(f"Loading model: {base_llm_name} ...")
        self.base_llm_name = base_llm_name

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base_llm_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if "qwen2.5-7b-instruct-1m" in base_llm_name.lower():
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                base_llm_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        elif "qwen" in base_llm_name.lower():
            rope = {
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
                "type": "yarn",
            }
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                base_llm_name,
                dtype=torch.float16,
                local_files_only=True,
                rope_scaling=rope,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                base_llm_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.model.eval()

        if "llama-3" in base_llm_name.lower():
            self.prompt_prefix = "<|start_header_id|>user<|end_header_id|>"
            self.prompt_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif "qwen" in base_llm_name.lower():
            self.prompt_prefix = "<|im_start|>user\n"
            self.prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        else:
            self.prompt_prefix = "[INST]"
            self.prompt_suffix = "[/INST]"

        self.sliding_window_size = sliding_window_size
        self.sliding_window_stride = sliding_window_stride


    def _create_prompt(self, query: str, doc_pool: List[str]) -> str:
        documents_prompt = "\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(doc_pool))
        return (
            f"{self.prompt_prefix} This is an intelligent assistant that can rank passages "
            f"based on their relevancy to the query.\n\n"
            f"The following are {len(doc_pool)} passages, each indicated by number identifier []. "
            f"I can rank them based on their relevance to query: \"{query}\"\n\n"
            f"{documents_prompt}\n\n"
            f"The search query is: \"{query}\". I will rank the {len(doc_pool)} passages above "
            f"based on their relevance to the search query. "
            f"The passages will be listed in descending order using identifiers, the most relevant "
            f"passages first and the format should be [1] > [2] > etc. "
            f"Be sure to list all {len(doc_pool)} ranked passages and do not explain your ranking "
            f"until after the list is done. "
            f"{self.prompt_suffix} Ranked Passages: ["
        )

    def _parse_output(self, output_str: str, doc_pool_size: int) -> Tuple[List[int], bool]:
        try:
            found = [int(n) - 1 for n in re.findall(r"\[(\d+)\]", output_str)]
            order: List[int] = []
            for idx in found:
                if 0 <= idx < doc_pool_size and idx not in order:
                    order.append(idx)
            if len(order) != doc_pool_size:
                for i in range(doc_pool_size):
                    if i not in order:
                        order.append(i)
                return order, False
            return order, True
        except Exception:
            return list(range(doc_pool_size)), False

class RankGPTOriginal(_ModelBase):
    def _generate_one(self, query: str, doc_pool: List[str]) -> str:
        prompt = self._create_prompt(query, doc_pool)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        max_new = max(100, len(doc_pool) * 5)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return "[" + self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

    def rerank(self, query: str, documents: List[str]) -> Tuple[List[int], OriginalTimingStats]:
        stats = OriginalTimingStats()
        t0 = time.perf_counter()

        N = len(documents)
        sorted_ids = list(reversed(range(N)))

        _i, _j = 0, min(self.sliding_window_size, N)
        while True:
            tw = time.perf_counter()
            # import ipdb; ipdb.set_trace()
            # build window (reversed so most-relevant candidates enter first)
            window_idx = list(reversed(sorted_ids[_i:_j]))
            docs = [documents[i] for i in window_idx]

            output = self._generate_one(query, docs)
            local_order, _ = self._parse_output(output, len(docs))
            local_order.reverse()
            sorted_ids[_i:_j] = [window_idx[i] for i in local_order]

            stats.window_times.append(time.perf_counter() - tw)

            if _j == N:
                break
            _i += self.sliding_window_stride
            _j = min(_i + self.sliding_window_size, N)

        sorted_ids.reverse()
        stats.total_time = time.perf_counter() - t0
        return sorted_ids, stats


class RankGPTTournament(_ModelBase):
    """
    Tournament / tree-sort reranker.

    Round structure
    ───────────────
    Repeat until the surviving pool fits in a single window:
      1.  Partition the pool into chunks of `window_size`.
      2.  Build one prompt per chunk.
      3.  Pad all prompts to the same length (left-pad for causal models).
      4.  Single model.generate(batch) — all chunks ranked in parallel.
      5.  Collect the top `top_k` documents from each chunk.

    Final round:
      The surviving pool (≤ window_size documents) is ranked once more
      to produce a fully ordered result.

    Why left-padding matters for batched causal generation
    ───────────────────────────────────────────────────────
    A causal model must not attend to future tokens.  With right-padding
    the padding tokens at the END of a short sequence would be attended
    to by the real tokens of a longer sequence in the same batch — that
    leaks information across examples.  Left-padding pushes padding to
    positions that are already masked by the causal mask.
    """

    def __init__(
        self,
        base_llm_name: str,
        window_size: int = 20,
        top_k_per_window: int = 5,
        max_gpu_batch_size: int = 2
    ) -> None:
        # stride is unused in tournament mode; pass window_size as a dummy
        super().__init__(base_llm_name, window_size, window_size)
        self.top_k_per_window = top_k_per_window
        self.max_gpu_batch_size = max_gpu_batch_size
        print(window_size, top_k_per_window)

   
    def _batch_generate(
        self, query: str, windows_of_docs: List[List[str]]
    ) -> List[str]:
        if not windows_of_docs:
            return []

        all_results = []
        
        # SUB-BATCHING LOOP: Process chunks in smaller groups to prevent OOM
        for i in range(0, len(windows_of_docs), self.max_gpu_batch_size):
            sub_batch = windows_of_docs[i : i + self.max_gpu_batch_size]
            prompts = [self._create_prompt(query, docs) for docs in sub_batch]
            
            # FIXED: max_new_tokens should be based on number of documents, not text length
            # A list of 40 docs [1] > [2]... needs roughly 5-8 tokens per doc.
            max_new = max(100, self.sliding_window_size * 10)

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True, # Safety: don't exceed model max position
                max_length=self.tokenizer.model_max_length 
            ).to(self.model.device)

            padded_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    temperature=0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            for j in range(len(sub_batch)):
                text = self.tokenizer.decode(
                    output_ids[j, padded_len:], skip_special_tokens=True
                )
                all_results.append("[" + text)
                
            # Clear cache for next sub-batch
            torch.cuda.empty_cache()

        return all_results

    
    @staticmethod
    def _chunk_pool(pool: List[int], chunk_size: int) -> List[List[int]]:
        """Split pool into contiguous chunks of at most chunk_size."""
        return [pool[i : i + chunk_size] for i in range(0, len(pool), chunk_size)]

    def _run_one_round(
        self,
        query: str,
        documents: List[str],
        pool: List[int],
        top_k: int,
    ) -> Tuple[List[int], List[int], float]:
        """
        Execute one tournament round.

        Returns
        survivors  : top_k doc indices from each chunk (the "winners")
        eliminated : the rest, ordered best-to-worst within each chunk
                     (used to build the tail of the final ranking)
        elapsed    : wall-clock seconds for the batched GPU call
        """
        chunks = self._chunk_pool(pool, self.sliding_window_size)
        doc_lists = [[documents[idx] for idx in chunk] for chunk in chunks]

        t0 = time.perf_counter()
        raw_outputs = self._batch_generate(query, doc_lists)
        elapsed = time.perf_counter() - t0

        survivors: List[int] = []
        eliminated_best_first: List[int] = []

        for chunk, raw in zip(chunks, raw_outputs):
            local_order, _ = self._parse_output(raw, len(chunk))
            # local_order[0] = best local rank, …, local_order[-1] = worst
            k = min(top_k, len(chunk))
            survivors.extend(chunk[local_order[i]] for i in range(k))
            eliminated_best_first.extend(
                chunk[local_order[i]] for i in range(k, len(chunk))
            )

        return survivors, eliminated_best_first, elapsed

    
    def rerank_tournament(
        self, query: str, documents: List[str]
    ) -> Tuple[List[int], TournamentTimingStats]:
        """
        Full tournament rerank.

        Output ranking:
        1.  Final window survivors  → fully ranked by the last LLM call.
        2.  Earlier-round survivors that did not reach the final window
            → appended in reverse-round order (later survival = higher rank).
        3.  Round-1 losers          → appended last (least likely to be relevant).
        """
        # stats = TournamentTimingStats()
        # t0 = time.perf_counter()

        N = len(documents)
        current_pool = list(range(N))  # original doc indices still in contention

        # per-round eliminated lists, indexed by round (0 = round 1)
        # we store them separately so we can append them in reverse-round order.
        eliminated_per_round: List[List[int]] = []

        round_num = 0
        while len(current_pool) > self.sliding_window_size:
            round_num += 1
            pool_in = len(current_pool)
            num_chunks = len(self._chunk_pool(current_pool, self.sliding_window_size))

            survivors, eliminated, elapsed = self._run_one_round(
                query, documents, current_pool, self.top_k_per_window
            )
            import ipdb; ipdb.set_trace()
            # stats.rounds.append(
            #     RoundStats(
            #         round_num=round_num,
            #         pool_size_in=pool_in,
            #         num_chunks=num_chunks,
            #         chunk_size=self.sliding_window_size,
            #         survivors=len(survivors),
            #         elapsed=elapsed,
            #     )
            # )
            eliminated_per_round.append(eliminated)
            current_pool = survivors

        tf = time.perf_counter()
        final_docs = [documents[idx] for idx in current_pool]
        [raw_final] = self._batch_generate(query, [final_docs])
        final_elapsed = time.perf_counter() - tf

        final_local_order, _ = self._parse_output(raw_final, len(current_pool))
        fully_ranked = [current_pool[i] for i in final_local_order]

        # stats.final_round = RoundStats(
        #     round_num=round_num + 1,
        #     pool_size_in=len(current_pool),
        #     num_chunks=1,
        #     chunk_size=len(current_pool),
        #     survivors=len(current_pool),
        #     elapsed=final_elapsed,
        # )

        tail: List[int] = []
        for elim_round in reversed(eliminated_per_round):
            tail.extend(elim_round)

        final_ranking = fully_ranked + tail

        # stats.total_time = time.perf_counter() - t0
        return final_ranking



def run_benchmark(
    model_name: str,
    documents: List[str],
    query: str,
    window_size: int = 20,
    stride: int = 5,
    top_k_per_window: int = 5,
) -> Tuple[Optional[OriginalTimingStats], Optional[TournamentTimingStats]]:
    """
    Load each variant in turn, run reranking, print timing, compare.
    Models are deleted between runs to avoid GPU OOM.
    """
    print(f"\n{'═' * 62}")
    print(f"  Benchmark configuration")
    print(f"{'─' * 62}")
    print(f"  {'Documents':<30} {len(documents)}")
    print(f"  {'Window size':<30} {window_size}")
    print(f"  {'Stride (original only)':<30} {stride}")
    print(f"  {'Top-K per window (tournament)':<30} {top_k_per_window}")
    print(f"  {'Query':<30} {query[:50]}...")
    print(f"{'═' * 62}")

    stats_orig: Optional[OriginalTimingStats] = None
    stats_tourn: Optional[TournamentTimingStats] = None
    ranked_orig: Optional[List[int]] = None
    ranked_tourn: Optional[List[int]] = None

    print("\n  [1/2]  Running Original RankGPT  (sequential, batch_size=1)")
    orig = RankGPTOriginal(model_name, window_size, stride)
    ranked_orig, stats_orig = orig.rerank(query, documents)
    stats_orig.report()
    print(f"  Top-10 (original): {ranked_orig[:10]}")
    del orig
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n  [2/2]  Running Tournament RankGPT  (batched, batch_size=N/W)")
    tourn = RankGPTTournament(model_name, window_size, top_k_per_window)
    ranked_tourn, stats_tourn = tourn.rerank_tournament(query, documents)
    stats_tourn.report()
    print(f"  Top-10 (tournament): {ranked_tourn[:10]}")
    del tourn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    compare_both(stats_orig, stats_tourn)

    for k in [1, 3, 5, 10]:
        if k <= len(documents):
            orig_set = set(ranked_orig[:k])
            tourn_set = set(ranked_tourn[:k])
            overlap = len(orig_set & tourn_set)
            print(
                f"  Top-{k:<3} overlap (original ∩ tournament): "
                f"{overlap}/{k}  ({100*overlap/k:.0f} %)"
            )

    return stats_orig, stats_tourn


if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"   

    DOCUMENTS = [
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower was constructed from 1887 to 1889 as the centrepiece of the 1889 World's Fair.",
        "Python is a high-level, general-purpose programming language created by Guido van Rossum.",
        "The French Revolution began in 1789 and fundamentally transformed French society and politics.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "The Louvre Museum in Paris is the world's largest art museum and a historic monument.",
        "Deep learning uses multi-layer neural networks to learn hierarchical representations.",
        "Napoleon Bonaparte was a French military commander who conquered much of continental Europe.",
        "Natural language processing enables computers to understand and generate human language.",
        "The Seine River flows 775 km through northern France, bisecting the heart of Paris.",
        "Versailles Palace was the principal royal residence of France from 1682 until the Revolution.",
        "Transformer networks use multi-head self-attention to process sequences in parallel.",
        "The storming of the Bastille on 14 July 1789 is the defining symbol of the Revolution.",
        "Retrieval-augmented generation grounds LLM outputs with documents fetched at query time.",
        "Notre-Dame Cathedral on Île de la Cité is one of the finest examples of Gothic architecture.",
        "The Arc de Triomphe stands at the western end of the Champs-Élysées in central Paris.",
        "Convolutional neural networks excel at image recognition and computer vision tasks.",
        "The Battle of Waterloo in 1815 ended Napoleon's rule and reshaped the map of Europe.",
        "The Musée d'Orsay houses the world's largest collection of Impressionist masterpieces.",
        "Gradient descent is the optimisation algorithm that underlies most deep learning training.",
    ] * 5  

    QUERY = "What are the most important historical landmarks and events in Paris, France?"

    run_benchmark(
        model_name=MODEL_NAME,
        documents=DOCUMENTS,
        query=QUERY,
        window_size=20,
        stride=5,
        top_k_per_window=5,   
    )