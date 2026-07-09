# Efficient Query Routing Harnessing LLMs as Retrievers

This repository contains the implementation and experiments for **efficient query routing using Large Language Models (LLMs) as retrievers**.  
The project studies how LLMs can select the most relevant database, tool, or document from a large candidate pool under long-context settings.

Instead of relying only on expensive generation-based reranking, this work investigates whether the internal attention patterns of LLMs can be used as efficient retrieval signals.

---

## Overview

Modern retrieval-augmented systems often require selecting the right item before generation:

- selecting the correct database schema for Text-to-SQL,
- selecting the right tool for an agent,
- reranking candidate documents,
- routing a query to the most relevant knowledge source.

This project explores the accuracy–efficiency trade-off between:

1. traditional retrievers,
2. generation-based LLM routing,
3. RankGPT-style reranking,
4. attention-based retrieval using selected LLM heads.

The central idea is that even when LLM generation becomes unreliable in long contexts, some internal attention heads still behave like strong retrieval mechanisms.

---

## Key Contributions

- Benchmarked LLM-based database routing on **SPIDER** and **BIRD**.
- Compared one-by-one LLM routing against all-in-one long-context routing.
- Benchmarked dense, sparse, late-interaction, and hybrid retrievers.
- Analyzed query-to-schema attention flow across layers and heads.
- Identified strong positional bias in long-context LLM routing.
- Used anchor-based correction to reduce prompt-position effects.
- Studied retrieval-specialized attention heads for efficient routing.
- Evaluated **OLR-Heads** as a single-forward-pass retrieval method.
- Explored efficiency improvements using:
  - Parallel Context Windows,
  - Virtual Query Repeaters,
  - Tournament-style RankGPT reranking.
