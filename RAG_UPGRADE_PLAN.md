# RAG Upgrade Plan

## Document Purpose

This document defines the upgrade path for the current `ai-engine` RAG stack.
It is intended to be the single implementation reference for future changes.

This plan is written in English on purpose because the current environment may display Chinese text as garbled output.

## Implementation Constraints

All future code changes based on this document should follow these rules:

1. Use English for comments, log messages, markdown documentation, prompt descriptions, and new identifiers where practical.
2. Preserve the existing project style and architecture instead of introducing a new coding style.
3. Apply changes incrementally, with each phase independently reviewable and reversible.
4. Prefer extending the current retrieval pipeline over replacing it all at once.
5. Keep feature flags or configuration switches where a rollout may affect answer quality.

## Current Project Baseline

The current project already has a usable first-stage RAG pipeline:

- Prompt-based intent routing in `src/ai_engine/chains/nodes/router_node.py`
- Vector retrieval in `src/ai_engine/chains/nodes/rag_node.py`
- Reranking in `src/ai_engine/utils/retrieval_utils.py`
- Knowledge ingestion in `scripts/init_knowledge_db.py`
- Business-specific prompt separation in `resource/prompts`
- Business-specific knowledge separation in `resource/knowledge`

Current strengths:

- Clear FastAPI + LangServe service structure
- Existing RAG entry point is centralized
- Existing rerank stage already exists
- Knowledge ingestion is script-driven and can be evolved

Current limitations:

- Retrieval is still mostly single-path vector retrieval
- Query understanding before retrieval is weak
- Chunk context is not rich enough for complex rule-style documents
- Retrieval quality control is limited
- The pipeline does not yet combine lexical and semantic recall

## Technique Review

The following section evaluates the techniques you listed against the current project state.

### Keep as Baseline

#### Fixed-size chunking

Status:
- Already implemented

Why keep it:
- Stable
- Cheap
- Good fallback behavior

Why not rely on it alone:
- It may split semantic units badly
- It may lose section-level meaning for policy, tutorial, and FAQ documents

Decision:
- Keep it as the base fallback chunking strategy

### Strong Candidates for Upgrade

#### Query Transformation

Value:
- Very high

Why it fits this project:
- Many user queries are conversational, incomplete, or under-specified
- The current routing layer identifies intent but does not really optimize the retrieval query

Recommended capabilities:
- Query rewrite
- History-aware query completion
- Multi-query expansion
- Optional sub-query decomposition for complex requests

Decision:
- Phase 1 priority

#### Fusion Retrieval

Definition:
- Combine semantic retrieval and keyword retrieval, then fuse results with RRF

Value:
- Very high

Why it fits this project:
- Your knowledge base contains policy terms, fees, product names, API names, and exact phrases
- Vector retrieval alone is weak on exact keyword matching

Recommended fusion stack:
- Semantic retrieval from PGVector or Chroma
- BM25 or equivalent lexical retrieval
- Reciprocal Rank Fusion
- Existing reranker after fusion

Decision:
- Phase 1 priority

#### Contextual Chunk Headers

Value:
- High

Why it fits this project:
- Many documents are naturally hierarchical
- A chunk becomes more useful when the title path is included

Recommended use:
- Add document title, section title, subsection title, and business type into chunk text or metadata

Decision:
- Phase 1 priority

#### Context Enriched Retrieval

Value:
- High

Why it fits this project:
- The current chunk may match well but miss adjacent explanatory text
- It is easier to implement than full parent-document retrieval

Recommended use:
- Retrieve the matched chunk
- Expand with previous and next sibling chunks during final context building

Decision:
- Phase 1 or Phase 2 priority

#### Small-to-Big Retrieval

Value:
- High

Why it fits this project:
- Good fit for tutorial and rule documents
- It improves precision during retrieval and completeness during generation

Recommended use:
- Match on small chunks
- Resolve to parent chunk, parent section, or parent document window before generation

Decision:
- Phase 2 priority

#### Contextual Compression

Value:
- High

Why it fits this project:
- Once context expansion is added, raw prompt context will become noisy
- Compression helps preserve only the most relevant evidence

Recommended use:
- Compress after fusion and rerank
- Compress before final prompt assembly

Decision:
- Phase 2 priority

#### Lightweight CRAG

Definition:
- Retrieval quality check before generation

Value:
- Medium to high

Why it fits this project:
- It can block weak retrieval results from going directly into answer generation
- It reduces low-confidence answers

Recommended lightweight strategy:
- Check recall count
- Check rerank score threshold
- Check score distribution
- Retry with relaxed strategy if retrieval is weak
- Refuse or degrade gracefully if evidence is still poor

Decision:
- Phase 2 priority

### Useful Later, But Not Immediate Priorities

#### Semantic Chunking

Value:
- Medium to high

Why not first:
- Harder to tune than fixed chunking
- Better to optimize retrieval strategy first

Decision:
- Phase 3

#### Document Augmentation

Definition:
- Generate synthetic QA pairs or retrieval hints for chunks

Value:
- Medium to high

Why not first:
- Requires offline generation quality control
- Adds ingestion complexity

Decision:
- Phase 3

#### RSE (Relevant Segment Extraction)

Value:
- Medium to high

Why not first:
- It is effectively an advanced retrieval post-processing framework
- Better after fusion and compression are already in place

Decision:
- Phase 3

### Not Recommended as Early Investments

#### Self-RAG

Why not now:
- Too much control is delegated to the model
- Harder to debug
- Higher latency
- Higher implementation complexity

Decision:
- Not an immediate priority

#### Knowledge Graph

Why not now:
- Current knowledge patterns are still mostly document-centric
- Graph construction and maintenance costs are high

Decision:
- Not an immediate priority

#### Hierarchical Indices

Why not now:
- More useful when corpus size grows much larger
- Current bottleneck is quality of retrieval logic, not index scale

Decision:
- Not an immediate priority

#### HyDE

Why not now:
- It adds another model call before retrieval
- It helps more in open-domain retrieval than structured business knowledge

Decision:
- Not an immediate priority

#### Feedback Loop

Why not now:
- Requires a complete online feedback and evaluation loop
- Better after the retrieval core becomes stable

Decision:
- Long-term item

## Selected Upgrade Strategy

The recommended upgrade path is:

### Phase 1: Improve Recall Quality

Scope:
- Query Transformation
- Fusion Retrieval
- Contextual Chunk Headers
- Context Enriched Retrieval

Goal:
- Retrieve the right evidence more consistently

Why this phase comes first:
- Retrieval quality is the current main bottleneck
- These changes improve quality without forcing a full architecture rewrite

### Phase 2: Improve Context Quality

Scope:
- Small-to-Big Retrieval
- Contextual Compression
- Lightweight CRAG

Goal:
- Provide cleaner and more complete context to the generator

Why this phase comes second:
- Better recall alone will increase context size and noise
- This phase turns good recall into good final prompts

### Phase 3: Improve Ingestion and Post-processing Quality

Scope:
- Semantic Chunking
- Document Augmentation
- RSE

Goal:
- Improve chunk quality and offline retrieval assets

Why this phase comes later:
- These are more expensive to implement and evaluate
- They are best added after the main online retrieval loop is stable

## Detailed Step-by-Step Plan

The following steps are intentionally detailed because future implementation work should follow this sequence.

### Step 1: Introduce Retrieval Strategy Configuration

What to do:
- Add retrieval strategy switches in settings or prompt config
- Allow enabling or disabling:
  - query rewrite
  - lexical retrieval
  - fusion
  - context expansion
  - compression
  - retrieval quality checks

Why:
- Upgrades should be incremental
- We need safe rollback if answer quality regresses

Expected impact:
- Better rollout safety
- Easier debugging by feature isolation

Likely files:
- `src/ai_engine/core/settings.py`
- `resource/prompts/*.yaml`
- possibly `src/ai_engine/core/constants.py`

### Step 2: Add Query Transformation Layer

What to do:
- Create a reusable pre-retrieval query transformation module
- Support:
  - simple rewrite
  - history-aware rewrite
  - optional multi-query expansion

Why:
- User input is often not retrieval-optimized
- A better query usually gives a larger quality gain than changing the embedding model

Expected impact:
- Better recall on incomplete or conversational input
- Better handling of follow-up questions

Likely files:
- new module under `src/ai_engine/chains/common/`
- `src/ai_engine/chains/nodes/rag_node.py`
- `resource/prompts/intent_router.yaml` or a new retrieval prompt file

### Step 3: Add Lexical Retrieval

What to do:
- Build a lightweight BM25-based retriever over the knowledge corpus
- Keep semantic retrieval as the existing primary retriever
- Execute both retrievers for the same transformed query

Why:
- Exact terms matter for your business domains
- Lexical retrieval covers the gap where embedding search underperforms

Expected impact:
- Better performance on exact phrase, keyword, product-name, and identifier queries

Likely files:
- new retrieval utility module
- `scripts/init_knowledge_db.py` if a lexical index needs offline assets
- `src/ai_engine/chains/nodes/rag_node.py`

### Step 4: Fuse Semantic and Lexical Results with RRF

What to do:
- Implement Reciprocal Rank Fusion over semantic and BM25 result sets
- Apply reranker after fusion, not before

Why:
- Fusion improves recall without overcommitting to one retrieval style
- RRF is simple and robust

Expected impact:
- Better balanced retrieval quality across exact-match and semantic-match queries

Likely files:
- `src/ai_engine/utils/retrieval_utils.py`
- `src/ai_engine/chains/nodes/rag_node.py`

### Step 5: Enrich Chunk Context with Header Paths

What to do:
- Store section path information in chunk metadata
- Optionally prepend a compact header path into chunk text before indexing

Why:
- The meaning of many chunks depends on where they came from
- Header-aware chunks are easier to retrieve and easier for the model to use

Expected impact:
- Better section-level recall
- Better answer grounding

Likely files:
- `scripts/init_knowledge_db.py`
- processor modules under `scripts/processors/`

### Step 6: Add Context Enriched Retrieval

What to do:
- When a chunk is retrieved, also fetch neighboring chunks
- Keep window size configurable

Why:
- The best-matching chunk often lacks surrounding explanation
- Neighbor expansion is simple and effective

Expected impact:
- Better completeness of generated answers

Likely files:
- `scripts/init_knowledge_db.py` for chunk ordering metadata
- `src/ai_engine/utils/retrieval_utils.py`
- `src/ai_engine/chains/nodes/rag_node.py`

### Step 7: Add Small-to-Big Retrieval

What to do:
- Track parent-child relationships during ingestion
- Retrieve using small chunks
- Build final context using parent sections or parent windows

Why:
- Small chunks improve recall precision
- Parent context improves answer completeness

Expected impact:
- Stronger results on long rule documents and learning material

Likely files:
- `scripts/init_knowledge_db.py`
- schema or metadata conventions
- `src/ai_engine/utils/retrieval_utils.py`

### Step 8: Add Contextual Compression

What to do:
- Compress expanded context before final generation
- Remove low-value spans while preserving evidence

Why:
- Context expansion without compression increases noise
- Compression makes larger-context retrieval practical

Expected impact:
- Better final prompt quality
- Lower token waste

Likely files:
- new compression utility
- `src/ai_engine/chains/nodes/rag_node.py`

### Step 9: Add Lightweight CRAG Checks

What to do:
- Add retrieval quality scoring before final answer generation
- If quality is low:
  - retry with relaxed retrieval
  - or answer with low-confidence fallback behavior

Why:
- Weak retrieval should not directly lead to confident answers

Expected impact:
- Better trustworthiness
- Better refusal behavior when evidence is weak

Likely files:
- `src/ai_engine/utils/retrieval_utils.py`
- `src/ai_engine/chains/nodes/rag_node.py`

### Step 10: Revisit Chunking Strategy

What to do:
- After the online retrieval path is stable, evaluate semantic chunking
- Compare against existing fixed-size chunking

Why:
- Chunking quality matters, but retrieval control matters more first

Expected impact:
- Potentially better retrieval precision
- Higher ingestion complexity

Likely files:
- `scripts/init_knowledge_db.py`
- processor modules

## Recommended File-Level Refactor Direction

This section maps the plan to the current codebase.

### Retrieval Orchestration

Primary file:
- `src/ai_engine/chains/nodes/rag_node.py`

Expected responsibility after upgrade:
- query transformation
- multi-retriever execution
- fusion
- rerank
- quality check
- context expansion
- compression
- final answer generation

Recommendation:
- Keep orchestration here
- Move reusable retrieval logic into helper modules
- Do not let this file become a monolith

### Retrieval Utilities

Primary file:
- `src/ai_engine/utils/retrieval_utils.py`

Expected responsibility after upgrade:
- RRF fusion
- score normalization
- rerank handling
- retrieval quality checks
- context assembly helpers

Recommendation:
- Expand this module first
- Split later only if it becomes too large

### Ingestion Pipeline

Primary file:
- `scripts/init_knowledge_db.py`

Expected responsibility after upgrade:
- chunk metadata enrichment
- parent-child metadata
- chunk order metadata
- optional header injection
- optional synthetic retrieval assets

Recommendation:
- Keep ingestion centralized here
- Move document-specific rules into processors

### Processor Layer

Primary directory:
- `scripts/processors/`

Expected responsibility after upgrade:
- document-specific metadata extraction
- language extraction
- section path extraction
- optional QA augmentation

Recommendation:
- Keep business-specific transformations here

## Design Principles for Implementation

### Principle 1: Do Not Replace the Current RAG Flow All at Once

Why:
- You already have a working pipeline
- Full replacement increases regression risk

How:
- Add feature flags
- Add new retrieval stages around the existing rerank path

### Principle 2: Keep Retrieval Deterministic Where Possible

Why:
- Model-driven control loops are harder to debug

How:
- Use explicit thresholds and staged fallbacks before introducing self-reflective retrieval

### Principle 3: Keep Ingestion Metadata-Rich

Why:
- Better retrieval quality starts with better offline data shape

How:
- Add structured metadata for:
  - language
  - business type
  - document name
  - title path
  - chunk position
  - parent block reference

### Principle 4: Separate Retrieval from Generation Concerns

Why:
- Retrieval problems should be diagnosable without changing generation prompts

How:
- Keep retrieval utilities modular
- Log retrieval steps clearly
- Keep prompt logic independent from retrieval scoring logic

## Suggested Rollout Order for This Project

The implementation order should be:

1. Retrieval feature flags
2. Query transformation
3. Lexical retrieval
4. RRF fusion
5. Header-enriched chunk metadata
6. Context enrichment
7. Compression
8. Lightweight CRAG
9. Small-to-Big retrieval
10. Advanced ingestion improvements

Why this order:
- Each step builds on a stable previous step
- It minimizes rewrites
- It gives measurable quality checkpoints

## Immediate Next Milestone

The best next milestone for this project is:

### Milestone A: Retrieval Quality Upgrade

Deliverables:
- Query transformation
- Vector + BM25 fusion via RRF
- Header-enriched chunks
- Context-enriched retrieval

Why this should be first:
- Highest impact relative to complexity
- Directly addresses current retrieval weaknesses
- Fits the current codebase cleanly

## Out of Scope for the First Upgrade Cycle

The following should not be part of the first upgrade cycle:

- Knowledge graph integration
- Self-RAG controller loop
- Full feedback learning loop
- HyDE-based retrieval
- Large-scale index hierarchy refactor

Why:
- These are high-complexity features with lower short-term payoff for the current codebase

## Final Recommendation

For this project, the most practical upgrade route is:

- First improve how evidence is retrieved
- Then improve how evidence is assembled
- Then improve how evidence is compressed and validated
- Only after that, consider advanced autonomous retrieval behaviors

This order matches the current codebase, keeps risk manageable, and should produce the fastest visible quality gains.
