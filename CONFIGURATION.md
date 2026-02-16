# Frozen Configuration

## Chunking Parameters
- TARGET_CHARS = 1200
- OVERLAP_CHARS = 150

## Embedding Model
- Type: AraGemma
- API: http://192.168.2.30:6001/embed
- Dimension: 768

## Fusion Method
- Type: RRF (Reciprocal Rank Fusion)
- RRF_K = 60
- TOP_K_DENSE = 10
- TOP_K_KEYWORD = 10
- TOP_K_FINAL = 6

## Reranker Model
- Type: Fallback (keyword-based)
- Not using cross-encoder for reproducibility

## Collection
- Name: realsoft_chunks_hybrid
- Total chunks: 202 (from PDF)
- Dense vectors: 768 dim
- Sparse vectors: BM25

## Date
Rebuilt: 2026-02-16
