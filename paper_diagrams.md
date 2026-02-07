# Research Paper Diagram Designs

This document contains the finalized Mermaid diagrams for the framework's architecture and experimental design.

## Design 1: End-to-End Architecture

```mermaid
graph TD
    Docs[Raw Documents] --> Chunk[Chunking Module]
    Chunk -->|Naive/Semantic| MetaGen[LLM Metadata Generation]
    Chunk --> Enrich[Enrichment Module]
    MetaGen -->|Summary/Keywords| Enrich
    
    subgraph "Offline Indexing"
        direction TB
        Enrich -->|Prefix-Fusion| Emb[Embedding Model]
        Emb --> Index[(Vector Index)]
    end
    
    User[User Query] --> Ret[Retriever]
    Index --> Ret
    Ret -->|Recall Top-N| Rerank[Cross-Encoder Reranker]
    Rerank -->|Top-K Context| Gen[LLM Generator]
    Gen --> Ans[Final Answer]
    
    style Index fill:#f9f,stroke:#333,stroke-width:2px
    style Rerank fill:#bbf,stroke:#333,stroke-width:2px
```

## Design 2: Embedding Construction Mechanisms

```mermaid
flowchart LR
    subgraph "Prefix Fusion Strategy"
        direction TB
        M1[Metadata String]:::meta & C1[Content String]:::content --> Cat[Concatenate]
        Cat --> Enc1[Transformer Encoder]
        Enc1 --> V1[Single Dense Vector]:::vector
    end

    subgraph "TF-IDF Hybrid Strategy"
        direction TB
        M2[Metadata Tokens]:::meta --> Sparse[Sparse TF-IDF]
        C2[Content Tokens]:::content --> Dense[Dense Encoder]
        Sparse -->|Weight 0.3| Add((+))
        Dense -->|Weight 0.7| Add
        Add --> V2[Hybrid Vector]:::vector
    end

    classDef meta fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef content fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef vector fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
```

## Design 3: Retrieval & Reranking Workflow

```mermaid
graph TD
    Query[User Query] --> KW[Keyword Extraction]
    Query --> Emb[Embedding Model]
    KW --> Sparse[Sparse Vector]
    Emb --> Dense[Dense Vector]
    
    Sparse --> Hybrid[Hybrid Search]
    Dense --> Hybrid
    
    subgraph "Retrieval Pipeline"
        Hybrid -->|Top-50 Candidates| Rerank[Cross-Encoder Reranker]
        Rerank -->|Re-score & Sort| Final[Top-10 Final Results]
    end
    
    Query --> Rerank

    style Rerank fill:#ff9,stroke:#333,stroke-width:2px
    style Final fill:#9f9,stroke:#333,stroke-width:2px
```
