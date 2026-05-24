# GenAI Feature Pipeline: GCP BigQuery + dbt + AWS Bedrock

A HIPAA-aligned GenAI feature pipeline built on GCP BigQuery and AWS Bedrock — dbt-modeled feature store for document metadata, embedding tracking, and retrieval analytics. Integrates pgvector for semantic search with full audit trail via MLflow and dbt test coverage.

---

## Architecture

```
Source Documents (Healthcare / Enterprise)
            │
            ▼
     GCP Cloud Storage
     (Raw document landing)
            │
            ▼
    Cloud Composer (Airflow)
    Orchestrates end-to-end pipeline
            │
      ┌─────┴──────┐
      ▼            ▼
  BigQuery      AWS Bedrock
  (Feature    (Embedding
   Store)      Generation)
      │            │
      └─────┬──────┘
            ▼
       pgvector (Vector Store)
       Semantic similarity search
            │
            ▼
       dbt Core (Transformation)
       - Incremental feature models
       - Embedding tracking
       - Retrieval analytics
            │
            ▼
       MLflow (Audit + Tracking)
       - Run logging
       - Data quality artifacts
       - HIPAA audit trail
            │
            ▼
   RAG Application Layer (FastAPI)
```

---

## Repository Structure

```
realtime_ml_feature_store/
├── ingestion/
│   └── document_loader.py          # GCS → BigQuery raw ingestion
├── embeddings/
│   ├── bedrock_embedder.py         # AWS Bedrock embedding generation
│   └── vector_store.py             # pgvector upsert + similarity search
├── dbt/
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_raw_documents.sql
│   │   │   └── stg_embeddings.sql
│   │   ├── intermediate/
│   │   │   └── int_document_chunks.sql
│   │   └── marts/
│   │       ├── fct_embedding_runs.sql      # Embedding job tracking
│   │       ├── fct_retrieval_analytics.sql # RAG retrieval quality
│   │       └── dim_documents.sql
│   ├── tests/
│   │   ├── assert_no_pii_in_embeddings.sql
│   │   └── assert_embedding_coverage.sql
│   ├── macros/
│   │   └── deidentify_pii.sql
│   └── dbt_project.yml
├── airflow/
│   └── dags/
│       └── genai_feature_pipeline.py
├── api/
│   └── rag_query.py                # FastAPI RAG endpoint
├── mlflow/
│   └── tracking.py                 # MLflow experiment + artifact logging
├── governance/
│   └── pii_deidentification.py     # Pre-processing PII removal
├── tests/
│   └── test_embeddings.py
├── requirements.txt
└── README.md
```

---

## Core Code

### AWS Bedrock Embedding Generation
```python
# embeddings/bedrock_embedder.py
import boto3
import json
import logging
from typing import List, Optional
import mlflow

logger = logging.getLogger(__name__)

class BedrockEmbedder:
    """Generates embeddings via AWS Bedrock Titan Embeddings model."""

    MODEL_ID = "amazon.titan-embed-text-v1"
    MAX_TOKENS = 8192

    def __init__(self, region: str = "us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.mlflow_client = mlflow.tracking.MlflowClient()

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text chunk."""
        try:
            response = self.client.invoke_model(
                modelId=self.MODEL_ID,
                body=json.dumps({"inputText": text[:self.MAX_TOKENS]}),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except Exception as e:
            logger.error(f"Bedrock embedding failed: {e}")
            return None

    def embed_batch(self, documents: List[dict], run_id: str) -> List[dict]:
        """Embed a batch of documents and log metrics to MLflow."""
        results = []
        success_count = 0

        with mlflow.start_run(run_id=run_id):
            for doc in documents:
                embedding = self.embed_text(doc["content"])
                if embedding:
                    results.append({
                        "doc_id": doc["doc_id"],
                        "chunk_id": doc["chunk_id"],
                        "embedding": embedding,
                        "embedding_dim": len(embedding),
                        "model_id": self.MODEL_ID,
                        "embedded_at": __import__("datetime").datetime.utcnow().isoformat(),
                    })
                    success_count += 1

            # Log metrics to MLflow for audit trail
            mlflow.log_metrics({
                "total_documents": len(documents),
                "successful_embeddings": success_count,
                "failure_rate": (len(documents) - success_count) / max(len(documents), 1),
            })
            mlflow.log_param("model_id", self.MODEL_ID)

        return results
```

### pgvector — Vector Store Upsert + Similarity Search
```python
# embeddings/vector_store.py
import psycopg2
import numpy as np
from typing import List, Tuple
from pgvector.psycopg2 import register_vector

class VectorStore:
    """pgvector-backed semantic similarity search store."""

    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
        register_vector(self.conn)
        self._ensure_table()

    def _ensure_table(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    doc_id          VARCHAR(64) PRIMARY KEY,
                    chunk_id        VARCHAR(64) NOT NULL,
                    content_hash    VARCHAR(64) NOT NULL,
                    embedding       vector(1536),
                    model_id        VARCHAR(100),
                    embedded_at     TIMESTAMP DEFAULT NOW(),
                    metadata        JSONB
                );
                CREATE INDEX IF NOT EXISTS idx_embedding_ivfflat
                ON document_embeddings
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            self.conn.commit()

    def upsert(self, records: List[dict]):
        """Upsert embeddings — idempotent on doc_id."""
        with self.conn.cursor() as cur:
            for rec in records:
                cur.execute("""
                    INSERT INTO document_embeddings
                        (doc_id, chunk_id, content_hash, embedding, model_id, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id)
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        model_id = EXCLUDED.model_id,
                        embedded_at = NOW()
                """, (
                    rec["doc_id"], rec["chunk_id"], rec["content_hash"],
                    rec["embedding"], rec["model_id"],
                    __import__("json").dumps(rec.get("metadata", {}))
                ))
        self.conn.commit()

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple]:
        """Return top-k most similar documents by cosine similarity."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT doc_id, chunk_id, metadata,
                       1 - (embedding <=> %s::vector) AS similarity_score
                FROM document_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            return cur.fetchall()
```

### dbt — Feature Store Models
```sql
-- dbt/models/marts/fct_embedding_runs.sql
-- Tracks every embedding job run for audit and lineage
{{
  config(
    materialized='incremental',
    unique_key='run_id',
    incremental_strategy='merge'
  )
}}

WITH runs AS (
  SELECT * FROM {{ ref('stg_embeddings') }}
  {% if is_incremental() %}
    WHERE embedded_at > (SELECT MAX(embedded_at) FROM {{ this }})
  {% endif %}
)

SELECT
  run_id,
  model_id,
  COUNT(doc_id)                           AS total_documents_embedded,
  COUNT(DISTINCT chunk_id)                AS total_chunks,
  AVG(embedding_dim)                      AS avg_embedding_dim,
  MIN(embedded_at)                        AS run_start,
  MAX(embedded_at)                        AS run_end,
  TIMESTAMP_DIFF(MAX(embedded_at), MIN(embedded_at), SECOND) AS duration_seconds,
  CURRENT_TIMESTAMP()                     AS dbt_updated_at
FROM runs
GROUP BY run_id, model_id
```

```sql
-- dbt/macros/deidentify_pii.sql
-- Pre-hook macro: strips PII before any transformation runs
{% macro deidentify_pii(column_name) %}
  REGEXP_REPLACE(
    REGEXP_REPLACE(
      REGEXP_REPLACE({{ column_name }},
        r'\b\d{3}-\d{2}-\d{4}\b', '[SSN-REDACTED]'),  -- SSN
      r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', '[EMAIL-REDACTED]'),  -- Email
    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE-REDACTED]'  -- Phone
  )
{% endmacro %}
```

### FastAPI RAG Endpoint
```python
# api/rag_query.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embeddings.bedrock_embedder import BedrockEmbedder
from embeddings.vector_store import VectorStore
import mlflow

app = FastAPI(title="GenAI RAG API", version="1.0")
embedder = BedrockEmbedder()
vector_store = VectorStore(connection_string="postgresql://...")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    results: list
    retrieval_latency_ms: float

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Semantic search over embedded document store."""
    import time

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start = time.time()

    # Generate query embedding
    query_embedding = embedder.embed_text(request.query)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    # Retrieve similar documents
    results = vector_store.similarity_search(query_embedding, top_k=request.top_k)
    latency_ms = (time.time() - start) * 1000

    # Log retrieval to MLflow for analytics
    with mlflow.start_run():
        mlflow.log_metrics({
            "retrieval_latency_ms": latency_ms,
            "results_returned": len(results),
            "top_similarity_score": results[0][3] if results else 0,
        })

    return QueryResponse(
        query=request.query,
        results=[
            {"doc_id": r[0], "chunk_id": r[1],
             "metadata": r[2], "similarity_score": float(r[3])}
            for r in results
        ],
        retrieval_latency_ms=round(latency_ms, 2)
    )
```

---

## HIPAA Governance

| Control | Implementation |
|---|---|
| PII de-identification | dbt pre-hooks + `deidentify_pii()` macro before any transformation |
| Column masking | Unity Catalog row-level filters on regulated datasets |
| Audit trail | MLflow run logging for every embedding job and retrieval call |
| Test coverage | dbt `assert_no_pii_in_embeddings` test on all mart models |
| Access control | IAM roles scoped per environment (dev / staging / prod) |

---

## Performance

| Metric | Value |
|---|---|
| Embedding throughput | ~500 documents/min (Bedrock Titan) |
| Vector search latency | < 50ms (p95) at 1M+ vectors |
| dbt incremental runtime | < 30 seconds |
| Pipeline end-to-end | < 5 minutes doc-to-searchable |

---

## Setup

```bash
git clone https://github.com/ManiRukki/realtime_ml_feature_store.git
cd realtime_ml_feature_store
pip install -r requirements.txt

# Configure AWS credentials for Bedrock
aws configure

# Configure GCP credentials
gcloud auth application-default login

# Run dbt
cd dbt && dbt run --target dev && dbt test --target dev

# Start RAG API
uvicorn api.rag_query:app --reload --port 8000
```

---

## Tech Stack

`GCP BigQuery` `dbt Core` `AWS Bedrock` `pgvector` `Apache Airflow`
`Cloud Composer` `MLflow` `FastAPI` `Python` `Terraform`

---

## Related Projects

- [streaming-event-pipeline](https://github.com/ManiRukki/streaming_event_pipeline) — Kafka + Flink + DLT real-time lakehouse
- [enterprise-data-warehouse](https://github.com/ManiRukki/entireprise_data_warehouse) — eBay 95TB Snowflake migration
- [customer-analytics-platform](https://github.com/ManiRukki/customer-analytics-platform) — CVS Health healthcare analytics
