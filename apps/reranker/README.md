# Rag_Issdhu Reranker

Microservicio local de rerank para Fase 3 avanzada.

## Contrato

- `POST /rerank`
- request:
  - `query`
  - `documents`
  - `top_n`
  - `model`
- response:
  - `results`
    - `index`
    - `relevance_score`

## Modelo recomendado

- `BAAI/bge-reranker-v2-m3`

## Arranque local recomendado

1. Crear un entorno separado:

```powershell
py -3.12 -m venv .venv-reranker
.\.venv-reranker\Scripts\Activate.ps1
```

2. Instalar PyTorch con CUDA desde la guia oficial de PyTorch.

3. Instalar el microservicio:

```powershell
pip install --upgrade pip
pip install -e .\apps\reranker
```

4. Ejecutar:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 7997 --app-dir .\apps\reranker
```

## Variables

- `RERANKER_HOST`
- `RERANKER_PORT`
- `RERANKER_DEFAULT_MODEL`
- `RERANKER_DEVICE`
- `RERANKER_MAX_BATCH_SIZE`
- `RERANKER_TRUST_REMOTE_CODE`

## Integracion con la API principal

En `.env` de la API:

```env
RERANK_MODE=local
RERANK_LOCAL_BASE_URL=http://localhost:7997
RERANK_LOCAL_MODEL=BAAI/bge-reranker-v2-m3
RERANK_LOCAL_TIMEOUT=20
```
