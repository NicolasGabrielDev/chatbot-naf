# Assistente Virtual Fiscal

Aplicação de chatbot fiscal com FastAPI, RAG em Python, ChromaDB local e frontend web simples.

## Configuração

1. Crie o arquivo `.env` a partir do exemplo:

```bash
cp .env.example .env
```

2. Ajuste as variáveis no `.env`:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sua-chave
APP_UID=1000
APP_GID=1001
CONTEXT_FILE_PATH=./data/contexto.md
TOP_K=4
```

No Linux, `APP_UID` e `APP_GID` devem ser o ID do seu usuário e grupo. Para descobrir os valores, rode:

```bash
id -u
id -g
```

Para Gemini:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
GEMINI_API_KEY=sua-chave
```

O arquivo de contexto usado pelo chat deve ser Markdown ou TXT já extraído. O PDF bruto não é lido durante uma requisição de chat.

Por padrão, coloque o PDF original em:

```text
data/context.pdf
```

Depois extraia o texto para:

```text
data/contexto.md
```

O `CONTEXT_FILE_PATH` deve apontar para o Markdown extraído:

```env
CONTEXT_FILE_PATH=./data/contexto.md
```

## Rodar localmente

Instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Extraia o PDF para Markdown:

```bash
python -m scripts.extract_context
```

Indexe ou reindexe o Markdown extraído:

```bash
python -m scripts.index_context
```

Suba a API:

```bash
uvicorn app.main:app --reload --port 8374
```

Acesse o chatbot em:

```text
http://localhost:8374
```

API:

```text
GET  /health
POST /chat
```

Exemplo de chamada:

```bash
curl -X POST http://localhost:8374/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Quem precisa declarar Imposto de Renda?"}'
```

## Rodar com Docker Compose

Antes de subir com Docker, garanta que o arquivo `.env` existe na raiz do projeto. O Compose monta esse arquivo dentro do container para a aplicação ler as configurações.

```bash
docker compose up --build
```

Em outro terminal, indexe o contexto dentro do container:

```bash
docker compose exec api python -m scripts.extract_context
```

```bash
docker compose exec api python -m scripts.index_context
```

Depois acesse:

```text
http://localhost:8374
```

## Trocar provedor ou modelo

A troca é feita pelo `.env`:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sua-chave
```

ou:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
GEMINI_API_KEY=sua-chave
```

Os embeddings usam o mesmo provedor configurado. Os modelos de embedding podem ser ajustados por `OPENAI_EMBEDDING_MODEL` e `GEMINI_EMBEDDING_MODEL`.

Se a cota de embeddings do provedor estiver limitada, use embeddings locais:

```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_DIMENSIONS=768
```

Com `EMBEDDING_PROVIDER=local`, a aplicação continua usando `LLM_PROVIDER` para gerar a resposta final, mas a indexação e a busca no Chroma não consomem quota de embeddings da API.

Para Gemini, use:

```env
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

Para reduzir chamadas na indexação, ajuste também:

```env
CHUNK_SIZE=4000
CHUNK_OVERLAP=400
EMBEDDING_BATCH_SIZE=50
```

## Observação

Se a resposta não estiver no contexto recuperado, o assistente deve responder que não foi possível localizar uma resposta segura na base consultada.
