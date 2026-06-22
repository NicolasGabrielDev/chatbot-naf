# Assistente Virtual Fiscal

Chatbot fiscal desenvolvido com FastAPI, RAG em Python, ChromaDB local e uma interface web simples. O assistente responde dúvidas sobre Imposto de Renda usando exclusivamente o material oficial carregado no projeto.

## Visão geral

O projeto transforma o manual de perguntas e respostas da Receita Federal em uma base consultável. A preparação do conteúdo ocorre em duas etapas:

1. O PDF é extraído para `data/contexto.md`, preservando a página de origem de cada trecho.
2. As perguntas e seções do manual são organizadas em `data/context_index.json`, formando um catálogo de temas com ID, título e páginas relacionadas.

O catálogo evita que uma pergunta seja pesquisada indiscriminadamente em todo o manual. Antes da busca, o provedor de IA configurado classifica a dúvida em até três temas. A consulta ao ChromaDB fica restrita às páginas desses temas.

## Fluxo de uma pergunta

1. A API recebe e valida a pergunta.
2. O texto é simplificado para remover palavras pouco relevantes para a classificação.
3. O provedor configurado, OpenAI ou Gemini, escolhe os IDs mais relacionados no catálogo de temas.
4. Os IDs são convertidos nas páginas correspondentes do manual.
5. A busca vetorial consulta somente os chunks dessas páginas.
6. Uma ordenação lexical complementa a busca vetorial e prioriza termos presentes na pergunta.
7. Os melhores trechos são enviados ao modelo para geração da resposta final.
8. A API retorna a resposta, as fontes recuperadas e o modelo utilizado.

Não existe busca global silenciosa quando nenhum tema válido é identificado. Se o conteúdo recuperado não sustentar uma resposta, o assistente informa que não encontrou uma resposta segura na base.

Cada pergunta normalmente utiliza duas chamadas ao provedor de IA: uma para classificar o tema e outra para gerar a resposta. Isso deve ser considerado ao utilizar planos com limite diário de requisições. Quando o provedor informa que a cota foi atingida, a API retorna HTTP `429` e o frontend apresenta uma mensagem específica.

## Componentes principais

- `app/api/routes.py`: endpoints e tratamento das respostas HTTP.
- `app/services/topic_catalog.py`: leitura do catálogo, simplificação da pergunta e resolução das páginas.
- `app/services/rag_service.py`: indexação, filtro por páginas, busca vetorial e ordenação lexical.
- `app/services/llm_service.py`: classificação temática, embeddings e geração das respostas com OpenAI ou Gemini.
- `app/services/context_loader.py`: normalização do Markdown ou TXT antes da indexação.
- `scripts/extract_context.py`: extração do PDF e geração do Markdown e do catálogo de temas.
- `scripts/index_context.py`: criação ou substituição da coleção no ChromaDB.
- `app/prompts/system_prompt.md`: regras que limitam a resposta ao contexto oficial.
- `frontend/`: interface web do chat.

## Arquivos de dados

O PDF original deve ficar em:

```text
data/context.pdf
```

O comando de extração gera:

```text
data/contexto.md
data/context_index.json
```

O `contexto.md` contém o texto normalizado e marcadores de página. O `context_index.json` contém os temas reconhecidos no manual. Sempre execute novamente a extração e a indexação quando o PDF for substituído.

O ChromaDB é persistido em:

```text
app/vectorstore
```

## Configuração

Crie o arquivo `.env` a partir do exemplo:

```bash
cp .env.example .env
```

Configuração básica com OpenAI:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sua-chave

CONTEXT_FILE_PATH=./data/contexto.md
TOPIC_INDEX_PATH=./data/context_index.json
SYSTEM_PROMPT_PATH=./app/prompts/system_prompt.md
VECTORSTORE_PATH=./app/vectorstore
CHROMA_COLLECTION_NAME=ir_context

CHUNK_SIZE=4000
CHUNK_OVERLAP=400
TOP_K=4
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_DIMENSIONS=768
```

Configuração com Gemini:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=sua-chave
```

O `LLM_PROVIDER` controla tanto a classificação dos temas quanto a geração da resposta. O `EMBEDDING_PROVIDER` controla separadamente os embeddings:

- `local`: usa embeddings locais baseados em hashing e não consome cota do provedor durante indexação e busca.
- `llm`: usa o modelo de embedding correspondente ao provedor configurado.

Modelos de embedding externos podem ser ajustados por:

```env
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
EMBEDDING_BATCH_SIZE=50
```

No Linux, `APP_UID` e `APP_GID` devem corresponder ao usuário e grupo que executarão o container:

```bash
id -u
id -g
```

Informe os valores no `.env`:

```env
APP_UID=1000
APP_GID=1001
```

## Executar localmente

Crie e ative o ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Extraia o PDF e gere o catálogo de temas:

```bash
python -m scripts.extract_context
```

Crie ou atualize o índice vetorial:

```bash
python -m scripts.index_context
```

Inicie a aplicação:

```bash
uvicorn app.main:app --reload --port 8374
```

Acesse:

```text
http://localhost:8374
```

## Executar com Docker Compose

Garanta que o `.env` esteja configurado e construa a aplicação:

```bash
docker compose up -d --build
```

Extraia o PDF e gere o catálogo:

```bash
docker compose exec api python -m scripts.extract_context
```

Indexe o conteúdo:

```bash
docker compose exec api python -m scripts.index_context
```

O chatbot fica disponível em:

```text
http://localhost:8374
```

Para acompanhar os logs:

```bash
docker compose logs -f api
```

## API

Verificação de saúde:

```text
GET /health
```

Envio de pergunta:

```text
POST /chat
```

Exemplo:

```bash
curl -X POST http://localhost:8374/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Quem precisa declarar Imposto de Renda?"}'
```

A resposta contém:

- `answer`: resposta produzida com base nos trechos recuperados.
- `sources`: trechos e páginas usados como fundamentação.
- `model_used`: modelo configurado para a resposta.

## Testes

Execute a suíte com:

```bash
python -m unittest discover -s tests -v
```

Os testes cobrem o fluxo de contexto, a restrição por páginas, a ordenação lexical, a geração do catálogo e o tratamento de limite do provedor.

## Limitações

- O assistente depende da qualidade e da atualização do PDF carregado.
- A classificação temática e a resposta final dependem da disponibilidade e da cota do provedor configurado.
- O conteúdo é informativo e não substitui orientação oficial da Receita Federal ou atendimento profissional especializado.
