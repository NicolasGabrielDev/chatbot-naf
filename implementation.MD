* Desenvolva uma aplicação de chatbot fiscal utilizando FastAPI, Python e um frontend web simples.
* O backend deve funcionar como uma API JSON.
* Todo processamento de IA, RAG e consulta ao contexto deve ser feito em Python.
* O objetivo do agente é responder dúvidas sobre Imposto de Renda com base em um arquivo de contexto fornecido pelo projeto.
* Utilize RAG como abordagem principal para consultar o arquivo de contexto antes de gerar qualquer resposta.
* O arquivo de contexto já existe e deve ser tratado como fonte única da verdade.
* O agente não deve responder com conhecimento próprio da LLM quando a resposta não estiver no contexto.
* Caso a resposta não seja encontrada no contexto, retorne uma mensagem informando que não foi possível localizar uma resposta segura na base consultada.
* O projeto deve aceitar arquivos de contexto em Markdown, TXT ou PDF.
* Internamente, normalize o conteúdo do arquivo de contexto para texto puro antes da indexação.
* Divida o conteúdo do contexto em chunks.
* Gere embeddings dos chunks.
* Armazene os embeddings localmente utilizando ChromaDB.
* Crie um processo de indexação/reindexação do contexto.
* A busca semântica deve recuperar os trechos mais relevantes do contexto antes de chamar a LLM.
* A quantidade de trechos recuperados deve ser configurável por variável de ambiente.
* O modelo de LLM deve ser configurável por variável de ambiente.
* O provedor da LLM deve ser configurável por variável de ambiente.
* Implemente suporte inicial para OpenAI e Gemini, escolhendo o provedor com base no .env.
* Crie um arquivo .env.example com todas as configurações necessárias.
* Use o arquivo .env para controlar provedor, modelo, chave de API, caminho do contexto, caminho do prompt, caminho do banco vetorial, tamanho dos chunks, overlap e top_k.
* O prompt de sistema da LLM deve ficar em um arquivo separado.
* Crie o arquivo prompts/system_prompt.md.
* O prompt de sistema deve instruir a LLM a agir como um assistente fiscal, responder em linguagem simples, evitar juridiquês/tributarês e usar apenas o contexto recuperado.
* A resposta da LLM deve ser clara, acolhedora e acessível para cidadãos comuns.
* Não use linguagem excessivamente técnica nas respostas.
* Quando um termo técnico for inevitável, explique-o de forma simples.
* Crie o endpoint POST /chat.
* O endpoint POST /chat deve receber uma pergunta do usuário em JSON.
* O endpoint POST /chat deve retornar apenas JSON.
* A resposta do POST /chat deve conter answer, sources e model_used.
* O campo answer deve conter a resposta final ao usuário.
* O campo sources deve conter os trechos recuperados do contexto que fundamentaram a resposta.
* O campo model_used deve informar qual modelo foi utilizado.
* Crie o endpoint GET /health.
* O endpoint GET /health deve retornar o status da API.
* Use Pydantic para validar entrada e saída dos endpoints.
* Valide perguntas vazias.
* Trate erros de configuração ausente no .env.
* Trate erros de arquivo de contexto inexistente.
* Trate erros de falha na indexação.
* Trate erros de falha na chamada da LLM.
* Crie um frontend web simples para interação com o chatbot.
* O frontend deve ser focado apenas em conversa, sem dashboards, gráficos, painéis de dados, relatórios ou telas administrativas.
* A tela principal deve conter uma interface de chat com histórico de mensagens, campo de texto e botão de envio.
* O usuário deve conseguir enviar mensagens pressionando Enter.
* Exiba estado de carregamento enquanto o backend processa a resposta.
* Mostre mensagens do usuário e do assistente em balões visualmente distintos.
* Inclua sugestões de perguntas iniciais, como “Quem precisa declarar Imposto de Renda?”, “Como declarar dependentes?” e “O que é restituição?”.
* Ao clicar em uma sugestão, envie a pergunta para o chat.
* O frontend deve consumir o endpoint POST /chat do backend.
* Use azul institucional #1351B4 como cor primária, aplicado em cabeçalho, botões, links e elementos de destaque.
* Mantenha o fundo geral em branco ou cinza muito claro #F8F8F8.
* Use amarelo institucional #FFCD07 apenas em acentos pontuais, como pequenos ícones, selos ou destaques discretos.
* Não use amarelo como cor dominante.
* Use tons de cinza neutro para textos, divisores, bordas e mensagens secundárias.
* Use verde e vermelho apenas se houver algum indicador simples de status, erro ou sucesso.
* Use tipografia sem serifa.
* Use uma fonte de título com aparência mais geométrica, estilo Rawline.
* Use uma fonte de corpo neutra, estilo Open Sans ou Roboto.
* Estabeleça hierarquia tipográfica clara, com título principal em negrito e subtítulo em peso médio.
* Inclua uma faixa superior fina e discreta com links de acessibilidade, como “Ir para conteúdo”, “Alto contraste” e “Acessibilidade”.
* Adicione uma faixa institucional discreta no topo usando as cores azul, verde e amarelo do Governo Federal.
* Estruture o cabeçalho principal com identificação do projeto à esquerda e navegação simples à direita.
* O cabeçalho deve conter o nome “Assistente Virtual Fiscal”.
* A navegação pode conter links simples para “Início”, “Sobre” e “Fonte oficial”.
* Use botões primários sólidos em azul institucional com texto branco.
* Use botões secundários com contorno fino em azul institucional.
* Prefira ícones simples em estilo linha fina ou flat, monocromáticos na cor primária.
* Aplique cantos levemente arredondados, sombra sutil e borda fina nos blocos principais.
* Use espaçamento generoso entre seções para reforçar organização institucional.
* Construa a interface de forma responsiva e mobile-first.
* Em telas pequenas, a estrutura deve colapsar para uma coluna única.
* Garanta estados de foco visíveis em todos os elementos interativos, com contorno perceptível.
* Siga boas práticas de acessibilidade e contraste.
* Evite fotografias grandes de pessoas.
* Priorize ícones, pictogramas simples e elementos visuais institucionais.
* Inclua uma seção curta “Sobre o assistente” explicando que ele responde com base no material de contexto oficial carregado no sistema.
* Inclua uma observação informando que a resposta não substitui orientação oficial da Receita Federal ou atendimento especializado.
* Finalize a página com rodapé em azul ou cinza institucional escuro.
* O rodapé deve conter links simples, identificação acadêmica do projeto e referência à fonte oficial do conteúdo.
* Organize o backend com uma estrutura limpa.
* Use a seguinte estrutura base para o backend: app/main.py, app/api/routes.py, app/core/config.py, app/schemas/chat.py, app/services/llm_service.py, app/services/rag_service.py, app/services/context_loader.py, app/prompts, app/vectorstore, scripts/index_context.py.
* Organize o frontend em uma pasta frontend.
* Use componentes simples no frontend, como ChatPage, ChatMessage, ChatInput, SuggestedQuestions, Header, Footer e AccessibilityBar.
* Crie um arquivo requirements.txt com as dependências Python necessárias.
* Crie um Dockerfile para executar a aplicação.
* O Dockerfile deve usar uma imagem Python estável e leve.
* O Dockerfile deve instalar as dependências do requirements.txt.
* O Dockerfile deve copiar o código da aplicação para dentro do container.
* O Dockerfile deve expor a porta 8000.
* O Dockerfile deve executar a API com uvicorn, usando host 0.0.0.0 e porta 8000.
* Crie um arquivo docker-compose.yml para subir a aplicação localmente.
* O docker-compose.yml deve conter um serviço para a API FastAPI.
* O serviço da API deve usar o Dockerfile criado no projeto.
* O serviço da API deve carregar as variáveis do arquivo .env.
* O serviço da API deve mapear a porta 8000 do container para a porta 8000 da máquina local.
* O serviço da API deve montar volumes para permitir persistência do vector store local e acesso ao arquivo de contexto.
* O docker-compose.yml não deve criar banco de dados relacional.
* O docker-compose.yml não deve criar serviços desnecessários.
* Crie um README.md explicando como configurar o .env, instalar dependências, indexar o contexto, rodar a API e acessar o frontend.
* O README.md deve incluir comandos para rodar localmente e com Docker Compose.
* A API deve poder ser executada localmente com uvicorn.
* A aplicação também deve poder ser executada com docker compose up --build.
* Não implemente login.
* Não implemente banco de dados relacional.
* Não implemente WhatsApp, Telegram ou Streamlit neste momento.
* Não crie dashboards, gráficos, cards analíticos ou telas de observatório fiscal.
* Foque no chatbot fiscal, na API FastAPI, no RAG, na configuração por .env, no Docker e em uma interface web simples e institucional.
* Ao final, explique resumidamente quais arquivos foram criados, como executar o projeto localmente, como executar com Docker, como acessar o chatbot e como trocar o modelo/provedor da LLM pelo .env.
