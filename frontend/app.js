const initialQuestions = [
  "Quem precisa declarar Imposto de Renda?",
  "Como declarar dependentes?",
  "O que é restituição?",
];

const state = {
  sessionId: crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2),
  messages: [
    {
      role: "assistant",
      content:
        "Olá. Envie sua dúvida sobre Imposto de Renda para eu consultar a base carregada no sistema.",
      sources: [],
    },
  ],
  loading: false,
};

function GovernmentStrip() {
  return `
    <div class="gov-strip" aria-hidden="true"><span></span><span></span><span></span></div>
  `;
}

function Header() {
  return `
    <header class="header">
      <div class="header-inner">
        <div class="brand">
          <span class="brand-icon" aria-hidden="true">CEFET</span>
          <span>
            <strong>Assistente Virtual Fiscal</strong>
            <small>CEFET Varginha</small>
          </span>
        </div>
        <div class="local-mark" aria-label="Referência visual à cidade de Varginha">
          <span aria-hidden="true">ET</span>
        </div>
        <nav class="nav" aria-label="Navegação principal">
          <a href="#conteudo">Início</a>
          <a href="#sobre">Sobre</a>
          <a href="/tecnologias.html">Tecnologias</a>
          <a href="https://www.gov.br/receitafederal" target="_blank" rel="noopener noreferrer">Fonte oficial</a>
        </nav>
      </div>
    </header>
  `;
}

function SuggestedQuestions() {
  return `
    <div class="suggestions" aria-label="Sugestões de perguntas">
      ${initialQuestions
        .map((question) => `<button class="suggestion-button" type="button" data-question="${question}">${question}</button>`)
        .join("")}
    </div>
  `;
}

function ChatMessage(message) {
  const sources =
    message.sources?.length && message.role === "assistant"
      ? `<div class="sources">Base consultada: ${message.sources.length} trecho(s) recuperado(s).</div>`
      : "";
  const content = message.role === "assistant" ? renderMarkdown(message.content) : escapeHtml(message.content);
  return `<div class="message ${message.role}">${content}${sources}</div>`;
}

function ChatInput() {
  return `
    <form class="chat-form" id="chat-form">
      <textarea class="chat-input" id="chat-input" rows="2" placeholder="Digite sua pergunta" aria-label="Digite sua pergunta"></textarea>
      <button class="primary-button" type="submit" ${state.loading ? "disabled" : ""}>Enviar</button>
    </form>
  `;
}

function ChatPage() {
  const loadingMessage = state.loading
    ? '<div class="message assistant loading">Consultando a base e preparando a resposta...</div>'
    : "";

  return `
    ${GovernmentStrip()}
    ${Header()}
    <main class="main" id="conteudo">
      <section class="intro">
        <h1>Assistente Virtual Fiscal</h1>
        <p>Converse com um assistente que responde dúvidas de Imposto de Renda com base no contexto oficial carregado no sistema.</p>
      </section>
      <section class="chat-shell" aria-label="Chat fiscal">
        ${SuggestedQuestions()}
        <div class="messages" id="messages" aria-live="polite">
          ${state.messages.map(ChatMessage).join("")}
          ${loadingMessage}
        </div>
        ${ChatInput()}
      </section>
      <section class="about" id="sobre">
        <h2>Sobre o assistente</h2>
        <p>Este assistente responde com base no material de contexto oficial carregado no sistema. Quando a resposta não está na base consultada, ele informa que não encontrou uma resposta segura.</p>
        <p>Este site é um apoio do Núcleo de Apoio Fiscal do CEFET Varginha e integra uma iniciativa acadêmica vinculada ao curso de Bacharelado em Sistemas de Informação do CEFET Varginha.</p>
        <p class="notice">As respostas não substituem orientação oficial da Receita Federal nem atendimento especializado.</p>
      </section>
    </main>
    ${Footer()}
  `;
}

function Footer() {
  return `
    <footer class="footer">
      <div class="footer-inner">
        <span>Projeto acadêmico - Bacharelado em Sistemas de Informação do CEFET Varginha</span>
        <span>Apoio: Núcleo de Apoio Fiscal do CEFET Varginha</span>
        <span><a href="https://www.gov.br/receitafederal" target="_blank" rel="noopener noreferrer">Receita Federal</a></span>
      </div>
    </footer>
  `;
}

function render() {
  document.querySelector("#app").innerHTML = ChatPage();
  bindEvents();
  scrollMessagesToBottom();
}

function bindEvents() {
  document.querySelector("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = document.querySelector("#chat-input");
    submitQuestion(input.value);
  });

  document.querySelector("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitQuestion(event.currentTarget.value);
    }
  });

  document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => submitQuestion(button.dataset.question));
  });
}

async function submitQuestion(rawQuestion) {
  const question = rawQuestion.trim();
  if (!question || state.loading) {
    return;
  }

  state.messages.push({ role: "user", content: question, sources: [] });
  state.loading = true;
  render();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: state.sessionId }),
    });

    const data = await response.json();
    const requestId = response.headers.get("X-Request-ID");
    if (!response.ok) {
      console.error("Falha na requisição do chat", {
        status: response.status,
        requestId,
      });
      const errorMessage = typeof data.detail === "string" ? data.detail : "Falha ao consultar o assistente.";
      throw new Error(errorMessage);
    }

    state.messages.push({
      role: "assistant",
      content: data.answer,
      sources: data.sources || [],
    });
  } catch (error) {
    state.messages.push({
      role: "assistant",
      content: error.message || "Não foi possível processar sua pergunta agora. Tente novamente mais tarde.",
      sources: [],
    });
  } finally {
    state.loading = false;
    render();
  }
}

function scrollMessagesToBottom() {
  const messages = document.querySelector("#messages");
  messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value;
  return element.innerHTML;
}

function renderMarkdown(value) {
  const escaped = escapeHtml(value);
  const lines = escaped.split("\n");
  const html = [];
  let inList = false;

  lines.forEach((line) => {
    const listMatch = line.match(/^\s*[-*]\s+(.+)/);
    if (listMatch) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${formatInlineMarkdown(listMatch[1])}</li>`);
      return;
    }

    if (inList) {
      html.push("</ul>");
      inList = false;
    }

    if (!line.trim()) {
      html.push("<br>");
      return;
    }

    html.push(`<p>${formatInlineMarkdown(line)}</p>`);
  });

  if (inList) {
    html.push("</ul>");
  }

  return html.join("");
}

function formatInlineMarkdown(value) {
  return value
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

render();
