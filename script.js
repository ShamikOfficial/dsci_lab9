const sendBtn = document.getElementById("sendBtn");
const questionInput = document.getElementById("questionInput");
const chatWindow = document.getElementById("chatWindow");
const modelLoadingOverlay = document.getElementById("modelLoadingOverlay");
const thinkingIndicator = document.getElementById("thinkingIndicator");
const backendTrigger = document.getElementById("backendTrigger");
const backendSelect = document.getElementById("backendSelect");
const backendSelectedText = document.getElementById("backendSelectedText");
const backendMenu = document.getElementById("backendMenu");

const userAvatar =
  "https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png";
const botAvatar =
  "https://upload.wikimedia.org/wikipedia/commons/0/0c/Chatbot_img.png";

let chatHistory = [];
let selectedBackend = "openai";
let backendOptions = []; // { id, label }
let isLoadingModel = false;

async function fetchBackends() {
  const response = await fetch("/backends");
  const data = await response.json();
  return data;
}

function buildBackendMenu(options, defaultId) {
  if (!backendMenu) return;
  backendMenu.innerHTML = "";
  options.forEach((opt) => {
    const div = document.createElement("div");
    div.className = "custom-select-option" + (opt.id === defaultId ? " selected" : "");
    div.dataset.value = opt.id;
    div.textContent = opt.label;
    backendMenu.appendChild(div);
  });
  backendOptions = document.querySelectorAll(".custom-select-option");
  attachBackendListeners();
}

function attachBackendListeners() {
  backendOptions.forEach((option) => {
    option.addEventListener("click", async () => {
      const backend = option.dataset.value;
      const label = option.textContent.trim();

      if (backend === selectedBackend) {
        backendSelect.classList.remove("open");
        return;
      }

      if (backend !== "openai") {
        const result = await ensureBackendLoaded(backend);
        if (result !== true) {
          addMessage("bot", `Could not load model: ${result}`);
          backendSelect.classList.remove("open");
          return;
        }
      }

      backendOptions.forEach((opt) => opt.classList.remove("selected"));
      option.classList.add("selected");
      selectedBackend = backend;
      backendSelectedText.textContent = label;
      backendSelect.classList.remove("open");
    });
  });
}

function showModelLoading(show) {
  isLoadingModel = show;
  backendTrigger.disabled = show;
  if (modelLoadingOverlay) {
    modelLoadingOverlay.setAttribute("aria-hidden", !show);
    modelLoadingOverlay.classList.toggle("show", show);
  }
}

function showThinking(show) {
  if (thinkingIndicator) {
    thinkingIndicator.setAttribute("aria-hidden", !show);
    thinkingIndicator.classList.toggle("show", show);
  }
}

async function ensureBackendLoaded(backend) {
  if (backend === "openai") return true;
  showModelLoading(true);
  try {
    const response = await fetch("/load_backend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend: backend }),
    });
    const data = await response.json();
    if (data.success) return true;
    return data.error || "Failed to load model";
  } catch (err) {
    return err.message || "Network error";
  } finally {
    showModelLoading(false);
  }
}

if (backendTrigger) {
  backendTrigger.addEventListener("click", () => {
    if (isLoadingModel) return;
    backendSelect.classList.toggle("open");
  });
}

document.addEventListener("click", (e) => {
  if (backendSelect && !backendSelect.contains(e.target)) {
    backendSelect.classList.remove("open");
  }
});

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function createMessageHTML(sender, text) {
  const avatar = sender === "user" ? userAvatar : botAvatar;
  return `
    <div class="chat-message ${sender}">
      <div class="avatar">
        <img src="${avatar}" alt="${sender} avatar">
      </div>
      <div class="message">${escapeHtml(text)}</div>
    </div>
  `;
}

function renderChatHistory() {
  chatWindow.innerHTML = "";
  chatHistory.forEach((item) => {
    chatWindow.innerHTML += createMessageHTML(item.sender, item.text);
  });
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addMessage(sender, text) {
  chatHistory.push({ sender, text });
  renderChatHistory();
}

async function sendQuestion() {
  const question = questionInput.value.trim();
  const backend = selectedBackend;

  if (!question) return;

  if (backend !== "openai") {
    const loaded = await ensureBackendLoaded(backend);
    if (loaded !== true) {
      addMessage("bot", `Error: ${loaded}`);
      return;
    }
  }

  addMessage("user", question);
  questionInput.value = "";

  showThinking(true);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, backend: backend }),
    });

    const data = await response.json();

    if (data.success) {
      addMessage("bot", data.answer);
    } else {
      addMessage("bot", `Error: ${data.error}`);
    }
  } catch (error) {
    addMessage("bot", `Error: ${error.message}`);
  } finally {
    showThinking(false);
  }
}

sendBtn.addEventListener("click", sendQuestion);

questionInput.addEventListener("keypress", (event) => {
  if (event.key === "Enter") sendQuestion();
});

// Load backends and build dropdown on page load
(async () => {
  try {
    const data = await fetchBackends();
    const options = data.backends || [];
    const defaultId = data.default || "openai";
    selectedBackend = defaultId;
    const defaultOpt = options.find((o) => o.id === defaultId);
    backendSelectedText.textContent = defaultOpt ? defaultOpt.label : "OpenAI";
    buildBackendMenu(options, defaultId);
  } catch (err) {
    backendSelectedText.textContent = "OpenAI";
    buildBackendMenu([{ id: "openai", label: "OpenAI" }], "openai");
  }
})();
