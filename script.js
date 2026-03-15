const analyzeBtn = document.getElementById("analyzeBtn");
const sendBtn = document.getElementById("sendBtn");
const pdfInput = document.getElementById("pdfs");
const questionInput = document.getElementById("questionInput");
const chatWindow = document.getElementById("chatWindow");
const statusMsg = document.getElementById("statusMsg");
const fileList = document.getElementById("fileList");

/* backend select */
const backendSelect = document.getElementById("backendSelect");
const backendTrigger = document.getElementById("backendTrigger");
const backendMenu = document.getElementById("backendMenu");
const backendSelectedText = document.getElementById("backendSelectedText");
const backendOptions = document.querySelectorAll(".custom-select-option");

/* avatars */
const userAvatar =
  "https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png";
const botAvatar =
  "https://upload.wikimedia.org/wikipedia/commons/0/0c/Chatbot_img.png";

/* state */
let chatHistory = [];
let selectedBackend = "openai";

/* select */
if (backendTrigger) {
  backendTrigger.addEventListener("click", () => {
    backendSelect.classList.toggle("open");
  });
}

backendOptions.forEach((option) => {
  option.addEventListener("click", () => {
    backendOptions.forEach((opt) => opt.classList.remove("selected"));
    option.classList.add("selected");

    selectedBackend = option.dataset.value;
    backendSelectedText.textContent = option.textContent.trim();

    backendSelect.classList.remove("open");
  });
});

document.addEventListener("click", (e) => {
  if (backendSelect && !backendSelect.contains(e.target)) {
    backendSelect.classList.remove("open");
  }
});

/* helpers */
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

function showStatus(text) {
  if (!text || !text.trim()) {
    hideStatus();
    return;
  }
  statusMsg.textContent = text;
  statusMsg.classList.add("show");
}

function hideStatus() {
  statusMsg.textContent = "";
  statusMsg.classList.remove("show");
}

function showFileList(html) {
  if (!html || !html.trim()) {
    hideFileList();
    return;
  }
  fileList.innerHTML = html;
  fileList.classList.add("show");
}

function hideFileList() {
  fileList.innerHTML = "";
  fileList.classList.remove("show");
}

/* file upload */
pdfInput.addEventListener("change", () => {
  const files = pdfInput.files;

  if (!files || !files.length) {
    hideFileList();
    hideStatus();
    return;
  }

  let html = "<strong>Selected files:</strong><br>";
  for (const file of files) {
    html += `• ${escapeHtml(file.name)}<br>`;
  }

  showFileList(html);
  hideStatus();
});

/* analyze PDFs */
analyzeBtn.addEventListener("click", async () => {
  const files = pdfInput.files;
  const backend = selectedBackend;

  if (!files || !files.length) {
    hideFileList();
    hideStatus();
    return;
  }

  const formData = new FormData();
  formData.append("backend", backend);

  for (const file of files) {
    formData.append("pdfs", file);
  }

  showStatus("Analyzing PDFs... Please wait.");

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      showStatus(data.message);
      chatHistory = [];
      addMessage(
        "bot",
        "PDFs processed successfully. You can start asking questions now."
      );
    } else {
      showStatus(`Error: ${data.error}`);
    }
  } catch (error) {
    showStatus(`Error: ${error.message}`);
  }
});

/* ask question */
async function sendQuestion() {
  const question = questionInput.value.trim();
  const backend = selectedBackend;

  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        backend: backend,
      }),
    });

    const data = await response.json();

    if (data.success) {
      addMessage("bot", data.answer);
    } else {
      addMessage("bot", `Error: ${data.error}`);
    }
  } catch (error) {
    addMessage("bot", `Error: ${error.message}`);
  }
}

sendBtn.addEventListener("click", sendQuestion);

questionInput.addEventListener("keypress", (event) => {
  if (event.key === "Enter") {
    sendQuestion();
  }
});