/**
 * Shukketsu Chat — WebSocket client with streaming LLM responses.
 *
 * Connects to /ws/chat, sends user messages as JSON, receives
 * streamed tokens, and renders final responses as sanitized Markdown.
 */

// --- State ---
let socket = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 10;
let currentAssistantEl = null;
let fullResponse = "";
let isStreaming = false;
let lastTraceId = null;

// --- DOM ---
const messagesEl = document.getElementById("chat-messages");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const stopBtn = document.getElementById("stop-btn");
const welcomeEl = document.getElementById("welcome-state");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

// --- Markdown config ---
marked.setOptions({ breaks: true, gfm: true });

// ============================================================
// WebSocket
// ============================================================

function connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${location.host}/ws/chat`);

    socket.onopen = () => {
        reconnectAttempts = 0;
        updateStatus("connected");
    };

    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };

    socket.onclose = () => {
        updateStatus("disconnected");
        if (isStreaming) endStreaming();
        scheduleReconnect();
    };

    socket.onerror = () => {
        // onclose fires after this — no action needed here
    };
}

function scheduleReconnect() {
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        updateStatus("failed");
        return;
    }
    const delay = Math.min(2000 * Math.pow(1.5, reconnectAttempts), 30000);
    reconnectAttempts++;
    updateStatus("reconnecting");
    setTimeout(connectWebSocket, delay);
}

function updateStatus(state) {
    const states = {
        connected:    { color: "bg-green-500",                  text: "Connected" },
        disconnected: { color: "bg-gray-500",                   text: "Disconnected" },
        reconnecting: { color: "bg-yellow-500 status-pulse",    text: "Reconnecting..." },
        failed:       { color: "bg-red-500",                    text: "Connection failed" },
    };
    const s = states[state] || states.disconnected;
    statusDot.className = `w-2 h-2 rounded-full ${s.color}`;
    statusText.textContent = s.text;
}

// ============================================================
// Message Handling
// ============================================================

function handleMessage(msg) {
    switch (msg.type) {
        case "token":
            if (currentAssistantEl) {
                fullResponse += msg.content;
                currentAssistantEl.textContent = fullResponse;
                scrollToBottom();
            }
            break;

        case "done":
            if (currentAssistantEl) {
                fullResponse = msg.content;
                lastTraceId = msg.trace_id || null;
                if (fullResponse) {
                    currentAssistantEl.innerHTML =
                        DOMPurify.sanitize(marked.parse(fullResponse));
                    currentAssistantEl.classList.add("markdown-body");
                }
                // Add feedback buttons
                if (lastTraceId) {
                    const feedbackRow = document.createElement("div");
                    feedbackRow.className = "flex gap-2 mt-2 feedback-row";

                    const upBtn = document.createElement("button");
                    upBtn.className = "thumb-btn text-parchment-dim hover:text-green-400 transition-colors text-sm px-2 py-1 rounded border border-white/10 hover:border-green-400/50";
                    upBtn.textContent = "👍";
                    upBtn.dataset.traceId = lastTraceId;
                    upBtn.addEventListener("click", function() {
                        sendFeedback(1, this.dataset.traceId, this.parentElement);
                    });

                    const downBtn = document.createElement("button");
                    downBtn.className = "thumb-btn text-parchment-dim hover:text-red-400 transition-colors text-sm px-2 py-1 rounded border border-white/10 hover:border-red-400/50";
                    downBtn.textContent = "👎";
                    downBtn.dataset.traceId = lastTraceId;
                    downBtn.addEventListener("click", function() {
                        sendFeedback(0, this.dataset.traceId, this.parentElement);
                    });

                    feedbackRow.appendChild(upBtn);
                    feedbackRow.appendChild(downBtn);
                    currentAssistantEl.parentElement.appendChild(feedbackRow);
                }
            }
            endStreaming();
            break;

        case "error":
            appendError(msg.content);
            if (isStreaming) endStreaming();
            break;

        case "status":
            if (msg.content !== "connected" && currentAssistantEl) {
                const statusSpan = document.createElement("span");
                statusSpan.className = "text-parchment-dim text-sm loading-dots";
                statusSpan.textContent = msg.content;
                currentAssistantEl.replaceChildren(statusSpan);
                scrollToBottom();
            }
            break;
    }
}

// ============================================================
// UI Helpers
// ============================================================

function appendUserMessage(text) {
    hideWelcome();
    const el = document.createElement("div");
    el.className = "flex justify-end";
    const bubble = document.createElement("div");
    bubble.className =
        "max-w-[80%] bg-shadow-mid border border-wow-gold/20 rounded-lg px-4 py-3";
    const p = document.createElement("p");
    p.className = "whitespace-pre-wrap";
    p.textContent = text;
    bubble.appendChild(p);
    el.appendChild(bubble);
    messagesEl.appendChild(el);
    scrollToBottom();
}

function appendAssistantBubble() {
    hideWelcome();
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start";
    const bubble = document.createElement("div");
    bubble.className =
        "max-w-[80%] bg-shadow border border-white/10 rounded-lg px-4 py-3 min-h-[2rem]";
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    scrollToBottom();
    return bubble;
}

function appendError(text) {
    const el = document.createElement("div");
    el.className = "flex justify-start";
    const bubble = document.createElement("div");
    bubble.className =
        "max-w-[80%] bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm";
    bubble.textContent = text;
    el.appendChild(bubble);
    messagesEl.appendChild(el);
    scrollToBottom();
}

function hideWelcome() {
    if (welcomeEl) {
        welcomeEl.style.display = "none";
    }
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ============================================================
// Streaming State
// ============================================================

function startStreaming() {
    isStreaming = true;
    inputEl.disabled = true;
    inputEl.classList.add("opacity-50");
    sendBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    currentAssistantEl = appendAssistantBubble();
    fullResponse = "";
}

function endStreaming() {
    isStreaming = false;
    inputEl.disabled = false;
    inputEl.classList.remove("opacity-50");
    sendBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    currentAssistantEl = null;
    fullResponse = "";
    inputEl.focus();
}

// ============================================================
// Send / Stop
// ============================================================

function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || !socket || socket.readyState !== WebSocket.OPEN) return;
    if (isStreaming) return;

    appendUserMessage(text);
    socket.send(JSON.stringify({ type: "message", content: text }));
    inputEl.value = "";
    inputEl.style.height = "auto";
    startStreaming();
}

function stopGeneration() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "stop" }));
    }
}

// ============================================================
// Input Handling
// ============================================================

inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
    if (e.key === "Escape") {
        inputEl.value = "";
        inputEl.style.height = "auto";
    }
});

// Auto-resize textarea (1 line to max 6 lines)
inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + "px";
});

sendBtn.addEventListener("click", sendMessage);
stopBtn.addEventListener("click", stopGeneration);

// Example prompts
document.querySelectorAll(".example-prompt").forEach((btn) => {
    btn.addEventListener("click", () => {
        inputEl.value = btn.dataset.prompt;
        sendMessage();
    });
});

// ============================================================
// Feedback
// ============================================================

function sendFeedback(score, traceId, row) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "feedback", score: score, trace_id: traceId }));
    }
    // Disable buttons after click
    row.querySelectorAll(".thumb-btn").forEach(btn => {
        btn.disabled = true;
        btn.classList.add("opacity-30", "cursor-not-allowed");
    });
    // Highlight selected
    const btns = row.querySelectorAll(".thumb-btn");
    if (score === 1) btns[0].classList.add("text-green-400", "border-green-400/50");
    else btns[1].classList.add("text-red-400", "border-red-400/50");
}

// ============================================================
// Init
// ============================================================

inputEl.focus();
connectWebSocket();
