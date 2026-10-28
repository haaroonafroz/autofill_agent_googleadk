// Autofill Agent side panel.
// Holds the WebSocket to the backend and acts as its bridge: when the backend
// sends a `request`, the panel forwards it to the service worker (background.js)
// and returns the result. The panel also has developer tools for testing.

// The backend address. Defaults to localhost:8000; tests can change it with
// chrome.storage (key: backend_url) to run a separate backend.
let API_URL = "http://localhost:8000";
let WS_URL = "ws://localhost:8000/ws";
const RECONNECT_MS = 2000;
const KEEPALIVE_MS = 20000;

const el = {
  userId: document.getElementById("userId"),
  connection: document.getElementById("connection"),
  cvStatus: document.getElementById("cvStatus"),
  cvFile: document.getElementById("cvFile"),
  uploadBtn: document.getElementById("uploadBtn"),
  uploadStatus: document.getElementById("uploadStatus"),
  readBtn: document.getElementById("readBtn"),
  runBtn: document.getElementById("runBtn"),
  actionsInput: document.getElementById("actionsInput"),
  devStatus: document.getElementById("devStatus"),
  pageInfo: document.getElementById("pageInfo"),
  elementsBody: document.querySelector("#elements tbody"),
  log: document.getElementById("log"),
  goalInput: document.getElementById("goalInput"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  agentStatus: document.getElementById("agentStatus"),
  agentLog: document.getElementById("agentLog"),
  askBox: document.getElementById("askBox"),
  askQuestion: document.getElementById("askQuestion"),
  askAnswer: document.getElementById("askAnswer"),
  askSend: document.getElementById("askSend"),
};

const DEFAULT_GOAL =
  "Fill out this job application using my CV. Answer only what the CV supports, " +
  "ask me when something is missing, and stop at the review page. Never submit.";
el.goalInput.value = DEFAULT_GOAL;

let askResolve = null;   // resolves the pending ask_human request
let agentRunning = false;

let ws = null;
let keepalive = null;
let devCounter = 0;

function generateUUID() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function log(message) {
  const time = new Date().toLocaleTimeString();
  el.log.textContent = `[${time}] ${message}\n` + el.log.textContent;
}

function setConnection(state, text) {
  el.connection.className = `pill pill-${state}`;
  el.connection.textContent = text;
}

// --- User id ----------------------------------------------------------------

function applyBackendUrl(url) {
  API_URL = url;
  WS_URL = `${url.replace(/^http/, "ws")}/ws`;
}

function loadUserId() {
  chrome.storage.local.get(["user_id", "backend_url"], (result) => {
    if (result.backend_url) applyBackendUrl(result.backend_url);
    if (result.user_id) {
      el.userId.value = result.user_id;
    } else {
      const id = generateUUID();
      chrome.storage.local.set({ user_id: id });
      el.userId.value = id;
    }
    refreshStatus();
    connect(); // connect only once the user id is known, so `hello` carries it
  });
}

el.userId.addEventListener("change", () => {
  const id = el.userId.value.trim();
  if (!id) return;
  chrome.storage.local.set({ user_id: id }, () => {
    reconnect();
    refreshStatus();
  });
});

// --- CV status and upload -------------------------------------------------

async function refreshStatus() {
  try {
    const res = await fetch(`${API_URL}/status?user_id=${encodeURIComponent(el.userId.value)}`);
    const data = await res.json();
    if (data.chunks > 0) {
      el.cvStatus.textContent = `CV indexed: ${data.chunks} chunks${data.has_profile ? ", profile ready" : ""}.`;
      el.cvStatus.className = "muted ok";
    } else {
      el.cvStatus.textContent = "No CV yet. Upload one below.";
      el.cvStatus.className = "muted";
    }
  } catch (_) {
    el.cvStatus.textContent = "Backend not reachable. Start the server on port 8000.";
    el.cvStatus.className = "muted error";
  }
}

el.uploadBtn.addEventListener("click", async () => {
  const file = el.cvFile.files[0];
  if (!file) {
    el.uploadStatus.textContent = "Choose a PDF first.";
    return;
  }
  const form = new FormData();
  form.append("file", file);
  form.append("user_id", el.userId.value);

  el.uploadBtn.disabled = true;
  el.uploadStatus.textContent = "Reading CV (this can take a minute the first time)...";
  el.uploadStatus.className = "muted";
  try {
    const res = await fetch(`${API_URL}/cv`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    el.uploadStatus.textContent = `Stored ${data.chunks} chunks for ${data.name || "the candidate"}.`;
    el.uploadStatus.className = "muted ok";
    log(`CV uploaded: ${data.chunks} chunks`);
    refreshStatus();
  } catch (err) {
    el.uploadStatus.textContent = `Upload failed: ${err.message}`;
    el.uploadStatus.className = "muted error";
  } finally {
    el.uploadBtn.disabled = false;
  }
});

// --- WebSocket to the backend ---------------------------------------------

function connect() {
  if (ws && ws.readyState <= WebSocket.OPEN) return;
  setConnection("wait", "Connecting...");
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "hello", user_id: el.userId.value }));
    keepalive = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "ping" }));
    }, KEEPALIVE_MS);
  };

  ws.onmessage = (event) => handleMessage(JSON.parse(event.data));

  ws.onclose = () => {
    clearInterval(keepalive);
    setConnection("off", "Offline");
    setTimeout(connect, RECONNECT_MS);
  };
}

function reconnect() {
  if (ws) {
    ws.onclose = null;
    ws.close();
  }
  connect();
}

function send(message) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    throw new Error("not connected to the backend");
  }
  ws.send(JSON.stringify(message));
}

async function handleMessage(msg) {
  switch (msg.type) {
    case "hello_ack":
      setConnection("on", "Connected");
      log(`Connected as ${msg.user_id} (${msg.chunks} CV chunks)`);
      break;

    case "request": {
      if (msg.command === "ask_human") {
        // The agent needs something only the candidate knows. Wait for the typed answer.
        const answer = await askCandidate(msg.payload.question);
        send({ type: "response", request_id: msg.request_id, ok: true, data: answer, error: null });
        break;
      }
      // The backend asked us to do something in the browser.
      const reply = await chrome.runtime.sendMessage({
        type: "exec",
        command: msg.command,
        payload: msg.payload,
      });
      send({
        type: "response",
        request_id: msg.request_id,
        ok: !!(reply && reply.ok),
        data: reply ? reply.data : null,
        error: reply && !reply.ok ? reply.error : (reply ? null : "no reply from background"),
      });
      break;
    }

    case "dev_result":
      if (String(msg.id || "").startsWith("agent-")) {
        handleAgentReply(msg);
      } else {
        handleDevResult(msg);
      }
      break;

    case "agent_event":
      renderAgentEvent(msg.kind, msg.data);
      break;

    case "pong":
      break;

    case "error":
      log(`Backend error: ${msg.error}`);
      break;
  }
}

// --- Developer tools ----------------------------------------------------------

function nextId() {
  devCounter += 1;
  return `dev-${devCounter}`;
}

el.readBtn.addEventListener("click", () => {
  try {
    send({ type: "dev.snapshot", id: nextId() });
    el.devStatus.textContent = "Reading page...";
    el.devStatus.className = "muted";
  } catch (err) {
    el.devStatus.textContent = err.message;
    el.devStatus.className = "muted error";
  }
});

el.runBtn.addEventListener("click", () => {
  let actions;
  try {
    actions = JSON.parse(el.actionsInput.value || "[]");
    if (!Array.isArray(actions)) throw new Error("actions must be a JSON array");
  } catch (err) {
    el.devStatus.textContent = `Invalid JSON: ${err.message}`;
    el.devStatus.className = "muted error";
    return;
  }
  try {
    send({ type: "dev.run", id: nextId(), actions });
    el.devStatus.textContent = "Running actions...";
    el.devStatus.className = "muted";
  } catch (err) {
    el.devStatus.textContent = err.message;
    el.devStatus.className = "muted error";
  }
});

function renderElements(data) {
  el.pageInfo.textContent = `${data.title || "(no title)"} - ${data.url}`;
  el.elementsBody.replaceChildren();
  for (const item of data.elements) {
    const row = document.createElement("tr");
    row.dataset.ref = item.ref;
    const value = item.checked !== undefined ? String(item.checked) : item.value ?? "";
    for (const text of [item.ref, item.input_type, item.label || "", value]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      if (text === item.ref) cell.className = "ref";
      row.appendChild(cell);
    }
    el.elementsBody.appendChild(row);
  }
}

function handleDevResult(msg) {
  if (!msg.ok) {
    el.devStatus.textContent = `Failed: ${msg.error}`;
    el.devStatus.className = "muted error";
    log(`Failed: ${msg.error}`);
    return;
  }
  if (msg.data && msg.data.elements) {
    renderElements(msg.data);
    el.devStatus.textContent = `Read ${msg.data.elements.length} elements.`;
    el.devStatus.className = "muted ok";
    log(`Snapshot: ${msg.data.elements.length} elements on ${msg.data.url}`);
    return;
  }
  if (msg.data && msg.data.results) {
    const allOk = msg.data.completed;
    el.devStatus.textContent = allOk ? "All actions completed." : "Stopped early. See log.";
    el.devStatus.className = `muted ${allOk ? "ok" : "error"}`;
    for (const result of msg.data.results) {
      const line = result.ok
        ? `ok    ${result.action} ${result.ref || ""} ${result.value_after !== undefined ? `-> ${result.value_after}` : ""}`
        : `${result.blocked ? "BLOCKED" : "FAIL "} ${result.action || ""} ${result.ref || ""}: ${result.error}`;
      log(line.trim());
    }
    return;
  }
  log(JSON.stringify(msg.data));
}

// --- Agent -----------------------------------------------------------------

function agentStatus(text, kind = "muted") {
  el.agentStatus.textContent = text;
  el.agentStatus.className = kind;
}

function setAgentRunning(running) {
  agentRunning = running;
  el.startBtn.disabled = running;
  el.stopBtn.disabled = !running;
}

// Adds one line to the agent feed. Text only, never HTML.
function feed(tag, text, cls = "") {
  const item = document.createElement("li");
  const label = document.createElement("span");
  label.className = "tag";
  label.textContent = tag;
  const body = document.createElement("span");
  body.className = cls;
  body.textContent = text;
  item.append(label, body);
  el.agentLog.append(item);
  while (el.agentLog.children.length > 200) el.agentLog.firstChild.remove();
  el.agentLog.lastElementChild.scrollIntoView({ block: "end" });
}

function describeCall(call) {
  const args = Object.entries(call.args || {})
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ");
  return `${call.name}(${args})`;
}

function renderAgentEvent(kind, data) {
  switch (kind) {
    case "observe":
      setAgentRunning(true);
      feed("page", `${data.title || "(untitled)"}: ${data.fields} fields`);
      break;
    case "decide":
      if (data.text) feed("model", data.text);
      feed(`turn ${data.step}`, data.calls.map(describeCall).join("  ") || "(no tool calls)");
      break;
    case "act":
      for (const r of data.results) {
        if (r.ok) feed("ok", r.tool || "turn", "good");
        else if (r.blocked) feed("blocked", r.error, "fail");
        else feed("failed", `${r.tool || "turn"}: ${r.error}`, "fail");
      }
      for (const note of data.notes || []) feed("note", note);
      break;
    case "verify":
      for (const issue of data.issues) feed("check", issue, "fail");
      break;
    case "ask":
      feed("asked", data.question);
      break;
    case "done":
      finishRun(data);
      break;
  }
}

function finishRun(data) {
  setAgentRunning(false);
  const outcome = data.outcome || {};
  const filled = (data.filled || []).length;
  if (outcome.status === "done") {
    agentStatus(`Reached the review page. ${filled} fields set. Review it and submit yourself.`, "ok");
  } else if (outcome.status === "stopped") {
    agentStatus(`Stopped: ${outcome.reason || "no reason given"}`, "muted");
  } else {
    agentStatus(`Failed: ${outcome.reason || "unknown error"}`, "error");
  }
  feed("done", `${outcome.status} after ${data.steps} turns`, outcome.status === "done" ? "good" : "fail");
}

function handleAgentReply(msg) {
  if (!msg.ok) {
    setAgentRunning(false);
    agentStatus(`Could not run: ${msg.error}`, "error");
    return;
  }
  if (msg.data && msg.data.run_id) {
    setAgentRunning(true);
    agentStatus("Agent running. Watch the progress below.", "muted");
    el.agentLog.replaceChildren();
  }
}

// Shows the question and resolves with the candidate's answer.
function askCandidate(question) {
  return new Promise((resolve) => {
    el.askQuestion.textContent = question;
    el.askAnswer.value = "";
    el.askBox.hidden = false;
    el.askAnswer.focus();
    askResolve = resolve;
  });
}

function answerQuestion(answer) {
  if (!askResolve) return;
  const resolve = askResolve;
  askResolve = null;
  el.askBox.hidden = true;
  resolve(answer);
}

el.askSend.addEventListener("click", () => answerQuestion(el.askAnswer.value.trim()));
el.askAnswer.addEventListener("keydown", (e) => {
  if (e.key === "Enter") answerQuestion(el.askAnswer.value.trim());
});

el.startBtn.addEventListener("click", () => {
  const goal = el.goalInput.value.trim() || DEFAULT_GOAL;
  try {
    send({ type: "agent.start", id: `agent-${nextId()}`, goal });
    setAgentRunning(true);
    el.agentLog.replaceChildren();
    agentStatus("Starting...", "muted");
  } catch (err) {
    agentStatus(err.message, "error");
  }
});

el.stopBtn.addEventListener("click", () => {
  try {
    send({ type: "agent.stop", id: `agent-${nextId()}` });
    agentStatus("Stopping after the current step...", "muted");
  } catch (err) {
    agentStatus(err.message, "error");
  }
  answerQuestion(""); // unblock a pending question
});

// --- Start -----------------------------------------------------------------

loadUserId();
