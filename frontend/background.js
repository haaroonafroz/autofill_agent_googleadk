// Autofill Agent service worker.
// It is the only part of the extension that can use chrome.debugger, so all
// real mouse/keyboard input and screenshots go through here. The side panel
// asks for work with runtime messages of type "exec".

const REAL_ACTIONS = new Set(["click", "type", "key", "scroll", "wait"]);
const KEY_CODES = {
  Enter: 13, Tab: 9, Escape: 27, Backspace: 8, Delete: 46, " ": 32, Space: 32,
  ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40, Home: 36, End: 35,
  PageUp: 33, PageDown: 34,
};

chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});

// --- Choosing the page to work on -------------------------------------------

let lastWebTabId = null;

function isWebTab(tab) {
  return !!tab && !!tab.url && /^https?:/.test(tab.url);
}

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  const tab = await chrome.tabs.get(tabId).catch(() => null);
  if (isWebTab(tab)) lastWebTabId = tabId;
});

async function findTargetTab() {
  const focused = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  const anyActive = await chrome.tabs.query({ active: true });
  let tab = [...focused, ...anyActive].find(isWebTab);
  if (!tab && lastWebTabId !== null) {
    tab = await chrome.tabs.get(lastWebTabId).catch(() => null);
  }
  if (!isWebTab(tab)) {
    throw new Error("No web page found. Bring the page you want to fill to the front.");
  }
  return tab;
}

// --- Content script plumbing -------------------------------------------------

async function ensureContentScript(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: "aa.ping" }, { frameId: 0 });
  } catch (_) {
    // The page was open before the extension loaded: inject into every frame.
    await chrome.scripting.executeScript({
      target: { tabId, allFrames: true },
      files: ["content_script.js"],
    });
  }
}

async function sendToFrame(tabId, frameId, message) {
  return chrome.tabs.sendMessage(tabId, message, { frameId });
}

// Reads every frame of the page. Refs in sub-frames get a prefix so they stay unique.
async function snapshot(tabId) {
  await ensureContentScript(tabId);
  const frames = await chrome.webNavigation.getAllFrames({ tabId });
  const results = [];

  for (const frame of frames) {
    if (!/^https?:/.test(frame.url)) continue;
    const prefix = frame.frameId === 0 ? "" : `f${frame.frameId}_`;
    try {
      const reply = await sendToFrame(tabId, frame.frameId, { type: "aa.snapshot", prefix });
      if (reply && reply.ok) {
        results.push({
          frame_id: frame.frameId,
          url: reply.url,
          title: reply.title,
          text: reply.text,
          alerts: reply.alerts,
          elements: reply.elements,
        });
      }
    } catch (_) {
      // Frame cannot be reached (for example a sandboxed frame); skip it.
    }
  }

  const top = results.find((f) => f.frame_id === 0);
  if (!top) throw new Error("Could not read the page. Try reloading it.");
  return {
    url: top.url,
    title: top.title,
    text: top.text,
    alerts: top.alerts,
    frames: results,
  };
}

// --- Real mouse and keyboard input (Chrome DevTools Protocol) ----------------

async function withDebugger(tabId, fn) {
  const target = { tabId };
  await chrome.debugger.attach(target, "1.3");
  try {
    return await fn(target);
  } finally {
    await chrome.debugger.detach(target).catch(() => {});
  }
}

const cdp = (target, method, params = {}) => chrome.debugger.sendCommand(target, method, params);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function clickAt(target, x, y) {
  await cdp(target, "Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await cdp(target, "Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", clickCount: 1 });
  await cdp(target, "Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", clickCount: 1 });
}

async function typeText(target, text) {
  for (const ch of text) {
    await cdp(target, "Input.dispatchKeyEvent", { type: "keyDown", key: ch, text: ch, unmodifiedText: ch });
    await cdp(target, "Input.dispatchKeyEvent", { type: "keyUp", key: ch });
    await sleep(15);
  }
}

async function pressKey(target, key) {
  const code = KEY_CODES[key];
  const params = { key, code: key, windowsVirtualKeyCode: code || 0, nativeVirtualKeyCode: code || 0 };
  await cdp(target, "Input.dispatchKeyEvent", { type: "keyDown", ...params });
  await cdp(target, "Input.dispatchKeyEvent", { type: "keyUp", ...params });
}

// Locates an element in the top frame and returns its viewport centre.
async function locate(tabId, ref) {
  const reply = await sendToFrame(tabId, 0, { type: "aa.locate", ref });
  if (!reply || !reply.ok) throw new Error(reply ? reply.error : "could not locate element");
  return reply;
}

async function runRealAction(tabId, action, frameId) {
  if (action.type === "wait") {
    await sleep(action.ms);
    return { ok: true };
  }
  if (action.type === "scroll") {
    const info = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => ({ w: window.innerWidth, h: window.innerHeight }),
    });
    const { w, h } = info[0].result;
    return withDebugger(tabId, async (target) => {
      await cdp(target, "Input.dispatchMouseEvent", {
        type: "mouseWheel", x: w / 2, y: h / 2, deltaX: 0, deltaY: action.dy,
      });
      return { ok: true };
    });
  }

  if (frameId !== 0 && frameId !== undefined) {
    return { ok: false, error: "real input works on the top page only; use mode 'dom' inside frames" };
  }

  if (action.type === "click") {
    const spot = await locate(tabId, action.ref);
    if (spot.covered) {
      return { ok: false, error: "element is covered by another element (overlay or popup?)" };
    }
    return withDebugger(tabId, async (target) => {
      await clickAt(target, spot.x, spot.y);
      return { ok: true };
    });
  }

  if (action.type === "type") {
    return withDebugger(tabId, async (target) => {
      if (action.ref) {
        const spot = await locate(tabId, action.ref);
        if (spot.covered) return { ok: false, error: "element is covered by another element" };
        await clickAt(target, spot.x, spot.y);
        await sleep(50);
      }
      await typeText(target, action.text);
      return { ok: true };
    });
  }

  if (action.type === "key") {
    return withDebugger(tabId, async (target) => {
      await pressKey(target, action.key);
      return { ok: true };
    });
  }

  return { ok: false, error: `unsupported real action '${action.type}'` };
}

// Real actions go through chrome.debugger; everything else runs inside the element's frame.
function isRealAction(action) {
  if (action.type === "click") return action.mode !== "dom";
  return REAL_ACTIONS.has(action.type);
}

async function runAction(tabId, action, frameId) {
  if (isRealAction(action)) {
    return runRealAction(tabId, action, frameId);
  }
  return sendToFrame(tabId, frameId || 0, { type: "aa.act", action });
}

// --- Commands from the side panel -----------------------------------------

async function handleExec(command, payload) {
  if (command === "snapshot") {
    const tab = await findTargetTab();
    return snapshot(tab.id);
  }
  if (command === "act") {
    const tab = await findTargetTab();
    await ensureContentScript(tab.id);
    return runAction(tab.id, payload.action, payload.frame_id);
  }
  if (command === "screenshot") {
    const tab = await findTargetTab();
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
    return { dataUrl, url: tab.url };
  }
  throw new Error(`unknown command '${command}'`);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.type !== "exec") return false;
  handleExec(message.command, message.payload || {})
    .then((data) => sendResponse({ ok: true, data }))
    .catch((err) => sendResponse({ ok: false, error: String((err && err.message) || err) }));
  return true;
});
