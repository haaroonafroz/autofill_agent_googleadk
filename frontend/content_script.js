// Autofill Agent content script. Runs in every frame of the page.
// It builds a DOM snapshot with stable element refs and performs DOM actions.
// Real mouse/keyboard input is sent by background.js via chrome.debugger.
(() => {
  if (window.__autofillAgentLoaded) return;
  window.__autofillAgentLoaded = true;

  const REF_ATTR = "data-aa-ref";
  const MAX_ELEMENTS = 300;
  const INTERACTIVE_SELECTOR = [
    "input", "textarea", "select", "button", "a[href]",
    "[role='button']", "[role='checkbox']", "[role='radio']", "[role='switch']",
    "[role='combobox']", "[role='textbox']", "[role='option']", "[role='listbox']",
    "[contenteditable='true']", "[contenteditable='']",
  ].join(",");
  const SUBMIT_WORDS = ["submit", "apply", "send application", "finish", "confirm", "pay", "place order"];

  // Walks into open shadow roots so custom components are found too.
  function deepQueryAll(root, selector, out = []) {
    root.querySelectorAll(selector).forEach((el) => out.push(el));
    root.querySelectorAll("*").forEach((el) => {
      if (el.shadowRoot) deepQueryAll(el.shadowRoot, selector, out);
    });
    return out;
  }

  function clean(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  // Text of a label, without the text of the controls inside it (e.g. select options).
  function labelText(node) {
    const copy = node.cloneNode(true);
    copy.querySelectorAll("select, textarea, input, button, option, script, style").forEach((n) => n.remove());
    return clean(copy.textContent);
  }

  function isVisible(el) {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    const style = getComputedStyle(el);
    return style.visibility !== "hidden" && style.display !== "none";
  }

  function fieldLabel(el) {
    const labelledBy = el.getAttribute("aria-labelledby");
    if (labelledBy) {
      const text = labelledBy
        .split(/\s+/)
        .map((id) => (document.getElementById(id) ? document.getElementById(id).innerText : ""))
        .join(" ");
      if (clean(text)) return clean(text);
    }
    if (el.labels && el.labels.length) {
      const text = [...el.labels].map(labelText).join(" ");
      if (clean(text)) return clean(text);
    }
    for (const attr of ["aria-label", "placeholder", "title", "alt"]) {
      if (el.getAttribute(attr)) return clean(el.getAttribute(attr));
    }
    if (el.tagName === "BUTTON" || el.getAttribute("role") === "button" || el.type === "submit" || el.type === "button") {
      return clean(el.innerText || el.value || "");
    }
    return clean(el.name || el.id || "");
  }

  function errorText(el) {
    const messages = [];
    for (const attr of ["aria-errormessage", "aria-describedby"]) {
      const ids = el.getAttribute(attr);
      if (!ids) continue;
      ids.split(/\s+/).forEach((id) => {
        const node = document.getElementById(id);
        if (!node) return;
        const looksLikeError = node.getAttribute("role") === "alert" || /error|invalid/i.test(node.className || "");
        if (looksLikeError) messages.push(clean(node.innerText));
      });
    }
    if (el.getAttribute("aria-invalid") === "true" || (el.validity && !el.validity.valid)) {
      if (el.validationMessage) messages.push(el.validationMessage);
    }
    return clean(messages.join(" | ")) || null;
  }

  function refFor(el, prefix) {
    let ref = el.getAttribute(REF_ATTR);
    if (!ref) {
      window.__aaCounter = (window.__aaCounter || 0) + 1;
      ref = `${prefix}e${window.__aaCounter}`;
      el.setAttribute(REF_ATTR, ref);
    }
    return ref;
  }

  function describe(el, prefix) {
    const tag = el.tagName.toLowerCase();
    const inputType = tag === "input" ? (el.type || "text") : tag === "select" ? "select" : tag === "textarea" ? "textarea" : tag;
    const entry = {
      ref: refFor(el, prefix),
      tag,
      input_type: inputType,
      role: el.getAttribute("role") || null,
      label: fieldLabel(el),
      name: el.name || null,
      id: el.id || null,
      required: !!el.required || el.getAttribute("aria-required") === "true",
      disabled: !!el.disabled,
      readonly: !!el.readOnly,
      error: errorText(el),
    };

    if (inputType === "password") {
      entry.value = "[redacted]";
    } else if (el.isContentEditable) {
      entry.value = clean(el.innerText).slice(0, 200);
    } else if ("value" in el) {
      // Full value (capped at 2000 chars), so verification compares real content.
      // The prompt builder shortens it for the model.
      entry.value = String(el.value).slice(0, 2000);
    }

    if (inputType === "checkbox" || inputType === "radio") {
      entry.checked = el.checked;
    }
    if (tag === "select") {
      entry.options = [...el.options].slice(0, 100).map((o) => ({
        value: o.value,
        text: clean(o.textContent),
        selected: o.selected,
      }));
    }
    if (tag === "a") {
      entry.href = el.getAttribute("href");
    }
    if (tag === "a" || tag === "button" || el.getAttribute("role") === "button") {
      entry.text = clean(el.innerText).slice(0, 80);
    }
    if (el.placeholder) entry.placeholder = el.placeholder;
    return entry;
  }

  function snapshot(prefix) {
    const elements = [];
    const seen = new Set();
    for (const el of deepQueryAll(document, INTERACTIVE_SELECTOR)) {
      if (seen.has(el)) continue;
      seen.add(el);
      if (el.tagName === "INPUT" && el.type === "hidden") continue;
      if (!isVisible(el)) continue;
      elements.push(describe(el, prefix));
      if (elements.length >= MAX_ELEMENTS) break;
    }

    const alerts = [...document.querySelectorAll("[role='alert']")]
      .map((n) => clean(n.innerText))
      .filter(Boolean)
      .slice(0, 5);

    return {
      ok: true,
      title: document.title,
      url: location.href,
      text: clean(document.body ? document.body.innerText : "").slice(0, 2000),
      alerts,
      elements,
    };
  }

  // Finds an element by its ref, including inside open shadow roots.
  function findByRef(ref) {
    return deepQueryAll(document, `[${REF_ATTR}]`).find((el) => el.getAttribute(REF_ATTR) === ref) || null;
  }

  function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 50)));
  }

  function fail(message, extra = {}) {
    return { ok: false, error: message, ...extra };
  }

  function valueAfter(el) {
    if (el.tagName === "SELECT") return el.options[el.selectedIndex] ? el.options[el.selectedIndex].text : "";
    if (el.isContentEditable) return clean(el.innerText);
    if (el.type === "checkbox" || el.type === "radio") return el.checked;
    return el.value;
  }

  function setNativeValue(el, value) {
    const proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    setter.call(el, value); // bypasses framework value trackers (React etc.)
  }

  function fireChange(el) {
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function submitLike(el) {
    const text = `${fieldLabel(el)} ${el.type || ""}`.toLowerCase();
    return SUBMIT_WORDS.some((w) => text.includes(w)) || el.type === "submit";
  }

  function doFill(el, action) {
    if (el.tagName === "SELECT") return fail("use a select action for dropdowns");
    if (el.type === "file") return fail("file inputs need human approval");
    if (el.type === "password") return fail("passwords are never filled by the agent");
    if (el.disabled || el.readOnly) return fail("field is disabled or read-only");
    el.focus();
    if (el.isContentEditable) {
      el.textContent = action.value;
      el.dispatchEvent(new InputEvent("input", { bubbles: true, data: action.value, inputType: "insertText" }));
    } else {
      setNativeValue(el, action.value);
      fireChange(el);
    }
    el.dispatchEvent(new Event("blur", { bubbles: true }));
    return { ok: true, value_after: valueAfter(el) };
  }

  function doSelect(el, action) {
    if (el.tagName !== "SELECT") return fail("select action needs a dropdown element");
    const wanted = (action.value ?? "").toString();
    const label = (action.label ?? "").toString().trim().toLowerCase();
    const options = [...el.options];
    const option =
      (action.value != null && options.find((o) => o.value === wanted)) ||
      (label && options.find((o) => clean(o.textContent).toLowerCase() === label));
    if (!option) {
      return fail("no matching option", { available: options.map((o) => ({ value: o.value, text: clean(o.textContent) })) });
    }
    el.focus();
    el.value = option.value;
    fireChange(el);
    return { ok: true, value_after: clean(option.textContent) };
  }

  function doCheck(el, action) {
    if (!(el.type === "checkbox" || el.type === "radio")) return fail("check action needs a checkbox or radio");
    const want = action.checked !== false;
    if (el.checked !== want) {
      if (el.type === "radio" && !want) return fail("a radio button cannot be unchecked directly");
      el.click(); // a real click runs the page's own handlers
    }
    if (el.checked !== want) return fail("the page did not change the checkbox state");
    return { ok: true, value_after: el.checked };
  }

  function doDomClick(el) {
    if (el.type === "file") return fail("file pickers need human approval");
    if (submitLike(el)) return fail("clicking a submit-like control needs human approval");
    el.click();
    return { ok: true };
  }

  const handlers = {
    "aa.ping": () => ({ ok: true }),

    "aa.snapshot": (msg) => snapshot(msg.prefix || ""),

    // Returns the viewport centre of an element, for real mouse input.
    "aa.locate": async (msg) => {
      const el = findByRef(msg.ref);
      if (!el) return fail(`unknown ref '${msg.ref}'; read the page again`);
      el.scrollIntoView({ block: "center", inline: "center" });
      await nextFrame();
      const rect = el.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return fail("element has no size on screen");
      const x = rect.left + rect.width / 2;
      const y = rect.top + rect.height / 2;
      const topmost = document.elementFromPoint(x, y);
      const covered = !!topmost && topmost !== el && !el.contains(topmost);
      return { ok: true, x, y, covered };
    },

    "aa.act": async (msg) => {
      const action = msg.action;
      if (action.type === "scroll" || action.type === "wait") return fail("handled by background");
      const el = findByRef(action.ref);
      if (!el) return fail(`unknown ref '${action.ref}'; read the page again`);
      if (el.disabled && action.type !== "wait") return fail("element is disabled");
      switch (action.type) {
        case "fill": return doFill(el, action);
        case "select": return doSelect(el, action);
        case "check": return doCheck(el, action);
        case "click": return doDomClick(el);
        default: return fail(`action '${action.type}' is not a DOM action`);
      }
    },
  };

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    const handler = handlers[msg && msg.type];
    if (!handler) return false;
    Promise.resolve()
      .then(() => handler(msg))
      .then((result) => sendResponse(result))
      .catch((err) => sendResponse(fail(String((err && err.message) || err))));
    return true; // keep the channel open for async replies
  });
})();
