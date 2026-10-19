// Frontend for the Gemini Router Tauri app.
//
// Talks to the Rust backend exclusively through Tauri commands and two events
// (`router:status` and `router:log`). No bundler, no framework.

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );

const MODES = ["flash", "flash_lite", "both"];

let state = null;
let statusSlot = null;
// Only rebuild the key/token inputs when their shape changes, so we never wipe
// what the user is typing.
let renderedKeySlotCount = -1;
let renderedTokenSignature = "";
let chainsDirty = false;
let bannerTimer = null;

// ---------------------------------------------------------------------------
// Plumbing
// ---------------------------------------------------------------------------

function setBanner(message) {
  const banner = $("banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
  if (bannerTimer) clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => banner.classList.add("hidden"), 9000);
}

async function call(command, args) {
  try {
    return await invoke(command, args ?? {});
  } catch (error) {
    setBanner(String(error));
    return null;
  }
}

async function refresh() {
  const args = statusSlot == null ? {} : { statusSlot };
  const snapshot = await call("refresh", args);
  if (snapshot) {
    state = snapshot;
    statusSlot = snapshot.status_slot;
    render();
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render() {
  if (!state) return;
  renderKeyArea();
  renderMode();
  renderModelStatus();
  renderServer();
  renderTokens();
  renderLog();
}

function renderKeyArea() {
  const slots = state.key_slots;
  const container = $("key-slots");

  if (renderedKeySlotCount !== slots.length) {
    container.innerHTML = slots
      .map(
        (slot) => `
        <div class="key-row">
          <span class="key-label">${esc(slot.label)}</span>
          <input type="password" id="key-input-${slot.index}" placeholder="Paste a Gemini API key"
                 autocomplete="off" spellcheck="false" />
          <button class="ghost" data-action="toggle" data-slot="${slot.index}">Show</button>
          <button data-action="save" data-slot="${slot.index}">Save</button>
          <button data-action="test" data-slot="${slot.index}">Test</button>
          <button class="ghost" data-action="clear" data-slot="${slot.index}">Clear</button>
          <div class="note" id="key-status-${slot.index}"></div>
        </div>`
      )
      .join("");
    container
      .querySelectorAll("[data-action]")
      .forEach((button) => button.addEventListener("click", onKeyAction));

    // Enter saves, and a paste is stripped of whitespace: API keys are one
    // continuous token, and terminal copies often carry a trailing newline.
    container.querySelectorAll("input[type='password'], input[type='text']").forEach((input) => {
      input.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        const slot = input.id.replace("key-input-", "");
        container.querySelector(`[data-action="save"][data-slot="${slot}"]`)?.click();
      });
      input.addEventListener("paste", () => {
        setTimeout(() => {
          input.value = input.value.replace(/\s+/g, "");
        }, 0);
      });
    });

    renderedKeySlotCount = slots.length;
  }

  for (const slot of slots) {
    const line = $(`key-status-${slot.index}`);
    if (!line) continue;
    const where =
      slot.source === "keychain"
        ? "OS keychain"
        : slot.source === "file"
        ? "restricted file"
        : slot.source.startsWith("env:")
        ? slot.source
        : null;
    line.textContent = slot.present
      ? `${slot.status} · ${slot.preview} · ${where}`
      : slot.status;
    line.className =
      "note " + (slot.status === "Key rejected by API" ? "bad" : slot.present ? "good" : "");
  }

  const note = $("keychain-note");
  if (!state.keychain_ok) {
    note.textContent =
      `OS keychain unavailable (${state.keychain_error || "reason unknown"}). ` +
      "Saved keys are written to a file in the app data directory with permissions restricted " +
      "to your user account instead.";
    note.className = "note warn";
  } else if (state.env_key_present) {
    note.textContent =
      "GEMINI_API_KEY is set in the environment and is used as a fallback for slot 1 when no key " +
      "is saved there. Keys are otherwise stored in the OS keychain.";
    note.className = "note";
  } else {
    note.textContent =
      "Keys are stored in the OS keychain. GEMINI_API_KEY is also accepted as a fallback for slot 1.";
    note.className = "note";
  }
}

function renderMode() {
  const active = state.settings.mode;
  const buttons = $("mode-buttons");
  buttons.innerHTML = MODES.map(
    (mode) =>
      `<button data-mode="${mode}" class="${mode === active ? "active" : ""}">${mode}</button>`
  ).join("");
  buttons.querySelectorAll("button").forEach((button) =>
    button.addEventListener("click", async () => {
      const snapshot = await call("set_mode", { mode: button.dataset.mode });
      if (snapshot) {
        state = snapshot;
        render();
      }
    })
  );

  const config = state.config;
  const both = [...config.flash_chain, ...config.flash_lite_chain];
  const line = (chip, models) =>
    `<div><span class="chip">${chip}</span> ${
      models.length ? models.map(esc).join(" &rarr; ") : "<em>none configured</em>"
    }</div>`;
  $("chains").innerHTML =
    line("flash", config.flash_chain) +
    line("flash_lite", config.flash_lite_chain) +
    line("both", both);

  // Never overwrite text the user is editing with a polled snapshot.
  if (!chainsDirty) {
    $("flash-chain-input").value = config.flash_chain.join(", ");
    $("flash-lite-chain-input").value = config.flash_lite_chain.join(", ");
  }
  $("config-path").textContent = state.config_path || "";
}

function renderModelStatus() {
  const select = $("status-slot");
  const signature = state.key_slots.map((s) => s.index).join(",");
  if (select.dataset.signature !== signature) {
    select.innerHTML = state.key_slots
      .map((slot) => `<option value="${slot.index}">${esc(slot.label)}</option>`)
      .join("");
    select.dataset.signature = signature;
  }
  select.value = String(state.status_slot);

  $("model-rows").innerHTML = state.models
    .map(
      (model) => `
      <tr>
        <td class="mono">${esc(model.model)}</td>
        <td>${esc(model.chain)}</td>
        <td class="status-${esc(model.status)}">${esc(model.detail)}</td>
      </tr>`
    )
    .join("");
}

function renderServer() {
  const server = state.server;
  const line = $("server-state");
  line.textContent = server.running
    ? `running · ${server.address}`
    : server.error
    ? `stopped · ${server.error}`
    : "stopped";
  line.className = "note " + (server.running ? "good" : server.error ? "bad" : "");

  $("address").value = server.address;
  if (document.activeElement !== $("port")) $("port").value = server.port;

  const wait = $("api-wait");
  wait.checked = state.settings.api_wait_enabled;
  const maxWait = $("api-max-wait");
  if (document.activeElement !== maxWait) maxWait.value = state.settings.api_max_wait_secs;
}

function renderTokens() {
  const tokens = state.settings.access_tokens;
  const container = $("tokens");
  const signature = tokens.map((token) => `${token.id}:${token.token}`).join("|");

  if (container.dataset.signature !== signature) {
    container.innerHTML = tokens.length
      ? tokens
          .map(
            (token) => `
            <div class="token-row">
              <span class="token-label">${esc(token.label)}</span>
              <input readonly class="mono" value="${esc(token.token)}" data-token="${esc(token.id)}" />
              <button class="ghost" data-token-action="copy" data-id="${esc(token.id)}">Copy</button>
              <button class="ghost" data-token-action="regen" data-id="${esc(token.id)}">Regenerate</button>
              <button class="ghost" data-token-action="remove" data-id="${esc(token.id)}">Remove</button>
            </div>`
          )
          .join("")
      : `<p class="note">No access tokens yet. While this list is empty the server accepts any request.</p>`;
    container
      .querySelectorAll("[data-token-action]")
      .forEach((button) => button.addEventListener("click", onTokenAction));
    container.dataset.signature = signature;
  }
}

// Targets are static per machine, so fetch them once.
let injectTargets = null;

async function loadInjectTargets() {
  injectTargets = await call("inject_targets");
  renderInject();
}

function renderInject() {
  const container = $("inject-targets");
  if (!injectTargets) {
    container.innerHTML = `<p class="note">Looking for the coding tools' config files…</p>`;
    return;
  }
  container.innerHTML = injectTargets
    .map(
      (target) => `
      <div class="inject-card">
        <div class="title">
          <span>${esc(target.label)}</span>
          <span class="note ${target.exists ? "good" : ""}">${
            target.exists ? "config found" : "no config yet"
          }</span>
        </div>
        <div class="path">${esc(target.path)}</div>
        <button class="primary" data-inject="${esc(target.id)}">Inject</button>
      </div>`
    )
    .join("");
  container
    .querySelectorAll("[data-inject]")
    .forEach((button) => button.addEventListener("click", onInject));
}

async function onInject(event) {
  const button = event.currentTarget;
  const target = button.dataset.inject;
  const mode = $("inject-mode").value;
  const label = button.textContent;

  button.disabled = true;
  button.textContent = "Injecting…";
  const result = await call("inject_tool", { target, mode });
  button.disabled = false;
  button.textContent = label;

  if (!result) return;
  const results = $("inject-results");
  const card = document.createElement("div");
  card.className = "inject-result" + (result.warning ? " warn" : "");
  card.innerHTML = `
    <div class="head">${esc(result.label)} · mode ${esc(result.mode)}</div>
    <div class="detail">${esc(result.message)}</div>
    <div class="detail mono">${esc(result.path)}</div>
    ${result.warning ? `<div class="detail">${esc(result.warning)}</div>` : ""}`;
  results.prepend(card);
  while (results.children.length > 6) results.lastElementChild.remove();

  // The server may have been started and a token created.
  await refresh();
  await loadInjectTargets();
}

function renderLog() {
  const entries = state.log.slice().reverse();
  $("log-rows").innerHTML = entries
    .map(
      (entry) => `
      <tr>
        <td class="mono">${esc(entry.timestamp)}</td>
        <td>${esc(entry.source)}</td>
        <td>${esc(entry.mode)}</td>
        <td class="mono">${esc(entry.model ?? "—")}</td>
        <td class="status-${esc(entry.status)}">${esc(entry.status)}</td>
        <td>${entry.latency_ms == null ? "—" : `${entry.latency_ms} ms`}</td>
        <td>${esc(entry.detail)}</td>
      </tr>`
    )
    .join("");
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async function onKeyAction(event) {
  const button = event.currentTarget;
  const slot = Number(button.dataset.slot);
  const action = button.dataset.action;
  const input = $(`key-input-${slot}`);

  if (action === "toggle") {
    const hidden = input.type === "password";
    input.type = hidden ? "text" : "password";
    button.textContent = hidden ? "Hide" : "Show";
    return;
  }

  if (action === "save") {
    const value = input.value.trim();
    if (!value) {
      setBanner("Enter a key before saving.");
      return;
    }
    const result = await call("save_key", { slot, value });
    if (result) {
      state = result.snapshot;
      if (result.saved) {
        input.value = "";
      } else {
        // Keep the pasted text so it can be corrected.
        input.focus();
        input.select();
      }
      setBanner(result.warning || result.message);
      render();

      // Verify immediately, so a truncated or wrong key is caught here rather
      // than halfway through a coding session.
      if (result.saved) {
        const tested = await call("test_key", { slot });
        if (tested) {
          state = tested;
          render();
          const status = tested.key_slots.find((s) => s.index === slot)?.status;
          setBanner(
            [result.warning, status === "Key rejected by API"
              ? "The API rejected this key. Copy it again from Google AI Studio."
              : `Key saved · ${status}`]
              .filter(Boolean)
              .join(" · ")
          );
        }
      }
    }
    return;
  }

  if (action === "clear") {
    const snapshot = await call("clear_key", { slot });
    if (snapshot) {
      state = snapshot;
      render();
    }
    return;
  }

  if (action === "test") {
    const label = button.textContent;
    button.disabled = true;
    button.textContent = "Testing…";
    const snapshot = await call("test_key", { slot });
    button.disabled = false;
    button.textContent = label;
    if (snapshot) {
      state = snapshot;
      render();
    }
  }
}

/// Split a comma/whitespace-separated model list, removing blanks and duplicates.
function parseModels(text) {
  const seen = new Set();
  const models = [];
  for (const raw of String(text).split(/[\s,]+/)) {
    const model = raw.trim();
    if (model && !seen.has(model)) {
      seen.add(model);
      models.push(model);
    }
  }
  return models;
}

async function saveChains() {
  const flash = parseModels($("flash-chain-input").value);
  const lite = parseModels($("flash-lite-chain-input").value);
  if (!flash.length || !lite.length) {
    setBanner("Both chains need at least one model.");
    return;
  }
  const config = { ...state.config, flash_chain: flash, flash_lite_chain: lite };
  const snapshot = await call("set_config_json", {
    jsonText: JSON.stringify(config, null, 2),
  });
  if (snapshot) {
    state = snapshot;
    chainsDirty = false;
    render();
    setBanner("Model chains saved.");
  }
}

async function reloadConfig() {
  const snapshot = await call("reload_config");
  if (snapshot) {
    state = snapshot;
    chainsDirty = false;
    render();
    setBanner("Reloaded config.json.");
  }
}

async function onTokenAction(event) {
  const button = event.currentTarget;
  const id = button.dataset.id;

  if (button.dataset.tokenAction === "copy") {
    const input = document.querySelector(`input[data-token="${id}"]`);
    try {
      await navigator.clipboard.writeText(input.value);
      setBanner("Access token copied to the clipboard.");
    } catch {
      input.select();
      setBanner("Press Ctrl+C to copy the selected token.");
    }
    return;
  }

  const snapshot =
    button.dataset.tokenAction === "regen"
      ? await call("regenerate_access_token", { id })
      : await call("remove_access_token", { id });
  if (snapshot) {
    state = snapshot;
    render();
  }
}

async function sendPrompt() {
  const prompt = $("prompt").value;
  if (!prompt.trim()) {
    setBanner("Write a prompt first.");
    return;
  }
  const button = $("send");
  button.disabled = true;
  button.textContent = "Sending…";
  const result = await call("send_prompt", { prompt });
  button.disabled = false;
  button.textContent = "Send";

  if (!result) return;
  $("response").textContent = result.text || "(empty response)";
  const extras = [
    `${result.latency_ms} ms`,
    `key slot ${result.key_slot + 1}`,
    result.fallbacks ? `${result.fallbacks} fallback(s)` : null,
    result.restarts ? `${result.restarts} restart(s)` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  $("handled").textContent = `handled by ${result.model} · ${extras}`;
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

$("add-key-slot").addEventListener("click", async () => {
  const snapshot = await call("add_key_slot");
  if (snapshot) {
    state = snapshot;
    render();
  }
});

$("flash-chain-input").addEventListener("input", () => {
  chainsDirty = true;
});
$("flash-lite-chain-input").addEventListener("input", () => {
  chainsDirty = true;
});
$("save-chains").addEventListener("click", saveChains);
$("reload-config").addEventListener("click", reloadConfig);

$("send").addEventListener("click", sendPrompt);
$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) sendPrompt();
});

$("status-slot").addEventListener("change", async () => {
  statusSlot = Number($("status-slot").value);
  await refresh();
});

$("server-start").addEventListener("click", async () => {
  const port = Number($("port").value) || 8787;
  const snapshot = await call("start_server", { port });
  if (snapshot) {
    state = snapshot;
    render();
  }
});

$("server-stop").addEventListener("click", async () => {
  const snapshot = await call("stop_server");
  if (snapshot) {
    state = snapshot;
    render();
  }
});

$("port").addEventListener("change", async () => {
  const snapshot = await call("update_server_settings", { port: Number($("port").value) });
  if (snapshot) {
    state = snapshot;
    render();
  }
});

$("api-wait").addEventListener("change", async () => {
  const snapshot = await call("update_server_settings", { apiWaitEnabled: $("api-wait").checked });
  if (snapshot) {
    state = snapshot;
    render();
  }
});

$("api-max-wait").addEventListener("change", async () => {
  const snapshot = await call("update_server_settings", {
    apiMaxWaitSecs: Number($("api-max-wait").value),
  });
  if (snapshot) {
    state = snapshot;
    render();
  }
});

$("add-token").addEventListener("click", async () => {
  const snapshot = await call("add_access_token");
  if (snapshot) {
    state = snapshot;
    render();
  }
});

$("clear-log").addEventListener("click", async () => {
  const snapshot = await call("clear_log");
  if (snapshot) {
    state = snapshot;
    render();
  }
});

async function init() {
  await listen("router:status", (event) => {
    state = event.payload;
    statusSlot = state.status_slot;
    render();
  });

  await listen("router:log", (event) => {
    if (!state) return;
    // A status snapshot can already contain this entry; do not render it twice.
    if (state.log.some((entry) => entry.id === event.payload.id)) return;
    state.log.push(event.payload);
    if (state.log.length > 500) state.log.shift();
    renderLog();
  });

  await refresh();
  await loadInjectTargets();
  // A slow poll keeps cooldown countdowns and expiry accurate between events.
  setInterval(refresh, 5000);
}

init();
