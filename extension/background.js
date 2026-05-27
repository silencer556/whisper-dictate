// Whisper Dictate — background service worker
// Polls the local HTTP server for transcribed text and injects it into the
// active tab via Chrome DevTools Protocol so CRD forwards it to the remote Mac.

const POLL_MS = 250;

// Build char → {vk, code, modifiers, unmodified} map for printable ASCII.
const KEY_MAP = (() => {
  const m = {};
  const add = (ch, vk, code, mods, unmod) => {
    m[ch] = { vk, code, mods: mods ?? 0, unmod: unmod ?? ch };
  };

  add(' ', 32, 'Space', 0);

  // Digits 0-9
  for (let i = 0; i <= 9; i++) add(String(i), 48 + i, 'Digit' + i, 0);

  // Shifted digit symbols
  [['!','1',49],['@','2',50],['#','3',51],['$','4',52],['%','5',53],
   ['^','6',54],['&','7',55],['*','8',56],['(','9',57],[')','0',48]
  ].forEach(([sym, un, vk]) => add(sym, vk, 'Digit' + un, 8, un));

  // Letters a-z / A-Z
  for (let i = 0; i < 26; i++) {
    const lo = String.fromCharCode(97 + i);
    const up = String.fromCharCode(65 + i);
    const vk = 65 + i;
    const code = 'Key' + up;
    add(lo, vk, code, 0);
    add(up, vk, code, 8, lo);
  }

  // Punctuation pairs [base, shifted, vk, code]
  [
    ['-','_',189,'Minus'],   ['=','+',187,'Equal'],
    ['[','{',219,'BracketLeft'],  [']','}',221,'BracketRight'],
    ['\\','|',220,'Backslash'],   [';',':',186,'Semicolon'],
    ["'",'"',222,'Quote'],        ['`','~',192,'Backquote'],
    [',','<',188,'Comma'],        ['.', '>',190,'Period'],
    ['/','?',191,'Slash'],
  ].forEach(([b, s, vk, code]) => {
    add(b, vk, code, 0);
    add(s, vk, code, 8, b);
  });

  // Special / non-printing keys
  add('\n', 13, 'Enter',     0, '\n');
  add('\r', 13, 'Enter',     0, '\r');
  add('\t',  9, 'Tab',       0, '\t');
  add('\b',  8, 'Backspace', 0, '');

  return m;
})();

function keyName(ch) {
  if (ch === '\n' || ch === '\r') return 'Enter';
  if (ch === '\t') return 'Tab';
  if (ch === '\b') return 'Backspace';
  return ch;
}

let attachedTabId = null;
let isTyping = false;
let serverPort = 9754;

// Persist port from storage
chrome.storage.local.get(['port'], (res) => {
  if (res.port) serverPort = res.port;
});

async function focusedCrdTab() {
  // Returns the CRD tab only if its Edge window actually has OS focus right now.
  // active:true means it's the selected tab in its window (not a background tab).
  const crdTabs = await chrome.tabs.query({ url: '*://remotedesktop.google.com/*', active: true });
  for (const tab of crdTabs) {
    const win = await chrome.windows.get(tab.windowId);
    if (win.focused) return tab;
  }
  return null;
}

async function poll() {
  if (isTyping) return;

  // Only report crd=1 when the CRD window actually has Windows focus —
  // so dictating into a local app while CRD is open in the background still
  // uses keystroke, not the extension.
  const crdTab = await focusedCrdTab();

  let text;
  try {
    const resp = await fetch(
      `http://localhost:${serverPort}/pending?crd=${crdTab ? 1 : 0}`,
      { signal: AbortSignal.timeout(200) }
    );
    if (!resp.ok) return;
    const data = await resp.json();
    text = data.text;
  } catch {
    return;
  }
  if (!text || !crdTab) return;

  await typeInTab(crdTab.id, text);
}

const SHIFT_DOWN = { type: 'keyDown', key: 'Shift', code: 'ShiftLeft', windowsVirtualKeyCode: 16, nativeVirtualKeyCode: 16, modifiers: 8 };
const SHIFT_UP   = { type: 'keyUp',   key: 'Shift', code: 'ShiftLeft', windowsVirtualKeyCode: 16, nativeVirtualKeyCode: 16, modifiers: 0 };

async function dispatchChar(tabId, char) {
  const info = KEY_MAP[char];
  if (!info) return;

  const needsShift = info.mods === 8;

  // CRD tracks modifier key state, so send real Shift down/up events —
  // just setting modifiers:8 on the character event is not enough.
  if (needsShift) {
    await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchKeyEvent', SHIFT_DOWN);
  }

  // Non-printing keys (Backspace, Enter, Tab) should not carry a text payload.
  const isPrinting = char.charCodeAt(0) >= 32;
  const base = {
    key: keyName(char),
    code: info.code,
    windowsVirtualKeyCode: info.vk,
    nativeVirtualKeyCode: info.vk,
    modifiers: info.mods,
    text: isPrinting ? char : '',
    unmodifiedText: isPrinting ? info.unmod : '',
  };
  await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchKeyEvent', { ...base, type: 'keyDown' });
  await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchKeyEvent', { ...base, type: 'keyUp' });

  if (needsShift) {
    await chrome.debugger.sendCommand({ tabId }, 'Input.dispatchKeyEvent', SHIFT_UP);
  }
}

async function typeInTab(tabId, text) {
  isTyping = true;
  try {
    // Attach debugger if not already attached to this tab
    if (tabId !== attachedTabId) {
      if (attachedTabId !== null) {
        try { await chrome.debugger.detach({ tabId: attachedTabId }); } catch {}
        attachedTabId = null;
      }
      await chrome.debugger.attach({ tabId }, '1.3');
      attachedTabId = tabId;
    }

    for (const char of text) {
      await dispatchChar(tabId, char);
    }
  } catch (err) {
    console.error('[Whisper Dictate] typeInTab error:', err);
  } finally {
    // Always detach immediately — keeps the debugger banner from lingering.
    // (The --silent-debugger-extension-api Edge flag suppresses it entirely.)
    try { await chrome.debugger.detach({ tabId }); } catch {}
    attachedTabId = null;
    isTyping = false;
  }
}

// Detach when the tab navigates or closes
chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === attachedTabId) attachedTabId = null;
});
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (tabId === attachedTabId && changeInfo.status === 'loading') {
    chrome.debugger.detach({ tabId }).catch(() => {});
    attachedTabId = null;
  }
});
chrome.debugger.onDetach.addListener((source) => {
  if (source.tabId === attachedTabId) attachedTabId = null;
});

setInterval(poll, POLL_MS);
