'use strict';
/**
 * background.js - Service Worker for AI Sentiment Analyzer
 * No "type":"module" — plain service worker script for reliable MV3 behavior.
 */

var API_URL = 'http://127.0.0.1:8000/predict';
var MENU_ID = 'sentiment-analyze';

// Track which tabs already have content.js injected to avoid double-injection
var injectedTabs = {};

// ── Register context menu on install / update ─────────────────────────────────
chrome.runtime.onInstalled.addListener(function () {
  chrome.contextMenus.removeAll(function () {
    chrome.contextMenus.create({
      id: MENU_ID,
      title: '\uD83E\uDDE0 Analyze with Sentiment AI',
      contexts: ['selection']
    });
    console.log('[Sentiment Extension] Context menu registered.');
  });
});

// ── Context menu click ────────────────────────────────────────────────────────
chrome.contextMenus.onClicked.addListener(function (info, tab) {
  console.log('[Sentiment Extension] context menu clicked', info.menuItemId);

  if (info.menuItemId !== MENU_ID) return;

  var text = (info.selectionText || '').trim();
  console.log('[Sentiment Extension] selected text:', text);

  if (!text) {
    console.warn('[Sentiment Extension] No text selected, aborting.');
    return;
  }

  if (!tab || !tab.id) {
    console.error('[Sentiment Extension] No valid tab, aborting.');
    return;
  }

  var tabId = tab.id;
  var tabUrl = tab.url || '';

  // Guard: restricted pages
  if (tabUrl.startsWith('chrome://') || tabUrl.startsWith('edge://') ||
      tabUrl.startsWith('chrome-extension://') || tabUrl.startsWith('about:')) {
    console.warn('[Sentiment Extension] Restricted page, cannot inject overlay:', tabUrl);
    return;
  }

  // Show loading, then fetch, then show result
  sendToTab(tabId, { type: 'sa_loading', text: text })
    .then(function () {
      console.log('[Sentiment Extension] Loading message sent, fetching API...');
      return fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
      });
    })
    .then(function (resp) {
      if (!resp.ok) {
        return resp.json().catch(function () { return {}; }).then(function (e) {
          throw new Error(e.detail || 'API error ' + resp.status);
        });
      }
      return resp.json();
    })
    .then(function (data) {
      console.log('[Sentiment Extension] API response:', JSON.stringify(data));
      return sendToTab(tabId, { type: 'sa_result', result: data, text: text });
    })
    .then(function () {
      // Cache for popup
      chrome.storage.session.set({
        lastText: text,
        lastResult: null, // will be re-fetched by popup if needed
        lastTs: Date.now()
      });
    })
    .catch(function (err) {
      var msg = (err instanceof TypeError)
        ? 'Cannot reach API. Run: uvicorn api.main:app --reload'
        : (err.message || 'Unknown error');
      console.error('[Sentiment Extension] Error:', msg);
      sendToTab(tabId, { type: 'sa_error', error: msg });
    });
});

// ── sendToTab: inject content.js + CSS if needed, then send message ───────────
function sendToTab(tabId, message) {
  return new Promise(function (resolve, reject) {

    // First attempt: maybe content.js is already running
    tryMessage(tabId, message, function (ok) {
      if (ok) { resolve(); return; }

      // Failed: inject content.js and overlay.css, then retry
      console.log('[Sentiment Extension] Injecting content.js + overlay.css into tab', tabId);

      // Inject CSS first
      chrome.scripting.insertCSS(
        { target: { tabId: tabId }, files: ['overlay.css'] },
        function () {
          if (chrome.runtime.lastError) {
            console.warn('[Sentiment Extension] CSS inject error:', chrome.runtime.lastError.message);
            // Continue anyway — content.js has inline styles as fallback
          }

          // Inject JS
          chrome.scripting.executeScript(
            { target: { tabId: tabId }, files: ['content.js'] },
            function () {
              if (chrome.runtime.lastError) {
                console.error('[Sentiment Extension] JS inject error:', chrome.runtime.lastError.message);
                reject(new Error(chrome.runtime.lastError.message));
                return;
              }

              // Wait for content script to register its listener
              setTimeout(function () {
                tryMessage(tabId, message, function (ok2) {
                  if (ok2) { resolve(); }
                  else {
                    console.error('[Sentiment Extension] Message delivery failed after injection');
                    reject(new Error('Message delivery failed'));
                  }
                });
              }, 300);
            }
          );
        }
      );
    });
  });
}

// ── tryMessage: attempt chrome.tabs.sendMessage, callback(true/false) ─────────
function tryMessage(tabId, message, callback) {
  chrome.tabs.sendMessage(tabId, message, function (response) {
    if (chrome.runtime.lastError) {
      callback(false);
    } else {
      callback(true);
    }
  });
}
