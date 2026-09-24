'use strict';
/**
 * popup.js - AI Sentiment Analyzer Extension Popup
 * Workflows:
 *   A. Context menu → right-click → result shown in page overlay (via background.js)
 *   B. Select text → open popup → auto-fill → Analyze
 *   C. Open popup → type/paste → Analyze
 */

const API_BASE_URL = 'http://127.0.0.1:8000';
const RESULT_CACHE_TTL_MS = 60000; // 60 seconds

document.addEventListener('DOMContentLoaded', function () {

  // ── Element refs ─────────────────────────────────────────────────────────────
  var reviewInput     = document.getElementById('reviewInput');
  var analyzeBtn      = document.getElementById('analyzeBtn');
  var clearBtn        = document.getElementById('clearBtn');
  var btnText         = document.getElementById('btnText');
  var spinner         = document.getElementById('spinner');
  var errorAlert      = document.getElementById('errorAlert');
  var errorMessage    = document.getElementById('errorMessage');
  var resultsCard     = document.getElementById('resultsCard');
  var sentimentBadge  = document.getElementById('sentimentBadge');
  var confidenceValue = document.getElementById('confidenceValue');
  var posProb         = document.getElementById('posProb');
  var neuProb         = document.getElementById('neuProb');
  var negProb         = document.getElementById('negProb');
  var posBar          = document.getElementById('posBar');
  var neuBar          = document.getElementById('neuBar');
  var negBar          = document.getElementById('negBar');
  var statusDot       = document.getElementById('statusDot');
  var statusText      = document.getElementById('statusText');
  var selectionHint   = document.getElementById('selectionHint');
  var apiLabel        = document.getElementById('apiLabel');

  if (!analyzeBtn || !reviewInput) {
    console.error('[SentimentExt] Critical DOM elements missing — popup cannot initialize.');
    return;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────────
  function setLoading(on) {
    analyzeBtn.disabled = on;
    if (spinner) spinner.style.display = on ? 'inline-block' : 'none';
    if (btnText) btnText.textContent = on ? 'Analyzing...' : 'Analyze';
  }

  function showError(msg) {
    if (errorMessage) errorMessage.textContent = msg;
    if (errorAlert) errorAlert.style.display = 'flex';
    if (resultsCard) resultsCard.style.display = 'none';
  }

  function hideError() {
    if (errorAlert) errorAlert.style.display = 'none';
  }

  function setBar(barEl, textEl, val) {
    var p = ((val || 0) * 100).toFixed(1);
    if (textEl) textEl.textContent = p + '%';
    if (barEl)  barEl.style.width  = p + '%';
  }

  function renderResults(data) {
    hideError();
    var label = (data.sentiment || '').toLowerCase();
    if (sentimentBadge) {
      sentimentBadge.textContent = label.toUpperCase();
      sentimentBadge.className   = 'sentiment-badge';
      if (label === 'positive') sentimentBadge.classList.add('badge-pos');
      else if (label === 'negative') sentimentBadge.classList.add('badge-neg');
      else sentimentBadge.classList.add('badge-neu');
    }
    if (confidenceValue) {
      confidenceValue.textContent = ((data.confidence || 0) * 100).toFixed(1) + '%';
    }
    var probs = data.probabilities || {};
    setBar(posBar, posProb, probs.positive);
    setBar(neuBar, neuProb, probs.neutral);
    setBar(negBar, negProb, probs.negative);
    if (resultsCard) resultsCard.style.display = 'block';
  }

  function setHint(msg, color) {
    if (selectionHint) {
      selectionHint.textContent = msg;
      selectionHint.style.color = color || '';
    }
  }

  // ── API call ──────────────────────────────────────────────────────────────────
  function analyzeSentiment(text) {
    setLoading(true);
    hideError();

    fetch(API_BASE_URL + '/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    })
      .then(function (resp) {
        if (!resp.ok) {
          return resp.json().catch(function () { return {}; }).then(function (e) {
            throw new Error(e.detail || 'Server error ' + resp.status);
          });
        }
        return resp.json();
      })
      .then(renderResults)
      .catch(function (err) {
        if (err instanceof TypeError) {
          showError('Cannot reach API.\nStart: uvicorn api.main:app --reload');
        } else {
          showError(err.message || 'Unknown error.');
        }
      })
      .finally(function () { setLoading(false); });
  }

  // ── Health check ──────────────────────────────────────────────────────────────
  function checkHealth() {
    fetch(API_BASE_URL + '/health')
      .then(function (r) {
        if (r.ok) {
          if (statusDot) statusDot.className = 'status-dot dot-ok';
          if (statusText) statusText.textContent = 'API Online';
        } else {
          if (statusDot) statusDot.className = 'status-dot dot-warn';
          if (statusText) statusText.textContent = 'API Error';
        }
      })
      .catch(function () {
        if (statusDot) statusDot.className = 'status-dot dot-offline';
        if (statusText) statusText.textContent = 'API Offline';
      });
  }

  // ── Auto-fill: selected text from active tab ───────────────────────────────
  function tryAutoFillFromTab() {
    if (typeof chrome === 'undefined' || !chrome.tabs) return;

    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      if (!tabs || !tabs[0]) return;
      var tab = tabs[0];
      if (!tab.id) return;
      var url = tab.url || '';
      if (url.startsWith('chrome://') || url.startsWith('edge://') || url.startsWith('about:')) return;

      chrome.tabs.sendMessage(tab.id, { action: 'getSelection' }, function (response) {
        if (chrome.runtime.lastError) {
          // Content script not injected yet, try injecting
          chrome.scripting.executeScript(
            { target: { tabId: tab.id }, files: ['content.js'] },
            function () {
              if (chrome.runtime.lastError) return;
              setTimeout(function () {
                chrome.tabs.sendMessage(tab.id, { action: 'getSelection' }, function (r2) {
                  if (chrome.runtime.lastError) return;
                  applySelection(r2);
                });
              }, 150);
            }
          );
          return;
        }
        applySelection(response);
      });
    });
  }

  function applySelection(response) {
    if (response && response.text && response.text.trim().length > 0) {
      reviewInput.value = response.text.trim();
      setHint('\u2714 Text loaded from page', '#4ade80');
    }
  }

  // ── Check session storage for recent context-menu result ─────────────────────
  function tryLoadCachedResult() {
    if (typeof chrome === 'undefined' || !chrome.storage) return;

    chrome.storage.session.get(['lastText', 'lastResult', 'lastTs'], function (items) {
      if (chrome.runtime.lastError) return;
      if (!items.lastResult || !items.lastText) return;
      var age = Date.now() - (items.lastTs || 0);
      if (age > RESULT_CACHE_TTL_MS) return; // stale

      // Show the cached result from context-menu analysis
      reviewInput.value = items.lastText;
      setHint('\u2714 From right-click analysis', '#a78bfa');
      renderResults(items.lastResult);

      // Clear cache so next open is fresh
      chrome.storage.session.remove(['lastText', 'lastResult', 'lastTs']);
    });
  }

  // ── Event listeners ───────────────────────────────────────────────────────────
  analyzeBtn.addEventListener('click', function () {
    var text = reviewInput.value.trim();
    if (!text) {
      showError('Please enter or select text to analyze.');
      return;
    }
    analyzeSentiment(text);
  });

  clearBtn.addEventListener('click', function () {
    reviewInput.value = '';
    hideError();
    if (resultsCard) resultsCard.style.display = 'none';
    setHint('Selection auto-detected');
    analyzeBtn.disabled = false;
  });

  reviewInput.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyzeBtn.click();
  });

  // ── Init ──────────────────────────────────────────────────────────────────────
  if (apiLabel) apiLabel.textContent = 'API: ' + API_BASE_URL;
  checkHealth();

  // Priority: 1) cached context-menu result → 2) live page selection
  tryLoadCachedResult();
  setTimeout(tryAutoFillFromTab, 80); // slight delay so cache check runs first

}); // end DOMContentLoaded
