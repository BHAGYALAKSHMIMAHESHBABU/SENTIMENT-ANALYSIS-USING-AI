'use strict';
/**
 * content.js - AI Sentiment Analyzer Content Script
 *
 * Critical design: ALL layout/positioning styles are set inline via JS.
 * The overlay will appear correctly whether overlay.css is loaded or not.
 */

console.log('[Sentiment Extension] content.js loaded');

var _dismissTimer = null;
var _overlayEl = null;

// ── Keyframe animation injected once ─────────────────────────────────────────
(function injectKeyframes() {
  if (document.getElementById('__sa_keyframes__')) return;
  var style = document.createElement('style');
  style.id = '__sa_keyframes__';
  style.textContent = [
    '@keyframes sa-spin { to { transform: rotate(360deg); } }',
    '@keyframes sa-fadein { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }'
  ].join('\n');
  (document.head || document.documentElement).appendChild(style);
})();

// ── Message listener ──────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log('[Sentiment Extension] message received:', JSON.stringify(request));

  // Popup: get selected text
  if (request.action === 'getSelection') {
    var sel = window.getSelection();
    var text = sel ? sel.toString().trim() : '';
    sendResponse({ text: text });
    return true;
  }

  // Background: loading state
  if (request.type === 'sa_loading') {
    showLoading(request.text || '');
    sendResponse({ ok: true });
    return true;
  }

  // Background: result
  if (request.type === 'sa_result') {
    console.log('[Sentiment Extension] rendering result:', JSON.stringify(request.result));
    showResult(request.result, request.text || '');
    sendResponse({ ok: true });
    return true;
  }

  // Background: error
  if (request.type === 'sa_error') {
    showError(request.error || 'Unknown error');
    sendResponse({ ok: true });
    return true;
  }

  // Debug test
  if (request.type === 'sa_test') {
    showTestBox();
    sendResponse({ ok: true });
    return true;
  }

  sendResponse({ ok: false, reason: 'unknown message type' });
  return true;
});

// ── Overlay root element ──────────────────────────────────────────────────────

function getOverlay() {
  // Reuse if exists in DOM
  var existing = document.getElementById('__sa_overlay__');
  if (existing) return existing;

  console.log('[Sentiment Extension] creating overlay');

  var el = document.createElement('div');
  el.id = '__sa_overlay__';

  // ALL critical styles inline — never rely on external CSS for these
  el.style.cssText = [
    'position: fixed',
    'bottom: 20px',
    'right: 20px',
    'z-index: 2147483647',
    'width: 310px',
    'max-width: calc(100vw - 40px)',
    'font-family: Segoe UI, system-ui, Arial, sans-serif',
    'font-size: 13px',
    'line-height: 1.4',
    'color: #e2e8f0',
    'background: #0f172a',
    'border: 1px solid #334155',
    'border-radius: 12px',
    'padding: 14px 16px',
    'box-shadow: 0 20px 60px rgba(0,0,0,0.8), 0 0 0 1px rgba(59,130,246,0.2)',
    'box-sizing: border-box',
    'animation: sa-fadein 0.3s ease forwards',
    'display: block'
  ].join('; ');

  // Append to documentElement so it works even if body is missing or overridden
  document.documentElement.appendChild(el);
  _overlayEl = el;

  console.log('[Sentiment Extension] overlay appended');
  return el;
}

function removeOverlay() {
  if (_dismissTimer) { clearTimeout(_dismissTimer); _dismissTimer = null; }
  var el = document.getElementById('__sa_overlay__');
  if (el && el.parentNode) {
    el.style.opacity = '0';
    el.style.transform = 'translateY(10px)';
    el.style.transition = 'opacity 0.3s, transform 0.3s';
    setTimeout(function () {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, 320);
  }
  _overlayEl = null;
}

// ── Inline style helpers ──────────────────────────────────────────────────────

function sentimentColor(label) {
  if (label === 'positive') return '#4ade80';
  if (label === 'negative') return '#f87171';
  return '#fbbf24';
}

function barBg(label) {
  if (label === 'positive') return 'linear-gradient(90deg,#16a34a,#4ade80)';
  if (label === 'negative') return 'linear-gradient(90deg,#b91c1c,#f87171)';
  return 'linear-gradient(90deg,#b45309,#fbbf24)';
}

function pct(val) { return ((val || 0) * 100).toFixed(1) + '%'; }

function makeHeader(titleText) {
  var hdr = document.createElement('div');
  hdr.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:10px';

  var icon = document.createElement('span');
  icon.textContent = '\uD83E\uDDE0';
  icon.style.cssText = 'font-size:18px;line-height:1;flex-shrink:0';

  var title = document.createElement('span');
  title.textContent = titleText;
  title.style.cssText = 'font-size:13px;font-weight:700;color:#f1f5f9;flex:1';

  var closeBtn = document.createElement('button');
  closeBtn.textContent = '\u00D7';
  closeBtn.title = 'Dismiss';
  closeBtn.style.cssText = [
    'all:unset',
    'cursor:pointer',
    'font-size:18px',
    'color:#64748b',
    'line-height:1',
    'padding:2px 6px',
    'border-radius:4px'
  ].join(';');
  closeBtn.addEventListener('mouseenter', function () { this.style.color = '#e2e8f0'; this.style.background = '#1e293b'; });
  closeBtn.addEventListener('mouseleave', function () { this.style.color = '#64748b'; this.style.background = 'transparent'; });
  closeBtn.addEventListener('click', removeOverlay);

  hdr.appendChild(icon);
  hdr.appendChild(title);
  hdr.appendChild(closeBtn);
  return hdr;
}

function makePreview(text) {
  var el = document.createElement('div');
  var preview = text.length > 80 ? text.slice(0, 80) + '\u2026' : text;
  el.textContent = '\u201C' + preview + '\u201D';
  el.style.cssText = [
    'font-size:10px',
    'color:#64748b',
    'font-style:italic',
    'margin-bottom:10px',
    'border-left:2px solid #334155',
    'padding-left:8px',
    'word-break:break-word',
    'line-height:1.4'
  ].join(';');
  return el;
}

function makeProbRow(label, val, colorLabel) {
  var p = ((val || 0) * 100).toFixed(1);

  var row = document.createElement('div');
  row.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:6px';

  var lbl = document.createElement('span');
  lbl.textContent = label;
  lbl.style.cssText = 'font-size:10px;font-weight:600;width:52px;flex-shrink:0;color:' + sentimentColor(colorLabel);

  var pctEl = document.createElement('span');
  pctEl.textContent = p + '%';
  pctEl.style.cssText = 'font-size:10px;color:#94a3b8;width:36px;text-align:right;flex-shrink:0';

  var track = document.createElement('div');
  track.style.cssText = 'flex:1;height:5px;background:#1e293b;border-radius:3px;overflow:hidden';

  var fill = document.createElement('div');
  fill.style.cssText = 'height:100%;border-radius:3px;transition:width 0.5s ease;background:' + barBg(colorLabel) + ';width:0%';

  track.appendChild(fill);
  row.appendChild(lbl);
  row.appendChild(pctEl);
  row.appendChild(track);

  // Animate bar after paint
  requestAnimationFrame(function () { fill.style.width = p + '%'; });

  return row;
}

// ── Show states ───────────────────────────────────────────────────────────────

function showLoading(text) {
  if (_dismissTimer) { clearTimeout(_dismissTimer); _dismissTimer = null; }
  var el = getOverlay();
  el.innerHTML = '';

  el.appendChild(makeHeader('Analyzing\u2026'));
  if (text) el.appendChild(makePreview(text));

  var spinWrap = document.createElement('div');
  spinWrap.style.cssText = 'display:flex;justify-content:center;padding:12px 0';

  var spin = document.createElement('div');
  spin.style.cssText = [
    'width:28px',
    'height:28px',
    'border:3px solid #1e293b',
    'border-top-color:#3b82f6',
    'border-radius:50%',
    'animation:sa-spin 0.7s linear infinite'
  ].join(';');

  spinWrap.appendChild(spin);
  el.appendChild(spinWrap);
}

function showResult(data, text) {
  if (_dismissTimer) { clearTimeout(_dismissTimer); _dismissTimer = null; }
  var el = getOverlay();
  el.innerHTML = '';

  var label = (data.sentiment || '').toLowerCase();
  var conf  = ((data.confidence || 0) * 100).toFixed(1);
  var probs = data.probabilities || {};
  var col   = sentimentColor(label);

  el.appendChild(makeHeader('Sentiment Result'));
  if (text) el.appendChild(makePreview(text));

  // Badge + confidence row
  var badgeRow = document.createElement('div');
  badgeRow.style.cssText = 'display:flex;align-items:center;justify-content:space-between;margin-bottom:12px';

  var badge = document.createElement('div');
  badge.textContent = label.toUpperCase();
  badge.style.cssText = [
    'padding:4px 12px',
    'border-radius:20px',
    'font-size:12px',
    'font-weight:700',
    'letter-spacing:0.08em',
    'border:1px solid ' + col,
    'color:' + col,
    'background:rgba(0,0,0,0.3)'
  ].join(';');

  var confEl = document.createElement('div');
  confEl.style.cssText = 'font-size:11px;color:#94a3b8';
  confEl.innerHTML = 'Confidence: <strong style="color:#38bdf8;font-weight:700">' + conf + '%</strong>';

  badgeRow.appendChild(badge);
  badgeRow.appendChild(confEl);
  el.appendChild(badgeRow);

  // Probability bars
  el.appendChild(makeProbRow('Positive', probs.positive, 'positive'));
  el.appendChild(makeProbRow('Neutral',  probs.neutral,  'neutral'));
  el.appendChild(makeProbRow('Negative', probs.negative, 'negative'));

  // Auto-dismiss after 14 seconds
  _dismissTimer = setTimeout(removeOverlay, 14000);
}

function showError(msg) {
  if (_dismissTimer) { clearTimeout(_dismissTimer); _dismissTimer = null; }
  var el = getOverlay();
  el.innerHTML = '';
  el.style.borderColor = '#7f1d1d';

  el.appendChild(makeHeader('Analysis Failed'));

  var errBox = document.createElement('div');
  errBox.textContent = msg;
  errBox.style.cssText = [
    'font-size:11px',
    'color:#fca5a5',
    'background:#1e0a0a',
    'border:1px solid #7f1d1d',
    'border-radius:6px',
    'padding:8px 10px',
    'line-height:1.5',
    'word-break:break-word'
  ].join(';');
  el.appendChild(errBox);
}

// ── Test handler ──────────────────────────────────────────────────────────────
function showTestBox() {
  var el = document.getElementById('__sa_test_box__');
  if (el && el.parentNode) el.parentNode.removeChild(el);

  el = document.createElement('div');
  el.id = '__sa_test_box__';
  el.textContent = 'Sentiment Extension Connected';
  el.style.cssText = [
    'position:fixed',
    'top:20px',
    'right:20px',
    'z-index:2147483647',
    'background:red',
    'color:white',
    'font-family:sans-serif',
    'font-size:14px',
    'font-weight:bold',
    'padding:12px 18px',
    'border-radius:8px',
    'box-shadow:0 4px 20px rgba(0,0,0,0.5)'
  ].join(';');
  document.documentElement.appendChild(el);
  setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 4000);
}
