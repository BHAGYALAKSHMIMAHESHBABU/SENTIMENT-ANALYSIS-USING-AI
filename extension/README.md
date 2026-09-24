# AI Sentiment Analyzer — Browser Extension

A Manifest V3 Chrome/Edge extension that connects to the local FastAPI backend
to analyze sentiment of any selected webpage text or manually entered text.

---

## Setup

### Prerequisites
- The FastAPI backend must be running: `uvicorn api.main:app --reload`
- Chrome or Microsoft Edge browser

### Load the Extension (Developer Mode)

1. Open Chrome/Edge and navigate to `chrome://extensions` (or `edge://extensions`)
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `extension/` folder inside the project directory
5. The "AI Sentiment Analyzer" extension icon should appear in the toolbar

---

## Usage

### Analyze Selected Webpage Text
1. Highlight any text on a webpage
2. Click the extension icon in the browser toolbar
3. The selected text is automatically loaded into the text area
4. Click **Analyze Sentiment**

### Analyze Custom Text
1. Click the extension icon
2. Type or paste any text into the text area
3. Click **Analyze Sentiment** (or press `Ctrl+Enter`)

### Reading Results
- **Sentiment badge**: POSITIVE / NEUTRAL / NEGATIVE with color coding
- **Confidence**: The model's confidence for the predicted class
- **Probability bars**: Full distribution across all three classes
- **API status indicator**: Green = API Online, Red = API Offline

---

## File Structure

```
extension/
├── manifest.json   # Manifest V3 configuration
├── popup.html      # Extension popup UI
├── popup.js        # Popup logic — fetches API, renders results
├── popup.css       # Popup styling (dark theme)
├── content.js      # Content script — captures selected text
└── icons/
    ├── icon16.png
    ├── icon32.png
    ├── icon48.png
    └── icon128.png
```

---

## API Communication

The extension sends POST requests to `http://127.0.0.1:8000/predict`.
The API endpoint must be accessible on localhost. No external network calls are made.

---

## Permissions Used

| Permission | Purpose |
|------------|---------|
| `activeTab` | Access the currently active tab |
| `scripting` | Inject content script to capture selection |
| `tabs` | Query the active tab URL and ID |
| `host_permissions: 127.0.0.1:8000` | Allow fetch to local FastAPI server |

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| "API Offline" shown | Start the FastAPI server: `uvicorn api.main:app --reload` |
| Selected text not loaded | Reload the extension; some pages block content scripts |
| Extension not loading | Ensure Developer Mode is enabled in browser |
