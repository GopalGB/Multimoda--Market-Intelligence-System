# Audience Intelligence Analyzer Chrome Extension

A browser extension that analyzes web content for audience engagement potential using the Cross-Modal Audience Intelligence Platform (CAIP).

## Features

- **Content Analysis**: Analyze any webpage for engagement potential
- **Causal Insights**: Discover what factors drive audience engagement
- **Sentiment Analysis**: Measure sentiment of content
- **Optimization Suggestions**: Get recommendations to improve engagement
- **Counterfactual Analysis**: Explore "what-if" scenarios

## Installation

### Development Mode

1. Clone this repository or download the source code
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" (toggle in top-right corner)
4. Click "Load unpacked" and select the `chrome-extension` directory
5. The extension should now appear in your browser toolbar

### Production Mode

1. Download the latest release `.zip` file
2. Extract the contents to a folder
3. Follow steps 2-5 from the Development Mode instructions

## Configuration

Before using the extension, you need to configure your API credentials:

1. Click the extension icon in your browser toolbar
2. Click the ⚙️ (settings) icon in the popup
3. Enter your API endpoint and API key
4. Click "Save Settings"

You can obtain API credentials by:
1. Setting up your own CAIP server instance
2. Requesting access to a hosted CAIP API
3. Using the demo endpoint (limited functionality)

## Usage

### Analyzing Content

1. Navigate to any webpage you want to analyze
2. Click the extension icon in your browser toolbar
3. Click "Analyze This Page"
4. View the engagement score and sentiment results

### Viewing Causal Insights

1. After analyzing a page, click the "Insights" tab
2. Review the causal factors that influence engagement
3. Examine the causal graph visualization

### Getting Optimization Suggestions

1. After analyzing a page, click the "Optimize" tab
2. Review the generated suggestions to improve engagement
3. Use the counterfactual tool to experiment with different factors

## Privacy Notice

This extension:
- Only analyzes content when explicitly requested
- Does not track browsing history or collect personal data
- Only sends the current page's content to the configured API endpoint
- Stores settings and recent analyses locally on your device

## Development

### Project Structure

```
chrome-extension/
├── manifest.json       # Extension configuration
├── popup/              # User interface
│   ├── popup.html
│   ├── popup.css
│   └── popup.js
├── background.js       # Background service worker
├── content-script.js   # DOM interaction
└── api/                # Communication with backend
    └── client.js
```

### Building from Source

1. Install dependencies: `npm install`
2. Build the extension: `npm run build`
3. Load the unpacked extension from the `dist` directory

## Support

For issues, feature requests, or questions:
- Open an issue in the GitHub repository
- Contact support at support@example.com

## License

This project is licensed under the MIT License - see the LICENSE file for details. 