// Background service worker for Audience Intelligence Analyzer

// Listen for installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    // First time installation
    console.log('Extension installed');
    
    // Initialize default settings
    chrome.storage.sync.set({
      apiEndpoint: 'https://api.example.com/v1',
      apiKey: '',
      recentAnalyses: []
    });
    
    // Open options page for setup
    chrome.tabs.create({
      url: 'options.html'
    });
  } else if (details.reason === 'update') {
    // Extension updated
    console.log('Extension updated');
  }
});

// Listen for messages from popup or content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'storeAnalysisResult') {
    // Store analysis results for later reference
    storeAnalysisResult(message.data, sendResponse);
    return true; // Keep channel open for async response
  }
});

// Store analysis result in sync storage
async function storeAnalysisResult(data, callback) {
  try {
    // Get existing analyses
    const result = await chrome.storage.sync.get('recentAnalyses');
    let recentAnalyses = result.recentAnalyses || [];
    
    // Add new analysis with timestamp
    const newAnalysis = {
      url: data.url,
      title: data.title,
      timestamp: new Date().toISOString(),
      engagement: data.engagement_score,
      sentiment: data.sentiment_score
    };
    
    // Add to beginning of array and limit to 10 items
    recentAnalyses.unshift(newAnalysis);
    if (recentAnalyses.length > 10) {
      recentAnalyses = recentAnalyses.slice(0, 10);
    }
    
    // Save back to storage
    await chrome.storage.sync.set({ recentAnalyses });
    
    if (callback) callback({ success: true });
  } catch (error) {
    console.error('Error storing analysis:', error);
    if (callback) callback({ success: false, error: error.message });
  }
}

// Optional: Set up periodic API status checks
const CHECK_INTERVAL = 30 * 60 * 1000; // 30 minutes

async function scheduleApiCheck() {
  try {
    const { apiEndpoint, apiKey } = await chrome.storage.sync.get(['apiEndpoint', 'apiKey']);
    
    if (!apiEndpoint || !apiKey) return;
    
    // Check API status
    const response = await fetch(`${apiEndpoint}/status`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    const status = response.ok ? 'online' : 'offline';
    
    // Update badge if API is down
    if (status === 'offline') {
      chrome.action.setBadgeText({ text: '!' });
      chrome.action.setBadgeBackgroundColor({ color: '#dc3545' });
    } else {
      chrome.action.setBadgeText({ text: '' });
    }
    
  } catch (error) {
    console.error('API check failed:', error);
    chrome.action.setBadgeText({ text: '!' });
    chrome.action.setBadgeBackgroundColor({ color: '#dc3545' });
  }
  
  // Schedule next check
  setTimeout(scheduleApiCheck, CHECK_INTERVAL);
}

// Start periodic checks
scheduleApiCheck(); 