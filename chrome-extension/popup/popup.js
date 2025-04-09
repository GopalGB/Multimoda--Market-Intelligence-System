// DOM Elements
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingEl = document.getElementById('loading');
const resultsEl = document.getElementById('results');
const scoreValueEl = document.querySelector('.score-value');
const meterFillEl = document.querySelector('.meter-fill');
const sentimentValueEl = document.querySelector('.sentiment-value');
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const factorsList = document.getElementById('factorsList');
const suggestionsList = document.getElementById('suggestionsList');
const counterfactualTool = document.getElementById('counterfactualTool');
const selectedFactorEl = document.getElementById('selectedFactor');
const factorSlider = document.getElementById('factorSlider');
const predictedOutcomeEl = document.getElementById('predictedOutcome');
const settingsBtn = document.getElementById('settingsBtn');
const settingsModal = document.getElementById('settingsModal');
const closeModalBtn = document.querySelector('.close-btn');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');
const apiEndpointInput = document.getElementById('apiEndpoint');
const apiKeyInput = document.getElementById('apiKey');
const apiStatusEl = document.getElementById('apiStatus');

// State
let pageContent = null;
let analysisResults = null;
let causalFactors = null;
let settings = {
  apiEndpoint: '',
  apiKey: ''
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  // Load settings
  chrome.storage.sync.get(['apiEndpoint', 'apiKey'], (result) => {
    if (result.apiEndpoint) {
      settings.apiEndpoint = result.apiEndpoint;
      apiEndpointInput.value = result.apiEndpoint;
    }
    if (result.apiKey) {
      settings.apiKey = result.apiKey;
      apiKeyInput.value = result.apiKey;
    }
    
    // Check API status
    checkApiStatus();
  });
  
  // Set up event listeners
  setupEventListeners();
});

// Set up event listeners
function setupEventListeners() {
  // Analyze button
  analyzeBtn.addEventListener('click', handleAnalyzeClick);
  
  // Tab navigation
  tabButtons.forEach(button => {
    button.addEventListener('click', () => handleTabClick(button));
  });
  
  // Settings
  settingsBtn.addEventListener('click', () => {
    settingsModal.classList.remove('hidden');
  });
  
  closeModalBtn.addEventListener('click', () => {
    settingsModal.classList.add('hidden');
  });
  
  saveSettingsBtn.addEventListener('click', saveSettings);
  
  // Counterfactual slider
  factorSlider.addEventListener('input', handleSliderChange);
}

// Handle analyze button click
function handleAnalyzeClick() {
  // Check if API settings are configured
  if (!settings.apiEndpoint || !settings.apiKey) {
    alert('Please configure API settings first');
    settingsModal.classList.remove('hidden');
    return;
  }
  
  // Show loading state
  analyzeBtn.disabled = true;
  loadingEl.classList.remove('hidden');
  resultsEl.classList.add('hidden');
  
  // Get active tab and extract content
  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    const activeTab = tabs[0];
    
    // Send message to content script
    chrome.tabs.sendMessage(
      activeTab.id,
      {action: "extractContent"},
      handleContentExtracted
    );
  });
}

// Handle content extracted from page
function handleContentExtracted(response) {
  if (!response || !response.content) {
    showError('Could not extract content from page');
    return;
  }
  
  pageContent = response.content;
  console.log('Extracted content:', pageContent);
  
  // Send to API for analysis
  analyzeContent(pageContent);
}

// Analyze content through API
async function analyzeContent(content) {
  try {
    // Prepare data for API
    const data = {
      url: content.url,
      title: content.title,
      text: content.mainContent || content.text,
      primaryImage: content.primaryImage ? content.primaryImage.src : null
    };
    
    // Call API
    const response = await fetch(`${settings.apiEndpoint}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${settings.apiKey}`
      },
      body: JSON.stringify(data)
    });
    
    if (!response.ok) {
      throw new Error(`API responded with status ${response.status}`);
    }
    
    // Parse results
    const results = await response.json();
    analysisResults = results;
    
    // Display results
    displayResults(results);
    
    // Parse causal factors for later use
    if (results.causal_factors) {
      causalFactors = results.causal_factors;
      populateCausalFactors(causalFactors);
    }
    
    // Display optimization suggestions
    if (results.suggestions) {
      populateSuggestions(results.suggestions);
    }
    
    // Setup counterfactual tool if factors are available
    if (causalFactors && causalFactors.length > 0) {
      setupCounterfactualTool(causalFactors);
    }
    
  } catch (error) {
    console.error('Error analyzing content:', error);
    showError('Error analyzing content: ' + error.message);
  }
}

// Display analysis results
function displayResults(results) {
  // Hide loading
  loadingEl.classList.add('hidden');
  analyzeBtn.disabled = false;
  
  // Show results
  resultsEl.classList.remove('hidden');
  
  // Set engagement score
  const score = results.engagement_score.toFixed(2);
  scoreValueEl.textContent = score;
  meterFillEl.style.width = `${Math.min(results.engagement_score * 100, 100)}%`;
  
  // Set sentiment
  const sentiment = results.sentiment_score.toFixed(2);
  sentimentValueEl.textContent = `${sentiment} (${results.sentiment_category})`;
  
  // Color sentiment based on value
  if (results.sentiment_score > 0.33) {
    sentimentValueEl.style.color = 'var(--success-color)';
  } else if (results.sentiment_score < -0.33) {
    sentimentValueEl.style.color = 'var(--danger-color)';
  } else {
    sentimentValueEl.style.color = 'var(--secondary-color)';
  }
}

// Populate causal factors list
function populateCausalFactors(factors) {
  // Clear previous factors
  factorsList.innerHTML = '';
  
  // Hide placeholder
  document.querySelector('#causalFactors .placeholder').classList.add('hidden');
  factorsList.classList.remove('hidden');
  
  // Add factors to list
  factors.forEach(factor => {
    const li = document.createElement('li');
    
    const nameSpan = document.createElement('span');
    nameSpan.className = 'factor-name';
    nameSpan.textContent = factor.name;
    
    const valueSpan = document.createElement('span');
    valueSpan.className = 'factor-value';
    valueSpan.textContent = `${(factor.effect * 100).toFixed(1)}%`;
    
    li.appendChild(nameSpan);
    li.appendChild(valueSpan);
    factorsList.appendChild(li);
  });
}

// Populate suggestions list
function populateSuggestions(suggestions) {
  // Clear previous suggestions
  suggestionsList.innerHTML = '';
  
  // Hide placeholder
  document.querySelector('.optimization-suggestions .placeholder').classList.add('hidden');
  suggestionsList.classList.remove('hidden');
  
  // Add suggestions to list
  suggestions.forEach(suggestion => {
    const li = document.createElement('li');
    li.textContent = suggestion;
    suggestionsList.appendChild(li);
  });
}

// Setup counterfactual analysis tool
function setupCounterfactualTool(factors) {
  // Show counterfactual tool
  counterfactualTool.classList.remove('hidden');
  
  // Use first factor by default
  const factor = factors[0];
  selectedFactorEl.textContent = factor.name;
  
  // Set initial prediction
  predictedOutcomeEl.textContent = analysisResults.engagement_score.toFixed(2);
  
  // Store current factor for slider
  counterfactualTool.dataset.currentFactor = factor.name;
  counterfactualTool.dataset.currentEffect = factor.effect;
}

// Handle slider change for counterfactual analysis
function handleSliderChange() {
  const sliderValue = parseInt(factorSlider.value);
  const factorName = counterfactualTool.dataset.currentFactor;
  const factorEffect = parseFloat(counterfactualTool.dataset.currentEffect);
  
  // Calculate counterfactual score
  // This is a simplified calculation - in reality, you'd call the API
  const baseline = analysisResults.engagement_score;
  const defaultValue = 50; // Middle of slider
  const change = (sliderValue - defaultValue) / 100; // -0.5 to 0.5
  const adjustedScore = baseline + (change * factorEffect * 2); // Scale effect
  
  // Update prediction
  predictedOutcomeEl.textContent = Math.max(0, Math.min(1, adjustedScore)).toFixed(2);
}

// Handle tab click
function handleTabClick(clickedButton) {
  // Update active button
  tabButtons.forEach(button => {
    button.classList.remove('active');
  });
  clickedButton.classList.add('active');
  
  // Show selected tab content
  const tabName = clickedButton.dataset.tab;
  tabContents.forEach(content => {
    content.classList.add('hidden');
  });
  document.getElementById(tabName).classList.remove('hidden');
}

// Save settings
function saveSettings() {
  const newEndpoint = apiEndpointInput.value.trim();
  const newKey = apiKeyInput.value.trim();
  
  if (!newEndpoint) {
    alert('Please enter a valid API endpoint');
    return;
  }
  
  // Update settings
  settings.apiEndpoint = newEndpoint;
  settings.apiKey = newKey;
  
  // Save to storage
  chrome.storage.sync.set({
    apiEndpoint: newEndpoint,
    apiKey: newKey
  }, () => {
    // Check API status with new settings
    checkApiStatus();
    
    // Close modal
    settingsModal.classList.add('hidden');
  });
}

// Check API status
async function checkApiStatus() {
  if (!settings.apiEndpoint) {
    apiStatusEl.className = 'status-indicator offline';
    return;
  }
  
  try {
    const response = await fetch(`${settings.apiEndpoint}/status`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${settings.apiKey}`
      }
    });
    
    if (response.ok) {
      apiStatusEl.className = 'status-indicator online';
    } else {
      apiStatusEl.className = 'status-indicator offline';
    }
  } catch (error) {
    console.error('API status check failed:', error);
    apiStatusEl.className = 'status-indicator offline';
  }
}

// Show error message
function showError(message) {
  loadingEl.classList.add('hidden');
  analyzeBtn.disabled = false;
  alert(message);
} 