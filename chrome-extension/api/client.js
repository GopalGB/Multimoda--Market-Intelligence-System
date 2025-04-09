/**
 * API Client for the Audience Intelligence Platform
 * Handles communication with the backend API
 */
class AudienceIntelligenceAPI {
  /**
   * Initialize the API client
   * @param {string} endpoint - API endpoint URL
   * @param {string} apiKey - API key for authentication
   */
  constructor(endpoint, apiKey) {
    this.endpoint = endpoint;
    this.apiKey = apiKey;
  }

  /**
   * Set API credentials
   * @param {string} endpoint - API endpoint URL
   * @param {string} apiKey - API key for authentication
   */
  setCredentials(endpoint, apiKey) {
    this.endpoint = endpoint;
    this.apiKey = apiKey;
  }

  /**
   * Check API status
   * @returns {Promise<Object>} Status response
   */
  async checkStatus() {
    return this._request('/status', 'GET');
  }

  /**
   * Analyze content
   * @param {Object} content - Content data
   * @returns {Promise<Object>} Analysis results
   */
  async analyzeContent(content) {
    const data = {
      url: content.url,
      title: content.title,
      text: content.mainContent || content.text,
      primary_image: content.primaryImage ? content.primaryImage.src : null
    };

    return this._request('/analyze', 'POST', data);
  }

  /**
   * Get causal analysis
   * @param {Object} content - Content data
   * @returns {Promise<Object>} Causal analysis results
   */
  async getCausalAnalysis(content) {
    const data = {
      url: content.url,
      title: content.title,
      text: content.mainContent || content.text,
      primary_image: content.primaryImage ? content.primaryImage.src : null
    };

    return this._request('/causal-analysis', 'POST', data);
  }

  /**
   * Generate counterfactual prediction
   * @param {Object} params - Counterfactual parameters
   * @returns {Promise<Object>} Counterfactual results
   */
  async getCounterfactual(params) {
    return this._request('/counterfactual', 'POST', params);
  }

  /**
   * Get optimization suggestions
   * @param {Object} content - Content data
   * @returns {Promise<Object>} Optimization suggestions
   */
  async getOptimizationSuggestions(content) {
    const data = {
      url: content.url,
      title: content.title,
      text: content.mainContent || content.text,
      primary_image: content.primaryImage ? content.primaryImage.src : null
    };

    return this._request('/optimize', 'POST', data);
  }

  /**
   * Make an API request
   * @param {string} path - API path
   * @param {string} method - HTTP method
   * @param {Object} data - Request data
   * @returns {Promise<Object>} Response data
   * @private
   */
  async _request(path, method, data = null) {
    // Validate credentials
    if (!this.endpoint || !this.apiKey) {
      throw new Error('API credentials not configured');
    }

    // Build request options
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      }
    };

    // Add request body for non-GET requests
    if (data && method !== 'GET') {
      options.body = JSON.stringify(data);
    }

    try {
      // Make API request
      const response = await fetch(`${this.endpoint}${path}`, options);

      // Parse response
      const responseData = await response.json();

      // Handle errors
      if (!response.ok) {
        throw new Error(responseData.message || `API error: ${response.status}`);
      }

      return responseData;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }
}

// Create singleton instance
const apiClient = new AudienceIntelligenceAPI();

// Export API client
export default apiClient; 