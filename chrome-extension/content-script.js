// content-script.js
function extractPageContent() {
  const content = {
    url: window.location.href,
    title: document.title,
    text: document.body.innerText,
    mainContent: extractMainContent(),
    images: Array.from(document.images)
      .map(img => ({
        src: img.src,
        alt: img.alt,
        width: img.width,
        height: img.height
      }))
      .filter(img => img.width > 100 && img.height > 100) // Filter small images
  };
  return content;
}

// Attempt to identify and extract the main content of the page
function extractMainContent() {
  // Common content selectors in order of priority
  const contentSelectors = [
    'article',
    '[role="main"]',
    'main',
    '.main-content',
    '#content',
    '.content',
    '.post-content',
    '.entry-content'
  ];
  
  // Try each selector until we find content
  for (const selector of contentSelectors) {
    const element = document.querySelector(selector);
    if (element && element.innerText.length > 200) {
      return element.innerText;
    }
  }
  
  // Fallback: Try to identify the content area with the most text
  let bestElement = null;
  let maxLength = 0;
  
  const contentCandidates = document.querySelectorAll('div, section, article');
  for (const element of contentCandidates) {
    // Skip tiny elements or likely navigation/sidebar elements
    if (element.offsetWidth < 200 || element.innerText.length < 200) {
      continue;
    }
    
    // Avoid navigation, header, footer areas
    const tagName = element.tagName.toLowerCase();
    const className = element.className.toLowerCase();
    const id = element.id.toLowerCase();
    const isLikelyContent = !(['nav', 'header', 'footer'].includes(tagName) ||
                             className.includes('nav') || 
                             className.includes('menu') ||
                             className.includes('header') ||
                             className.includes('footer') ||
                             id.includes('nav') ||
                             id.includes('menu') ||
                             id.includes('header') ||
                             id.includes('footer'));
    
    if (isLikelyContent && element.innerText.length > maxLength) {
      maxLength = element.innerText.length;
      bestElement = element;
    }
  }
  
  return bestElement ? bestElement.innerText : document.body.innerText;
}

// Extract the primary image from the page (e.g., featured image, hero image)
function extractPrimaryImage() {
  // Common selectors for featured images
  const imageSelectors = [
    'meta[property="og:image"]',
    'meta[name="twitter:image"]',
    '.featured-image img',
    '.post-thumbnail img',
    'article img',
    '.hero-image img',
    'header img'
  ];
  
  // Try meta tags first
  for (const selector of imageSelectors.slice(0, 2)) {
    const metaTag = document.querySelector(selector);
    if (metaTag && metaTag.content) {
      return {
        src: metaTag.content,
        alt: "",
        isMeta: true
      };
    }
  }
  
  // Then try actual image elements
  for (const selector of imageSelectors.slice(2)) {
    const img = document.querySelector(selector);
    if (img && img.src && img.width > 200 && img.height > 200) {
      return {
        src: img.src,
        alt: img.alt || "",
        width: img.width,
        height: img.height
      };
    }
  }
  
  // Fallback: find the largest image
  const images = Array.from(document.images)
    .filter(img => img.width > 200 && img.height > 200 && isVisible(img));
  
  if (images.length > 0) {
    // Sort by size (area) descending
    images.sort((a, b) => (b.width * b.height) - (a.width * a.height));
    const largest = images[0];
    return {
      src: largest.src,
      alt: largest.alt || "",
      width: largest.width,
      height: largest.height
    };
  }
  
  return null;
}

// Check if an element is visible
function isVisible(element) {
  const style = window.getComputedStyle(element);
  return style.display !== 'none' && style.visibility !== 'hidden' && element.offsetWidth > 0 && element.offsetHeight > 0;
}

// Listen for messages from popup or background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "extractContent") {
    const content = extractPageContent();
    content.primaryImage = extractPrimaryImage();
    sendResponse({ content: content });
  }
}); 