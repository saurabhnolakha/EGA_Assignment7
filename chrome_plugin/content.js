// Send only the URL, title, and timestamp to the backend API
(function() {
  fetch('http://127.0.0.1:5000/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      url: window.location.href,
      title: document.title,
      timestamp: Date.now()
    })
  }).then(res => {
    if (!res.ok) throw new Error('Failed to send URL to backend');
    return res.json();
  }).then(data => {
    console.log('[EGA Plugin] Backend response:', data);
  }).catch(err => {
    console.error('[EGA Plugin] Error sending URL to backend:', err);
  });
})();

// Highlight all occurrences of a term on the page
function highlightTermOnPage(term) {
  if (!term) return;
  console.log('[EGA Plugin] Highlighting term on page:', term);
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(escaped, 'gi');
  const treeWalker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
  const nodes = [];
  while (treeWalker.nextNode()) {
    nodes.push(treeWalker.currentNode);
  }
  nodes.forEach(node => {
    if (node.parentNode && node.nodeValue.match(regex)) {
      const span = document.createElement('span');
      span.innerHTML = node.nodeValue.replace(regex, match => `<mark style="background-color: #ffd1dc; color: black;">${match}</mark>`);
      node.parentNode.replaceChild(span, node);
    }
  });
}

// Listen for highlight messages from the popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'highlight' && message.term) {
    console.log('[EGA Plugin] Received highlight message:', message.term);
    highlightTermOnPage(message.term);
  }
}); 