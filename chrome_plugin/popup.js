function highlightTerm(text, term) {
  if (!term) return text;
  // Escape special regex characters in the term
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(escaped, 'gi');
  // Use a pastel pink background for highlighting
  return text.replace(regex, match => `<mark style="background-color: #ffd1dc; color: black;">${match}</mark>`);
}

function highlightInTab(tabId, term) {
  chrome.tabs.sendMessage(tabId, { action: 'highlight', term });
}

async function getTopResult(query, rerank) {
  const response = await fetch('http://127.0.0.1:5000/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k: 1, model: 'ollama', rerank })
  });
  if (!response.ok) throw new Error('Server error');
  const data = await response.json();
  if (!data.results || data.results.length === 0) return null;
  return data.results[0];
}

document.getElementById('phraseSearchBtn').addEventListener('click', async () => {
  const query = document.getElementById('searchBox').value.trim();
  const resultsDiv = document.getElementById('results');
  if (!query) return;
  resultsDiv.innerText = 'Searching...';
  try {
    const r = await getTopResult(query, true); // rerank: true
    if (!r) {
      resultsDiv.innerText = 'No results found.';
      return;
    }
    // Use only the user's search query for the text fragment
    const searchTerm = encodeURIComponent(query.trim());
    const urlWithFragment = `${r.url}#:~:text=${searchTerm}`;
    chrome.tabs.create({ url: urlWithFragment });
    resultsDiv.innerText = 'Opening result with phrase highlight...';
  } catch (err) {
    resultsDiv.innerText = 'Error searching. Please try again.';
    console.error('[EGA Plugin] Search error:', err);
  }
});

document.getElementById('contentSearchBtn').addEventListener('click', async () => {
  const query = document.getElementById('searchBox').value.trim();
  const resultsDiv = document.getElementById('results');
  if (!query) return;
  resultsDiv.innerText = 'Searching...';
  try {
    const r = await getTopResult(query, false); // rerank: false
    if (!r) {
      resultsDiv.innerText = 'No results found.';
      return;
    }
    // Highlight the first 15 words of the chunk
    const chunkText = (r.chunk || '');
    const first15 = chunkText.split(/\s+/).slice(0, 15).join(' ');
    chrome.tabs.create({ url: r.url }, (tab) => {
      // Wait for the tab to load, then inject and highlight
      const listener = (tabId, changeInfo) => {
        if (tabId === tab.id && changeInfo.status === 'complete') {
          chrome.scripting.executeScript(
            {
              target: { tabId: tab.id },
              files: ['content.js']
            },
            () => {
              highlightInTab(tab.id, first15);
            }
          );
          chrome.tabs.onUpdated.removeListener(listener);
        }
      };
      chrome.tabs.onUpdated.addListener(listener);
    });
    resultsDiv.innerText = 'Opening result and highlighting content...';
  } catch (err) {
    resultsDiv.innerText = 'Error searching. Please try again.';
    console.error('[EGA Plugin] Search error:', err);
  }
}); 