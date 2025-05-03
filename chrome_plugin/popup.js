document.getElementById('searchBtn').addEventListener('click', async () => {
  const query = document.getElementById('searchBox').value.trim();
  if (!query) return;
  document.getElementById('results').innerText = 'Searching...';
  // TODO: Send query to backend API and display results
  console.log('[EGA Plugin] Search query:', query);
  // Placeholder result
  setTimeout(() => {
    document.getElementById('results').innerText = 'Results will appear here.';
  }, 500);
}); 