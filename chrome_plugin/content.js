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