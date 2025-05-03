// Log when the service worker starts
console.log('[EGA Plugin] Service worker started');

// Listen for completed navigations (real-time capture)
chrome.webNavigation.onCompleted.addListener(async (details) => {
  console.log('[EGA Plugin] webNavigation.onCompleted event:', details);
  if (details.frameId !== 0) return; // Only main frame
  chrome.tabs.get(details.tabId, (tab) => {
    if (!tab.url || tab.url.startsWith('chrome://')) return;
    const visit = { url: tab.url, title: tab.title, timestamp: Date.now() };
    // Store in local storage (append to history)
    chrome.storage.local.get({egaHistory: []}, (data) => {
      const history = data.egaHistory;
      history.push(visit);
      chrome.storage.local.set({egaHistory: history});
      console.log('[EGA Plugin] Added visit to egaHistory:', visit);
    });
    // Optionally, trigger content script injection
    chrome.scripting.executeScript({
      target: {tabId: details.tabId},
      files: ['content.js']
    }).then(() => {
      console.log('[EGA Plugin] Injected content.js into tab', details.tabId);
    }).catch((e) => {
      console.error('[EGA Plugin] Error injecting content.js:', e);
    });
  });
});

// Periodic catch-up (every hour)
chrome.alarms.create('ega_catchup', { periodInMinutes: 60 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'ega_catchup') {
    console.log('[EGA Plugin] ega_catchup alarm fired');
    // Fetch last 100 visited URLs in the last 24 hours
    const startTime = Date.now() - 24 * 60 * 60 * 1000;
    chrome.history.search({text: '', startTime, maxResults: 100}, (results) => {
      chrome.storage.local.get({egaHistory: []}, (data) => {
        const known = new Set(data.egaHistory.map(v => v.url));
        const newVisits = results.filter(r => !known.has(r.url)).map(r => ({
          url: r.url,
          title: r.title,
          timestamp: r.lastVisitTime
        }));
        if (newVisits.length > 0) {
          const updated = data.egaHistory.concat(newVisits);
          chrome.storage.local.set({egaHistory: updated});
          console.log('[EGA Plugin] Added new visits from catch-up:', newVisits);
        } else {
          console.log('[EGA Plugin] No new visits found in catch-up.');
        }
      });
    });
  }
}); 