// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "get_html") {
        sendResponse({ html: document.body.outerHTML });
    }
    else if (request.action === "execute_actions") {
        console.log("Autofill Agent: Executing actions...", request.data);
        let count = 0;
        request.data.forEach(task => {
            const el = document.querySelector(task.selector);
            if (el) {
                // Determine value to set
                let valToSet = task.value;
                
                if (task.type === 'checkbox') {
                    el.checked = (task.value === 'true');
                } else if (task.type === 'radio') {
                    if (task.value === 'true') el.checked = true;
                } else {
                    el.value = valToSet;
                }

                // Dispatch events to ensure React/Angular/Vue frameworks detect the change
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.dispatchEvent(new Event('blur', { bubbles: true }));
                count++;
            }
        });
        sendResponse({ status: "done", count: count });
    }
});
