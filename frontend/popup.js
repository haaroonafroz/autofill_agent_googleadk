// Helper to generate UUID
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const userIdInput = document.getElementById('userId');
    const cvFileInput = document.getElementById('cvFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const startBtn = document.getElementById('startBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const agentStatus = document.getElementById('agentStatus');
    
    const API_URL = "http://localhost:8000";

    // Load saved user ID
    chrome.storage.local.get(['user_id'], (result) => {
        if (result.user_id) {
            userIdInput.value = result.user_id;
        } else {
            const newId = generateUUID();
            chrome.storage.local.set({ user_id: newId });
            userIdInput.value = newId;
        }
    });

    // 1. Upload CV
    uploadBtn.addEventListener('click', async () => {
        const userId = userIdInput.value;
        const file = cvFileInput.files[0];

        if (!userId || !file) {
            uploadStatus.textContent = "Please enter User ID and select a file.";
            return;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("user_id", userId);

        uploadStatus.textContent = "Uploading...";
        
        try {
            const response = await fetch(`${API_URL}/upload_cv`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                uploadStatus.textContent = "Success: CV Indexed!";
                uploadStatus.style.color = "green";
                startBtn.disabled = false;
            } else {
                uploadStatus.textContent = "Error uploading.";
                uploadStatus.style.color = "red";
            }
        } catch (error) {
            uploadStatus.textContent = "Connection Error: Is server running?";
            console.error(error);
        }
    });

    // 2. Start Agent (Extension Native Flow)
    startBtn.addEventListener('click', async () => {
        const userId = userIdInput.value;
        agentStatus.textContent = "Agent starting...";
        agentStatus.style.color = "blue";

        // Get current tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        // 1. Get HTML from Content Script
        chrome.tabs.sendMessage(tab.id, {action: "get_html"}, async (response) => {
            if (chrome.runtime.lastError || !response || !response.html) {
                agentStatus.textContent = "Error: Cannot read page. Refresh?";
                return;
            }

            agentStatus.textContent = "Thinking...";

            try {
                // 2. Send to Backend
                const apiRes = await fetch(`${API_URL}/process_page`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: tab.url,
                        html: response.html,
                        user_id: userId
                    })
                });
                
                if (!apiRes.ok) throw new Error("API Error");
                
                const data = await apiRes.json();
                const actions = data.actions;
                
                if (!actions || actions.length === 0) {
                    agentStatus.textContent = "No fields found to fill.";
                    return;
                }

                agentStatus.textContent = `Found ${actions.length} fields. Filling...`;

                // 3. Send Actions back to Content Script
                chrome.tabs.sendMessage(tab.id, {
                    action: "execute_actions", 
                    data: actions
                }, (res) => {
                     if (res && res.status === "done") {
                         agentStatus.textContent = `Done! Filled ${res.count} fields.`;
                         agentStatus.style.color = "green";
                     }
                });

            } catch (error) {
                console.error(error);
                agentStatus.textContent = "Agent Error.";
                agentStatus.style.color = "red";
            }
        });
    });
});
