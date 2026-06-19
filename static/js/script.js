/**
 * GestureTalk - Client Controller
 * Handles webcam stream, canvas landmark overlays, Flask APIs communication,
 * gesture accumulation state machine, dual TTS, theme toggle, and keyboard shortcuts.
 */

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const body = document.documentElement;
    const themeToggle = document.getElementById('theme-toggle');
    const themeSun = document.getElementById('theme-sun');
    const themeMoon = document.getElementById('theme-moon');
    
    const hamburger = document.getElementById('hamburger');
    const navMenu = document.getElementById('nav-menu');
    
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('landmarks-canvas');
    const ctx = canvas ? canvas.getContext('2d') : null;
    
    const sentenceBubble = document.getElementById('sentence-bubble');
    const btnSpeak = document.getElementById('btn-speak');
    const btnCopy = document.getElementById('btn-copy');
    const btnClear = document.getElementById('btn-clear');
    
    const cameraBadge = document.getElementById('camera-badge');
    const cameraStatusText = document.getElementById('camera-status-text');
    const predictedLetter = document.getElementById('predicted-letter');
    const confidencePercentage = document.getElementById('confidence-percentage');
    const confidenceBar = document.getElementById('confidence-bar');
    
    const classifierModeSelect = document.getElementById('classifier-mode');
    const ttsEngineSelect = document.getElementById('tts-engine');
    const debounceSlider = document.getElementById('debounce-slider');
    const debounceValText = document.getElementById('debounce-val');
    const historyList = document.getElementById('history-list');
    
    // Application State Variables
    let currentSentence = "";
    let isCameraActive = false;
    let frameIntervalId = null;
    let predictionMode = "heuristic"; // Default mode
    let ttsEngine = "frontend"; // Default speech engine
    let debounceTimeMs = 1500; // Time (ms) a gesture must be held to register
    
    // Gesture Debounce State Machine
    let currentPrediction = null;
    let lastRegisteredPrediction = null;
    let gestureHoldStart = null;
    let isGestureRegistered = false; // Prevents repeating same hold without release
    
    // Toast Notification System
    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = 'fa-info-circle';
        if (type === 'success') icon = 'fa-check-circle';
        if (type === 'danger') icon = 'fa-circle-xmark';
        if (type === 'warning') icon = 'fa-triangle-exclamation';
        
        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        // Remove toast after animation finishes
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(20px)';
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 3000);
    }

    // --- Theme Switcher Logic ---
    const savedTheme = localStorage.getItem('theme') || 'dark';
    body.setAttribute('data-theme', savedTheme);
    updateThemeIcons(savedTheme);
    
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = body.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            body.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeIcons(newTheme);
            showToast(`Theme switched to ${newTheme} mode!`, 'success');
        });
    }
    
    function updateThemeIcons(theme) {
        if (!themeSun || !themeMoon) return;
        if (theme === 'dark') {
            themeSun.style.display = 'none';
            themeMoon.style.display = 'block';
        } else {
            themeSun.style.display = 'block';
            themeMoon.style.display = 'none';
        }
    }

    // --- Responsive Mobile Navbar Hamburger ---
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            hamburger.classList.toggle('active');
        });
        
        // Close menu when clicking outside or on a link
        document.querySelectorAll('.nav-item a').forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('active');
                hamburger.classList.remove('active');
            });
        });
    }

    // --- Webcam Capture & Frame Loop ---
    if (webcam) {
        startWebcam();
        
        // Settings Listeners
        if (classifierModeSelect) {
            classifierModeSelect.addEventListener('change', (e) => {
                predictionMode = e.target.value;
                showToast(`Classification set to: ${e.target.options[e.target.selectedIndex].text}`, 'info');
                // Reset hold state on mode change
                resetHoldState();
            });
        }
        
        if (ttsEngineSelect) {
            ttsEngineSelect.addEventListener('change', (e) => {
                ttsEngine = e.target.value;
                showToast(`Speech synthesis: ${e.target.options[e.target.selectedIndex].text}`, 'info');
            });
        }
        
        if (debounceSlider && debounceValText) {
            debounceSlider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                debounceTimeMs = val * 1000;
                debounceValText.innerText = `${val.toFixed(1)}s`;
            });
        }
    }
    
    async function startWebcam() {
        // Check if MediaPipe is available in the browser (client-side)
        if (typeof Hands !== 'undefined' && typeof Camera !== 'undefined') {
            console.log("Initializing client-side MediaPipe Hands...");
            initClientSideMediaPipe();
        } else {
            console.log("Client-side MediaPipe missing. Falling back to server-side image processing...");
            showToast("Offline mode: Using server-side classification fallback.", "warning");
            initServerSideFallback();
        }
    }

    // CLIENT-SIDE MEDIAPIPE PIPELINE (High performance)
    function hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay && overlay.style.display !== 'none') {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 400);
        }
    }

    function initClientSideMediaPipe() {
        const hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
        });

        hands.onResults((results) => {
            // Dismiss loading indicator on first frame result
            hideLoadingOverlay();
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                const landmarks = results.multiHandLandmarks[0];
                
                // Draw skeleton landmarks locally
                drawSkeletalMap(landmarks);
                
                // Send coordinates directly to Flask API
                sendLandmarksToBackend(landmarks);
            } else {
                // No hand detected
                predictedLetter.innerText = "-";
                confidencePercentage.innerText = "0%";
                confidenceBar.style.width = "0%";
                resetHoldState();
            }
        });

        const camera = new Camera(webcam, {
            onFrame: async () => {
                if (isCameraActive) {
                    try {
                        await hands.send({ image: webcam });
                    } catch (e) {
                        console.error("MediaPipe frame send error:", e);
                    }
                }
            },
            width: 640,
            height: 480
        });

        camera.start()
            .then(() => {
                isCameraActive = true;
                // Wait for video properties
                setTimeout(() => {
                    canvas.width = webcam.videoWidth || 640;
                    canvas.height = webcam.videoHeight || 480;
                }, 1000);
                
                if (cameraBadge) {
                    cameraBadge.className = 'status-badge active';
                    cameraStatusText.innerText = 'Connected';
                }
                showToast("Camera connected (GPU acceleration active)!", "success");
            })
            .catch((err) => {
                console.error("Camera start failed:", err);
                showToast("Camera block or permission error.", "danger");
                if (cameraBadge) {
                    cameraBadge.className = 'status-badge inactive';
                    cameraStatusText.innerText = 'Blocked';
                }
            });
    }

    async function sendLandmarksToBackend(landmarks) {
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    landmarks: landmarks,
                    mode: predictionMode
                })
            });
            const data = await response.json();
            if (data.success) {
                // Parse prediction response
                const prediction = data.prediction;
                const confidence = data.confidence;
                
                if (prediction) {
                    predictedLetter.innerText = prediction;
                    const pct = Math.round(confidence * 100);
                    confidencePercentage.innerText = `${pct}%`;
                    confidenceBar.style.width = `${pct}%`;
                    handleGestureState(prediction);
                } else {
                    predictedLetter.innerText = "-";
                    confidencePercentage.innerText = "0%";
                    confidenceBar.style.width = "0%";
                    resetHoldState();
                }
            }
        } catch (err) {
            console.error("Failed to send landmarks:", err);
        }
    }

    // SERVER-SIDE FALLBACK PIPELINE (Traditional frame capture)
    async function initServerSideFallback() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' },
                audio: false
            });
            webcam.srcObject = stream;
            isCameraActive = true;
            
            webcam.onloadedmetadata = () => {
                hideLoadingOverlay();
                canvas.width = webcam.videoWidth;
                canvas.height = webcam.videoHeight;
                if (cameraBadge) {
                    cameraBadge.className = 'status-badge active';
                    cameraStatusText.innerText = 'Connected (Fallback)';
                }
                frameIntervalId = setInterval(captureAndPredict, 250);
            };
        } catch (err) {
            console.error("Camera access failed:", err);
            showToast("Camera access failed.", "danger");
        }
    }

    async function captureAndPredict() {
        if (!isCameraActive || !webcam.videoWidth) return;
        
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = 320;
        offscreenCanvas.height = 240;
        const offContext = offscreenCanvas.getContext('2d');
        
        offContext.save();
        offContext.translate(offscreenCanvas.width, 0);
        offContext.scale(-1, 1);
        offContext.drawImage(webcam, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        offContext.restore();
        
        const base64Image = offscreenCanvas.toDataURL('image/jpeg', 0.6);
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: base64Image,
                    mode: predictionMode
                })
            });
            const data = await response.json();
            if (data.success) {
                handlePredictionResult(data);
            }
        } catch (err) {
            console.error("Fallback predict loop network error:", err);
        }
    }

    // Process predictions and draw skeletal lines
    function handlePredictionResult(result) {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const prediction = result.prediction;
        const confidence = result.confidence;
        const landmarks = result.landmarks;
        
        // Update stats UI
        if (prediction) {
            predictedLetter.innerText = prediction;
            const pct = Math.round(confidence * 100);
            confidencePercentage.innerText = `${pct}%`;
            confidenceBar.style.width = `${pct}%`;
            
            // Draw skeleton landmarks on Canvas
            if (landmarks) {
                drawSkeletalMap(landmarks);
            }
            
            // State Machine for Debounced Sentence Building
            handleGestureState(prediction);
        } else {
            // No hand detected
            predictedLetter.innerText = "-";
            confidencePercentage.innerText = "0%";
            confidenceBar.style.width = "0%";
            resetHoldState();
        }
    }

    // State machine resolving continuous holds
    function handleGestureState(prediction) {
        const now = Date.now();
        
        if (prediction !== currentPrediction) {
            // Gesture changed! Reset hold timer
            currentPrediction = prediction;
            gestureHoldStart = now;
            isGestureRegistered = false;
        } else if (!isGestureRegistered) {
            // Same gesture is held
            const timeHeld = now - gestureHoldStart;
            
            // Check if debounce cooldown reached
            if (timeHeld >= debounceTimeMs) {
                registerGesture(prediction);
                isGestureRegistered = true; // Lock out further registration for this hold
            }
        }
    }
    
    function resetHoldState() {
        currentPrediction = null;
        gestureHoldStart = null;
        isGestureRegistered = false;
    }

    // Trigger action on successful gesture registration
    function registerGesture(gesture) {
        let msg = "";
        
        if (gesture === 'SPACE') {
            appendChar(" ");
            msg = "Added SPACE";
            showToast(msg, "success");
        } else if (gesture === 'DELETE') {
            deleteLastChar();
            msg = "Removed last character";
            showToast(msg, "warning");
        } else if (gesture === 'CLEAR') {
            clearSentence();
            msg = "Cleared whole sentence";
            showToast(msg, "danger");
        } else {
            // Alphabet character (A, B, C, L, Y, etc.)
            appendChar(gesture);
            msg = `Registered: '${gesture}'`;
            showToast(msg, "success");
        }
        
        // Add to history log
        addToHistoryLog(gesture, msg);
    }

    // Draw lines between hand joints
    function drawSkeletalMap(landmarks) {
        if (!ctx) return;
        
        // MediaPipe connections map
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],       // Index
            [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
            [9, 13], [13, 14], [14, 15], [15, 16], // Ring
            [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
            [0, 17]                               // Palm base
        ];
        
        ctx.strokeStyle = getComputedStyle(body).getPropertyValue('--primary').trim();
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        
        // Draw connections
        connections.forEach(([p1, p2]) => {
            const pt1 = landmarks[p1];
            const pt2 = landmarks[p2];
            
            if (pt1 && pt2) {
                ctx.beginPath();
                ctx.moveTo(pt1.x * canvas.width, pt1.y * canvas.height);
                ctx.lineTo(pt2.x * canvas.width, pt2.y * canvas.height);
                ctx.stroke();
            }
        });
        
        // Draw joints points
        ctx.fillStyle = getComputedStyle(body).getPropertyValue('--secondary').trim();
        landmarks.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 5, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    // Append a letter or space to the sentence
    function appendChar(char) {
        // If placeholder exists, remove it
        if (currentSentence.length === 0) {
            sentenceBubble.innerHTML = "";
        }
        
        currentSentence += char;
        sentenceBubble.innerText = currentSentence;
    }
    
    // Backspace delete
    function deleteLastChar() {
        if (currentSentence.length > 0) {
            currentSentence = currentSentence.slice(0, -1);
            
            if (currentSentence.length === 0) {
                sentenceBubble.innerHTML = '<span class="sentence-placeholder">Awaiting hand gestures...</span>';
            } else {
                sentenceBubble.innerText = currentSentence;
            }
        }
    }
    
    // Clear whole sentence
    function clearSentence() {
        currentSentence = "";
        sentenceBubble.innerHTML = '<span class="sentence-placeholder">Awaiting hand gestures...</span>';
    }

    // Copy to clipboard
    if (btnCopy) {
        btnCopy.addEventListener('click', () => {
            if (currentSentence.length === 0) {
                showToast("No sentence text to copy!", "warning");
                return;
            }
            
            navigator.clipboard.writeText(currentSentence)
                .then(() => {
                    showToast("Sentence copied to clipboard!", "success");
                })
                .catch(err => {
                    console.error("Copy failed:", err);
                    showToast("Failed to copy text.", "danger");
                });
        });
    }

    // Clear Button trigger
    if (btnClear) {
        btnClear.addEventListener('click', () => {
            if (currentSentence.length > 0) {
                clearSentence();
                showToast("Text cleared!", "danger");
                addToHistoryLog("CLEAR", "Sentence manually cleared");
            }
        });
    }

    // Speech engine trigger
    if (btnSpeak) {
        btnSpeak.addEventListener('click', () => {
            speakText(currentSentence);
        });
    }
    
    function speakText(text) {
        const cleanText = text.trim();
        if (cleanText.length === 0) {
            showToast("No text to speak!", "warning");
            return;
        }
        
        // Cooldown: Disable speak button temporarily to prevent spamming
        btnSpeak.disabled = true;
        setTimeout(() => { btnSpeak.disabled = false; }, 2500);
        
        if (ttsEngine === "frontend") {
            // Method 1: Web Speech API (Client side browser)
            try {
                // Cancel active speech
                window.speechSynthesis.cancel();
                
                const utterance = new SpeechSynthesisUtterance(cleanText);
                utterance.rate = 1.0;
                utterance.volume = 1.0;
                
                // Select a default voice
                const voices = window.speechSynthesis.getVoices();
                if (voices.length > 0) {
                    // Try to find a clear English female voice
                    const preferredVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Zira') || v.name.includes('Natural')));
                    if (preferredVoice) utterance.voice = preferredVoice;
                }
                
                window.speechSynthesis.speak(utterance);
                showToast("Speaking sentence...", "success");
            } catch (err) {
                console.error("Web Speech API Failed:", err);
                showToast("Browser TTS failed. Trying backend speech.", "warning");
                speakTextBackend(cleanText);
            }
        } else {
            // Method 2: Backend TTS Endpoint (pyttsx3 python)
            speakTextBackend(cleanText);
        }
    }
    
    async function speakTextBackend(text) {
        try {
            showToast("Requesting backend speech...", "info");
            const response = await fetch('/api/speak', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await response.json();
            if (data.success) {
                showToast("Speaking from backend...", "success");
            } else {
                showToast(`Backend TTS error: ${data.error}`, "danger");
            }
        } catch (err) {
            console.error("Backend TTS network error:", err);
            showToast("Backend TTS network error.", "danger");
        }
    }

    // Add entries to History Panel
    function addToHistoryLog(action, detail) {
        if (!historyList) return;
        
        // If first entry, remove placeholder
        if (historyList.innerHTML.includes("No entries yet")) {
            historyList.innerHTML = "";
        }
        
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `
            <span><strong>${action}</strong>: ${detail}</span>
            <span class="history-time">${timestamp}</span>
        `;
        
        historyList.insertBefore(item, historyList.firstChild);
        
        // Cap history entries to 10
        if (historyList.children.length > 10) {
            historyList.removeChild(historyList.lastChild);
        }
    }

    // --- Contact Form Submission Handler ---
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const nameInput = document.getElementById('contact-name');
            const name = nameInput ? nameInput.value : 'Customer';
            showToast(`Thank you, ${name}! Your message has been sent successfully.`, 'success');
            contactForm.reset();
        });
    }

    // --- Document Keyboard Shortcuts ---
    document.addEventListener('keydown', (e) => {
        // Skip shortcuts if user is typing in a form input or selector
        const activeEl = document.activeElement;
        if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'SELECT' || activeEl.tagName === 'TEXTAREA')) {
            return;
        }
        
        const key = e.key;
        
        if (key === 'Enter') {
            e.preventDefault();
            speakText(currentSentence);
        } else if (key === 'Escape') {
            e.preventDefault();
            if (currentSentence.length > 0) {
                clearSentence();
                showToast("Text cleared via Esc shortcut!", "danger");
                addToHistoryLog("ESC SHORTCUT", "Sentence cleared");
            }
        } else if (key === ' ') {
            // Space key
            e.preventDefault();
            appendChar(" ");
            showToast("Added space via Spacebar shortcut", "success");
            addToHistoryLog("SPACE KEY", "Manual space added");
        } else if (key === 'Backspace') {
            // Backspace key
            e.preventDefault();
            if (currentSentence.length > 0) {
                deleteLastChar();
                showToast("Removed last character via Backspace shortcut", "warning");
                addToHistoryLog("BACKSPACE KEY", "Manual character deletion");
            }
        }
    });
});
