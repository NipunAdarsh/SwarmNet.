/**
 * SwarmNet — Live Webcam Real-Time Classification
 * Sends frames to server which runs a background NPU inference loop.
 * The server-side loop keeps the NPU continuously saturated.
 */
(function () {
    'use strict';

    const API_BASE = window.location.origin;
    const FRAME_SEND_INTERVAL = 200; // Send new frame every 200ms — server runs NPU continuously in between

    let stream = null;
    let canvas = null;
    let ctx = null;
    let running = false;
    let frameCount = 0;
    let fpsTimer = null;
    let pollTimer = null;

    // DOM elements
    let startBtn, stopBtn, webcamVideo, webcamOverlay;
    let labelEl, confEl, fpsEl, providerEl, latencyEl, npuUtilEl;
    let webcamPlaceholderIcon;

    function init() {
        startBtn = document.getElementById('webcam-start');
        stopBtn = document.getElementById('webcam-stop');
        webcamVideo = document.getElementById('webcam-video');
        webcamOverlay = document.getElementById('webcam-overlay');
        labelEl = document.getElementById('webcam-label');
        confEl = document.getElementById('webcam-confidence');
        fpsEl = document.getElementById('webcam-fps');
        providerEl = document.getElementById('webcam-provider');
        latencyEl = document.getElementById('webcam-latency');
        npuUtilEl = document.getElementById('webcam-npu-util');
        webcamPlaceholderIcon = document.getElementById('webcam-placeholder-icon');

        if (!startBtn) return;

        startBtn.addEventListener('click', startWebcam);
        stopBtn.addEventListener('click', stopWebcam);
    }

    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'environment' },
                audio: false,
            });
            webcamVideo.srcObject = stream;
            await webcamVideo.play();

            canvas = document.createElement('canvas');
            canvas.width = 320;
            canvas.height = 240;
            ctx = canvas.getContext('2d');

            running = true;
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline-flex';
            webcamOverlay.style.display = 'flex';
            webcamVideo.classList.remove('hidden');
            if (webcamPlaceholderIcon) webcamPlaceholderIcon.style.display = 'none';

            frameCount = 0;
            fpsTimer = setInterval(() => {
                if (fpsEl) fpsEl.textContent = frameCount + ' FPS';
                frameCount = 0;
            }, 1000);

            // Send frames periodically — server runs NPU inference continuously in background
            sendFrameLoop();
        } catch (err) {
            console.error('Webcam error:', err);
            if (labelEl) labelEl.textContent = 'Camera access denied';
        }
    }

    async function stopWebcam() {
        running = false;
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        webcamVideo.srcObject = null;
        startBtn.style.display = 'inline-flex';
        stopBtn.style.display = 'none';
        webcamOverlay.style.display = 'none';
        webcamVideo.classList.add('hidden');
        if (webcamPlaceholderIcon) webcamPlaceholderIcon.style.display = 'block';
        if (fpsTimer) clearInterval(fpsTimer);
        if (labelEl) labelEl.textContent = '—';
        if (confEl) confEl.textContent = '0%';
        if (latencyEl) latencyEl.textContent = '— ms';

        // Tell server to stop the NPU inference loop
        try {
            await fetch(`${API_BASE}/api/v1/webcam-stop`, { method: 'POST' });
        } catch (e) { /* ignore */ }
    }

    async function sendFrameLoop() {
        while (running) {
            try {
                ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
                const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
                const base64 = dataUrl.split(',')[1];

                const resp = await fetch(`${API_BASE}/api/v1/infer-frame`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frame: base64 }),
                });
                const data = await resp.json();

                frameCount++;

                if (labelEl) labelEl.textContent = data.label || '—';
                if (confEl) confEl.textContent = (data.confidence * 100).toFixed(1) + '%';
                if (latencyEl) latencyEl.textContent = data.processing_time_ms.toFixed(1) + ' ms';
                if (providerEl) {
                    const short = (data.provider || '').replace('ExecutionProvider', '');
                    providerEl.textContent = short;
                }
                if (npuUtilEl && data.npu_utilization !== undefined) {
                    npuUtilEl.textContent = data.npu_utilization.toFixed(1) + '%';
                }
            } catch (err) {
                console.error('Frame error:', err);
            }

            // Wait before sending next frame — NPU runs continuously in background anyway
            await new Promise(r => setTimeout(r, FRAME_SEND_INTERVAL));
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
