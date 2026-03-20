/* ============================================
   SwarmNet — Live Inference Demo Module
   Image upload, API calls, result rendering
   ============================================ */

(function () {
    'use strict';

    // ── Config ──
    const BACKEND_TIMEOUT = 15000; // ms — generous timeout for NPU inference
    const API_BASE = window.location.origin;

    // ── DOM Elements (visible in demos.html layout) ──
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');

    // Prediction result elements (visible in demos.html)
    const pred0Name = document.getElementById('pred-item-0-name');
    const pred0Score = document.getElementById('pred-item-0-score');
    const pred0Bar = document.getElementById('pred-item-0-bar');
    const pred1Name = document.getElementById('pred-item-1-name');
    const pred1Score = document.getElementById('pred-item-1-score');
    const pred1Bar = document.getElementById('pred-item-1-bar');
    const pred2Name = document.getElementById('pred-item-2-name');
    const pred2Score = document.getElementById('pred-item-2-score');
    const pred2Bar = document.getElementById('pred-item-2-bar');

    // ── State ──
    let selectedFile = null;
    let base64Payload = null;

    // ── Helpers ──

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const b64 = reader.result.split(',')[1];
                resolve(b64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    function fetchWithTimeout(url, options, timeout) {
        return Promise.race([
            fetch(url, options),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Request timed out')), timeout)
            ),
        ]);
    }

    function setProcessing() {
        if (pred0Name) pred0Name.textContent = 'Processing…';
        if (pred0Score) pred0Score.textContent = '…';
        if (pred0Bar) pred0Bar.style.width = '0%';
        if (pred1Name) pred1Name.textContent = '—';
        if (pred1Score) pred1Score.textContent = '…';
        if (pred1Bar) pred1Bar.style.width = '0%';
        if (pred2Name) pred2Name.textContent = '—';
        if (pred2Score) pred2Score.textContent = '…';
        if (pred2Bar) pred2Bar.style.width = '0%';
    }

    function setResults(data) {
        const result = data.result || {};
        const top5 = result.top5 || [];

        // Prediction 1 (top)
        if (top5.length > 0) {
            const p = top5[0];
            const pct = (p.confidence * 100).toFixed(1);
            if (pred0Name) pred0Name.textContent = p.label;
            if (pred0Score) pred0Score.textContent = pct + '%';
            if (pred0Bar) pred0Bar.style.width = pct + '%';
        } else if (result.label) {
            const pct = ((result.confidence || 0) * 100).toFixed(1);
            if (pred0Name) pred0Name.textContent = result.label;
            if (pred0Score) pred0Score.textContent = pct + '%';
            if (pred0Bar) pred0Bar.style.width = pct + '%';
        }

        // Prediction 2
        if (top5.length > 1) {
            const p = top5[1];
            const pct = (p.confidence * 100).toFixed(1);
            if (pred1Name) pred1Name.textContent = p.label;
            if (pred1Score) pred1Score.textContent = pct + '%';
            if (pred1Bar) pred1Bar.style.width = pct + '%';
        } else {
            if (pred1Name) pred1Name.textContent = '—';
            if (pred1Score) pred1Score.textContent = '0%';
            if (pred1Bar) pred1Bar.style.width = '0%';
        }

        // Prediction 3
        if (top5.length > 2) {
            const p = top5[2];
            const pct = (p.confidence * 100).toFixed(1);
            if (pred2Name) pred2Name.textContent = p.label;
            if (pred2Score) pred2Score.textContent = pct + '%';
            if (pred2Bar) pred2Bar.style.width = pct + '%';
        } else {
            if (pred2Name) pred2Name.textContent = '—';
            if (pred2Score) pred2Score.textContent = '0%';
            if (pred2Bar) pred2Bar.style.width = '0%';
        }
    }

    function setError(msg) {
        if (pred0Name) pred0Name.textContent = msg || 'Error';
        if (pred0Score) pred0Score.textContent = '—';
        if (pred0Bar) pred0Bar.style.width = '0%';
        if (pred1Name) pred1Name.textContent = '—';
        if (pred1Score) pred1Score.textContent = '—';
        if (pred2Name) pred2Name.textContent = '—';
        if (pred2Score) pred2Score.textContent = '—';
    }

    // ── Run inference automatically ──
    async function runInference(b64) {
        const taskId = `req_${Date.now()}`;
        setProcessing();

        try {
            const resp = await fetchWithTimeout(
                `${API_BASE}/api/v1/infer`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        task_id: taskId,
                        data_type: 'base64_image',
                        payload: b64,
                    }),
                },
                BACKEND_TIMEOUT
            );

            if (!resp.ok) throw new Error(`Server returned ${resp.status}`);

            const data = await resp.json();
            setResults(data);
        } catch (e) {
            console.error('Inference error:', e);
            setError('Inference failed');
        }
    }

    // ── File upload handling ──

    if (!uploadZone) return;

    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('border-primary', 'bg-white/10');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('border-primary', 'bg-white/10');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('border-primary', 'bg-white/10');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    async function handleFile(file) {
        // Validate type
        const validTypes = ['image/png', 'image/jpeg', 'image/bmp', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            alert('Unsupported file type. Please use PNG, JPEG, BMP, or WebP.');
            return;
        }

        selectedFile = file;

        // Show image preview in upload zone
        const previewURL = URL.createObjectURL(file);
        uploadZone.innerHTML = `
            <img src="${previewURL}" class="max-h-40 rounded-lg object-contain mb-2" alt="Preview">
            <p class="text-primary text-sm font-medium">Image uploaded — running inference…</p>
            <p class="text-slate-500 text-xs mt-1">Click to upload a different image</p>
        `;

        // Encode and auto-run
        base64Payload = await fileToBase64(file);
        runInference(base64Payload);
    }

})();
