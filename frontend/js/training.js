/**
 * SwarmNet — Live NPU Training Demo
 * Connects via WebSocket to stream ES training metrics in real-time.
 */
(function () {
    'use strict';

    let ws = null;
    let chartCanvas, chartCtx;
    let accuracyData = [];
    let rewardData = [];

    // DOM elements
    let startTrainBtn, stopTrainBtn, statusEl;
    let genEl, accuracyEl, rewardEl, timeEl, providerEl, paramsEl;
    let progressBar, progressText;
    let trainingPanel;

    function init() {
        startTrainBtn = document.getElementById('train-start');
        stopTrainBtn = document.getElementById('train-stop');
        statusEl = document.getElementById('train-status');
        genEl = document.getElementById('train-gen');
        accuracyEl = document.getElementById('train-accuracy');
        rewardEl = document.getElementById('train-reward');
        timeEl = document.getElementById('train-time');
        providerEl = document.getElementById('train-provider');
        paramsEl = document.getElementById('train-params');
        progressBar = document.getElementById('train-progress-bar');
        progressText = document.getElementById('train-progress-text');
        chartCanvas = document.getElementById('train-chart');
        trainingPanel = document.getElementById('training-results');

        if (!startTrainBtn) return;

        if (chartCanvas) {
            chartCtx = chartCanvas.getContext('2d');
            chartCanvas.width = chartCanvas.offsetWidth * 2;
            chartCanvas.height = chartCanvas.offsetHeight * 2;
            chartCtx.scale(2, 2);
        }

        startTrainBtn.addEventListener('click', startTraining);
        stopTrainBtn.addEventListener('click', stopTraining);
    }

    function startTraining() {
        accuracyData = [];
        rewardData = [];
        if (chartCtx) clearChart();

        const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProto}//${window.location.host}/ws/train`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            startTrainBtn.style.display = 'none';
            stopTrainBtn.style.display = 'inline-flex';
            if (trainingPanel) trainingPanel.style.display = 'block';

            const config = {
                generations: 30,
                pop_size: 40,
                sigma: 0.02,
                lr: 0.03,
            };
            ws.send(JSON.stringify(config));
            if (statusEl) statusEl.textContent = 'Connecting to NPU…';
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'status') {
                if (statusEl) statusEl.textContent = data.message;
            }

            if (data.type === 'init') {
                if (statusEl) statusEl.textContent = 'Training started on NPU';
                if (paramsEl) paramsEl.textContent = data.total_params.toLocaleString();
            }

            if (data.type === 'generation') {
                const pct = Math.round((data.gen / data.total_gens) * 100);

                if (genEl) genEl.textContent = `${data.gen} / ${data.total_gens}`;
                if (accuracyEl) accuracyEl.textContent = data.test_accuracy.toFixed(1) + '%';
                if (rewardEl) rewardEl.textContent = data.best_reward.toFixed(4);
                if (timeEl) timeEl.textContent = data.elapsed_s.toFixed(1) + 's';
                if (providerEl) {
                    const short = (data.provider || '').replace('ExecutionProvider', '');
                    providerEl.textContent = short;
                }
                if (progressBar) progressBar.style.width = pct + '%';
                if (progressText) progressText.textContent = pct + '%';

                if (statusEl) statusEl.textContent = `Gen ${data.gen} — ${data.gen_time_ms.toFixed(0)}ms/gen`;

                accuracyData.push(data.test_accuracy);
                rewardData.push(data.best_reward);
                drawChart();
            }

            if (data.type === 'complete') {
                if (statusEl) statusEl.textContent = `✅ Training Complete — ${data.final_accuracy.toFixed(1)}% accuracy`;
                if (progressBar) progressBar.style.width = '100%';
                if (progressText) progressText.textContent = '100%';
                startTrainBtn.style.display = 'inline-flex';
                stopTrainBtn.style.display = 'none';
                startTrainBtn.innerHTML = '<span>🔄</span> Train Again';
            }

            if (data.type === 'error') {
                if (statusEl) statusEl.textContent = '❌ Error: ' + data.message;
                startTrainBtn.style.display = 'inline-flex';
                stopTrainBtn.style.display = 'none';
            }
        };

        ws.onclose = () => {
            startTrainBtn.style.display = 'inline-flex';
            stopTrainBtn.style.display = 'none';
        };

        ws.onerror = () => {
            if (statusEl) statusEl.textContent = '❌ Connection failed — is the server running?';
            startTrainBtn.style.display = 'inline-flex';
            stopTrainBtn.style.display = 'none';
        };
    }

    function stopTraining() {
        if (ws) {
            ws.close();
            ws = null;
        }
        startTrainBtn.style.display = 'inline-flex';
        stopTrainBtn.style.display = 'none';
        if (statusEl) statusEl.textContent = 'Training stopped';
    }

    function clearChart() {
        if (!chartCtx || !chartCanvas) return;
        const w = chartCanvas.offsetWidth;
        const h = chartCanvas.offsetHeight;
        chartCtx.clearRect(0, 0, w, h);
    }

    function drawChart() {
        if (!chartCtx || !chartCanvas || accuracyData.length < 2) return;

        const w = chartCanvas.offsetWidth;
        const h = chartCanvas.offsetHeight;
        const padL = 40, padR = 12, padT = 12, padB = 28;
        const plotW = w - padL - padR;
        const plotH = h - padT - padB;

        chartCtx.clearRect(0, 0, w, h);

        // Grid lines
        chartCtx.strokeStyle = 'rgba(86, 91, 98, 0.2)';
        chartCtx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = padT + (plotH / 4) * i;
            chartCtx.beginPath();
            chartCtx.moveTo(padL, y);
            chartCtx.lineTo(w - padR, y);
            chartCtx.stroke();
        }

        // Y-axis labels
        chartCtx.fillStyle = 'rgba(160,160,168,0.7)';
        chartCtx.font = '10px Inter, sans-serif';
        chartCtx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
            const val = 100 - i * 25;
            const y = padT + (plotH / 4) * i;
            chartCtx.fillText(val + '%', padL - 6, y + 4);
        }

        // Draw accuracy line
        chartCtx.strokeStyle = '#3fb950';
        chartCtx.lineWidth = 2;
        chartCtx.beginPath();
        const n = accuracyData.length;
        for (let i = 0; i < n; i++) {
            const x = padL + (i / Math.max(n - 1, 1)) * plotW;
            const y = padT + plotH * (1 - accuracyData[i] / 100);
            if (i === 0) chartCtx.moveTo(x, y);
            else chartCtx.lineTo(x, y);
        }
        chartCtx.stroke();

        // Glow effect
        chartCtx.shadowColor = '#3fb950';
        chartCtx.shadowBlur = 8;
        chartCtx.stroke();
        chartCtx.shadowBlur = 0;

        // Current accuracy dot
        if (n > 0) {
            const lastX = padL + ((n - 1) / Math.max(n - 1, 1)) * plotW;
            const lastY = padT + plotH * (1 - accuracyData[n - 1] / 100);
            chartCtx.fillStyle = '#3fb950';
            chartCtx.beginPath();
            chartCtx.arc(lastX, lastY, 4, 0, Math.PI * 2);
            chartCtx.fill();
        }

        // X-axis label
        chartCtx.fillStyle = 'rgba(160,160,168,0.5)';
        chartCtx.textAlign = 'center';
        chartCtx.font = '9px Inter, sans-serif';
        chartCtx.fillText('Generation', w / 2, h - 2);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
