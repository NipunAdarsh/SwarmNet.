/**
 * SwarmNet — NPU vs CPU Benchmark Race
 * Sends the same image through both providers simultaneously, shows a race animation.
 */
(function () {
    'use strict';

    const API_BASE = window.location.origin;

    let benchFile = null;
    let benchBase64 = null;
    let history = [];

    let uploadZone, fileInput, previewImg, previewArea;
    let runBtn, clearBtn;
    let npuBar, cpuBar, npuTime, cpuTime, npuLabel, cpuLabel;
    let speedupEl, npuProvider, cpuProvider;
    let resultsPanel, promptPanel, historyContainer;

    function init() {
        uploadZone = document.getElementById('bench-upload-zone');
        fileInput = document.getElementById('bench-file-input');
        previewImg = document.getElementById('bench-preview-img');
        previewArea = document.getElementById('bench-preview-area');
        runBtn = document.getElementById('bench-run');
        clearBtn = document.getElementById('bench-clear');
        npuBar = document.getElementById('bench-npu-bar');
        cpuBar = document.getElementById('bench-cpu-bar');
        npuTime = document.getElementById('bench-npu-time');
        cpuTime = document.getElementById('bench-cpu-time');
        npuLabel = document.getElementById('bench-npu-label');
        cpuLabel = document.getElementById('bench-cpu-label');
        speedupEl = document.getElementById('bench-speedup');
        npuProvider = document.getElementById('bench-npu-provider');
        cpuProvider = document.getElementById('bench-cpu-provider');
        resultsPanel = document.getElementById('bench-results');
        promptPanel = document.getElementById('bench-prompt');
        historyContainer = document.getElementById('bench-history');

        if (!uploadZone) return;

        uploadZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFile);
        uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('border-primary', 'bg-white/10'); });
        uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('border-primary', 'bg-white/10'));
        uploadZone.addEventListener('drop', e => {
            e.preventDefault();
            uploadZone.classList.remove('border-primary', 'bg-white/10');
            if (e.dataTransfer.files.length) handleFileObj(e.dataTransfer.files[0]);
        });

        runBtn.addEventListener('click', runBenchmark);
        clearBtn.addEventListener('click', clearBench);
    }

    function handleFile(e) {
        if (e.target.files.length) handleFileObj(e.target.files[0]);
    }

    function handleFileObj(file) {
        const valid = ['image/png', 'image/jpeg', 'image/bmp', 'image/webp'];
        if (!valid.includes(file.type)) return;

        benchFile = file;
        previewImg.src = URL.createObjectURL(file);
        uploadZone.style.display = 'none';
        previewArea.style.display = 'flex';
        runBtn.disabled = false;

        // Pre-encode
        const reader = new FileReader();
        reader.onload = () => {
            benchBase64 = reader.result.split(',')[1];
        };
        reader.readAsDataURL(file);
    }

    function clearBench() {
        benchFile = null;
        benchBase64 = null;
        uploadZone.style.display = 'flex';
        previewArea.style.display = 'none';
        runBtn.disabled = true;
        if (resultsPanel) {
            resultsPanel.classList.add('hidden');
            resultsPanel.style.display = '';
        }
        if (promptPanel) promptPanel.style.display = 'flex';
        resetBars();
    }

    function resetBars() {
        if (npuBar) npuBar.style.width = '0%';
        if (cpuBar) cpuBar.style.width = '0%';
        if (npuTime) npuTime.textContent = '—';
        if (cpuTime) cpuTime.textContent = '—';
        if (speedupEl) speedupEl.textContent = '—';
    }

    async function runBenchmark() {
        if (!benchBase64) return;

        runBtn.disabled = true;
        runBtn.innerHTML = '<span class="material-symbols-outlined text-sm font-bold animate-spin">progress_activity</span> Racing…';
        resetBars();

        // Show results panel, hide prompt
        if (resultsPanel) {
            resultsPanel.classList.remove('hidden');
            resultsPanel.style.display = 'flex';
        }
        if (promptPanel) promptPanel.style.display = 'none';

        try {
            const resp = await fetch(`${API_BASE}/api/v1/benchmark`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame: benchBase64 }),
            });
            const data = await resp.json();

            animateRace(data);
            history.unshift(data);
            renderHistory();
        } catch (err) {
            console.error('Benchmark error:', err);
            if (speedupEl) speedupEl.textContent = 'Error';
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = '<span class="material-symbols-outlined text-sm font-bold">speed</span> RUN';
        }
    }

    function animateRace(data) {
        const maxTime = Math.max(data.npu.time_ms, data.cpu.time_ms);

        // Animate NPU bar
        const npuPct = Math.max(5, (data.npu.time_ms / maxTime) * 100);
        const cpuPct = Math.max(5, (data.cpu.time_ms / maxTime) * 100);

        setTimeout(() => {
            if (npuBar) npuBar.style.width = npuPct + '%';
            if (npuTime) npuTime.textContent = data.npu.time_ms.toFixed(1) + ' ms';
            if (npuLabel) npuLabel.textContent = 'Swarm Edge NPU';
            if (npuProvider) {
                const short = data.npu.provider.replace('ExecutionProvider', '');
                npuProvider.textContent = short;
            }
        }, 100);

        setTimeout(() => {
            if (cpuBar) cpuBar.style.width = cpuPct + '%';
            if (cpuTime) cpuTime.textContent = data.cpu.time_ms.toFixed(1) + ' ms';
            if (cpuLabel) cpuLabel.textContent = 'Generic CPU';
            if (cpuProvider) cpuProvider.textContent = 'Standard Compute';
        }, 300);

        setTimeout(() => {
            if (speedupEl) {
                speedupEl.textContent = data.speedup.toFixed(1) + '×';
                speedupEl.className = 'text-5xl font-mono font-bold drop-shadow-[0_0_15px_rgba(0,199,176,0.3)] ' +
                    (data.speedup > 1 ? 'text-primary' : 'text-slate-400');
            }
        }, 500);
    }

    function renderHistory() {
        if (!historyContainer) return;
        const recent = history.slice(0, 5);
        historyContainer.innerHTML = recent.map((d, i) =>
            `<div class="flex gap-4 items-center">
        <span class="text-slate-500">#${i + 1}</span>
        <span class="text-primary">${d.npu.time_ms.toFixed(1)}ms</span>
        <span class="text-slate-600">vs</span>
        <span class="text-slate-300">${d.cpu.time_ms.toFixed(1)}ms</span>
        <span class="${d.speedup > 1 ? 'text-primary' : 'text-slate-400'}">${d.speedup.toFixed(1)}×</span>
      </div>`
        ).join('');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => { init(); initControlled(); });
    } else {
        init();
        initControlled();
    }

    // ── Controlled Benchmark ──

    function initControlled() {
        const runBtn = document.getElementById('ctrl-bench-run');
        if (!runBtn) return;

        runBtn.addEventListener('click', async () => {
            const runs = parseInt(document.getElementById('ctrl-runs').value) || 5;
            const loading = document.getElementById('ctrl-bench-loading');
            const results = document.getElementById('ctrl-bench-results');
            const loadingMsg = document.getElementById('ctrl-loading-msg');

            loading.style.display = 'block';
            results.style.display = 'none';
            runBtn.disabled = true;
            loadingMsg.textContent = `Running ${runs} benchmark runs (warmup + ${runs * 10} timed iterations)…`;

            try {
                const resp = await fetch(`${API_BASE}/api/v1/benchmark-controlled`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ runs }),
                });
                const data = await resp.json();
                loading.style.display = 'none';
                results.style.display = 'block';

                // Populate table
                const tbody = document.getElementById('ctrl-results-body');
                const metrics = ['mean_ms', 'median_ms', 'p95_ms', 'min_ms', 'max_ms', 'samples'];
                const labels = ['Mean (ms)', 'Median (ms)', 'P95 (ms)', 'Min (ms)', 'Max (ms)', 'Samples'];
                tbody.innerHTML = metrics.map((m, i) =>
                    `<tr><td>${labels[i]}</td><td>${data.npu[m]}</td><td>${data.cpu[m]}</td></tr>`
                ).join('');

                // Speedup
                const speedEl = document.getElementById('ctrl-speedup');
                speedEl.textContent = data.speedup_median.toFixed(1) + '×';
                speedEl.className = 'text-3xl font-mono font-bold ' + (data.speedup_median > 1 ? 'text-primary' : 'text-slate-400');

                // Reproduce command
                if (data.reproduce) {
                    const repro = document.getElementById('ctrl-reproduce');
                    const cmd = document.getElementById('ctrl-reproduce-cmd');
                    repro.style.display = 'block';
                    cmd.textContent = data.reproduce;
                }
            } catch (err) {
                loading.style.display = 'none';
                console.error('Controlled benchmark error:', err);
                alert('Benchmark failed: ' + err.message);
            } finally {
                runBtn.disabled = false;
            }
        });
    }
})();
