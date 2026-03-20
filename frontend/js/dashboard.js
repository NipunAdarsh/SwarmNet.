/**
 * SwarmNet — Swarm Dashboard
 * Real-time monitoring of the decentralized edge NPU swarm.
 * Polls node status, renders topology, energy/cloud charts, model registry.
 */
(function () {
    'use strict';

    const API = window.location.origin;
    const POLL_INTERVAL = 2000; // ms

    let pollTimer = null;
    let topologyCtx = null;
    let topologyCanvas = null;
    let nodesData = [];

    // ── Initialization ──
    function init() {
        // Bind kill/revive buttons
        const killBtn = document.getElementById('btn-kill-random');
        const reviveBtn = document.getElementById('btn-revive-all');
        if (killBtn) killBtn.addEventListener('click', killRandomNode);
        if (reviveBtn) reviveBtn.addEventListener('click', reviveAllNodes);

        // Setup topology canvas
        topologyCanvas = document.getElementById('topology-canvas');
        if (topologyCanvas) {
            topologyCtx = topologyCanvas.getContext('2d');
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);
        }

        // Fetch static comparison data once
        fetchEnergy();
        fetchCloudComparison();
        fetchModels();

        // Start polling
        pollSwarm();
        pollTimer = setInterval(pollSwarm, POLL_INTERVAL);
    }

    // ── Canvas resize ──
    function resizeCanvas() {
        if (!topologyCanvas) return;
        const rect = topologyCanvas.parentElement.getBoundingClientRect();
        topologyCanvas.width = rect.width;
        topologyCanvas.height = 320;
    }

    // ── Poll swarm data ──
    async function pollSwarm() {
        try {
            const [nodesRes, metricsRes] = await Promise.all([
                fetch(`${API}/api/v1/swarm/nodes`),
                fetch(`${API}/api/v1/swarm/metrics`),
            ]);
            const nodesJson = await nodesRes.json();
            const metricsJson = await metricsRes.json();

            nodesData = nodesJson.nodes || [];
            renderSummary(metricsJson, nodesJson);
            renderNodeGrid(nodesData);
            renderTopology(nodesData);
        } catch (e) {
            console.warn('Swarm poll error:', e);
        }
    }

    // ── Summary metrics ──
    function renderSummary(metrics, nodesJson) {
        setText('summary-active', metrics.active_nodes || 0);
        setText('summary-throughput', (metrics.total_throughput || 0).toLocaleString());
        setText('summary-latency', `${metrics.avg_latency_ms || 0} ms`);
        setText('summary-npu-nodes', metrics.npu_nodes || 0);
        const strategy = (metrics.routing_strategy || 'lowest_queue_depth').replace(/_/g, ' ');
        setText('summary-strategy', strategy);
    }

    function renderNodeGrid(nodes) {
        const grid = document.getElementById('node-grid');
        if (!grid) return;

        grid.innerHTML = nodes.map(node => {
            const isOnline = node.status === 'online';
            const isHighLoad = node.status === 'high_load';

            // Dynamic visual states
            const dotColor = isOnline ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]'
                : isHighLoad ? 'bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.6)]'
                    : 'bg-primary shadow-[0_0_8px_rgba(228,26,12,0.6)]';

            const cardOpacity = (isOnline || isHighLoad) ? '' : 'opacity-80 grayscale-[0.5]';
            const statusTextColor = (isOnline || isHighLoad) ? 'text-off-white' : 'text-primary';

            const npuBadge = node.npu_available
                ? '<span class="px-2 py-0.5 text-[10px] font-bold border border-primary text-primary rounded-sm tracking-tighter">NPU</span>'
                : '<span class="px-2 py-0.5 text-[10px] font-bold border border-slate-custom text-slate-custom rounded-sm tracking-tighter">CPU</span>';

            const accelRaw = (node.available_accelerators || []).join(', ') || 'CPU only';
            const accel = accelRaw.replace(/ExecutionProvider/g, '').trim() || 'CPU only';

            const buttonHtml = (node.status === 'disabled' || node.status === 'offline')
                ? `<button class="w-full bg-green-600 hover:bg-green-700 text-off-white py-3 text-xs font-bold tracking-[0.2em] uppercase transition-colors" onclick="window._reviveNode('${node.device_id}')">Revive Node</button>`
                : `<button class="w-full bg-primary hover:bg-primary/90 text-off-white py-3 text-xs font-bold tracking-[0.2em] uppercase transition-colors" onclick="window._killNode('${node.device_id}')">Kill Node</button>`;

            return `
                <div class="glass-card rounded-lg flex flex-col overflow-hidden transition-all hover:border-primary/50 relative ${cardOpacity}">
                    <div class="p-4 flex items-center justify-between border-b border-slate-custom/30">
                        <div class="flex items-center gap-2">
                            <span class="size-2 rounded-full ${dotColor}"></span>
                            <h3 class="text-off-white font-bold tracking-tight">${node.device_id}</h3>
                        </div>
                        ${npuBadge}
                    </div>
                    <div class="p-4 grid grid-cols-2 gap-y-4 gap-x-2 flex-1 auto-rows-min">
                        <div class="flex flex-col">
                            <span class="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Status</span>
                            <span class="text-sm ${statusTextColor} font-medium">${node.status.toUpperCase()}</span>
                        </div>
                        <div class="flex flex-col">
                            <span class="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Queue</span>
                            <span class="text-sm text-off-white font-medium">${node.queue_size} TASKS</span>
                        </div>
                        <div class="flex flex-col">
                            <span class="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Latency</span>
                            <span class="text-sm text-off-white font-medium">${node.avg_latency_ms.toFixed(1)} MS</span>
                        </div>
                        <div class="flex flex-col">
                            <span class="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">IP</span>
                            <span class="text-sm text-off-white font-medium">${node.ip_address}</span>
                        </div>
                        <div class="flex flex-col">
                            <span class="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Cores/RAM</span>
                            <span class="text-sm text-off-white font-medium">${node.cpu_cores}C / ${(node.memory_mb / 1024).toFixed(0)}GB</span>
                        </div>
                        <div class="flex flex-col">
                            <span class="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Accelerator</span>
                            <span class="text-sm text-off-white font-medium" title="${accel}">${accel}</span>
                        </div>
                    </div>
                    ${buttonHtml}
                </div>
            `;
        }).join('');
    }

    // ── Topology visualization ──
    function renderTopology(nodes) {
        if (!topologyCtx || !topologyCanvas) return;
        const ctx = topologyCtx;
        const W = topologyCanvas.width;
        const H = topologyCanvas.height;
        ctx.clearRect(0, 0, W, H);

        if (!nodes.length) return;

        // Registry at center
        const cx = W / 2;
        const cy = H / 2;
        const radius = Math.min(W, H) * 0.35;

        // Position nodes in a circle
        const positions = nodes.map((n, i) => {
            const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
            return {
                x: cx + radius * Math.cos(angle),
                y: cy + radius * Math.sin(angle),
                node: n,
            };
        });

        // Draw connections (registry → nodes)
        positions.forEach(p => {
            const isOnline = p.node.status === 'online' || p.node.status === 'high_load';
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(p.x, p.y);
            ctx.strokeStyle = isOnline
                ? 'rgba(0, 220, 130, 0.4)'
                : 'rgba(229, 27, 12, 0.25)';
            ctx.lineWidth = isOnline ? 2 : 1;
            ctx.setLineDash(isOnline ? [] : [5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Animated packet dot on online connections
            if (isOnline) {
                const t = (Date.now() % 2000) / 2000;
                const px = cx + (p.x - cx) * t;
                const py = cy + (p.y - cy) * t;
                ctx.beginPath();
                ctx.arc(px, py, 3, 0, 2 * Math.PI);
                ctx.fillStyle = '#00dc82';
                ctx.fill();
            }
        });

        // Draw registry (center)
        ctx.beginPath();
        ctx.arc(cx, cy, 24, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(229, 27, 12, 0.15)';
        ctx.fill();
        ctx.strokeStyle = '#E51B0C';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#F9F7F7';
        ctx.font = '12px "Space Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('REG', cx, cy);

        // Draw nodes
        positions.forEach(p => {
            const isOnline = p.node.status === 'online' || p.node.status === 'high_load';
            const isHighLoad = p.node.status === 'high_load';

            ctx.beginPath();
            ctx.arc(p.x, p.y, 18, 0, 2 * Math.PI);
            ctx.fillStyle = isOnline
                ? (isHighLoad ? 'rgba(255, 190, 0, 0.15)' : 'rgba(0, 220, 130, 0.15)')
                : 'rgba(229, 27, 12, 0.1)';
            ctx.fill();
            ctx.strokeStyle = isOnline
                ? (isHighLoad ? '#ffbe00' : '#00dc82')
                : '#E51B0C';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Node label
            ctx.fillStyle = '#F9F7F7';
            ctx.font = '9px "Space Mono", monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const shortName = p.node.device_id.replace('edge-', '');
            ctx.fillText(shortName, p.x, p.y);

            // NPU badge
            if (p.node.npu_available) {
                ctx.fillStyle = '#00dc82';
                ctx.font = '7px "Space Mono", monospace';
                ctx.fillText('NPU', p.x, p.y + 26);
            }
        });
    }

    // ── Energy benchmark ──
    async function fetchEnergy() {
        try {
            const res = await fetch(`${API}/api/v1/energy`);
            const data = await res.json();
            const maxWatts = Math.max(data.cpu_watts, data.gpu_watts, data.npu_watts);
            setBar('energy-cpu', (data.cpu_watts / maxWatts) * 100);
            setBar('energy-gpu', (data.gpu_watts / maxWatts) * 100);
            setBar('energy-npu', (data.npu_watts / maxWatts) * 100);
            setText('energy-cpu-val', `${data.cpu_watts}W / ${data.cpu_inference_ms}ms`);
            setText('energy-gpu-val', `${data.gpu_watts}W / ${data.gpu_inference_ms}ms`);
            setText('energy-npu-val', `${data.npu_watts}W / ${data.npu_inference_ms}ms`);
        } catch (e) { console.warn('Energy fetch error:', e); }
    }

    // ── Cloud comparison ──
    async function fetchCloudComparison() {
        try {
            const res = await fetch(`${API}/api/v1/cloud-comparison`);
            const data = await res.json();
            const maxLat = Math.max(data.cloud_latency_ms, data.edge_latency_ms);
            setBar('cloud-lat-bar', (data.cloud_latency_ms / maxLat) * 100);
            setBar('edge-lat-bar', (data.edge_latency_ms / maxLat) * 100);
            setText('cloud-lat-val', `${data.cloud_latency_ms} ms`);
            setText('edge-lat-val', `${data.edge_latency_ms} ms`);

            const maxCost = Math.max(data.cloud_cost_per_million, data.edge_cost_per_million);
            setBar('cloud-cost-bar', (data.cloud_cost_per_million / maxCost) * 100);
            setBar('edge-cost-bar', (data.edge_cost_per_million / maxCost) * 100);
            setText('cloud-cost-val', `$${data.cloud_cost_per_million}`);
            setText('edge-cost-val', `$${data.edge_cost_per_million}`);
        } catch (e) { console.warn('Cloud comparison fetch error:', e); }
    }

    // ── Model registry ──
    async function fetchModels() {
        try {
            const res = await fetch(`${API}/api/v1/models`);
            const data = await res.json();
            const tbody = document.getElementById('model-table-body');
            if (!tbody) return;
            tbody.innerHTML = (data.models || []).map(m => {
                const statusClass = m.status === 'deployed' ? 'deployed'
                    : m.status === 'retired' ? 'retired' : 'staged';
                return `
                    <tr>
                        <td><code>${m.version}</code></td>
                        <td>${m.model_name}</td>
                        <td><strong>${m.accuracy}%</strong></td>
                        <td><span class="model-status-badge ${statusClass}">${m.status}</span></td>
                    </tr>
                `;
            }).join('');
        } catch (e) { console.warn('Models fetch error:', e); }
    }

    // ── Node actions ──
    async function killRandomNode() {
        const online = nodesData.filter(n => n.status === 'online' || n.status === 'high_load');
        if (!online.length) return;
        const target = online[Math.floor(Math.random() * online.length)];
        await killNode(target.device_id);
    }

    async function killNode(deviceId) {
        try {
            await fetch(`${API}/api/v1/swarm/node/${deviceId}/disable`, { method: 'POST' });
            pollSwarm(); // Immediate refresh
        } catch (e) { console.warn('Kill node error:', e); }
    }

    async function reviveAllNodes() {
        const disabled = nodesData.filter(n => n.status === 'disabled' || n.status === 'offline');
        for (const node of disabled) {
            try {
                await fetch(`${API}/api/v1/swarm/node/${node.device_id}/enable`, { method: 'POST' });
            } catch (e) { /* skip */ }
        }
        pollSwarm();
    }

    async function reviveNode(deviceId) {
        try {
            await fetch(`${API}/api/v1/swarm/node/${deviceId}/enable`, { method: 'POST' });
            pollSwarm();
        } catch (e) { console.warn('Revive node error:', e); }
    }

    // Expose to inline onclick handlers
    window._killNode = killNode;
    window._reviveNode = reviveNode;

    // ── Helpers ──
    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function setBar(id, pct) {
        const el = document.getElementById(id);
        if (el) el.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    }


    // ── Start ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
