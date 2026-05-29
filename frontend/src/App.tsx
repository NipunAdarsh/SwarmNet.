import { useState, useEffect, useRef } from 'react';
import { 
  Cpu, Layers, Award, Play, Pause, Camera, Upload, 
  Activity, Server, Globe, RefreshCw, Download, ArrowRight, 
  X, Menu, Info, BarChart2
} from 'lucide-react';
import { 
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip
} from 'recharts';

// --- MOCK CONSTANTS & DATA ---
const INITIAL_NODES = [
  { device_id: "edge-alpha", ip_address: "192.168.1.10", port: 8000, cpu_cores: 8, npu_available: true, gpu_available: false, memory_mb: 16384, queue_size: 2, avg_latency_ms: 8.24, inferences_run: 1245, status: "online" },
  { device_id: "edge-bravo", ip_address: "192.168.1.11", port: 8001, cpu_cores: 6, npu_available: true, gpu_available: false, memory_mb: 8192, queue_size: 0, avg_latency_ms: 9.12, inferences_run: 984, status: "online" },
  { device_id: "edge-charlie", ip_address: "192.168.1.12", port: 8002, cpu_cores: 12, npu_available: true, gpu_available: true, memory_mb: 32768, queue_size: 11, avg_latency_ms: 12.85, inferences_run: 2314, status: "high_load" },
  { device_id: "edge-delta", ip_address: "192.168.1.13", port: 8003, cpu_cores: 8, npu_available: true, gpu_available: false, memory_mb: 8192, queue_size: 4, avg_latency_ms: 7.95, inferences_run: 1102, status: "online" },
  { device_id: "edge-echo", ip_address: "192.168.1.14", port: 8004, cpu_cores: 4, npu_available: false, gpu_available: false, memory_mb: 16384, queue_size: 0, avg_latency_ms: 32.14, inferences_run: 412, status: "online" }
];

const INITIAL_LEADERBOARD = [
  { rank: 1, username: "neuron_architect", xp: 12840, badges: ["First Gradient", "100h Club", "24h Streak", "Super NPU"] },
  { rank: 2, username: "quantum_bee", xp: 10450, badges: ["First Gradient", "100h Club", "24h Streak"] },
  { rank: 3, username: "snapdragon_donor", xp: 9120, badges: ["First Gradient", "100h Club", "NPU Pioneer"] },
  { rank: 4, username: "vitis_ai_hero", xp: 8430, badges: ["First Gradient", "100h Club"] },
  { rank: 5, username: "apple_silicon_guy", xp: 7290, badges: ["First Gradient", "24h Streak"] },
  { rank: 6, username: "ryzen_ai_user", xp: 6810, badges: ["First Gradient"] }
];

const MOCK_LABELS = [
  { label: "Golden Retriever", confidence: 0.942 },
  { label: "Labrador Retriever", confidence: 0.038 },
  { label: "Irish Setter", confidence: 0.012 },
  { label: "Red Bone Coonhound", confidence: 0.005 },
  { label: "Tibetan Mastiff", confidence: 0.003 }
];

export default function App() {
  // Helper to determine initial tab from current URL path
  const getInitialTab = (): 'landing' | 'dashboard' | 'training' | 'benchmark' | 'classify' => {
    const path = window.location.pathname.replace(/^\/|\/$/g, '');
    if (path === 'dashboard') return 'dashboard';
    if (path === 'training') return 'training';
    if (path === 'benchmark') return 'benchmark';
    if (path === 'classify') return 'classify';
    return 'landing';
  };

  const [activeTab, setActiveTab] = useState<'landing' | 'dashboard' | 'training' | 'benchmark' | 'classify'>(getInitialTab);
  const [isScrolled, setIsScrolled] = useState(false);
  const [showDonateModal, setShowDonateModal] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Listen to activeTab and update browser history path
  useEffect(() => {
    const currentPath = window.location.pathname.replace(/^\/|\/$/g, '');
    const targetPath = activeTab === 'landing' ? '' : activeTab;
    if (currentPath !== targetPath) {
      window.history.pushState(null, '', activeTab === 'landing' ? '/' : `/${activeTab}`);
    }
  }, [activeTab]);

  // Handle browser back/forward buttons (popstate event)
  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname.replace(/^\/|\/$/g, '');
      if (path === 'dashboard') setActiveTab('dashboard');
      else if (path === 'training') setActiveTab('training');
      else if (path === 'benchmark') setActiveTab('benchmark');
      else if (path === 'classify') setActiveTab('classify');
      else setActiveTab('landing');
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // --- STATS FLUX STATE ---
  const [activeNodesCount, setActiveNodesCount] = useState(14820);
  const [throughput, setThroughput] = useState(4.25);
  const [totalInferences, setTotalInferences] = useState(14829340);
  const [nodes, setNodes] = useState(INITIAL_NODES);

  // --- BENCHMARK STATE ---
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<any>(null);

  // --- PLAYGROUND STATE ---
  const [isClassifying, setIsClassifying] = useState(false);
  const [showScanner, setShowScanner] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [feedType, setFeedType] = useState<'upload' | 'webcam'>('upload');
  const [selectedImg, setSelectedImg] = useState<string>('https://images.unsplash.com/photo-1552053831-71594a27632d?auto=format&fit=crop&q=80&w=600');

  // --- TRAINING STATE ---
  const [isTraining, setIsTraining] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [learningRate, setLearningRate] = useState(0.03);
  const [sigma, setSigma] = useState(0.02);
  const [popSize, setPopSize] = useState(50);
  const [trainLogs, setTrainLogs] = useState<string[]>([
    "System ready. Awaiting training initialization...",
    "Click 'Start Training' to begin Evolutionary Strategies on MNIST."
  ]);
  const [trainingData, setTrainingData] = useState<{ gen: number; acc: number; loss: number }[]>([]);

  const logContainerRef = useRef<HTMLDivElement>(null);

  // Scroll logic for sticky header
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 40);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Fluctuate stats to simulate real-time stream
  useEffect(() => {
    const interval = setInterval(() => {
      // Fluctuate stats
      setActiveNodesCount(prev => prev + Math.floor(Math.random() * 5) - 2);
      setThroughput(prev => parseFloat((prev + (Math.random() * 0.04 - 0.02)).toFixed(2)));
      setTotalInferences(prev => prev + Math.floor(Math.random() * 8));

      // Fluctuate node queues/latency
      setNodes(prev => prev.map(node => {
        if (node.status === "offline") return node;
        const newQueue = Math.max(0, node.queue_size + (Math.random() > 0.55 ? 1 : -1));
        const baseLatency = node.npu_available ? 8.0 : 30.0;
        const extraLatency = newQueue * 1.5;
        return {
          ...node,
          queue_size: newQueue,
          avg_latency_ms: parseFloat((baseLatency + extraLatency + Math.random() * 0.5).toFixed(2)),
          inferences_run: node.inferences_run + Math.floor(Math.random() * 3),
          status: newQueue > 9 ? "high_load" : "online"
        };
      }));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // Handle Training Simulation
  useEffect(() => {
    let timer: any;
    if (isTraining) {
      timer = setInterval(() => {
        setGeneration(prev => {
          const nextGen = prev + 1;
          const progress = nextGen / 100;
          
          // Mimic learning curve: start fast, decelerate
          const currentAcc = Math.min(0.965, 0.10 + (0.85 * Math.log10(1 + 9 * progress)));
          const currentLoss = Math.max(0.08, 2.30 - (2.1 * Math.log10(1 + 9 * progress)));

          setTrainingData(prevData => [
            ...prevData, 
            { gen: nextGen, acc: parseFloat((currentAcc * 100).toFixed(2)), loss: parseFloat(currentLoss.toFixed(3)) }
          ]);

          const timestamp = new Date().toLocaleTimeString();
          setTrainLogs(prevLogs => [
            ...prevLogs,
            `[${timestamp}] Generation ${nextGen} complete: Test Acc = ${(currentAcc * 100).toFixed(2)}%, Reward = ${(currentAcc).toFixed(4)}, Loss = ${(currentLoss).toFixed(3)}`
          ]);

          if (nextGen >= 100) {
            setIsTraining(false);
            setTrainLogs(prevLogs => [...prevLogs, `[${timestamp}] ✅ Training complete! Model saved to model/es_trained.onnx`]);
          }

          return nextGen;
        });
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isTraining]);

  // Auto-scroll logs
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [trainLogs]);

  const handleStartTraining = () => {
    if (generation >= 100) {
      setGeneration(0);
      setTrainingData([]);
      setTrainLogs(["Restarting Evolutionary Strategies training on MNIST..."]);
    }
    setIsTraining(true);
    const timestamp = new Date().toLocaleTimeString();
    setTrainLogs(prev => [...prev, `[${timestamp}] 🚀 Connected to Swarm — Launching ES Engine. Sig: ${sigma}, LR: ${learningRate}`]);
  };

  const handlePauseTraining = () => {
    setIsTraining(false);
    const timestamp = new Date().toLocaleTimeString();
    setTrainLogs(prev => [...prev, `[${timestamp}] ⏸️ Training paused by user.`]);
  };

  const runBenchmark = () => {
    setIsBenchmarking(true);
    setBenchmarkResult(null);
    setTimeout(() => {
      setIsBenchmarking(false);
      setBenchmarkResult({
        npu: 7.94,
        gpu: 16.48,
        cpu: 64.21,
        speedup: "8.1x",
        npuPower: 0.8,
        cpuPower: 3.2,
        powerSaving: "75%"
      });
    }, 2000);
  };

  const handleClassify = () => {
    setIsClassifying(true);
    setShowScanner(true);
    setResults([]);
    setTimeout(() => {
      setShowScanner(false);
      setIsClassifying(false);
      setResults(MOCK_LABELS);
    }, 2200);
  };

  return (
    <>
      {/* HEADER */}
      <header className={`header ${isScrolled ? 'scrolled' : ''}`}>
        <div className="container header-container">
          <a href="#" className="logo" onClick={() => { setActiveTab('landing'); window.scrollTo(0,0); }}>
            <div className="logo-dot"></div>
            SWARMNET
          </a>

          <nav className="nav">
            <a 
              href="#how-it-works" 
              className={`nav-link ${activeTab === 'landing' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); setActiveTab('landing'); document.getElementById('how-it-works')?.scrollIntoView(); }}
            >
              HOW IT WORKS
            </a>
            <a 
              href="#" 
              className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => setActiveTab('dashboard')}
            >
              SWARM & LEADERBOARD
            </a>
            <a 
              href="#" 
              className={`nav-link ${activeTab === 'training' ? 'active' : ''}`}
              onClick={() => setActiveTab('training')}
            >
              LIVE ES TRAINING
            </a>
            <a 
              href="#" 
              className={`nav-link ${activeTab === 'benchmark' ? 'active' : ''}`}
              onClick={() => setActiveTab('benchmark')}
            >
              NPU BENCHMARK
            </a>
            <a 
              href="#" 
              className={`nav-link ${activeTab === 'classify' ? 'active' : ''}`}
              onClick={() => setActiveTab('classify')}
            >
              PLAYGROUND
            </a>
          </nav>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button className="cta-button" onClick={() => setShowDonateModal(true)}>
              DONATE COMPUTE <ArrowRight size={16} />
            </button>
            <button className="mobile-menu-btn" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
              <Menu size={24} />
            </button>
          </div>
        </div>
      </header>

      {/* MOBILE MENU */}
      {mobileMenuOpen && (
        <div style={{
          position: 'fixed', top: '4.5rem', left: 0, right: 0, 
          background: '#ffffff', borderBottom: '1px solid var(--border-light)',
          padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.25rem',
          zIndex: 99, boxShadow: 'var(--shadow-md)'
        }}>
          <a href="#" className="nav-link" onClick={() => { setActiveTab('landing'); setMobileMenuOpen(false); }}>How It Works</a>
          <a href="#" className="nav-link" onClick={() => { setActiveTab('dashboard'); setMobileMenuOpen(false); }}>Swarm & Leaderboard</a>
          <a href="#" className="nav-link" onClick={() => { setActiveTab('training'); setMobileMenuOpen(false); }}>Live ES Training</a>
          <a href="#" className="nav-link" onClick={() => { setActiveTab('benchmark'); setMobileMenuOpen(false); }}>NPU Benchmark</a>
          <a href="#" className="nav-link" onClick={() => { setActiveTab('classify'); setMobileMenuOpen(false); }}>Playground</a>
        </div>
      )}

      {/* ---------------------------------------------------- */}
      {/* 1. LANDING PAGE VIEW */}
      {/* ---------------------------------------------------- */}
      {activeTab === 'landing' && (
        <>
          {/* HERO */}
          <section className="hero">
            <div className="container hero-grid">
              <div className="hero-content">
                <div className="hero-tag">
                  <div className="hero-dot-active"></div>
                  LIVE NETWORK ACTIVE
                </div>
                <h1 className="hero-title">
                  YOUR DEVICE.<br />THE FUTURE OF <span className="accent">SCIENCE.</span>
                </h1>
                <p className="hero-desc">
                  Donate your idle NPU compute cycles to power evolutionary AI algorithms. Help scientists solve complex genomic, medical, and protein-folding puzzles right from your laptop.
                </p>
                <div className="hero-actions">
                  <button className="cta-button" onClick={() => setShowDonateModal(true)}>
                    START DONATING
                  </button>
                  <button className="cta-button secondary" onClick={() => setActiveTab('benchmark')}>
                    RUN NPU BENCHMARK
                  </button>
                </div>
              </div>

              <div className="hero-visual">
                <div className="globe-wrapper">
                  <div className="network-sphere">
                    <div className="network-node" style={{ top: '10%', left: '30%' }}></div>
                    <div className="network-node purple" style={{ top: '25%', left: '75%' }}></div>
                    <div className="network-node green" style={{ top: '65%', left: '15%' }}></div>
                    <div className="network-node" style={{ top: '80%', left: '50%' }}></div>
                    <div className="network-node purple" style={{ top: '45%', left: '85%' }}></div>
                    <div className="network-node green" style={{ top: '55%', left: '30%' }}></div>
                    <div className="network-ring"></div>
                    <div className="network-ring-alt"></div>
                  </div>
                  <div className="network-center">
                    <Cpu size={32} />
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* QUICK STATS BAR */}
          <div className="quick-stats">
            <div className="container stats-bar-grid">
              <div className="stat-bar-item">
                <span className="stat-bar-val">{activeNodesCount.toLocaleString()}</span>
                <span className="stat-bar-label">Active Nodes</span>
              </div>
              <div className="stat-bar-item">
                <span className="stat-bar-val">{throughput} PFLOPS</span>
                <span className="stat-bar-label">Swarm Throughput</span>
              </div>
              <div className="stat-bar-item">
                <span className="stat-bar-val">{(totalInferences / 1000000).toFixed(2)}M</span>
                <span className="stat-bar-label">Total Inferences</span>
              </div>
              <div className="stat-bar-item">
                <span className="stat-bar-val">99.8%</span>
                <span className="stat-bar-label">Uptime</span>
              </div>
            </div>
          </div>

          {/* HOW THE SWARM WORKS (BENTO GRID) */}
          <section className="bento-section" id="how-it-works">
            <div className="container">
              <div className="section-header">
                <h2 className="section-title">THE ARCHITECTURE OF A DECENRALIZED SUPERCOMPUTER</h2>
                <p>Designed to leverage the hyper-efficient Neural Processing Units (NPUs) inside modern personal devices.</p>
              </div>

              <div className="bento-grid">
                {/* Bento Card 1 */}
                <div className="bento-card span-2 dark-theme">
                  <div className="bento-icon">
                    <Cpu size={24} />
                  </div>
                  <h3>NPU-First Execution</h3>
                  <p className="mb-4">
                    Most distributed computing applications rely heavily on CPUs and power-hungry GPUs. SwarmNet goes straight for the "AI Brain" - the dedicated NPU. By targeting Qualcomm Hexagon, AMD XDNA, and Apple Neural Engine via custom ONNX Runtime execution providers, we execute deep neural networks at a fraction of the battery cost.
                  </p>
                  <div style={{ display: 'flex', gap: '1rem', marginTop: 'auto', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, padding: '0.25rem 0.75rem', borderRadius: '4px', background: 'rgba(255, 255, 255, 0.05)', color: '#ffffff' }}>Qualcomm QNN</span>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, padding: '0.25rem 0.75rem', borderRadius: '4px', background: 'rgba(255, 255, 255, 0.05)', color: '#ffffff' }}>AMD VitisAI</span>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, padding: '0.25rem 0.75rem', borderRadius: '4px', background: 'rgba(255, 255, 255, 0.05)', color: '#ffffff' }}>Apple CoreML</span>
                  </div>
                </div>

                {/* Bento Card 2 */}
                <div className="bento-card">
                  <div className="bento-icon">
                    <Layers size={24} />
                  </div>
                  <h3>Antithetic Training</h3>
                  <p>
                    Instead of backpropagation (which is extremely memory intensive), SwarmNet uses OpenAI-style Evolutionary Strategies (ES). We generate pairs of mathematical perturbations (+ε and -ε), evaluate them locally on your NPU, and merge the results to train models gradient-free.
                  </p>
                </div>

                {/* Bento Card 3 */}
                <div className="bento-card">
                  <div className="bento-icon">
                    <Globe size={24} />
                  </div>
                  <h3>Zero-Config Discovery</h3>
                  <p>
                    Utilizing UDP Multicast network broadcast, nodes detect the local hub and peers automatically. No port-forwarding, no technical command line configuration. Plug-and-play AI supercomputing.
                  </p>
                </div>

                {/* Bento Card 4 */}
                <div className="bento-card span-2">
                  <div className="bento-icon">
                    <Award size={24} />
                  </div>
                  <h3>Gamified Contributions</h3>
                  <p style={{ marginBottom: '1.5rem' }}>
                    Every inference step completed earns you Experience Points (XP) towards climbing the global leaderboard. Earn unique hexagonal badges, level up your donor status, and compete with other computing nodes around the globe.
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginTop: 'auto' }}>
                    <div style={{ background: 'var(--bg-light)', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                      <span style={{ display: 'block', fontSize: '1.25rem', fontWeight: 800, color: 'var(--accent-blue)' }}>Level 12</span>
                      <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)' }}>Average Level</span>
                    </div>
                    <div style={{ background: 'var(--bg-light)', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                      <span style={{ display: 'block', fontSize: '1.25rem', fontWeight: 800, color: 'var(--accent-blue)' }}>300 XP</span>
                      <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)' }}>Daily Average</span>
                    </div>
                    <div style={{ background: 'var(--bg-light)', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                      <span style={{ display: 'block', fontSize: '1.25rem', fontWeight: 800, color: 'var(--accent-blue)' }}>6 Badges</span>
                      <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)' }}>To Earn</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* CALL TO ACTION */}
          <section style={{ backgroundColor: 'var(--bg-dark)', color: 'var(--text-light)', textAlign: 'center', padding: '6rem 0' }}>
            <div className="container" style={{ maxWidth: '720px' }}>
              <h2 style={{ fontSize: '2.5rem', marginBottom: '1rem', color: '#ffffff' }}>JOIN THE COMPUTATIONAL REVOLUTION</h2>
              <p style={{ color: 'var(--text-muted-light)', fontSize: '1.1rem', marginBottom: '2.5rem' }}>
                Join thousands of users donating spare computing cycles to push scientific boundaries. Secure, sandboxed, and optimized for battery life.
              </p>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
                <button className="cta-button" onClick={() => setShowDonateModal(true)}>
                  DOWNLOAD FOR WINDOWS <Download size={16} />
                </button>
                <button className="cta-button secondary" style={{ color: '#ffffff', borderColor: 'rgba(255,255,255,0.2)' }} onClick={() => setActiveTab('dashboard')}>
                  VIEW ONLINE SWARM
                </button>
              </div>
            </div>
          </section>
        </>
      )}

      {/* ---------------------------------------------------- */}
      {/* 2. DASHBOARD VIEW (SWARM NODES & LEADERBOARD) */}
      {/* ---------------------------------------------------- */}
      {activeTab === 'dashboard' && (
        <section className="dashboard-section" style={{ minHeight: '90vh', paddingTop: '7rem' }}>
          <div className="container">
            <div className="dashboard-controls">
              <div>
                <h2 style={{ fontSize: '2.25rem', marginBottom: '0.25rem' }}>SWARM MANAGEMENT CONSOLE</h2>
                <p>Monitor live nodes, review system-wide throughput, and track top XP donors.</p>
              </div>
              <div className="dashboard-tabs">
                <button className="tab-btn active">ACTIVE SWARM</button>
                <button className="tab-btn" onClick={() => setActiveTab('training')}>LIVE TRAINING</button>
              </div>
            </div>

            {/* Live Nodes Map and Grid */}
            <div style={{ marginBottom: '4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <h3 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Server size={20} className="text-accent-blue" />
                  CONNECTED EDGE NODES ({nodes.length})
                </h3>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <RefreshCw size={14} className="animate-spin" /> Auto-refreshing metrics
                </span>
              </div>

              <div className="node-grid">
                {nodes.map(node => (
                  <div key={node.device_id} className="node-card">
                    <div className="node-header">
                      <span className="node-name">{node.device_id}</span>
                      <span className={`node-status-badge ${node.status === 'online' ? 'online' : node.status === 'high_load' ? 'load' : 'offline'}`}>
                        {node.status.replace('_', ' ')}
                      </span>
                    </div>

                    <div className="node-metrics">
                      <div>
                        <span className="metric-box-title">IP ADDRESS</span>
                        <div className="metric-box-val">{node.ip_address}</div>
                      </div>
                      <div>
                        <span className="metric-box-title">CORES</span>
                        <div className="metric-box-val">{node.cpu_cores} Threads</div>
                      </div>
                      <div>
                        <span className="metric-box-title">QUEUE DEPTH</span>
                        <div className="metric-box-val">{node.queue_size} tasks</div>
                      </div>
                      <div>
                        <span className="metric-box-title">LATENCY</span>
                        <div className="metric-box-val">{node.avg_latency_ms} ms</div>
                      </div>
                    </div>

                    <div className="node-footer">
                      <div className="npu-tag">
                        <Cpu size={12} />
                        {node.npu_available ? 'NPU Active' : 'CPU Only'}
                      </div>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--mono-font)' }}>
                        {node.inferences_run} infs
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Leaderboard Table */}
            <div>
              <h3 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
                <Award size={20} className="text-accent-blue" />
                GLOBAL XP LEADERBOARD
              </h3>
              <div className="leaderboard-board">
                <div className="leaderboard-row header-row">
                  <span>RANK</span>
                  <span>DONOR USERNAME</span>
                  <span>TOTAL EXPERIENCE (XP)</span>
                  <span>BADGES</span>
                </div>
                {INITIAL_LEADERBOARD.map((user) => (
                  <div key={user.rank} className="leaderboard-row">
                    <span className={`rank-val ${user.rank <= 3 ? 'top-rank' : ''}`}>
                      #{user.rank.toString().padStart(2, '0')}
                    </span>
                    <div className="user-cell">
                      <div className="user-avatar">
                        {user.username.split('_').map(n => n[0]).join('').toUpperCase().slice(0, 2)}
                      </div>
                      {user.username}
                    </div>
                    <span style={{ fontFamily: 'var(--mono-font)', fontWeight: 700 }}>
                      {user.xp.toLocaleString()} XP
                    </span>
                    <div className="badge-row">
                      {user.badges.slice(0, 2).map((badge, idx) => (
                        <span key={idx} className="badge-pill">{badge}</span>
                      ))}
                      {user.badges.length > 2 && (
                        <span className="badge-pill" style={{ background: 'rgba(15, 23, 42, 0.05)', color: 'var(--text-muted)' }}>
                          +{user.badges.length - 2}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </section>
      )}

      {/* ---------------------------------------------------- */}
      {/* 3. LIVE TRAINING SIMULATION VIEW */}
      {/* ---------------------------------------------------- */}
      {activeTab === 'training' && (
        <section className="training-section" style={{ minHeight: '90vh', paddingTop: '7rem' }}>
          <div className="container">
            <div className="dashboard-controls" style={{ borderBottomColor: 'rgba(255,255,255,0.08)' }}>
              <div>
                <h2 style={{ fontSize: '2.25rem', marginBottom: '0.25rem', color: '#ffffff' }}>LIVE EDGE MODEL TRAINING</h2>
                <p style={{ color: 'var(--text-muted-light)' }}>Watch real-time Evolutionary Strategies (ES) optimization on the NPU swarm.</p>
              </div>
              <div className="dashboard-tabs">
                <button className="tab-btn" onClick={() => setActiveTab('dashboard')} style={{ color: 'var(--text-muted-light)' }}>ACTIVE SWARM</button>
                <button className="tab-btn active" style={{ color: 'var(--text-light)' }}>LIVE TRAINING</button>
              </div>
            </div>

            <div className="training-layout">
              {/* Controls Column */}
              <div className="training-card">
                <h3 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Activity size={20} className="text-accent-blue" />
                  TRAINING CONTROLS
                </h3>

                <div className="params-grid">
                  <div className="param-item">
                    <span className="param-label">Learning Rate</span>
                    <div className="param-input-group">
                      <input 
                        type="number" 
                        className="param-input" 
                        value={learningRate} 
                        onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                        disabled={isTraining}
                      />
                    </div>
                  </div>
                  <div className="param-item">
                    <span className="param-label">Sigma (Noise)</span>
                    <div className="param-input-group">
                      <input 
                        type="number" 
                        className="param-input" 
                        value={sigma} 
                        onChange={(e) => setSigma(parseFloat(e.target.value))}
                        disabled={isTraining}
                      />
                    </div>
                  </div>
                  <div className="param-item">
                    <span className="param-label">Population Size</span>
                    <div className="param-input-group">
                      <input 
                        type="number" 
                        className="param-input" 
                        value={popSize} 
                        onChange={(e) => setPopSize(parseInt(e.target.value))}
                        disabled={isTraining}
                      />
                    </div>
                  </div>
                  <div className="param-item">
                    <span className="param-label">Target Dataset</span>
                    <div className="param-input-group">
                      <input type="text" className="param-input" value="MNIST (Handwritten)" disabled />
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
                  {isTraining ? (
                    <button className="cta-button" style={{ flex: 1, background: 'var(--warning-amber)' }} onClick={handlePauseTraining}>
                      <Pause size={16} /> PAUSE TRAINING
                    </button>
                  ) : (
                    <button className="cta-button" style={{ flex: 1, background: 'var(--success-mint)' }} onClick={handleStartTraining}>
                      <Play size={16} /> START TRAINING
                    </button>
                  )}
                  <button 
                    className="cta-button secondary" 
                    style={{ flex: 0.5, color: '#ffffff', borderColor: 'rgba(255,255,255,0.1)' }}
                    onClick={() => { setGeneration(0); setTrainingData([]); setTrainLogs(["Logs cleared. Ready to start."]); }}
                    disabled={isTraining}
                  >
                    RESET
                  </button>
                </div>

                {/* Progress bar */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted-light)' }}>Training Progress</span>
                    <span style={{ fontWeight: 700 }}>{generation} / 100 Generations</span>
                  </div>
                  <div className="training-progress-bar">
                    <div className="training-progress-fill" style={{ width: `${generation}%` }}></div>
                  </div>
                </div>

                {/* Live parameter matrix values */}
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.25rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.03)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.5rem', marginBottom: '0.5rem' }}>
                    <span style={{ color: 'var(--text-muted-light)' }}>Accuracy</span>
                    <span style={{ fontFamily: 'var(--mono-font)', fontWeight: 700 }}>
                      {trainingData.length > 0 ? `${trainingData[trainingData.length - 1].acc}%` : 'N/A'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.5rem', marginBottom: '0.5rem' }}>
                    <span style={{ color: 'var(--text-muted-light)' }}>Current Loss</span>
                    <span style={{ fontFamily: 'var(--mono-font)', fontWeight: 700 }}>
                      {trainingData.length > 0 ? trainingData[trainingData.length - 1].loss : 'N/A'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                    <span style={{ color: 'var(--text-muted-light)' }}>Optimizer</span>
                    <span style={{ fontWeight: 700 }}>ES (Antithetic standard)</span>
                  </div>
                </div>
              </div>

              {/* Chart and Terminal logs column */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                {/* Accuracy Line Chart */}
                <div style={{ background: 'var(--bg-dark-card)', border: '1px solid var(--border-dark)', borderRadius: 'var(--border-radius-lg)', padding: '2rem' }}>
                  <h3 style={{ fontSize: '1.1rem', marginBottom: '1.5rem', color: '#ffffff' }}>TEST ACCURACY OVER TIME</h3>
                  <div style={{ width: '100%', height: 260 }}>
                    <ResponsiveContainer>
                      <LineChart data={trainingData}>
                        <XAxis dataKey="gen" stroke="#475569" fontSize={11} tickLine={false} />
                        <YAxis stroke="#475569" fontSize={11} domain={[0, 100]} tickLine={false} unit="%" />
                        <Tooltip contentStyle={{ background: '#131926', borderColor: 'rgba(255,255,255,0.1)' }} labelStyle={{ color: '#ffffff' }} />
                        <Line type="monotone" dataKey="acc" stroke="#7c3aed" strokeWidth={2} dot={false} activeDot={{ r: 4 }} name="Accuracy" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Console Logs */}
                <div style={{ background: '#05070c', borderRadius: 'var(--border-radius-lg)', border: '1px solid var(--border-dark)', overflow: 'hidden' }}>
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted-light)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Websocket log stream
                  </div>
                  <div 
                    ref={logContainerRef}
                    style={{ 
                      padding: '1.5rem', height: 180, overflowY: 'auto', 
                      fontFamily: 'var(--mono-font)', fontSize: '0.8rem', 
                      lineHeight: 1.5, color: '#22c55e', display: 'flex', flexDirection: 'column', gap: '0.35rem' 
                    }}
                  >
                    {trainLogs.map((log, index) => (
                      <div key={index}>{log}</div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ---------------------------------------------------- */}
      {/* 4. BENCHMARK RACE VIEW */}
      {/* ---------------------------------------------------- */}
      {activeTab === 'benchmark' && (
        <section className="benchmark-section" style={{ minHeight: '90vh', paddingTop: '7rem' }}>
          <div className="container">
            <div className="section-header" style={{ maxWidth: '640px' }}>
              <h2 className="section-title">NPU VS CPU SPEED &amp; POWER RACES</h2>
              <p>Compare the inference latency and power consumption of a standard device CPU, GPU, and Qualcomm NPU.</p>
            </div>

            <div className="benchmark-layout">
              {/* Benchmark trigger */}
              <div>
                <h3 style={{ fontSize: '1.5rem', marginBottom: '1.25rem' }}>Speedup Race Simulator</h3>
                <p style={{ marginBottom: '2.5rem' }}>
                  Test hardware acceleration under load. We will simulate running an image classification model (MobileNetV2) on an input image, measuring latency in milliseconds.
                </p>

                <div style={{ background: '#ffffff', border: '1px solid var(--border-light)', padding: '2rem', borderRadius: 'var(--border-radius-lg)', boxShadow: 'var(--shadow-sm)', marginBottom: '2rem' }}>
                  <h4 style={{ fontSize: '1rem', marginBottom: '1rem' }}>Test Parameters</h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '2rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem' }}>
                      <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Target Model</span>
                      <span style={{ fontWeight: 700 }}>MobileNetV2 (FP16 ONNX)</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem' }}>
                      <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Input Size</span>
                      <span style={{ fontWeight: 700 }}>224 x 224 pixels</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem' }}>
                      <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Batch Size</span>
                      <span style={{ fontWeight: 700 }}>1 (Real-time stream)</span>
                    </div>
                  </div>
                  <button 
                    className="cta-button" 
                    style={{ width: '100%' }} 
                    onClick={runBenchmark}
                    disabled={isBenchmarking}
                  >
                    {isBenchmarking ? <RefreshCw size={16} className="animate-spin" /> : null}
                    {isBenchmarking ? "MEASURING SPEED..." : "RUN HARDWARE RACE"}
                  </button>
                </div>
              </div>

              {/* Dynamic comparative bars */}
              <div className="comparison-card">
                <h3 style={{ fontSize: '1.25rem', marginBottom: '1.5rem' }}>LATENCY COMPARISON (FP16)</h3>

                {!benchmarkResult && !isBenchmarking && (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 260, textAlign: 'center', color: 'var(--text-muted)' }}>
                    <BarChart2 size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                    <p style={{ fontSize: '0.95rem' }}>Click "Run Hardware Race" to start test.</p>
                  </div>
                )}

                {isBenchmarking && (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 260 }}>
                    <RefreshCw size={36} className="animate-spin text-accent-blue" style={{ marginBottom: '1rem' }} />
                    <p style={{ fontWeight: 700 }}>Priming model &amp; running forward pass...</p>
                  </div>
                )}

                {benchmarkResult && !isBenchmarking && (
                  <div>
                    {/* NPU Bar */}
                    <div className="comparison-bar-item">
                      <div className="comparison-bar-header">
                        <span>Qualcomm Hexagon NPU (HTP)</span>
                        <span style={{ color: 'var(--accent-blue)' }}>{benchmarkResult.npu} ms (Fastest)</span>
                      </div>
                      <div className="comparison-bar-track">
                        <div className="comparison-bar-fill npu" style={{ width: '12%' }}></div>
                      </div>
                    </div>

                    {/* GPU Bar */}
                    <div className="comparison-bar-item">
                      <div className="comparison-bar-header">
                        <span>DirectX 12 GPU (GPU acceleration)</span>
                        <span>{benchmarkResult.gpu} ms</span>
                      </div>
                      <div className="comparison-bar-track">
                        <div className="comparison-bar-fill gpu" style={{ width: '25%' }}></div>
                      </div>
                    </div>

                    {/* CPU Bar */}
                    <div className="comparison-bar-item">
                      <div className="comparison-bar-header">
                        <span>x86-64 Host CPU (CPU fallback)</span>
                        <span>{benchmarkResult.cpu} ms</span>
                      </div>
                      <div className="comparison-bar-track">
                        <div className="comparison-bar-fill cpu" style={{ width: '100%' }}></div>
                      </div>
                    </div>

                    {/* Summary bento row */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '2.5rem', borderTop: '1px solid var(--border-light)', paddingTop: '1.5rem' }}>
                      <div style={{ background: 'var(--bg-light)', padding: '1rem', borderRadius: '8px' }}>
                        <span style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>NPU SPEEDUP</span>
                        <span style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-blue)', fontFamily: 'var(--mono-font)' }}>{benchmarkResult.speedup}</span>
                      </div>
                      <div style={{ background: 'var(--bg-light)', padding: '1rem', borderRadius: '8px' }}>
                        <span style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>POWER SAVING</span>
                        <span style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--success-mint)', fontFamily: 'var(--mono-font)' }}>{benchmarkResult.powerSaving}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ---------------------------------------------------- */}
      {/* 5. CLASSIFY PLAYGROUND VIEW */}
      {/* ---------------------------------------------------- */}
      {activeTab === 'classify' && (
        <section className="playground-section" style={{ minHeight: '90vh', paddingTop: '7rem' }}>
          <div className="container">
            <div className="section-header" style={{ maxWidth: '640px' }}>
              <h2 className="section-title">INFERENCE SANDBOX PLAYGROUND</h2>
              <p>Upload a custom image or use a mock webcam feed to experience local NPU-accelerated neural network classification.</p>
            </div>

            <div className="playground-grid">
              {/* Media viewer */}
              <div>
                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
                  <button 
                    className={`cta-button secondary ${feedType === 'upload' ? 'active' : ''}`}
                    onClick={() => { setFeedType('upload'); setResults([]); }}
                    style={{ flex: 1, background: feedType === 'upload' ? 'rgba(30, 64, 175, 0.05)' : '' }}
                  >
                    <Upload size={16} /> Image File
                  </button>
                  <button 
                    className={`cta-button secondary ${feedType === 'webcam' ? 'active' : ''}`}
                    onClick={() => { setFeedType('webcam'); setResults([]); }}
                    style={{ flex: 1, background: feedType === 'webcam' ? 'rgba(30, 64, 175, 0.05)' : '' }}
                  >
                    <Camera size={16} /> Live Webcam Feed
                  </button>
                </div>

                <div className="canvas-container">
                  {feedType === 'upload' ? (
                    <img 
                      src={selectedImg} 
                      alt="Uploaded classify target" 
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
                    />
                  ) : (
                    // Mock webcam static image with scan overlay
                    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
                      <img 
                        src="https://images.unsplash.com/photo-1543466835-00a7907e9de1?auto=format&fit=crop&q=80&w=600" 
                        alt="Webcam placeholder" 
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
                      />
                      <div style={{ position: 'absolute', top: '1rem', left: '1rem', background: 'rgba(239, 68, 68, 0.85)', color: '#ffffff', fontSize: '0.75rem', fontWeight: 700, padding: '0.25rem 0.5rem', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <div style={{ width: '6px', height: '6px', background: '#ffffff', borderRadius: '50%', animation: 'beacon 1s infinite' }}></div>
                        REC FEED
                      </div>
                    </div>
                  )}

                  {showScanner && (
                    <div className="canvas-overlay">
                      <div className="canvas-scanner"></div>
                    </div>
                  )}
                </div>

                <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
                  {feedType === 'upload' && (
                    <div style={{ display: 'flex', gap: '0.5rem', flex: 1 }}>
                      <input 
                        type="text" 
                        className="form-input" 
                        style={{ flex: 1 }} 
                        placeholder="Image URL" 
                        value={selectedImg} 
                        onChange={(e) => { setSelectedImg(e.target.value); setResults([]); }} 
                      />
                      <button className="cta-button" onClick={handleClassify} disabled={isClassifying}>
                        RUN INFERENCE
                      </button>
                    </div>
                  )}
                  {feedType === 'webcam' && (
                    <button className="cta-button" style={{ width: '100%' }} onClick={handleClassify} disabled={isClassifying}>
                      CAPTURE FRAME &amp; CLASSIFY
                    </button>
                  )}
                </div>
              </div>

              {/* Classification scores column */}
              <div>
                <h3 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Activity size={20} className="text-accent-blue" />
                  CLASSIFICATION RESULTS
                </h3>

                {results.length === 0 && !isClassifying && (
                  <div style={{ border: '1px dashed var(--border-light)', borderRadius: 'var(--border-radius-lg)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 280, color: 'var(--text-muted)' }}>
                    <Info size={32} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                    <p style={{ fontSize: '0.95rem' }}>Awaiting classification...</p>
                  </div>
                )}

                {isClassifying && (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 280 }}>
                    <RefreshCw size={36} className="animate-spin text-accent-blue" style={{ marginBottom: '1rem' }} />
                    <p style={{ fontWeight: 700 }}>Executing forward pass on NPU...</p>
                  </div>
                )}

                {results.length > 0 && !isClassifying && (
                  <div className="confidence-list">
                    {results.map((res, index) => (
                      <div key={index} className="confidence-item">
                        <div>
                          <div className="confidence-label">{res.label}</div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '0.25rem' }}>
                            <div style={{ width: 140, height: 6, background: 'rgba(15, 23, 42, 0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                              <div style={{ width: `${res.confidence * 100}%`, height: '100%', background: 'var(--accent-blue)' }}></div>
                            </div>
                          </div>
                        </div>
                        <span className="confidence-val">{(res.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* FOOTER */}
      <footer className="footer">
        <div className="container footer-grid">
          <div>
            <div className="footer-logo">
              <div className="logo-dot"></div>
              SWARMNET
            </div>
            <p className="footer-desc">
              Pioneering the decentralized AI supercomputer using efficient NPU cycles. Join the swarm today.
            </p>
          </div>
          <div>
            <h4 className="footer-title">Network</h4>
            <ul className="footer-links">
              <li><a href="#" className="footer-link" onClick={() => setActiveTab('dashboard')}>Swarm Console</a></li>
              <li><a href="#" className="footer-link" onClick={() => setActiveTab('training')}>Training Progress</a></li>
              <li><a href="#" className="footer-link" onClick={() => setActiveTab('benchmark')}>Benchmark Stats</a></li>
            </ul>
          </div>
          <div>
            <h4 className="footer-title">Science</h4>
            <ul className="footer-links">
              <li><a href="#" className="footer-link">Evolutionary Strategies</a></li>
              <li><a href="#" className="footer-link">Genomic Modeling</a></li>
              <li><a href="#" className="footer-link">Edge Inference</a></li>
            </ul>
          </div>
          <div>
            <h4 className="footer-title">Support</h4>
            <ul className="footer-links">
              <li><a href="#" className="footer-link">Documentation</a></li>
              <li><a href="#" className="footer-link">API Reference</a></li>
              <li><a href="#" className="footer-link">GitHub Repository</a></li>
            </ul>
          </div>
        </div>
        <div className="container footer-bottom">
          <p>&copy; 2026 SwarmNet. All rights reserved. Distributed NPU Supercomputing.</p>
          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <a href="#" className="footer-link">Privacy Policy</a>
            <a href="#" className="footer-link">Terms of Service</a>
          </div>
        </div>
      </footer>

      {/* LEAD CAPTURE / DONATE COMPUTE MODAL */}
      {showDonateModal && (
        <div className="modal-overlay" onClick={() => setShowDonateModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowDonateModal(false)}>
              <X size={20} />
            </button>
            <h3 style={{ fontSize: '1.75rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Download size={24} className="text-accent-blue" />
              JOIN THE SWARM
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', marginBottom: '2rem' }}>
              Enter your device details below to download the SwarmNet client tray agent and start earning XP immediately.
            </p>

            <form onSubmit={(e) => { e.preventDefault(); setShowDonateModal(false); alert("Success! Your mock download for SwarmNet Agent has started."); }}>
              <div className="form-group">
                <label className="form-label">Operating System</label>
                <select className="form-input" required>
                  <option value="windows">Windows 11 (Qualcomm Snapdragon / AMD Ryzen)</option>
                  <option value="macos">macOS Apple Silicon (M1/M2/M3/M4)</option>
                  <option value="linux">Linux (Intel/AMD)</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">NPU Accelerator Type</label>
                <select className="form-input" required>
                  <option value="hexagon">Qualcomm Hexagon (QNN HTP)</option>
                  <option value="xdna">AMD XDNA (VitisAI)</option>
                  <option value="ane">Apple Neural Engine (ANE)</option>
                  <option value="intel">Intel Meteor Lake (OpenVINO)</option>
                  <option value="none">No NPU (CPU/GPU fallback)</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Email Address</label>
                <input type="email" className="form-input" placeholder="you@company.com" required />
              </div>

              <button type="submit" className="cta-button" style={{ width: '100%', marginTop: '1rem' }}>
                GENERATE CLIENT AGENT <ArrowRight size={16} />
              </button>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
