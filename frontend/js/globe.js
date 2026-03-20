/* ============================================
   SwarmNet — Three.js Globe Visualization
   Wireframe globe with glowing swarm nodes
   ============================================ */

(function () {
    const canvas = document.getElementById('globe-canvas');
    if (!canvas) return;

    // Wait for Three.js to load
    if (typeof THREE === 'undefined') {
        console.warn('Three.js not loaded, skipping globe.');
        return;
    }

    // ── Setup ──
    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    camera.position.z = 3.5;

    const renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);

    const resizeRenderer = () => {
        const parent = canvas.parentElement;
        const size = Math.min(parent.clientWidth, 550);
        canvas.style.width = size + 'px';
        canvas.style.height = size + 'px';
        renderer.setSize(size, size);
        camera.updateProjectionMatrix();
    };

    resizeRenderer();
    window.addEventListener('resize', resizeRenderer);

    // ── Globe wireframe ──
    const globeGeometry = new THREE.IcosahedronGeometry(1.2, 3);
    const globeMaterial = new THREE.MeshBasicMaterial({
        color: 0x565b62,
        wireframe: true,
        transparent: true,
        opacity: 0.12,
    });
    const globe = new THREE.Mesh(globeGeometry, globeMaterial);
    scene.add(globe);

    // ── Inner glow sphere ──
    const innerGeometry = new THREE.SphereGeometry(1.15, 32, 32);
    const innerMaterial = new THREE.MeshBasicMaterial({
        color: 0xe51b0c,
        transparent: true,
        opacity: 0.02,
    });
    const innerGlow = new THREE.Mesh(innerGeometry, innerMaterial);
    scene.add(innerGlow);

    // ── Swarm nodes (particles on globe surface) ──
    const nodeCount = 120;
    const nodePositions = new Float32Array(nodeCount * 3);
    const nodeColors = new Float32Array(nodeCount * 3);
    const nodeSpeeds = [];

    const accentColor = new THREE.Color(0xe51b0c);
    const tealColor = new THREE.Color(0x7d9d9c);
    const whiteColor = new THREE.Color(0xf9f7f7);

    for (let i = 0; i < nodeCount; i++) {
        // Random points on sphere surface
        const phi = Math.acos(2 * Math.random() - 1);
        const theta = 2 * Math.PI * Math.random();
        const r = 1.22;

        nodePositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        nodePositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        nodePositions[i * 3 + 2] = r * Math.cos(phi);

        // Color variety
        const colorChoice = Math.random();
        let color;
        if (colorChoice < 0.5) color = accentColor;
        else if (colorChoice < 0.8) color = tealColor;
        else color = whiteColor;

        nodeColors[i * 3] = color.r;
        nodeColors[i * 3 + 1] = color.g;
        nodeColors[i * 3 + 2] = color.b;

        nodeSpeeds.push(0.2 + Math.random() * 0.5);
    }

    const nodesGeometry = new THREE.BufferGeometry();
    nodesGeometry.setAttribute('position', new THREE.BufferAttribute(nodePositions, 3));
    nodesGeometry.setAttribute('color', new THREE.BufferAttribute(nodeColors, 3));

    const nodesMaterial = new THREE.PointsMaterial({
        size: 0.035,
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        sizeAttenuation: true,
    });

    const nodes = new THREE.Points(nodesGeometry, nodesMaterial);
    scene.add(nodes);

    // ── Connection lines between nearby nodes ──
    const linePositions = [];
    const lineColors = [];
    const maxConnectionDist = 0.7;

    for (let i = 0; i < nodeCount; i++) {
        for (let j = i + 1; j < nodeCount; j++) {
            const dx = nodePositions[i * 3] - nodePositions[j * 3];
            const dy = nodePositions[i * 3 + 1] - nodePositions[j * 3 + 1];
            const dz = nodePositions[i * 3 + 2] - nodePositions[j * 3 + 2];
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

            if (dist < maxConnectionDist) {
                linePositions.push(
                    nodePositions[i * 3], nodePositions[i * 3 + 1], nodePositions[i * 3 + 2],
                    nodePositions[j * 3], nodePositions[j * 3 + 1], nodePositions[j * 3 + 2]
                );

                const alpha = 1 - dist / maxConnectionDist;
                lineColors.push(
                    accentColor.r * alpha, accentColor.g * alpha, accentColor.b * alpha,
                    accentColor.r * alpha, accentColor.g * alpha, accentColor.b * alpha
                );
            }
        }
    }

    const linesGeometry = new THREE.BufferGeometry();
    linesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
    linesGeometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

    const linesMaterial = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.15,
    });

    const lines = new THREE.LineSegments(linesGeometry, linesMaterial);
    scene.add(lines);

    // ── Outer ring orbits ──
    const ringGeometry = new THREE.RingGeometry(1.6, 1.62, 64);
    const ringMaterial = new THREE.MeshBasicMaterial({
        color: 0x565b62,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
    });

    const ring1 = new THREE.Mesh(ringGeometry, ringMaterial);
    ring1.rotation.x = Math.PI * 0.35;
    ring1.rotation.y = Math.PI * 0.15;
    scene.add(ring1);

    const ring2 = new THREE.Mesh(ringGeometry, ringMaterial.clone());
    ring2.rotation.x = Math.PI * 0.55;
    ring2.rotation.y = Math.PI * -0.3;
    scene.add(ring2);

    // ── Mouse interaction ──
    let mouseX = 0;
    let mouseY = 0;
    let targetRotX = 0;
    let targetRotY = 0;

    document.addEventListener('mousemove', (e) => {
        mouseX = (e.clientX / window.innerWidth) * 2 - 1;
        mouseY = (e.clientY / window.innerHeight) * 2 - 1;
    });

    // ── Animation loop ──
    const clock = new THREE.Clock();

    const animate = () => {
        requestAnimationFrame(animate);

        const elapsed = clock.getElapsedTime();

        // Auto-rotate
        globe.rotation.y = elapsed * 0.08;
        nodes.rotation.y = elapsed * 0.08;
        lines.rotation.y = elapsed * 0.08;

        // Mouse-reactive tilt
        targetRotX = mouseY * 0.15;
        targetRotY = mouseX * 0.15;

        globe.rotation.x += (targetRotX - globe.rotation.x) * 0.02;
        nodes.rotation.x += (targetRotX - nodes.rotation.x) * 0.02;
        lines.rotation.x += (targetRotX - lines.rotation.x) * 0.02;

        // Ring slow spin
        ring1.rotation.z = elapsed * 0.03;
        ring2.rotation.z = -elapsed * 0.02;

        // Pulse inner glow
        innerGlow.material.opacity = 0.02 + Math.sin(elapsed * 1.5) * 0.01;

        // Pulse node sizes
        nodesMaterial.size = 0.035 + Math.sin(elapsed * 2) * 0.008;

        renderer.render(scene, camera);
    };

    animate();
})();
