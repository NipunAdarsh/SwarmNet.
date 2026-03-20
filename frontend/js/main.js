/* ============================================
   SwarmNet — Main JavaScript
   Nav, Scroll Animations, Counters, Mobile Menu
   Multi-page aware (works on index + sub-pages)
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {

    // ── Navbar scroll effect ──
    const navbar = document.getElementById('navbar');

    if (navbar) {
        const handleNavScroll = () => {
            if (window.scrollY > 60) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        };
        window.addEventListener('scroll', handleNavScroll, { passive: true });
    }

    // ── Mobile hamburger menu ──
    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('nav-links');

    if (hamburger && navLinks) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navLinks.classList.toggle('active');
        });

        // Close mobile menu on link click
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navLinks.classList.remove('active');
            });
        });
    }

    // ── IntersectionObserver — scroll entrance animations ──
    const animatedElements = document.querySelectorAll('.animate-on-scroll');

    if (animatedElements.length) {
        const scrollObserver = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('is-visible');
                        scrollObserver.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
        );
        animatedElements.forEach(el => scrollObserver.observe(el));
    }

    // ── Animated counters (fetches real stats from API) ──
    const counters = document.querySelectorAll('.counter');
    let countersAnimated = false;

    const animateCounters = async () => {
        if (countersAnimated) return;
        countersAnimated = true;

        // Fetch real stats from backend
        try {
            const resp = await fetch('/api/v1/stats');
            if (resp.ok) {
                const stats = await resp.json();
                const mapping = {
                    'stat-inferences': stats.inferences_run || 0,
                    'stat-latency': Math.round(stats.avg_latency_ms || 0),
                    'stat-uptime': Math.round(stats.uptime_hours || 0),
                    'stat-sessions': stats.active_sessions || 0,
                };
                for (const [id, val] of Object.entries(mapping)) {
                    const el = document.getElementById(id);
                    if (el) el.setAttribute('data-target', val);
                }
            }
        } catch (e) {
            // Fallback: animate with whatever data-target is in HTML
        }

        counters.forEach(counter => {
            const target = parseInt(counter.getAttribute('data-target'));
            const duration = 2000; // ms
            const stepTime = 16;
            const totalSteps = Math.floor(duration / stepTime);
            let currentStep = 0;

            const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);

            const updateCounter = () => {
                currentStep++;
                const progress = easeOutQuart(currentStep / totalSteps);
                const current = Math.floor(target * progress);

                counter.textContent = current.toLocaleString();

                if (currentStep < totalSteps) {
                    requestAnimationFrame(updateCounter);
                } else {
                    counter.textContent = target.toLocaleString();
                }
            };

            requestAnimationFrame(updateCounter);
        });
    };

    // Observe the stats section to trigger counters
    const statsSection = document.getElementById('stats');
    if (statsSection) {
        const statsObserver = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        animateCounters();
                        statsObserver.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.3 }
        );
        statsObserver.observe(statsSection);
    }

    // ── Smooth scroll for same-page anchor links only ──
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const target = document.querySelector(targetId);
            if (target && navbar) {
                e.preventDefault();
                const navHeight = navbar.offsetHeight;
                const targetPosition = target.getBoundingClientRect().top + window.scrollY - navHeight;
                window.scrollTo({ top: targetPosition, behavior: 'smooth' });
            }
            // If target doesn't exist on this page, let it navigate normally
            // (e.g. /#how-it-works from a sub-page will go to index)
        });
    });

    // ── Parallax on hero section ──
    const heroBg = document.querySelector('.hero-bg-image');
    const heroGradient = document.querySelector('.hero-gradient-overlay');

    if (heroBg && heroGradient) {
        window.addEventListener('scroll', () => {
            const scrolled = window.scrollY;
            if (scrolled < window.innerHeight) {
                heroBg.style.transform = `translateY(${scrolled * 0.3}px) scale(1.1)`;
                heroGradient.style.transform = `translateY(${scrolled * 0.15}px)`;
            }
        }, { passive: true });
    }

    // ── Active nav link highlight ──
    // On sub-pages, highlight is set via the `active` class in HTML.
    // On the index page, use scroll-based highlighting for anchor sections.
    const pathname = window.location.pathname;
    const isIndexPage = pathname === '/' || pathname === '/index.html';

    if (isIndexPage) {
        const sections = document.querySelectorAll('section[id]');
        const navAnchors = document.querySelectorAll('.nav-links a:not(.nav-cta)');

        const highlightNav = () => {
            const scrollY = window.scrollY + 100;

            sections.forEach(section => {
                const top = section.offsetTop;
                const height = section.offsetHeight;
                const id = section.getAttribute('id');

                if (scrollY >= top && scrollY < top + height) {
                    navAnchors.forEach(a => {
                        a.style.color = '';
                        if (a.getAttribute('href') === `#${id}`) {
                            a.style.color = 'var(--text)';
                        }
                    });
                }
            });
        };

        window.addEventListener('scroll', highlightNav, { passive: true });
    }

});
