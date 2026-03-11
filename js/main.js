/* ============================================
   SwarmNet — Main JavaScript
   Nav, Scroll Animations, Counters, Mobile Menu
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {

  // ── Navbar scroll effect ──
  const navbar = document.getElementById('navbar');

  const handleNavScroll = () => {
    if (window.scrollY > 60) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  };

  window.addEventListener('scroll', handleNavScroll, { passive: true });

  // ── Mobile hamburger menu ──
  const hamburger = document.getElementById('hamburger');
  const navLinks = document.getElementById('nav-links');

  hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navLinks.classList.toggle('active');
  });

  // Close mobile menu on link click
  navLinks.querySelectorAll('a').forEach(navLink => {
    navLink.addEventListener('click', () => {
      hamburger.classList.remove('active');
      navLinks.classList.remove('active');
    });
  });

  // ── IntersectionObserver — scroll entrance animations ──
  const animatedElements = document.querySelectorAll('.animate-on-scroll');

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

  animatedElements.forEach(animatedElement => scrollObserver.observe(animatedElement));

  // ── Animated counters ──
  const counterElements = document.querySelectorAll('.counter');
  let haveCountersAnimated = false;

  const animateCounters = () => {
    if (haveCountersAnimated) return;
    haveCountersAnimated = true;

    counterElements.forEach(counterElement => {
      const targetValue = parseInt(counterElement.getAttribute('data-target'));
      const animationDurationMs = 2000; // ms
      const frameIntervalMs = 16;
      const totalAnimationSteps = Math.floor(animationDurationMs / frameIntervalMs);
      let currentAnimationStep = 0;

      const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);

      const updateCounter = () => {
        currentAnimationStep++;
        const progress = easeOutQuart(currentAnimationStep / totalAnimationSteps);
        const currentDisplayValue = Math.floor(targetValue * progress);

        counterElement.textContent = currentDisplayValue.toLocaleString();

        if (currentAnimationStep < totalAnimationSteps) {
          requestAnimationFrame(updateCounter);
        } else {
          counterElement.textContent = targetValue.toLocaleString();
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

  // ── Smooth scroll for anchor links ──
  document.querySelectorAll('a[href^="#"]').forEach(anchorLink => {
    anchorLink.addEventListener('click', function (e) {
      const targetSectionId = this.getAttribute('href');
      if (targetSectionId === '#') return;

      e.preventDefault();
      const targetElement = document.querySelector(targetSectionId);
      if (targetElement) {
        const navHeight = navbar.offsetHeight;
        const targetPosition = targetElement.getBoundingClientRect().top + window.scrollY - navHeight;
        window.scrollTo({ top: targetPosition, behavior: 'smooth' });
      }
    });
  });

  // ── Parallax on hero section ──
  const heroBg = document.querySelector('.hero-bg-image');
  const heroGradient = document.querySelector('.hero-gradient-overlay');

  if (heroBg) {
    window.addEventListener('scroll', () => {
      const parallaxScrollDistance = window.scrollY;
      if (parallaxScrollDistance < window.innerHeight) {
        heroBg.style.transform = `translateY(${parallaxScrollDistance * 0.3}px) scale(1.1)`;
        heroGradient.style.transform = `translateY(${parallaxScrollDistance * 0.15}px)`;
      }
    }, { passive: true });
  }

  // ── Active nav link highlight ──
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

});
