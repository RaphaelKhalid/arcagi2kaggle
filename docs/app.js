(() => {
  const progress = document.querySelector('#progress-bar');
  const updateProgress = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${max > 0 ? (window.scrollY / max) * 100 : 0}%`;
  };
  document.addEventListener('scroll', updateProgress, { passive: true });
  updateProgress();

  document.querySelectorAll('.term').forEach((term) => {
    term.addEventListener('click', (event) => {
      event.stopPropagation();
      document.querySelectorAll('.term.is-open').forEach((other) => {
        if (other !== term) other.classList.remove('is-open');
      });
      term.classList.toggle('is-open');
    });
    term.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        term.click();
      }
    });
  });
  document.addEventListener('click', () => {
    document.querySelectorAll('.term.is-open').forEach((term) => term.classList.remove('is-open'));
  });
})();
