// stepper.js — passo-a-passo reutilizável para percorrer os estágios de um
// fluxo (tokenização, forward pass, atenção…). Sem dependências. O chamador
// descreve os estágios e reage em `onEnter`; este módulo cuida da navegação,
// dos indicadores e do teclado (setas ← →).

const prefersReducedMotion = () =>
  window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * @param {HTMLElement} rootEl  container onde o stepper é montado
 * @param {Array<{id?:string,label:string,onEnter?:Function,onLeave?:Function}>} stages
 * @param {{autoplayMs?:number, loop?:boolean, onChange?:(i:number)=>void}} opts
 * @returns {{goTo,next,prev,play,pause,current,size}}
 */
export function createStepper(rootEl, stages, opts = {}) {
  const { autoplayMs = 0, loop = false, onChange = null } = opts;
  let index = 0;
  let timer = null;

  rootEl.classList.add('stepper');
  rootEl.innerHTML = '';

  const mkBtn = (text, aria) => {
    const b = document.createElement('button');
    b.className = 'stepper__btn';
    b.type = 'button';
    b.textContent = text;
    b.setAttribute('aria-label', aria);
    return b;
  };

  const btnPrev = mkBtn('◂', 'passo anterior');
  const btnPlay = mkBtn('▸', 'reproduzir');
  const btnNext = mkBtn('▸▸', 'próximo passo');

  const controls = document.createElement('div');
  controls.className = 'stepper__controls';
  controls.append(btnPrev, btnPlay, btnNext);

  const dots = document.createElement('ol');
  dots.className = 'stepper__dots';
  const dotEls = stages.map((s, i) => {
    const li = document.createElement('li');
    li.className = 'stepper__dot';
    li.tabIndex = 0;
    li.title = s.label || `passo ${i + 1}`;
    li.addEventListener('click', () => { pause(); goTo(i); });
    li.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pause(); goTo(i); }
    });
    dots.appendChild(li);
    return li;
  });

  const count = document.createElement('span');
  count.className = 'stepper__count';

  const bar = document.createElement('div');
  bar.className = 'stepper__bar';
  bar.append(controls, dots, count);

  const label = document.createElement('div');
  label.className = 'stepper__label';

  rootEl.append(bar, label);

  function render() {
    dotEls.forEach((d, i) => {
      d.dataset.state = i === index ? 'current' : (i < index ? 'done' : 'pending');
      if (i === index) d.setAttribute('aria-current', 'step');
      else d.removeAttribute('aria-current');
    });
    label.textContent = stages[index]?.label || '';
    count.textContent = `${index + 1} / ${stages.length}`;
    btnPrev.disabled = !loop && index === 0;
    btnNext.disabled = !loop && index === stages.length - 1;
  }

  function goTo(i) {
    const n = stages.length;
    const next = loop ? ((i % n) + n) % n : Math.max(0, Math.min(n - 1, i));
    if (next === index) return;
    const prev = index;
    stages[prev]?.onLeave?.(prev);
    index = next;
    render();
    stages[index]?.onEnter?.(index, controller);
    onChange?.(index);
  }

  const next = () => goTo(index + 1);
  const prev = () => goTo(index - 1);

  function play() {
    if (prefersReducedMotion() || autoplayMs <= 0) return;
    pause();
    btnPlay.textContent = '⏸';
    btnPlay.dataset.playing = 'true';
    timer = setInterval(() => {
      if (!loop && index === stages.length - 1) { pause(); return; }
      next();
    }, autoplayMs);
  }
  function pause() {
    if (timer) { clearInterval(timer); timer = null; }
    btnPlay.textContent = '▸';
    delete btnPlay.dataset.playing;
  }

  btnPrev.addEventListener('click', () => { pause(); prev(); });
  btnNext.addEventListener('click', () => { pause(); next(); });
  btnPlay.addEventListener('click', () => (timer ? pause() : play()));

  rootEl.tabIndex = 0;
  rootEl.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); pause(); prev(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); pause(); next(); }
  });

  const controller = { goTo, next, prev, play, pause, current: () => index, size: stages.length };

  render();
  stages[0]?.onEnter?.(0, controller);
  onChange?.(0);

  return controller;
}
