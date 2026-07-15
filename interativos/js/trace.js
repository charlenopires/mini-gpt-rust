// trace.js — utilidades de hover para as visualizações em canvas: um tooltip
// flutuante reutilizável e um mapeador ponteiro→dado com correção de DPR e de
// escala de exibição (o canvas é desenhado em unidades lógicas mas exibido a
// 100% da largura do container).

/** Um único tooltip flutuante, posicionado em coordenadas de viewport. */
export function createTooltip() {
  const el = document.createElement('div');
  el.className = 'viz-tooltip';
  document.body.appendChild(el);
  return {
    // `html` deve já vir com qualquer texto do usuário escapado (escapeHtml).
    show(clientX, clientY, html) {
      el.innerHTML = html;
      el.dataset.visible = 'true';
      el.style.left = `${clientX + 14}px`;
      el.style.top = `${clientY + 14}px`;
    },
    hide() { el.dataset.visible = 'false'; },
    destroy() { el.remove(); },
  };
}

/**
 * Traduz a posição do ponteiro para as coordenadas em que o canvas foi
 * desenhado (após `ctx.scale(dpr, dpr)`), chamando `hitTest(px, py)`. Se
 * `hitTest` devolver algo (≠ null/false), chama `onHover(cell, ev)`; senão
 * `onLeave()`.
 * @param {HTMLCanvasElement} canvas
 * @param {{hitTest:(px:number,py:number)=>any, onHover:Function, onLeave?:Function}} handlers
 */
export function createCanvasTrace(canvas, { hitTest, onHover, onLeave }) {
  function toLogical(ev) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const logicalW = canvas.width / dpr;
    const logicalH = canvas.height / dpr;
    const px = (ev.clientX - rect.left) * (logicalW / (rect.width || 1));
    const py = (ev.clientY - rect.top) * (logicalH / (rect.height || 1));
    return [px, py];
  }
  function handle(ev) {
    const [px, py] = toLogical(ev);
    const cell = hitTest(px, py);
    if (cell != null && cell !== false) onHover(cell, ev);
    else onLeave?.();
  }
  function leave() { onLeave?.(); }

  canvas.addEventListener('mousemove', handle);
  canvas.addEventListener('mouseleave', leave);
  return {
    destroy() {
      canvas.removeEventListener('mousemove', handle);
      canvas.removeEventListener('mouseleave', leave);
    },
  };
}

// ── Rampa sequencial (a mesma família --seq-* do design system) ────────
// Compartilhada pelo heatmap de atenção e pela previsão. 0 → 1.
export const SEQ_RAMP = ['#201d30', '#2c2748', '#3d3768', '#524b92', '#6c63b8', '#8b84d9', '#b7b3ef'];

export function rampColor(t) {
  const c = Math.max(0, Math.min(1, t));
  const idx = c * (SEQ_RAMP.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return SEQ_RAMP[lo];
  return mixHex(SEQ_RAMP[lo], SEQ_RAMP[hi], idx - lo);
}

function mixHex(a, b, t) {
  const pa = hexToRgb(a), pb = hexToRgb(b);
  return `rgb(${Math.round(pa[0] + (pb[0] - pa[0]) * t)},${Math.round(pa[1] + (pb[1] - pa[1]) * t)},${Math.round(pa[2] + (pb[2] - pa[2]) * t)})`;
}
function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// ── Linha de barra (token · trilha · valor) — distribuições de prob/peso ──
// Compartilhada por previsão, inferência e a linha fixada da atenção.
export function barRow(token, valueText, fillFrac, chosen) {
  const row = document.createElement('div');
  row.className = 'prob-bar';
  if (chosen) row.dataset.chosen = 'true';
  const tok = document.createElement('div');
  tok.className = 'prob-bar__token';
  tok.textContent = token || '∅';
  tok.title = token || '';
  const track = document.createElement('div');
  track.className = 'prob-bar__track';
  const fill = document.createElement('div');
  fill.className = 'prob-bar__fill';
  fill.style.width = `${Math.max(2, fillFrac * 100)}%`;
  track.appendChild(fill);
  const val = document.createElement('div');
  val.className = 'prob-bar__val';
  val.textContent = valueText;
  row.append(tok, track, val);
  return row;
}
