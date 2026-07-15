// embeddings.js — projeção 2D (PCA) dos embeddings reais. Agora o servidor
// devolve também a similaridade de cosseno entre os tokens do prompt e os
// vizinhos mais próximos no vocabulário inteiro: passar o mouse num ponto
// desenha linhas para os tokens parecidos do prompt e lista os vizinhos de
// vocabulário na barra lateral.

import { apiPost, refreshStatusPill, showError, requireModel, escapeHtml } from '/js/api.js';
import { explain, FORMULAS } from '/js/explain.js';
import { glossaryTerm, attachGlossary } from '/js/glossary.js';
import { PROMPT_SETS, mountPromptChips } from '/js/prompts.js';
import { createTooltip, createCanvasTrace, barRow } from '/js/trace.js';

refreshStatusPill();
attachGlossary(document.body);

const els = {
  input: document.getElementById('input-text'),
  chips: document.getElementById('prompt-chips'),
  btn: document.getElementById('btn-run'),
  notice: document.getElementById('notice'),
  result: document.getElementById('result-area'),
  neighbors: document.getElementById('neighbors'),
  explain: document.getElementById('explain'),
};

let data = null;       // { points, similarity, neighbors }
let cb = null;
let positions = [];    // px de cada ponto
const tooltip = createTooltip();

function makeCanvas() {
  const w = Math.max(320, els.result.clientWidth || 640);
  const h = Math.min(w, 480);
  const box = document.createElement('div');
  box.className = 'canvas-box';
  const canvas = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = `${h}px`;
  box.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { box, canvas, ctx, w, h };
}

function computePositions() {
  const { w, h } = cb;
  const pad = 40;
  const pts = data.points;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (x) => pad + ((x - minX) / ((maxX - minX) || 1)) * (w - 2 * pad);
  const sy = (y) => pad + (1 - (y - minY) / ((maxY - minY) || 1)) * (h - 2 * pad);
  positions = pts.map((p) => [sx(p.x), sy(p.y)]);
}

function draw(highlight) {
  const { ctx, w, h } = cb;
  const pts = data.points;
  ctx.clearRect(0, 0, w, h);

  // eixos recessivos
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(40, h / 2); ctx.lineTo(w - 40, h / 2);
  ctx.moveTo(w / 2, 40); ctx.lineTo(w / 2, h - 40);
  ctx.stroke();

  // linhas para os vizinhos (dentro do prompt) do ponto destacado
  if (highlight != null && data.similarity && data.similarity[highlight]) {
    const sims = data.similarity[highlight]
      .map((s, j) => [j, s])
      .filter(([j]) => j !== highlight)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
    const [hx, hy] = positions[highlight];
    sims.forEach(([j, s]) => {
      ctx.strokeStyle = `rgba(123,132,232,${Math.max(0.15, Math.min(0.9, s))})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(positions[j][0], positions[j][1]);
      ctx.stroke();
    });
  }

  // pontos
  pts.forEach((p, i) => {
    const [px, py] = positions[i];
    const on = highlight == null || highlight === i;
    ctx.globalAlpha = on ? 1 : 0.4;
    ctx.fillStyle = highlight === i ? '#ff7a35' : '#7b84e8';
    ctx.beginPath();
    ctx.arc(px, py, highlight === i ? 6 : 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#e7e9ec';
    ctx.font = '12px ui-monospace, monospace';
    ctx.textBaseline = 'bottom';
    ctx.textAlign = 'left';
    ctx.fillText(p.token, px + 7, py - 4);
  });
  ctx.globalAlpha = 1;
}

function hitTest(px, py) {
  for (let i = 0; i < positions.length; i++) {
    if (Math.hypot(positions[i][0] - px, positions[i][1] - py) < 10) return i;
  }
  return null;
}

function renderNeighbors(i) {
  els.neighbors.innerHTML = '';
  const title = document.createElement('p');
  title.className = 'hint';
  title.style.margin = '0 0 var(--sp-2)';
  if (i == null || !data.neighbors || !data.neighbors[i]) {
    title.textContent = 'Passe o mouse sobre um ponto para ver os vizinhos do vocabulário aqui.';
    els.neighbors.appendChild(title);
    return;
  }
  const nb = data.neighbors[i];
  title.textContent = `mais parecidos com "${nb.token}":`;
  const bars = document.createElement('div');
  bars.className = 'prob-bars';
  const max = Math.max(...nb.nearest.map((n) => n.sim), 1e-6);
  nb.nearest.forEach((n) => bars.appendChild(barRow(n.token, n.sim.toFixed(2), Math.max(0, n.sim) / max, false)));
  els.neighbors.append(title, bars);
}

function render() {
  cb = makeCanvas();
  els.result.innerHTML = '';
  els.result.appendChild(cb.box);
  computePositions();
  draw(null);
  createCanvasTrace(cb.canvas, {
    hitTest,
    onHover: (i, ev) => {
      draw(i);
      renderNeighbors(i);
      const p = data.points[i];
      tooltip.show(ev.clientX, ev.clientY, `${escapeHtml(p.token)} · (${p.x.toFixed(2)}, ${p.y.toFixed(2)})`);
    },
    onLeave: () => { tooltip.hide(); draw(null); },
  });
}

async function run() {
  const text = els.input.value.trim();
  if (!text) return;
  els.btn.disabled = true;
  const label = els.btn.textContent;
  els.btn.textContent = 'projetando…';
  try {
    data = await apiPost('/embeddings', { text });
    if (!data.points.length) { showError(els.result, 'Nenhum token reconhecido nesse texto.'); return; }
    render();
    renderNeighbors(null);
  } catch (e) {
    showError(els.result, `Erro: ${e.message}`);
  } finally {
    els.btn.disabled = false;
    els.btn.textContent = label;
  }
}

els.explain.appendChild(explain('Como ler esta projeção',
  `<p>Cada ponto é o ${glossaryTerm('embedding', 'embedding')} de um token, reduzido de 128 para 2 dimensões por ` +
  `${glossaryTerm('pca', 'PCA')}. Pontos próximos ≈ vetores parecidos, mas a projeção descarta dimensões, então a distância ` +
  `é aproximada. A "parecença" real é medida por ${glossaryTerm('cosseno', 'similaridade de cosseno')}.</p>` +
  `<p>${FORMULAS.cosine}</p>` +
  `<p class="hint">Modelo minúsculo (best loss ≈ 5,4): as vizinhanças são fracas — o interessante é que já são vetores reais e não aleatórios.</p>`));

mountPromptChips(els.chips, PROMPT_SETS.embeddings, (p) => { els.input.value = p; });
els.btn.addEventListener('click', run);
requireModel(els.notice, els.btn);
