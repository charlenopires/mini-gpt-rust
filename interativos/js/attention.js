// attention.js — mapa de atenção real, agora com TODAS as camadas × cabeças
// vindas numa única resposta (/api/attention). O usuário desliza por
// camada/cabeça sem novos round-trips, passa o mouse para destacar a linha de
// uma query, e clica para fixá-la e ver para onde ela olha. Um stepper narra
// como a atenção é computada.

import { apiPost, requireModel, refreshStatusPill, escapeHtml } from '/js/api.js';
import { createStepper } from '/js/stepper.js';
import { explain, FORMULAS } from '/js/explain.js';
import { glossaryTerm, attachGlossary } from '/js/glossary.js';
import { PROMPT_SETS, mountPromptChips } from '/js/prompts.js';
import { createTooltip, createCanvasTrace, rampColor, barRow } from '/js/trace.js';

refreshStatusPill();
attachGlossary(document.body);

const els = {
  prompt: document.getElementById('prompt'),
  chips: document.getElementById('prompt-chips'),
  btn: document.getElementById('btn-run'),
  notice: document.getElementById('notice'),
  layer: document.getElementById('layer'),
  layerVal: document.getElementById('layer-val'),
  head: document.getElementById('head'),
  headVal: document.getElementById('head-val'),
  avg: document.getElementById('avg'),
  stepper: document.getElementById('stepper'),
  viz: document.getElementById('viz'),
  pinned: document.getElementById('pinned'),
  explain: document.getElementById('explain'),
};

let attn = null;        // { tokens, num_layers, num_heads, layers }
let selLayer = 0;
let selHead = 0;
let useAvg = true;
let cb = null;          // canvas atual
let curMatrix = null;
let hoverRow = null;
let pinnedRow = null;
const tooltip = createTooltip();
const trunc = (s, n = 9) => (s.length > n ? s.slice(0, n) + '…' : s);

// ── matriz selecionada (camada/cabeça, ou média das cabeças) ─────────────
function matrixFor() {
  const layer = attn.layers[selLayer];
  const n = attn.tokens.length;
  if (!useAvg) return layer[selHead];
  const H = layer.length;
  const avg = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let hh = 0; hh < H; hh++) s += layer[hh][i][j];
      row.push(s / H);
    }
    avg.push(row);
  }
  return avg;
}

// ── desenho do heatmap ───────────────────────────────────────────────────
function drawHeatmap(highlight) {
  const { ctx, w, h } = cb;
  const tokens = attn.tokens;
  const n = tokens.length;
  ctx.clearRect(0, 0, w, h);
  const labelSpace = 78;
  const avail = Math.min(w - labelSpace - 12, h - labelSpace - 12);
  const cell = Math.max(9, Math.min(40, Math.floor(avail / Math.max(n, 1))));
  const originX = labelSpace, originY = labelSpace;
  for (let i = 0; i < n; i++) {
    const dim = highlight != null && highlight !== i;
    for (let j = 0; j < n; j++) {
      ctx.globalAlpha = dim ? 0.22 : 1;
      ctx.fillStyle = rampColor(curMatrix[i][j]);
      ctx.fillRect(originX + j * cell, originY + i * cell, cell - 1, cell - 1);
    }
  }
  ctx.globalAlpha = 1;
  ctx.font = '10px ui-monospace, monospace';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = highlight === i ? '#ff7a35' : '#8890a0';
    ctx.fillText(trunc(tokens[i]), originX - 6, originY + i * cell + cell / 2);
  }
  ctx.fillStyle = '#8890a0';
  for (let j = 0; j < n; j++) {
    ctx.save();
    ctx.translate(originX + j * cell + cell / 2, originY - 6);
    ctx.rotate(-Math.PI / 3);
    ctx.textAlign = 'left';
    ctx.fillText(trunc(tokens[j]), 0, 0);
    ctx.restore();
  }
  cb._geom = { originX, originY, cell, n };
}

function hitTest(px, py) {
  const g = cb && cb._geom;
  if (!g) return null;
  const j = Math.floor((px - g.originX) / g.cell);
  const i = Math.floor((py - g.originY) / g.cell);
  return (i >= 0 && i < g.n && j >= 0 && j < g.n) ? { i, j } : null;
}

function makeCanvas() {
  const w = Math.max(320, els.viz.clientWidth || 640);
  const h = Math.min(w, 520);
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

function renderHeatmap() {
  if (!attn) return;
  curMatrix = matrixFor();
  hoverRow = null;
  cb = makeCanvas();
  els.viz.innerHTML = '';
  els.viz.appendChild(cb.box);
  const legend = document.createElement('div');
  legend.className = 'legend';
  legend.innerHTML = '<span>0.0</span><span class="legend__ramp"></span><span>1.0 · peso de atenção</span>';
  els.viz.appendChild(legend);
  drawHeatmap(pinnedRow);

  createCanvasTrace(cb.canvas, {
    hitTest,
    onHover: ({ i, j }, ev) => {
      tooltip.show(ev.clientX, ev.clientY, `${escapeHtml(attn.tokens[i])} → ${escapeHtml(attn.tokens[j])}: ${curMatrix[i][j].toFixed(3)}`);
      if (i !== hoverRow) { hoverRow = i; drawHeatmap(i); }
    },
    onLeave: () => {
      tooltip.hide();
      if (hoverRow !== null) { hoverRow = null; drawHeatmap(pinnedRow); }
    },
  });

  cb.canvas.addEventListener('click', (ev) => {
    const rect = cb.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const px = (ev.clientX - rect.left) * (cb.canvas.width / dpr / (rect.width || 1));
    const py = (ev.clientY - rect.top) * (cb.canvas.height / dpr / (rect.height || 1));
    const hit = hitTest(px, py);
    if (!hit) return;
    pinnedRow = pinnedRow === hit.i ? null : hit.i;
    renderPinned();
    drawHeatmap(hoverRow != null ? hoverRow : pinnedRow);
  });
}

function renderPinned() {
  els.pinned.innerHTML = '';
  if (pinnedRow == null || !attn) return;
  const tokens = attn.tokens;
  const row = curMatrix[pinnedRow];
  const ranked = row.map((v, j) => [j, v]).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const topJ = ranked[0][0];
  const title = document.createElement('div');
  title.className = 'hint';
  title.style.margin = 'var(--sp-4) 0 var(--sp-2)';
  title.textContent = `"${tokens[pinnedRow]}" olha principalmente para: (clique de novo na linha para desafixar)`;
  const bars = document.createElement('div');
  bars.className = 'prob-bars';
  ranked.forEach(([j, v]) => bars.appendChild(barRow(tokens[j], v.toFixed(3), v / (ranked[0][1] || 1), j === topJ)));
  els.pinned.append(title, bars);
}

// ── controles de scrub ───────────────────────────────────────────────────
function syncControls() {
  els.layer.max = String(attn.num_layers - 1);
  els.layer.value = String(selLayer);
  els.layer.disabled = false;
  els.layerVal.textContent = `${selLayer + 1} / ${attn.num_layers}`;
  els.head.max = String(attn.num_heads - 1);
  els.head.value = String(selHead);
  els.head.disabled = useAvg;
  els.headVal.textContent = useAvg ? 'média' : `${selHead + 1} / ${attn.num_heads}`;
}

els.layer.addEventListener('input', () => {
  selLayer = Number(els.layer.value);
  els.layerVal.textContent = `${selLayer + 1} / ${attn.num_layers}`;
  pinnedRow = null; renderPinned(); renderHeatmap();
});
els.head.addEventListener('input', () => {
  selHead = Number(els.head.value);
  els.headVal.textContent = `${selHead + 1} / ${attn.num_heads}`;
  renderHeatmap();
});
els.avg.addEventListener('change', () => {
  useAvg = els.avg.checked;
  els.head.disabled = useAvg;
  els.headVal.textContent = useAvg ? 'média' : `${selHead + 1} / ${attn.num_heads}`;
  renderHeatmap();
});

// ── ação ─────────────────────────────────────────────────────────────────
async function run() {
  const prompt = els.prompt.value.trim();
  if (!prompt) return;
  els.btn.disabled = true;
  const label = els.btn.textContent;
  els.btn.textContent = 'calculando…';
  try {
    attn = await apiPost('/attention', { prompt });
    if (!attn.tokens.length) { els.viz.innerHTML = '<div class="empty-hint">Prompt sem tokens reconhecidos.</div>'; return; }
    selLayer = attn.num_layers - 1;
    selHead = 0;
    useAvg = true;
    els.avg.checked = true;
    pinnedRow = null;
    renderPinned();
    syncControls();
    renderHeatmap();
  } catch (e) {
    els.viz.innerHTML = '';
    const d = document.createElement('div');
    d.className = 'error-banner';
    d.textContent = `Erro: ${e.message}`;
    els.viz.appendChild(d);
  } finally {
    els.btn.disabled = false;
    els.btn.textContent = label;
  }
}

// ── stepper: como a atenção é computada ──────────────────────────────────
const STAGES = [
  {
    short: 'Q · K · V', label: '1 · Query, Key, Value',
    body: () => `<p>Cada token é projetado em três vetores: ${glossaryTerm('qkv', 'Query, Key e Value')}. ` +
      `A Query pergunta "de que preciso?", a Key anuncia "eu ofereço isto", e o Value carrega a informação transportada.</p>`,
  },
  {
    short: 'pontuação', label: '2 · pontuação Q·Kᵀ ÷ √d',
    body: () => `<p>Comparamos cada Query com cada Key por produto escalar e dividimos por √d para estabilizar a escala. ` +
      `Isso dá uma matriz de "afinidade" bruta entre todos os pares de tokens.</p><p>${FORMULAS.attention}</p>`,
  },
  {
    short: 'máscara', label: '3 · máscara causal',
    body: () => `<p>Zeramos (com −∞ antes do softmax) tudo acima da diagonal: a ${glossaryTerm('mascara-causal', 'máscara causal')} ` +
      `impede um token de olhar para o futuro. É por isso que o mapa fica triangular — só o triângulo inferior tem peso.</p>`,
  },
  {
    short: 'softmax', label: '4 · softmax por linha',
    body: () => `<p>O ${glossaryTerm('softmax', 'softmax')} normaliza cada linha para somar 1: viram pesos de atenção. ` +
      `Cada query reparte 100% da sua atenção entre as keys visíveis.</p><p>${FORMULAS.softmax}</p>`,
  },
  {
    short: 'leitura', label: '5 · leia o mapa',
    body: () => `<p>É o que você vê: <strong>linha</strong> = query (quem olha), <strong>coluna</strong> = key (olhado); ` +
      `mais claro = mais atenção. Deslize por camada/cabeça, passe o mouse para destacar uma linha e clique para fixá-la.</p>`,
  },
];

function setStage(i) {
  els.explain.innerHTML = '';
  els.explain.appendChild(explain(STAGES[i].label, STAGES[i].body(), { open: true }));
}

// ── init ─────────────────────────────────────────────────────────────────
mountPromptChips(els.chips, PROMPT_SETS.attention, (p) => { els.prompt.value = p; });
createStepper(els.stepper, STAGES.map((s) => ({ short: s.short, label: s.label, onEnter: (i) => setStage(i) })));
els.btn.addEventListener('click', run);
requireModel(els.notice, els.btn);
