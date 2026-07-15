// previsao.js — a peça central: um forward pass real, decomposto em 7 estágios
// que o aluno percorre com o stepper. O slider de temperatura remodela a
// distribuição do próximo token AO VIVO, recomputando o softmax no navegador
// sobre os logits crus (uma única ida ao servidor por prompt).

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
  temp: document.getElementById('temp'),
  tempVal: document.getElementById('temp-val'),
  preview: document.getElementById('preview'),
  btn: document.getElementById('btn-calc'),
  notice: document.getElementById('notice'),
  chips: document.getElementById('prompt-chips'),
  stepper: document.getElementById('stepper'),
  flow: document.getElementById('flow'),
  viz: document.getElementById('viz'),
  explain: document.getElementById('explain'),
};

let data = null;          // resposta de /predict do prompt computado
let computedPrompt = '';
let embData = null;       // cache lazy de /embeddings
let attnData = null;      // cache lazy de /attention
let temperature = parseFloat(els.temp.value);
let currentStage = 0;
let stepper = null;
const tooltip = createTooltip();

// ── utilidades ───────────────────────────────────────────────────────────
function softmax(logits, T) {
  const t = Math.max(0.05, T);
  let max = -Infinity;
  for (const l of logits) if (l > max) max = l;
  const exps = logits.map((l) => Math.exp((l - max) / t));
  let sum = 0;
  for (const e of exps) sum += e;
  return exps.map((e) => e / (sum || 1));
}
const trunc = (s, n = 8) => (s.length > n ? s.slice(0, n) + '…' : s);
const TOP_SHOWN = 12; // quantos candidatos mostrar nas barras (o endpoint traz 40)

function setViz(node) { els.viz.innerHTML = ''; els.viz.appendChild(node); }
function setExplain(node) { els.explain.innerHTML = ''; if (node) els.explain.appendChild(node); }
function hint(msg) { const d = document.createElement('div'); d.className = 'empty-hint'; d.textContent = msg; return d; }
function needData() { setViz(hint('Clique em Calcular para rodar um forward pass.')); setExplain(null); }

function canvasBox(h) {
  const w = Math.max(320, els.viz.clientWidth || 640);
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

// ── barras de probabilidade ──────────────────────────────────────────────
function renderNextBars(probs, caption) {
  const wrap = document.createElement('div');
  const cap = document.createElement('div');
  cap.className = 'hint';
  cap.style.marginBottom = 'var(--sp-3)';
  cap.textContent = caption;
  const bars = document.createElement('div');
  bars.className = 'prob-bars';
  const shown = data.top.slice(0, TOP_SHOWN);
  const maxP = Math.max(...shown.map((c) => probs[c.id]), 1e-9);
  shown.forEach((c, i) => {
    const p = probs[c.id];
    bars.appendChild(barRow(c.token, `${(p * 100).toFixed(1)}%`, p / maxP, i === 0));
  });
  wrap.append(cap, bars);
  setViz(wrap);
}

// ── desenhos compactos (previsão é a visão geral; as páginas dedicadas têm
//    as versões interativas completas) ────────────────────────────────────
function drawScatter(cb, points) {
  const { ctx, w, h } = cb;
  ctx.clearRect(0, 0, w, h);
  if (!points || points.length === 0) return;
  const pad = 44;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (x) => pad + ((x - minX) / ((maxX - minX) || 1)) * (w - 2 * pad);
  const sy = (y) => pad + (1 - (y - minY) / ((maxY - minY) || 1)) * (h - 2 * pad);
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, h / 2); ctx.lineTo(w - pad, h / 2);
  ctx.moveTo(w / 2, pad); ctx.lineTo(w / 2, h - pad);
  ctx.stroke();
  points.forEach((p) => {
    const x = sx(p.x), y = sy(p.y);
    ctx.fillStyle = '#7b84e8';
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#e7e9ec';
    ctx.font = '11px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(p.token, x + 7, y + 3);
  });
}

function drawHeatmapAvgLastLayer(cb, attn) {
  const { ctx, w, h } = cb;
  const tokens = attn.tokens;
  const n = tokens.length;
  const L = attn.layers.length;
  if (L === 0 || n === 0) return;
  const heads = attn.layers[L - 1];
  const H = heads.length;
  const avg = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let hh = 0; hh < H; hh++) s += heads[hh][i][j];
      row.push(s / H);
    }
    avg.push(row);
  }
  ctx.clearRect(0, 0, w, h);
  const labelSpace = 72;
  const cell = Math.max(10, Math.min(34, Math.floor((Math.min(w, h) - labelSpace - 12) / Math.max(n, 1))));
  const originX = labelSpace, originY = labelSpace;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      ctx.fillStyle = rampColor(avg[i][j]);
      ctx.fillRect(originX + j * cell, originY + i * cell, cell - 1, cell - 1);
    }
  }
  ctx.fillStyle = '#8890a0';
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < n; i++) ctx.fillText(trunc(tokens[i]), originX - 6, originY + i * cell + cell / 2);
  for (let j = 0; j < n; j++) {
    ctx.save();
    ctx.translate(originX + j * cell + cell / 2, originY - 6);
    ctx.rotate(-Math.PI / 3);
    ctx.textAlign = 'left';
    ctx.fillText(trunc(tokens[j]), 0, 0);
    ctx.restore();
  }
  createCanvasTrace(cb.canvas, {
    hitTest: (px, py) => {
      const j = Math.floor((px - originX) / cell), i = Math.floor((py - originY) / cell);
      return (i >= 0 && i < n && j >= 0 && j < n) ? { i, j } : null;
    },
    onHover: ({ i, j }, ev) =>
      tooltip.show(ev.clientX, ev.clientY, `${escapeHtml(tokens[i])} → ${escapeHtml(tokens[j])}: ${avg[i][j].toFixed(3)}`),
    onLeave: () => tooltip.hide(),
  });
}

// ── estágios ─────────────────────────────────────────────────────────────
function showTokens() {
  if (!data) return needData();
  const stream = document.createElement('div');
  stream.className = 'token-stream';
  data.tokens.forEach((t) => {
    const chip = document.createElement('span');
    chip.className = 'token-chip';
    chip.textContent = t.text;
    const id = document.createElement('span');
    id.className = 'id';
    id.textContent = `#${t.id}`;
    chip.appendChild(id);
    stream.appendChild(chip);
  });
  setViz(stream);
  setExplain(explain('O que são estes tokens?',
    `<p>O tokenizador ${glossaryTerm('bpe', 'BPE')} quebrou o texto em <strong>${data.tokens.length}</strong> ` +
    `${glossaryTerm('token', 'tokens')}. Cada um é um índice fixo no ${glossaryTerm('vocabulario', 'vocabulário')} ` +
    `de ${data.vocab_size}. O modelo só recebe esta sequência de números — as letras não entram.</p>`,
    { open: true }));
}

async function showEmbeddings() {
  if (!data) return needData();
  setExplain(explain('De tokens a vetores',
    `<p>Cada token vira um ${glossaryTerm('embedding', 'embedding')} de 128 dimensões, somado a uma ` +
    `${glossaryTerm('posicional', 'codificação posicional')} que informa a ordem. Projetamos os vetores para 2D com ` +
    `${glossaryTerm('pca', 'PCA')} só para caber na tela — proximidade ≈ vetores parecidos.</p>` +
    `<p><a href="/embeddings.html">Explorar embeddings a fundo →</a></p>`, { open: true }));
  if (!embData) {
    setViz(hint('projetando embeddings…'));
    try { embData = await apiPost('/embeddings', { text: computedPrompt }); }
    catch (e) { setViz(hint('erro ao projetar embeddings: ' + e.message)); return; }
  }
  if (currentStage !== 1) return;
  const cb = canvasBox(300);
  setViz(cb.box);
  drawScatter(cb, embData.points);
}

async function showAttention() {
  if (!data) return needData();
  setExplain(explain('Atenção: cada token olha para os outros',
    `<p>Aqui, a média das cabeças da última camada. Cada linha (uma ${glossaryTerm('atencao', 'query')}) ` +
    `distribui seu "olhar" pelas colunas (as ${glossaryTerm('qkv', 'keys')}). O triângulo inferior aparece por causa da ` +
    `${glossaryTerm('mascara-causal', 'máscara causal')}: nenhum token vê o futuro. Passe o mouse para ler cada peso.</p>` +
    `<p>${FORMULAS.attention}</p>` +
    `<p><a href="/attention.html">Ver todas as camadas e cabeças →</a></p>`, { open: true }));
  if (!attnData) {
    setViz(hint('calculando atenção…'));
    try { attnData = await apiPost('/attention', { prompt: computedPrompt }); }
    catch (e) { setViz(hint('erro ao calcular atenção: ' + e.message)); return; }
  }
  if (currentStage !== 2) return;
  const cb = canvasBox(360);
  setViz(cb.box);
  drawHeatmapAvgLastLayer(cb, attnData);
}

function showMLP() {
  if (!data) return needData();
  const flow = document.createElement('div');
  flow.className = 'stage-flow';
  flow.style.margin = 'var(--sp-5) 0';
  flow.innerHTML =
    '<span class="stage-flow__item" data-state="done">vetor · 128</span>' +
    '<span class="stage-flow__arrow">→</span>' +
    '<span class="stage-flow__item" data-state="active">expansão 4× · 512 + GELU</span>' +
    '<span class="stage-flow__arrow">→</span>' +
    '<span class="stage-flow__item" data-state="done">projeção · 128</span>';
  setViz(flow);
  setExplain(explain('O MLP (feed-forward)',
    `<p>Depois da atenção, cada posição passa por uma pequena rede: expande de 128 para 512 dimensões, aplica a ` +
    `não-linearidade <strong>GELU</strong> e projeta de volta para 128. É onde o modelo "processa" o que a atenção ` +
    `reuniu. (Mostramos a estrutura; as ativações internas não são expostas por este demo.)</p>`, { open: true }));
}

function showLogits() {
  if (!data) return needData();
  const shown = data.top.slice(0, TOP_SHOWN);
  const logs = shown.map((c) => c.logit);
  const min = Math.min(...logs), max = Math.max(...logs);
  const span = Math.max(max - min, 1e-6);
  const bars = document.createElement('div');
  bars.className = 'prob-bars';
  shown.forEach((c, i) => {
    bars.appendChild(barRow(c.token, c.logit.toFixed(2), (c.logit - min) / span, i === 0));
  });
  setViz(bars);
  setExplain(explain('Logits: as notas brutas',
    `<p>A camada final dá uma nota — um ${glossaryTerm('logits', 'logit')} — a cada um dos ${data.vocab_size} tokens do ` +
    `vocabulário. Mostramos só os ${TOP_SHOWN} maiores. Ainda não são probabilidades (podem ser negativas e não somam 1); ` +
    `o próximo passo conserta isso.</p>`, { open: true }));
}

function showSoftmax() {
  if (!data) return needData();
  renderNextBars(softmax(data.logits, 1.0), 'Softmax (T = 1): as notas viram probabilidades que somam 1');
  setExplain(explain('Softmax',
    `<p>O ${glossaryTerm('softmax', 'softmax')} exponencia cada logit e divide pela soma — transformando as notas em ` +
    `probabilidades entre 0 e 1 que somam 1. O maior logit continua o maior, mas agora dá para amostrar.</p>` +
    `<p>${FORMULAS.softmax}</p>`, { open: true }));
}

function showNext() {
  if (!data) return needData();
  renderNextBars(softmax(data.logits, temperature), `Próximo token · T = ${temperature.toFixed(1)}`);
  setExplain(explain('A distribuição do próximo token',
    `<p>Esta é a decisão do modelo. Arraste a ${glossaryTerm('temperatura', 'temperatura')} na barra lateral: ` +
    `<strong>T baixo</strong> agudiza a distribuição (o topo domina — mais previsível); <strong>T alto</strong> a achata ` +
    `(mais opções ganham chance — mais criativo e aleatório). O token de maior probabilidade (em destaque) não muda com T; ` +
    `o que muda é o quanto ele domina os demais.</p>` +
    `<p>${FORMULAS.temperature}</p>` +
    `<p class="hint">Modelo minúsculo (~1M parâmetros): as escolhas são fracas de propósito — o valor aqui é ver o ` +
    `mecanismo, não a eloquência. <a href="/inference.html">Gerar uma frase inteira →</a></p>`, { open: true }));
}

const STAGES = [
  { short: 'tokens', label: '1 · tokens', onEnter: showTokens },
  { short: 'embeddings', label: '2 · embeddings (+ posicional)', onEnter: showEmbeddings },
  { short: 'atenção', label: '3 · atenção', onEnter: showAttention },
  { short: 'MLP', label: '4 · MLP (feed-forward)', onEnter: showMLP },
  { short: 'logits', label: '5 · logits', onEnter: showLogits },
  { short: 'softmax', label: '6 · softmax', onEnter: showSoftmax },
  { short: 'próximo token', label: '7 · próximo token', onEnter: showNext },
];

// ── fluxo de estágios (breadcrumb) ───────────────────────────────────────
function buildFlow() {
  els.flow.innerHTML = '';
  STAGES.forEach((s, i) => {
    if (i > 0) {
      const a = document.createElement('span');
      a.className = 'stage-flow__arrow';
      a.textContent = '→';
      els.flow.appendChild(a);
    }
    const item = document.createElement('span');
    item.className = 'stage-flow__item';
    item.textContent = s.short;
    item.addEventListener('click', () => stepper && stepper.goTo(i));
    els.flow.appendChild(item);
  });
}
function updateFlow(active) {
  els.flow.querySelectorAll('.stage-flow__item').forEach((el, i) => {
    el.dataset.state = i === active ? 'active' : (i < active ? 'done' : 'pending');
  });
}

// ── previsão ao vivo (readout do topo, sempre visível) ───────────────────
function updatePreview() {
  if (!data) { els.preview.textContent = '—'; return; }
  const probs = softmax(data.logits, temperature);
  const top1 = data.top[0];
  els.preview.textContent = `previsão: "${top1.token}" · ${(probs[top1.id] * 100).toFixed(1)}%`;
}

// ── ações ────────────────────────────────────────────────────────────────
async function calcular() {
  const prompt = els.prompt.value.trim();
  if (!prompt) return;
  els.btn.disabled = true;
  const label = els.btn.textContent;
  els.btn.textContent = 'calculando…';
  try {
    data = await apiPost('/predict', { prompt });
    computedPrompt = prompt;
    embData = null;
    attnData = null;
    updatePreview();
    if (stepper.current() === 0) showTokens();
    else stepper.goTo(0);
  } catch (e) {
    setViz(hint('erro: ' + e.message));
    setExplain(null);
  } finally {
    els.btn.disabled = false;
    els.btn.textContent = label;
  }
}

// ── init ─────────────────────────────────────────────────────────────────
mountPromptChips(els.chips, PROMPT_SETS.previsao, (p) => { els.prompt.value = p; });
buildFlow();
stepper = createStepper(els.stepper, STAGES, {
  onChange: (i) => { currentStage = i; updateFlow(i); },
});

els.temp.addEventListener('input', () => {
  temperature = parseFloat(els.temp.value);
  els.tempVal.textContent = temperature.toFixed(1);
  updatePreview();
  if (data && currentStage === 6) showNext();
});
els.btn.addEventListener('click', calcular);
els.prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); calcular(); }
});

requireModel(els.notice, els.btn);
