import { apiPost, refreshStatusPill, showError } from '/js/api.js';

refreshStatusPill();

// Rampa sequencial violeta (0 → 1), a mesma família de --violet do design system.
const RAMP = ['#201d30', '#2c2748', '#3d3768', '#524b92', '#6c63b8', '#8b84d9', '#b7b3ef'];

function rampColor(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (RAMP.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return RAMP[lo];
  const frac = idx - lo;
  return mixHex(RAMP[lo], RAMP[hi], frac);
}

function mixHex(a, b, t) {
  const pa = hexToRgb(a), pb = hexToRgb(b);
  const r = Math.round(pa[0] + (pb[0] - pa[0]) * t);
  const g = Math.round(pa[1] + (pb[1] - pa[1]) * t);
  const bl = Math.round(pa[2] + (pb[2] - pa[2]) * t);
  return `rgb(${r},${g},${bl})`;
}
function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

let selectedLayer = null;
let selectedHead = 'avg';

function drawHeatmap(container, data) {
  container.textContent = '';

  const tokens = data.tokens;
  const n = tokens.length;
  const labelSpace = Math.min(120, Math.max(60, n * 6));
  const cell = Math.max(14, Math.min(38, Math.floor(640 / n)));
  const size = n * cell;
  const width = size + labelSpace + 24;
  const height = size + labelSpace + 24;

  const box = document.createElement('div');
  box.className = 'canvas-box';
  const canvas = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = `${height}px`;
  box.appendChild(canvas);
  container.appendChild(box);

  const legend = document.createElement('div');
  legend.className = 'legend';
  const rampEl = document.createElement('span');
  rampEl.className = 'legend__ramp';
  const lo = document.createElement('span');
  lo.textContent = '0.0';
  const hi = document.createElement('span');
  hi.textContent = '1.0 · peso de atenção';
  legend.appendChild(lo);
  legend.appendChild(rampEl);
  legend.appendChild(hi);
  container.appendChild(legend);

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const originX = labelSpace;
  const originY = labelSpace;

  // Células
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = data.weights[i][j];
      ctx.fillStyle = rampColor(v);
      ctx.fillRect(originX + j * cell, originY + i * cell, cell - 1, cell - 1);
    }
  }

  // Rótulos das colunas (query, no topo, rotacionados)
  ctx.fillStyle = '#8890a0';
  ctx.font = '11px ui-monospace, monospace';
  ctx.save();
  for (let j = 0; j < n; j++) {
    ctx.save();
    ctx.translate(originX + j * cell + cell / 2, originY - 6);
    ctx.rotate(-Math.PI / 3);
    ctx.textAlign = 'right';
    ctx.fillText(truncate(tokens[j]), 0, 0);
    ctx.restore();
  }
  ctx.restore();

  // Rótulos das linhas (key, à esquerda)
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < n; i++) {
    ctx.fillText(truncate(tokens[i]), originX - 8, originY + i * cell + cell / 2);
  }

  // Tooltip via hover
  const tooltip = document.createElement('div');
  tooltip.style.cssText = 'position:fixed;pointer-events:none;background:#1c2029;border:1px solid rgba(255,255,255,0.16);' +
    'border-radius:4px;padding:4px 8px;font-family:ui-monospace,monospace;font-size:12px;color:#e7e9ec;display:none;z-index:50;';
  document.body.appendChild(tooltip);

  canvas.addEventListener('mousemove', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const j = Math.floor((x - originX) / cell);
    const i = Math.floor((y - originY) / cell);
    if (i >= 0 && i < n && j >= 0 && j < n) {
      const v = data.weights[i][j];
      tooltip.style.display = 'block';
      tooltip.style.left = `${ev.clientX + 12}px`;
      tooltip.style.top = `${ev.clientY + 12}px`;
      tooltip.textContent = `${tokens[i]} → ${tokens[j]}: ${v.toFixed(3)}`;
    } else {
      tooltip.style.display = 'none';
    }
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
}

function truncate(s, n = 10) {
  return s.length > n ? s.slice(0, n) + '…' : s;
}

function populateSelectors(data) {
  const layerSelect = document.getElementById('layer');
  const headSelect = document.getElementById('head');

  if (layerSelect.dataset.numLayers !== String(data.num_layers)) {
    layerSelect.textContent = '';
    for (let i = 0; i < data.num_layers; i++) {
      const opt = document.createElement('option');
      opt.value = String(i);
      opt.textContent = `${i + 1} / ${data.num_layers}`;
      layerSelect.appendChild(opt);
    }
    layerSelect.dataset.numLayers = String(data.num_layers);
  }
  layerSelect.value = String(data.layer);

  if (headSelect.dataset.numHeads !== String(data.num_heads)) {
    headSelect.textContent = '';
    const avgOpt = document.createElement('option');
    avgOpt.value = 'avg';
    avgOpt.textContent = 'média';
    headSelect.appendChild(avgOpt);
    for (let h = 0; h < data.num_heads; h++) {
      const opt = document.createElement('option');
      opt.value = String(h);
      opt.textContent = `cabeça ${h + 1}`;
      headSelect.appendChild(opt);
    }
    headSelect.dataset.numHeads = String(data.num_heads);
  }
}

async function run() {
  const prompt = document.getElementById('prompt').value;
  const resultArea = document.getElementById('result-area');
  if (!prompt.trim()) return;

  const btn = document.getElementById('btn-run');
  btn.disabled = true;
  try {
    const body = { prompt };
    if (selectedLayer !== null) body.layer = selectedLayer;
    if (selectedHead !== 'avg') body.head = Number(selectedHead);

    const data = await apiPost('/attention', body);
    populateSelectors(data);
    drawHeatmap(resultArea, data);
  } catch (e) {
    showError(resultArea, `Erro: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('btn-run').addEventListener('click', run);
document.getElementById('layer').addEventListener('change', (e) => {
  selectedLayer = Number(e.target.value);
  run();
});
document.getElementById('head').addEventListener('change', (e) => {
  selectedHead = e.target.value;
  run();
});
