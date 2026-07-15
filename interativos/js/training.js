// training.js — treino real transmitido ao vivo por WebSocket. Além da curva
// de loss e do log de eventos, agora: hover na curva (loss/perplexidade/melhor
// loss por ponto), um marcador da melhor loss, um painel de amostras por época
// (o aprendizado ficando visível) e callouts explicando cada etapa do laço.

import { apiGet, apiPost, wsUrl, refreshStatusPill, formatTime } from '/js/api.js';
import { explain, FORMULAS } from '/js/explain.js';
import { glossaryTerm, attachGlossary } from '/js/glossary.js';
import { createTooltip, createCanvasTrace } from '/js/trace.js';

refreshStatusPill();
attachGlossary(document.body);

const lossPoints = []; // { step, loss, ppl, best }
let chartCanvas = null;
let chartCtx = null;
let chartPix = [];     // posições em px de cada ponto, para o hover
const tooltip = createTooltip();
const CHART_W = 560, CHART_H = 280, PAD = 36;

const readouts = {
  epoch: document.querySelector('#ro-epoch .readout__value'),
  step: document.querySelector('#ro-step .readout__value'),
  loss: document.querySelector('#ro-loss .readout__value'),
  best: document.querySelector('#ro-best .readout__value'),
  ppl: document.querySelector('#ro-ppl .readout__value'),
};

const pipelineStages = document.querySelectorAll('.stage-flow__item');
const samplesEl = document.getElementById('samples');

// Callout fixo explicando o laço de treino.
document.getElementById('pipeline-explain').appendChild(
  explain('O que acontece em cada passo do laço',
    `<p>Para cada lote (batch): o modelo faz um ${glossaryTerm('forward', 'forward pass')} e prevê o próximo token de cada posição; ` +
    `a ${glossaryTerm('cross-entropy', 'cross-entropy')} mede o quanto ele errou; o ${glossaryTerm('backprop', 'backward pass')} ` +
    `calcula os gradientes; e o ${glossaryTerm('adam', 'Adam')} ajusta os pesos um pouquinho na direção certa. Repetido milhares de vezes, ` +
    `a loss cai.</p>` +
    `<p>${FORMULAS.crossEntropy} &nbsp; · &nbsp; ${FORMULAS.perplexity} — a ${glossaryTerm('perplexidade', 'perplexidade')} é ` +
    `"entre quantos tokens ele hesita, em média". Neste modelo minúsculo a loss fica alta de propósito: o valor é ver o mecanismo funcionando.</p>`));

function setPipelineStage(stage) {
  pipelineStages.forEach((el) => {
    if (el.dataset.stage === stage) el.dataset.state = 'active';
    else if (el.dataset.state === 'active') el.dataset.state = 'done';
  });
}

function ensureChart() {
  const area = document.getElementById('chart-area');
  if (chartCanvas) return;
  area.textContent = '';
  const box = document.createElement('div');
  box.className = 'canvas-box';
  chartCanvas = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  chartCanvas.width = CHART_W * dpr;
  chartCanvas.height = CHART_H * dpr;
  chartCanvas.style.height = `${CHART_H}px`;
  box.appendChild(chartCanvas);
  area.appendChild(box);
  chartCtx = chartCanvas.getContext('2d');
  chartCtx.scale(dpr, dpr);

  createCanvasTrace(chartCanvas, {
    hitTest: (px) => {
      let best = null, bestDx = 14;
      for (let i = 0; i < chartPix.length; i++) {
        const dx = Math.abs(chartPix[i].x - px);
        if (dx < bestDx) { bestDx = dx; best = i; }
      }
      return best != null ? best : null;
    },
    onHover: (i, ev) => {
      const p = lossPoints[i];
      tooltip.show(ev.clientX, ev.clientY,
        `step ${p.step} · loss ${p.loss.toFixed(3)} · ppl ${p.ppl.toFixed(1)} · melhor ${p.best.toFixed(3)}`);
    },
    onLeave: () => tooltip.hide(),
  });
}

function drawChart() {
  if (!chartCtx || lossPoints.length === 0) return;
  const ctx = chartCtx;
  ctx.clearRect(0, 0, CHART_W, CHART_H);

  const losses = lossPoints.map((p) => p.loss);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const span = Math.max(maxLoss - minLoss, 1e-3);
  const maxStep = lossPoints[lossPoints.length - 1].step || 1;
  const best = lossPoints[lossPoints.length - 1].best;

  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = PAD + (i / 4) * (CHART_H - 2 * PAD);
    ctx.beginPath();
    ctx.moveTo(PAD, y);
    ctx.lineTo(CHART_W - PAD, y);
    ctx.stroke();
  }

  const toPx = (p) => [
    PAD + (p.step / maxStep) * (CHART_W - 2 * PAD),
    PAD + (1 - (p.loss - minLoss) / span) * (CHART_H - 2 * PAD),
  ];

  // Marcador da melhor loss (linha tracejada ember).
  if (best >= minLoss && best <= maxLoss) {
    const yBest = PAD + (1 - (best - minLoss) / span) * (CHART_H - 2 * PAD);
    ctx.save();
    ctx.strokeStyle = 'rgba(232,97,28,0.55)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(PAD, yBest);
    ctx.lineTo(CHART_W - PAD, yBest);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = '#e8611c';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`melhor ${best.toFixed(3)}`, CHART_W - PAD, yBest - 4);
  }

  ctx.strokeStyle = '#2fa86b';
  ctx.lineWidth = 2;
  ctx.beginPath();
  chartPix = lossPoints.map((p) => {
    const [x, y] = toPx(p);
    return { x, y };
  });
  chartPix.forEach((pt, i) => {
    if (i === 0) ctx.moveTo(pt.x, pt.y);
    else ctx.lineTo(pt.x, pt.y);
  });
  ctx.stroke();

  ctx.fillStyle = '#8890a0';
  ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'left';
  ctx.fillText(maxLoss.toFixed(3), 4, PAD + 4);
  ctx.fillText(minLoss.toFixed(3), 4, CHART_H - PAD + 4);
}

function logEvent(kind, message) {
  const log = document.getElementById('event-log');
  const line = document.createElement('div');
  line.className = 'event-log__line';
  const time = document.createElement('span');
  time.className = 'event-log__time';
  time.textContent = formatTime();
  const kindEl = document.createElement('span');
  kindEl.className = 'event-log__kind';
  kindEl.dataset.kind = kind;
  kindEl.textContent = kind;
  const msg = document.createElement('span');
  msg.className = 'event-log__msg';
  msg.textContent = message;
  line.append(time, kindEl, msg);
  log.appendChild(line);
  while (log.children.length > 300) log.removeChild(log.firstChild);
}

function addSample(epoch, text) {
  if (samplesEl.querySelector('.empty-hint')) samplesEl.textContent = '';
  const row = document.createElement('div');
  row.className = 'event-log__line';
  const ep = document.createElement('span');
  ep.className = 'event-log__kind';
  ep.textContent = `época ${epoch}`;
  const t = document.createElement('span');
  t.className = 'event-log__msg';
  t.textContent = text || '(vazio)';
  row.append(ep, t);
  samplesEl.insertBefore(row, samplesEl.firstChild);
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'Started':
      setPipelineStage('batches');
      logEvent('Started', `${ev.total_epochs} épocas · ${ev.total_steps} steps · batch=${ev.batch_size} · lr=${ev.learning_rate} · ${ev.total_tokens} tokens`);
      break;
    case 'ChunkingComplete':
      setPipelineStage('corpus');
      logEvent('ChunkingComplete', `${ev.chunk_count} chunks · tamanho médio ${ev.avg_chunk_size.toFixed(0)} tokens`);
      break;
    case 'Step':
      setPipelineStage('step');
      readouts.epoch.textContent = ev.epoch;
      readouts.step.textContent = `${ev.step} / ${ev.total_steps}`;
      readouts.loss.textContent = ev.loss.toFixed(4);
      readouts.best.textContent = ev.best_loss.toFixed(4);
      readouts.ppl.textContent = ev.perplexity.toFixed(2);
      ensureChart();
      lossPoints.push({ step: ev.step, loss: ev.loss, ppl: ev.perplexity, best: ev.best_loss });
      drawChart();
      break;
    case 'EpochComplete':
      setPipelineStage('epoch');
      logEvent('EpochComplete', `época ${ev.epoch}/${ev.total_epochs} · loss médio ${ev.avg_loss.toFixed(4)}`);
      break;
    case 'SampleGenerated':
      addSample(ev.epoch, ev.text);
      logEvent('SampleGenerated', `época ${ev.epoch}: ${ev.text}`);
      break;
    case 'Finished':
      setPipelineStage('finished');
      logEvent('Finished', `${ev.duration_secs.toFixed(1)}s · ${ev.tokens_per_sec.toFixed(0)} tokens/s · loss final ${ev.final_loss.toFixed(4)}`);
      document.getElementById('btn-start').disabled = false;
      refreshStatusPill();
      break;
    default:
      break;
  }
}

function connectWs() {
  const ws = new WebSocket(wsUrl('/train/ws'));
  ws.addEventListener('message', (msg) => {
    try { handleEvent(JSON.parse(msg.data)); } catch { /* ignora frame inválido */ }
  });
  ws.addEventListener('close', () => setTimeout(connectWs, 2000));
}

async function start() {
  const epochs = Number(document.getElementById('epochs').value) || 30;
  const btn = document.getElementById('btn-start');
  btn.disabled = true;
  // reinicia o estado visual para um novo treino
  lossPoints.length = 0;
  chartPix = [];
  try {
    await apiPost('/train/start', { epochs });
    logEvent('Started', 'requisição de treinamento enviada…');
  } catch (e) {
    logEvent('Error', e.message);
    btn.disabled = false;
  }
}

async function init() {
  try {
    const status = await apiGet('/train/status');
    if (status.running) document.getElementById('btn-start').disabled = true;
    for (const ev of status.recent_events) handleEvent(ev);
  } catch { /* servidor pode ainda não ter histórico */ }
  connectWs();
}

document.getElementById('btn-start').addEventListener('click', start);
init();
