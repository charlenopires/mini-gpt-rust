import { apiGet, apiPost, wsUrl, refreshStatusPill, formatTime } from '/js/api.js';

refreshStatusPill();

const lossPoints = []; // {step, loss}
let chartCanvas = null;
let chartCtx = null;

const readouts = {
  epoch: document.querySelector('#ro-epoch .readout__value'),
  step: document.querySelector('#ro-step .readout__value'),
  loss: document.querySelector('#ro-loss .readout__value'),
  best: document.querySelector('#ro-best .readout__value'),
  ppl: document.querySelector('#ro-ppl .readout__value'),
};

const pipelineStages = document.querySelectorAll('.pipeline__stage');

function setPipelineStage(stage) {
  pipelineStages.forEach((el) => {
    if (el.dataset.stage === stage) {
      el.dataset.state = 'active';
    } else if (el.dataset.state === 'active') {
      el.dataset.state = 'done';
    }
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
  const width = 560, height = 280;
  chartCanvas.width = width * dpr;
  chartCanvas.height = height * dpr;
  chartCanvas.style.height = `${height}px`;
  box.appendChild(chartCanvas);
  area.appendChild(box);
  chartCtx = chartCanvas.getContext('2d');
  chartCtx.scale(dpr, dpr);
}

function drawChart() {
  if (!chartCtx || lossPoints.length === 0) return;
  const width = 560, height = 280, pad = 36;
  const ctx = chartCtx;
  ctx.clearRect(0, 0, width, height);

  const losses = lossPoints.map((p) => p.loss);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const span = Math.max(maxLoss - minLoss, 1e-3);
  const maxStep = lossPoints[lossPoints.length - 1].step || 1;

  // Grade recessiva
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (i / 4) * (height - 2 * pad);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  const toPx = (p) => [
    pad + (p.step / maxStep) * (width - 2 * pad),
    pad + (1 - (p.loss - minLoss) / span) * (height - 2 * pad),
  ];

  ctx.strokeStyle = '#2fa86b';
  ctx.lineWidth = 2;
  ctx.beginPath();
  lossPoints.forEach((p, i) => {
    const [x, y] = toPx(p);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Labels de eixo (min/max loss)
  ctx.fillStyle = '#8890a0';
  ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'left';
  ctx.fillText(maxLoss.toFixed(3), 4, pad + 4);
  ctx.fillText(minLoss.toFixed(3), 4, height - pad + 4);
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
  line.appendChild(time);
  line.appendChild(kindEl);
  line.appendChild(msg);
  log.appendChild(line);
  while (log.children.length > 300) log.removeChild(log.firstChild);
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
      lossPoints.push({ step: ev.step, loss: ev.loss });
      drawChart();
      break;
    case 'EpochComplete':
      setPipelineStage('epoch');
      logEvent('EpochComplete', `época ${ev.epoch}/${ev.total_epochs} · loss médio ${ev.avg_loss.toFixed(4)}`);
      break;
    case 'SampleGenerated':
      logEvent('SampleGenerated', ev.text);
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
  ws.addEventListener('close', () => {
    setTimeout(connectWs, 2000);
  });
}

async function start() {
  const epochs = Number(document.getElementById('epochs').value) || 30;
  const btn = document.getElementById('btn-start');
  btn.disabled = true;
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
