import { apiPost, refreshStatusPill, showError } from '/js/api.js';

refreshStatusPill();

function drawScatter(container, points) {
  container.textContent = '';

  const width = 640;
  const height = 480;
  const pad = 32;

  const box = document.createElement('div');
  box.className = 'canvas-box';
  const canvas = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = `${height}px`;
  box.appendChild(canvas);
  container.appendChild(box);

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);

  const toPx = (p) => [
    pad + ((p.x - minX) / spanX) * (width - 2 * pad),
    height - pad - ((p.y - minY) / spanY) * (height - 2 * pad),
  ];

  // Eixos recessivos
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, height / 2); ctx.lineTo(width - pad, height / 2);
  ctx.moveTo(width / 2, pad); ctx.lineTo(width / 2, height - pad);
  ctx.stroke();

  const positions = points.map(toPx);

  // Pontos
  ctx.fillStyle = '#7b84e8';
  for (const [px, py] of positions) {
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Rótulos
  ctx.fillStyle = '#e7e9ec';
  ctx.font = '12px ui-monospace, monospace';
  ctx.textBaseline = 'bottom';
  points.forEach((p, i) => {
    const [px, py] = positions[i];
    ctx.fillText(p.token, px + 7, py - 4);
  });

  // Tooltip
  const tooltip = document.createElement('div');
  tooltip.style.cssText = 'position:fixed;pointer-events:none;background:#1c2029;border:1px solid rgba(255,255,255,0.16);' +
    'border-radius:4px;padding:4px 8px;font-family:ui-monospace,monospace;font-size:12px;color:#e7e9ec;display:none;z-index:50;';
  document.body.appendChild(tooltip);

  canvas.addEventListener('mousemove', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    let hit = null;
    for (let i = 0; i < positions.length; i++) {
      const [px, py] = positions[i];
      if (Math.hypot(px - mx, py - my) < 8) { hit = i; break; }
    }
    if (hit !== null) {
      tooltip.style.display = 'block';
      tooltip.style.left = `${ev.clientX + 12}px`;
      tooltip.style.top = `${ev.clientY + 12}px`;
      tooltip.textContent = `${points[hit].token} · (${points[hit].x.toFixed(2)}, ${points[hit].y.toFixed(2)})`;
    } else {
      tooltip.style.display = 'none';
    }
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
}

async function run() {
  const text = document.getElementById('input-text').value;
  const resultArea = document.getElementById('result-area');
  if (!text.trim()) return;

  const btn = document.getElementById('btn-run');
  btn.disabled = true;
  try {
    const data = await apiPost('/embeddings', { text });
    if (!data.points.length) {
      showError(resultArea, 'Nenhum token reconhecido nesse texto.');
      return;
    }
    drawScatter(resultArea, data.points);
  } catch (e) {
    showError(resultArea, `Erro: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('btn-run').addEventListener('click', run);
