// inference.js — geração autoregressiva via SSE. Agora cada evento traz também
// os top-k candidatos daquele passo (com o token sorteado marcado), então dá
// para VER a distribuição de onde cada token saiu — e que amostrar não é o
// mesmo que pegar sempre o mais provável.

import { refreshStatusPill, showError, requireModel } from '/js/api.js';
import { attachGlossary } from '/js/glossary.js';
import { PROMPT_SETS, mountPromptChips } from '/js/prompts.js';
import { barRow } from '/js/trace.js';

refreshStatusPill();
attachGlossary(document.body);

const maxTokensInput = document.getElementById('max-tokens');
const temperatureInput = document.getElementById('temperature');
const stepBars = document.getElementById('step-bars');
const notice = document.getElementById('notice');
const btn = document.getElementById('btn-run');

maxTokensInput.addEventListener('input', () => {
  document.getElementById('max-tokens-value').textContent = maxTokensInput.value;
});
temperatureInput.addEventListener('input', () => {
  document.getElementById('temperature-value').textContent = temperatureInput.value;
});

mountPromptChips(document.getElementById('prompt-chips'), PROMPT_SETS.inference, (p) => {
  document.getElementById('prompt').value = p;
});

let currentSource = null;

function renderStepBars(top, chosen) {
  if (!top || !top.length) return;
  stepBars.innerHTML = '';
  const bars = document.createElement('div');
  bars.className = 'prob-bars';
  const max = Math.max(...top.map((t) => t.prob), 1e-9);
  top.forEach((t) => bars.appendChild(barRow(t.token, `${(t.prob * 100).toFixed(1)}%`, t.prob / max, t.id === chosen)));
  stepBars.appendChild(bars);
}

function run() {
  const prompt = document.getElementById('prompt').value;
  const resultArea = document.getElementById('result-area');
  if (!prompt.trim()) return;

  if (currentSource) currentSource.close();

  resultArea.textContent = '';
  stepBars.innerHTML = '<div class="empty-hint">gerando…</div>';
  const pre = document.createElement('div');
  pre.className = 'generated-text';
  const promptSpan = document.createElement('span');
  promptSpan.textContent = prompt;
  promptSpan.style.color = 'var(--ink-dim)';
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  pre.appendChild(promptSpan);
  pre.appendChild(cursor);
  resultArea.appendChild(pre);

  btn.disabled = true;

  const params = new URLSearchParams({
    prompt,
    max_tokens: maxTokensInput.value,
    temperature: temperatureInput.value,
    top_k: '8',
  });
  const source = new EventSource(`/api/generate?${params.toString()}`);
  currentSource = source;

  source.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data);
      const tokenSpan = document.createElement('span');
      tokenSpan.textContent = data.text;
      pre.insertBefore(tokenSpan, cursor);
      renderStepBars(data.top, data.chosen);
    } catch { /* ignora frame inválido */ }
  };

  source.addEventListener('done', () => {
    cursor.remove();
    source.close();
    currentSource = null;
    btn.disabled = false;
  });

  source.onerror = () => {
    if (currentSource !== source) return;
    cursor.remove();
    source.close();
    currentSource = null;
    btn.disabled = false;
    if (!pre.textContent.trim().length || pre.textContent === prompt) {
      stepBars.innerHTML = '';
      showError(resultArea, 'Erro ao gerar — verifique se há um modelo carregado (treine em /treinamento).');
    }
  };
}

btn.addEventListener('click', run);
requireModel(notice, btn);
