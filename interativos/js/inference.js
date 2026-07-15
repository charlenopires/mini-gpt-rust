import { refreshStatusPill, showError } from '/js/api.js';

refreshStatusPill();

const maxTokensInput = document.getElementById('max-tokens');
const temperatureInput = document.getElementById('temperature');
maxTokensInput.addEventListener('input', () => {
  document.getElementById('max-tokens-value').textContent = maxTokensInput.value;
});
temperatureInput.addEventListener('input', () => {
  document.getElementById('temperature-value').textContent = temperatureInput.value;
});

let currentSource = null;

function run() {
  const prompt = document.getElementById('prompt').value;
  const resultArea = document.getElementById('result-area');
  if (!prompt.trim()) return;

  if (currentSource) currentSource.close();

  resultArea.textContent = '';
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

  const btn = document.getElementById('btn-run');
  btn.disabled = true;

  const params = new URLSearchParams({
    prompt,
    max_tokens: maxTokensInput.value,
    temperature: temperatureInput.value,
  });
  const source = new EventSource(`/api/generate?${params.toString()}`);
  currentSource = source;

  source.onmessage = (msg) => {
    try {
      const { text } = JSON.parse(msg.data);
      const tokenSpan = document.createElement('span');
      tokenSpan.textContent = text;
      pre.insertBefore(tokenSpan, cursor);
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
      showError(resultArea, 'Erro ao gerar — verifique se há um modelo carregado (treine em /treinamento).');
    }
  };
}

document.getElementById('btn-run').addEventListener('click', run);
