import { apiGet, apiPost, refreshStatusPill, showError } from '/js/api.js';

refreshStatusPill();

async function loadCorpusStats() {
  try {
    const stats = await apiGet('/corpus/stats');
    document.getElementById('c-chars').innerText = stats.chars.toLocaleString('pt-BR');
    document.getElementById('c-lines').innerText = stats.lines.toLocaleString('pt-BR');
    document.getElementById('c-words').innerText = stats.words.toLocaleString('pt-BR');
    document.getElementById('c-tokens').innerText = stats.tokens ? stats.tokens.toLocaleString('pt-BR') : '—';
  } catch (e) {
    showError(document.getElementById('result-area'), `Erro ao carregar estatísticas do corpus: ${e.message}`);
  }
}

function renderTokens(container, tokens) {
  container.textContent = '';
  const stream = document.createElement('div');
  stream.className = 'token-stream';
  for (const t of tokens) {
    const chip = document.createElement('span');
    chip.className = 'token-chip';
    const text = document.createElement('span');
    text.textContent = t.text;
    const id = document.createElement('span');
    id.className = 'id';
    id.textContent = `#${t.id}`;
    chip.appendChild(text);
    chip.appendChild(id);
    stream.appendChild(chip);
  }
  container.appendChild(stream);

  const summary = document.createElement('p');
  summary.className = 'hint';
  summary.style.marginTop = 'var(--sp-4)';
  summary.textContent = `${tokens.length} tokens`;
  container.appendChild(summary);
}

async function tokenize() {
  const text = document.getElementById('input-text').value;
  const resultArea = document.getElementById('result-area');
  if (!text.trim()) return;

  const btn = document.getElementById('btn-tokenize');
  btn.disabled = true;
  try {
    const res = await apiPost('/tokenize', { text });
    renderTokens(resultArea, res.tokens);
  } catch (e) {
    showError(resultArea, `Erro ao tokenizar: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('btn-tokenize').addEventListener('click', tokenize);
loadCorpusStats();
tokenize();
