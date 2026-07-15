// tokenization.js — mostra o caminho de letras a tokens em 4 estágios: texto
// cru → palavras → subpalavras (BPE) → IDs no vocabulário. Os tokens vêm do
// tokenizador BPE real (/api/tokenize); as palavras são o split por espaços.

import { apiGet, apiPost, refreshStatusPill, showError } from '/js/api.js';
import { createStepper } from '/js/stepper.js';
import { explain } from '/js/explain.js';
import { glossaryTerm, attachGlossary } from '/js/glossary.js';
import { PROMPT_SETS, mountPromptChips } from '/js/prompts.js';

refreshStatusPill();
attachGlossary(document.body);

const resultArea = document.getElementById('result-area');
const explainEl = document.getElementById('explain');

let lastText = '';
let lastTokens = null;

async function loadCorpusStats() {
  try {
    const stats = await apiGet('/corpus/stats');
    document.getElementById('c-chars').textContent = stats.chars.toLocaleString('pt-BR');
    document.getElementById('c-lines').textContent = stats.lines.toLocaleString('pt-BR');
    document.getElementById('c-words').textContent = stats.words.toLocaleString('pt-BR');
    document.getElementById('c-tokens').textContent = stats.tokens ? stats.tokens.toLocaleString('pt-BR') : '—';
  } catch (e) {
    showError(resultArea, `Erro ao carregar estatísticas do corpus: ${e.message}`);
  }
}

// ── renderizadores de cada estágio (escrevem em #result-area) ─────────────
function caption(text) {
  const p = document.createElement('p');
  p.className = 'hint';
  p.style.margin = '0 0 var(--sp-3)';
  p.textContent = text;
  return p;
}

function chipStream(items, withId) {
  const stream = document.createElement('div');
  stream.className = 'token-stream';
  items.forEach((it) => {
    const chip = document.createElement('span');
    chip.className = 'token-chip';
    const text = document.createElement('span');
    text.textContent = withId ? it.text : it;
    chip.appendChild(text);
    if (withId) {
      const id = document.createElement('span');
      id.className = 'id';
      id.textContent = `#${it.id}`;
      chip.appendChild(id);
    }
    stream.appendChild(chip);
  });
  return stream;
}

function showRaw() {
  if (!lastTokens) return needTokens();
  const wrap = document.createElement('div');
  const box = document.createElement('div');
  box.className = 'generated-text';
  box.style.color = 'var(--ink-dim)';
  box.textContent = lastText;
  wrap.append(caption('O texto cru, como você digitou. O modelo não lê letras — precisa virar números.'), box);
  setResult(wrap);
}

function showWords() {
  if (!lastTokens) return needTokens();
  const words = lastText.split(/\s+/).filter(Boolean);
  const wrap = document.createElement('div');
  wrap.append(caption(`${words.length} palavras, separadas por espaço. É o ponto de partida do BPE.`), chipStream(words, false));
  setResult(wrap);
}

function showSubwords() {
  if (!lastTokens) return needTokens();
  const texts = lastTokens.map((t) => t.text);
  const wrap = document.createElement('div');
  wrap.append(
    caption(`O BPE quebrou tudo em ${texts.length} subpalavras (inclui <BOS>/<EOS>). Palavras raras viram pedaços conhecidos.`),
    chipStream(texts, false),
  );
  setResult(wrap);
}

function showIds() {
  if (!lastTokens) return needTokens();
  const wrap = document.createElement('div');
  wrap.append(
    caption(`Cada subpalavra é um índice no vocabulário. Esta sequência de ${lastTokens.length} IDs é o que o modelo realmente recebe.`),
    chipStream(lastTokens, true),
  );
  setResult(wrap);
}

function needTokens() { setResult(hint('Clique em "Tokenizar" para começar.')); }
function hint(msg) { const d = document.createElement('div'); d.className = 'empty-hint'; d.textContent = msg; return d; }
function setResult(node) { resultArea.innerHTML = ''; resultArea.appendChild(node); }

const STAGES = [
  { short: 'texto', label: '1 · texto cru', onEnter: showRaw },
  { short: 'palavras', label: '2 · palavras (split por espaço)', onEnter: showWords },
  { short: 'subpalavras', label: '3 · subpalavras (BPE)', onEnter: showSubwords },
  { short: 'IDs', label: '4 · IDs no vocabulário', onEnter: showIds },
];
let stepper = null;

async function tokenize() {
  const text = document.getElementById('input-text').value;
  if (!text.trim()) return;
  const btn = document.getElementById('btn-tokenize');
  btn.disabled = true;
  try {
    const res = await apiPost('/tokenize', { text });
    lastText = text;
    lastTokens = res.tokens;
    if (stepper.current() === 3) showIds();
    else stepper.goTo(3); // vai direto ao resultado final; o usuário volta para ver os passos
  } catch (e) {
    showError(resultArea, `Erro ao tokenizar: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

explainEl.appendChild(explain('O que é BPE, afinal?',
  `<p>${glossaryTerm('bpe', 'Byte Pair Encoding')} aprende, a partir do corpus, quais pares de símbolos aparecem juntos com mais ` +
  `frequência e os funde em ${glossaryTerm('subpalavra', 'subpalavras')}. Começando por caracteres, ele repete essa fusão até ` +
  `montar um ${glossaryTerm('vocabulario', 'vocabulário')} de 1000 tokens. Palavras comuns viram um token só; raras, vários pedaços.</p>` +
  `<p class="hint">Passe pelos passos acima (◂ ▸) para ver a mesma frase em cada nível de granularidade.</p>`));

mountPromptChips(document.getElementById('prompt-chips'), PROMPT_SETS.tokenization, (p) => {
  document.getElementById('input-text').value = p;
});
stepper = createStepper(document.getElementById('stepper'), STAGES);
document.getElementById('btn-tokenize').addEventListener('click', tokenize);
loadCorpusStats();
tokenize();
