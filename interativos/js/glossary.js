// glossary.js — termos de ML/redes neurais definidos em pt-BR, com um popover
// compartilhado. `glossaryTerm(...)` marca uma palavra no texto; um único
// `attachGlossary(root)` por página cuida de mostrar/esconder a definição
// (hover, foco de teclado ou toque; Esc fecha).

import { escapeHtml } from '/js/api.js';

// term (chave, kebab-case) → definição de 1–2 frases. Definições curtas e
// concretas, ancoradas neste modelo minúsculo quando faz diferença.
export const GLOSSARY = {
  token: 'A menor unidade que o modelo lê — pode ser uma palavra, um pedaço de palavra (subpalavra) ou um caractere. O modelo nunca vê letras soltas, só a sequência de IDs de tokens.',
  bpe: 'Byte Pair Encoding: algoritmo que descobre, a partir do corpus, quais pares de símbolos aparecem juntos com mais frequência e os funde em subpalavras. É assim que o vocabulário de 1000 tokens é construído aqui.',
  subpalavra: 'Um pedaço de palavra que virou um token só (ex.: "otimiza" + "ção"). Permite representar palavras raras juntando pedaços já conhecidos.',
  vocabulario: 'O conjunto de todos os tokens que o modelo conhece (aqui, 1000). Cada token tem um índice fixo — o seu "id".',
  embedding: 'O vetor de números (aqui, 128 dimensões) que representa um token. Tokens de significado parecido tendem a ter vetores próximos. É a primeira coisa que o modelo aprende.',
  posicional: 'Codificação de posição: um segundo vetor somado ao embedding para dizer ONDE o token está na sequência. Sem ele o modelo não distinguiria "o gato" de "gato o".',
  qkv: 'Query, Key e Value: três projeções de cada token. A Query pergunta "de que eu preciso?", a Key responde "eu ofereço isto" e o Value carrega a informação de fato transportada.',
  atencao: 'O mecanismo pelo qual cada token decide quanto "olhar" para os outros: compara Queries com Keys, normaliza com softmax e usa os pesos para combinar os Values.',
  cabeca: 'Uma "cabeça" de atenção é uma cópia independente do mecanismo, com suas próprias projeções Q/K/V. Várias cabeças em paralelo captam relações diferentes (ex.: sintaxe vs. concordância).',
  'mascara-causal': 'Máscara triangular que impede um token de olhar para tokens futuros. Garante que a previsão da posição i use só o que veio antes — essencial para gerar da esquerda para a direita.',
  softmax: 'Transforma uma lista de números quaisquer (logits) em probabilidades que somam 1: exponencia cada valor e divide pela soma. Realça o maior sem zerar os outros.',
  logits: 'As "notas" brutas que o modelo dá a cada token do vocabulário, antes do softmax. Quanto maior o logit, mais o modelo aposta naquele token como o próximo.',
  temperatura: 'Divisor aplicado aos logits antes do softmax. T baixo (→0) agudiza a distribuição (mais previsível); T alto (→2) a achata (mais criativo e aleatório).',
  perplexidade: 'e elevado à loss. Intuição: entre quantos tokens o modelo hesita, em média. Perplexidade 10 ≈ "tão indeciso quanto escolher entre 10 opções". Menor é melhor.',
  'cross-entropy': 'A função de perda (loss) do treino: mede quão baixa foi a probabilidade que o modelo deu ao token que de fato veio. Menos surpresa = menor loss.',
  adam: 'O otimizador que ajusta os pesos a cada passo. Usa médias móveis do gradiente para dar passos maiores onde é seguro e menores onde é instável.',
  backprop: 'Backpropagation: calcula, para cada peso, o quanto ele contribuiu para o erro — os gradientes. O otimizador usa isso para corrigir os pesos.',
  checkpoint: 'Um instantâneo salvo do modelo (pesos + configuração + tokenizador). Permite recarregar exatamente o modelo treinado depois, sem treinar de novo.',
  pca: 'Análise de Componentes Principais: reduz vetores de 128 dimensões para 2, mantendo as direções de maior variação. É como projetamos os embeddings num gráfico plano.',
  cosseno: 'Similaridade de cosseno: mede o ângulo entre dois vetores (não o tamanho). 1 = mesma direção (muito parecidos), 0 = perpendiculares (sem relação).',
  forward: 'Forward pass: a passagem dos dados pela rede, da entrada (tokens) até a saída (logits). É o que produz uma previsão; o treino a compara com a resposta certa.',
  autoregressivo: 'Geração em que cada token produzido é reinserido como entrada para prever o próximo. O texto cresce um token de cada vez, sempre condicionado ao que já saiu.',
};

/**
 * Marca uma palavra como termo do glossário (para innerHTML).
 * @param {string} term  chave em GLOSSARY (ex.: "softmax")
 * @param {string} [label] texto visível (default = term)
 */
export function glossaryTerm(term, label) {
  const text = escapeHtml(label ?? term);
  return `<span class="glossary-term" data-term="${escapeHtml(term)}" tabindex="0" role="button">${text}</span>`;
}

/** Liga um único popover a todos os `.glossary-term` dentro de `rootEl`. */
export function attachGlossary(rootEl = document.body) {
  let pop = document.querySelector('.glossary-pop');
  if (!pop) {
    pop = document.createElement('div');
    pop.className = 'glossary-pop';
    pop.setAttribute('role', 'tooltip');
    document.body.appendChild(pop);
    pop.addEventListener('mouseenter', () => clearTimeout(hideTimer));
    pop.addEventListener('mouseleave', () => hide());
  }
  let hideTimer = null;

  function show(el) {
    const def = GLOSSARY[el.dataset.term];
    if (!def) return;
    clearTimeout(hideTimer);
    // Nome do termo via textContent (pode vir de um token do usuário);
    // a definição é conteúdo estático confiável.
    pop.innerHTML = '';
    const termEl = document.createElement('span');
    termEl.className = 'glossary-pop__term';
    termEl.textContent = el.textContent;
    const defEl = document.createElement('div');
    defEl.innerHTML = def;
    pop.append(termEl, defEl);

    pop.dataset.visible = 'true';
    const r = el.getBoundingClientRect();
    pop.style.top = `${r.bottom + 8}px`;
    pop.style.left = `${r.left}px`;
    requestAnimationFrame(() => {
      const pr = pop.getBoundingClientRect();
      if (pr.right > window.innerWidth - 12) {
        pop.style.left = `${Math.max(12, window.innerWidth - pr.width - 12)}px`;
      }
      if (pr.bottom > window.innerHeight - 12) {
        pop.style.top = `${Math.max(12, r.top - pr.height - 8)}px`;
      }
    });
  }
  function hide() {
    hideTimer = setTimeout(() => { pop.dataset.visible = 'false'; }, 80);
  }

  rootEl.addEventListener('mouseover', (e) => {
    const el = e.target.closest?.('.glossary-term');
    if (el && rootEl.contains(el)) show(el);
  });
  rootEl.addEventListener('mouseout', (e) => {
    if (e.target.closest?.('.glossary-term')) hide();
  });
  rootEl.addEventListener('focusin', (e) => {
    const el = e.target.closest?.('.glossary-term');
    if (el) show(el);
  });
  rootEl.addEventListener('focusout', (e) => {
    if (e.target.closest?.('.glossary-term')) hide();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') pop.dataset.visible = 'false';
  });
}
