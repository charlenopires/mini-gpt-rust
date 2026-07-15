// explain.js — callouts explicativos e fórmulas renderizadas como HTML/CSS
// (sem MathJax/KaTeX; nada de CDN, tudo self-contained). `explain(...)`
// devolve um <details> que o chamador insere onde quiser no fluxo.

/**
 * Cria um callout expansível "por quê / como funciona".
 * @param {string} title    texto do resumo (sempre visível)
 * @param {string} bodyHtml HTML do corpo (aceita fórmulas e <span> de glossário)
 * @param {{open?:boolean}} opts
 * @returns {HTMLDetailsElement}
 */
export function explain(title, bodyHtml, opts = {}) {
  const details = document.createElement('details');
  details.className = 'explain';
  if (opts.open) details.open = true;

  const summary = document.createElement('summary');
  summary.className = 'explain__summary';
  summary.innerHTML = `<span class="explain__icon">?</span><span>${title}</span>`;

  const body = document.createElement('div');
  body.className = 'explain__body';
  body.innerHTML = bodyHtml;

  details.append(summary, body);
  return details;
}

/** Fração renderizada com barra (numerador sobre denominador). */
export function frac(num, den) {
  return `<span class="m-frac"><span class="m-num">${num}</span><span class="m-den">${den}</span></span>`;
}

/** Envolve uma fórmula em um "chip" monoespaçado inline. */
export function m(html) {
  return `<span class="m">${html}</span>`;
}

// Fórmulas reais usadas nas páginas — strings HTML prontas para innerHTML.
export const FORMULAS = {
  softmax: m(`softmax(z<sub>i</sub>) = ${frac('e<sup>z<sub>i</sub></sup>', '&Sigma;<sub>j</sub> e<sup>z<sub>j</sub></sup>')}`),
  attention: m(`Aten&ccedil;&atilde;o(Q,K,V) = softmax(${frac('Q &middot; K<sup>T</sup>', '&radic;d<sub>k</sub>')}) &middot; V`),
  crossEntropy: m('L = &minus;log p(token correto)'),
  perplexity: m('PPL = e<sup>L</sup>'),
  temperature: m('p<sub>i</sub> = softmax(z<sub>i</sub> / T)'),
  cosine: m(`cos(a,b) = ${frac('a &middot; b', '&#8214;a&#8214; &middot; &#8214;b&#8214;')}`),
};
