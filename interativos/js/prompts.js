// prompts.js — prompts-exemplo curados por fluxo, escolhidos no domínio do
// corpus (IA/tech) e do vocabulário real para tornar cada conceito legível
// apesar do modelo minúsculo. Clique num chip preenche o input da página.

export const PROMPT_SETS = {
  // Palavras com split de subpalavra claro (ex.: otimiza+ção).
  tokenization: [
    'otimização',
    'armazenamento',
    'aprendizado de máquina',
    'processamento de linguagem natural',
    'A inteligência artificial aprende com dados.',
  ],
  // Estrutura referencial curta, para os pesos de atenção "fazerem sentido".
  attention: [
    'A inteligência artificial aprende com dados.',
    'O modelo de linguagem processa o texto.',
    'Redes neurais são inspiradas no cérebro.',
    'O mecanismo de atenção pesa cada token.',
  ],
  // Mistura de palavras relacionadas e não, para ver a vizinhança se formar.
  embeddings: [
    'modelo rede neurônio dados atenção token Brasil linguagem',
    'inteligência artificial aprendizado máquina',
    'gato cachorro casa cidade número',
  ],
  // Prefixos de frases frequentes → melhor comportamento do próximo token.
  previsao: [
    'A inteligência artificial',
    'O aprendizado de máquina',
    'Redes neurais',
    'Tokens são',
    'O mecanismo de atenção',
    'GPT significa',
  ],
  // O que o modelo continua de forma menos ruim.
  inference: [
    'A inteligência artificial está',
    'O aprendizado de máquina permite',
    'GPT significa',
    'Redes neurais artificiais',
    'O Brasil é',
  ],
};

/**
 * Renderiza uma fileira de chips clicáveis.
 * @param {HTMLElement} containerEl
 * @param {string[]} prompts
 * @param {(prompt:string)=>void} onPick
 */
export function mountPromptChips(containerEl, prompts, onPick) {
  containerEl.classList.add('prompt-chips');
  containerEl.innerHTML = '';

  const legenda = document.createElement('span');
  legenda.className = 'prompt-chips__label';
  legenda.textContent = 'experimente:';
  containerEl.appendChild(legenda);

  for (const p of prompts) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'prompt-chip';
    b.textContent = p;
    b.title = p;
    b.addEventListener('click', () => onPick(p));
    containerEl.appendChild(b);
  }
}
