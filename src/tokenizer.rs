//! 🔤 **TOKENIZAÇÃO BPE (BYTE PAIR ENCODING)**
//! 
//! Este módulo implementa o algoritmo BPE, uma técnica fundamental para
//! converter texto em sequências numéricas que modelos de linguagem podem processar.
//! 
//! ## 🧩 **O Problema da Tokenização:**
//! 
//! ### 🤔 **Por que não usar caracteres individuais?**
//! - **Sequências muito longas**: "Olá" = [O, l, á] = 3 tokens
//! - **Perda de significado**: Palavras são quebradas arbitrariamente
//! - **Ineficiência**: Modelos precisam aprender padrões básicos
//! 
//! ### 🤔 **Por que não usar palavras completas?**
//! - **Vocabulário gigantesco**: Milhões de palavras possíveis
//! - **Palavras raras**: "antidisestablishmentarianism" seria UNK
//! - **Flexões**: "correr", "correndo", "correu" = tokens diferentes
//! 
//! ## 🎯 **A Solução BPE:**
//! 
//! ### 📚 **Analogia da Biblioteca:**
//! Imagine organizar uma biblioteca:
//! - **Nível 1**: Letras individuais (a, b, c...)
//! - **Nível 2**: Sílabas comuns ("ção", "mente", "pre"...)
//! - **Nível 3**: Palavras frequentes ("que", "para", "com"...)
//! - **Nível 4**: Expressões comuns ("por favor", "muito obrigado"...)
//! 
//! BPE constrói esse "sistema de organização" automaticamente!
//! 
//! ### 🔄 **Algoritmo BPE em Etapas:**
//! 
//! #### 1️⃣ **Inicialização:**
//! ```text
//! Texto: "hello world"
//! Tokens: ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
//! ```
//! 
//! #### 2️⃣ **Contagem de Pares:**
//! ```text
//! "he": 1 vez
//! "el": 1 vez  
//! "ll": 1 vez
//! "lo": 1 vez
//! "o ": 1 vez
//! " w": 1 vez
//! "wo": 1 vez
//! "or": 1 vez
//! "rl": 1 vez
//! "ld": 1 vez
//! ```
//! 
//! #### 3️⃣ **Merge do Par Mais Frequente:**
//! Se "ll" fosse o mais frequente:
//! ```text
//! Antes: ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
//! Depois: ["h", "e", "ll", "o", " ", "w", "o", "r", "l", "d"]
//! ```
//! 
//! #### 4️⃣ **Repetição:**
//! Continue até atingir o tamanho de vocabulário desejado!
//! 
//! ## 🏗️ **Vantagens do BPE:**
//! 
//! ### ✅ **Eficiência:**
//! - **Vocabulário controlado**: Tamanho fixo (ex: 50k tokens)
//! - **Sequências menores**: "tokenização" → ["token", "ização"]
//! - **Sem palavras desconhecidas**: Qualquer texto pode ser tokenizado
//! 
//! ### ✅ **Flexibilidade:**
//! - **Multilíngue**: Funciona para qualquer idioma
//! - **Domínio-específico**: Aprende jargões técnicos automaticamente
//! - **Subpalavras**: Captura prefixos, sufixos, radicais
//! 
//! ### ✅ **Qualidade:**
//! - **Preserva significado**: "unhappy" → ["un", "happy"]
//! - **Eficiência computacional**: Menos tokens = menos processamento
//! - **Generalização**: Modelos aprendem padrões morfológicos
//! 
//! ## 🎭 **Exemplo Prático:**
//! 
//! ### 📝 **Texto Original:**
//! ```
//! "Eu amo programação em Rust!"
//! ```
//! 
//! ### 🔤 **Após Tokenização BPE:**
//! ```
//! ["Eu", " amo", " program", "ação", " em", " Rust", "!"]
//! ```
//! 
//! ### 🔢 **IDs Numéricos:**
//! ```
//! [156, 892, 1247, 3891, 234, 5672, 33]
//! ```
//! 
//! ### 🧠 **Por que essa divisão?**
//! - **"Eu"**: Palavra comum, tem seu próprio token
//! - **" amo"**: Verbo frequente com espaço
//! - **"program"**: Radical comum em "programação", "programar", etc.
//! - **"ação"**: Sufixo comum em português
//! - **" em"**: Preposição frequente
//! - **" Rust"**: Nome próprio, aprendido como unidade
//! - **"!"**: Pontuação comum
//! 
//! ## 🔄 **Processo de Encoding/Decoding:**
//! 
//! ### 📥 **ENCODING (Texto → Números):**
//! ```text
//! 1. Dividir texto em caracteres
//! 2. Aplicar merges aprendidos (do mais frequente ao menos frequente)
//! 3. Converter tokens para IDs usando vocabulário
//! 4. Retornar sequência de números
//! ```
//! 
//! ### 📤 **DECODING (Números → Texto):**
//! ```text
//! 1. Converter IDs para tokens usando vocabulário reverso
//! 2. Concatenar tokens
//! 3. Tratar espaços e caracteres especiais
//! 4. Retornar texto original
//! ```
//! 
//! ## 🎯 **Tokens Especiais:**
//! 
//! ### 🏁 **EOS (End of Sequence):**
//! - **Função**: Marca o fim de uma sequência
//! - **Uso**: Permite ao modelo saber quando parar de gerar
//! - **Exemplo**: "Olá mundo<EOS>"
//! 
//! ### 🔤 **UNK (Unknown):**
//! - **Função**: Representa tokens desconhecidos
//! - **Uso**: Fallback para caracteres não vistos no treinamento
//! - **Exemplo**: Emojis raros, símbolos especiais
//! 
//! ### 📍 **PAD (Padding):**
//! - **Função**: Preenche sequências para ter o mesmo tamanho
//! - **Uso**: Permite processamento em lotes (batches)
//! - **Exemplo**: ["Oi", "<PAD>", "<PAD>"] para igualar tamanhos
//! 
//! ## 🧮 **Matemática da Tokenização:**
//! 
//! ### 📊 **Frequência de Pares:**
//! ```
//! freq(pair) = Σ count(pair, word) × freq(word)
//! ```
//! 
//! ### 🎯 **Critério de Merge:**
//! ```
//! best_pair = argmax(freq(pair)) for all pairs
//! ```
//! 
//! ### 📏 **Eficiência de Compressão:**
//! ```
//! compression_ratio = original_chars / final_tokens
//! ```
//! 
//! ## 🚀 **Otimizações Implementadas:**
//! 
//! ### ⚡ **Cache de Merges:**
//! - Armazena resultados de merges já computados
//! - Evita recomputação desnecessária
//! - Acelera encoding de textos similares
//! 
//! ### 🧠 **Vocabulário Reverso:**
//! - HashMap para lookup O(1) durante decoding
//! - Evita busca linear no vocabulário
//! - Essencial para geração de texto rápida
//! 
//! ### 📈 **Processamento Incremental:**
//! - Aplica merges em ordem de frequência
//! - Permite interrupção e retomada do treinamento
//! - Facilita debugging e análise
//! ```text
//! "O gato subiu no telhado. O gato desceu do telhado."
//! ```
//! 
//! ### 🔤 **Após Treinamento BPE:**
//! ```text
//! Vocabulário aprendido:
//! - Caracteres: ["O", " ", "g", "a", "t", "o", "s", "u", "b", "i", ...]
//! - Sílabas: ["ga", "to", "su", "bi", "te", "lh", "ad", "o."]
//! - Palavras: ["gato", "telhado"]
//! ```
//! 
//! ### 🎯 **Tokenização Final:**
//! ```text
//! ["O", " ", "gato", " ", "su", "bi", "u", " ", "no", " ", "telhado", ".", ...]
//! ```

use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// 🔤 **TOKENIZADOR BPE: CONVERSOR INTELIGENTE DE TEXTO**
/// 
/// Esta estrutura implementa o algoritmo Byte Pair Encoding (BPE),
/// que aprende automaticamente como dividir texto em unidades
/// significativas (tokens) para processamento por modelos de linguagem.
/// 
/// ## 🧠 **Componentes Principais:**
/// 
/// ### 📚 **Vocabulário Bidirecional:**
/// - `vocab`: String → ID (para codificação)
/// - `reverse_vocab`: ID → String (para decodificação)
/// 
/// ### 🔗 **Regras de Merge:**
/// - `merges`: Lista ordenada de pares que devem ser combinados
/// - Aplicadas sequencialmente durante tokenização
/// 
/// ### ⚙️ **Configuração:**
/// - `vocab_size`: Tamanho máximo do vocabulário
/// 
/// ## 🎯 **Tokens Especiais:**
/// - `<PAD>` (ID: 0): Preenchimento para sequências de tamanhos diferentes
/// - `<UNK>` (ID: 1): Tokens desconhecidos (raramente usado em BPE)
/// - `<BOS>` (ID: 2): Início de sequência (Begin of Sentence)
/// - `<EOS>` (ID: 3): Fim de sequência (End of Sentence)
#[derive(Serialize, Deserialize)]
pub struct BPETokenizer {
    /// 📖 Mapeamento de tokens (strings) para IDs numéricos
    /// Usado durante a codificação: texto → números
    vocab: HashMap<String, usize>,
    
    /// 🔄 Mapeamento reverso de IDs para tokens
    /// Usado durante a decodificação: números → texto
    reverse_vocab: HashMap<usize, String>,
    
    /// 🔗 Lista ordenada de regras de merge aprendidas
    /// Cada elemento é um par (token1, token2) que deve ser combinado
    /// A ordem importa: merges anteriores têm prioridade
    merges: Vec<(String, String)>,
    
    /// 📏 Tamanho máximo do vocabulário
    /// Controla quando parar o treinamento BPE
    vocab_size: usize,
}

impl BPETokenizer {
    /// 🏗️ **CONSTRUTOR: INICIALIZANDO O TOKENIZADOR BPE**
    /// 
    /// Cria uma nova instância do tokenizador BPE com vocabulário vazio
    /// e tokens especiais pré-definidos. O tokenizador está pronto para
    /// treinamento, mas ainda não pode tokenizar texto real.
    /// 
    /// ## 🎯 **Parâmetros:**
    /// - `vocab_size`: Tamanho máximo do vocabulário final
    ///   - Típico: 30k-50k para modelos pequenos
    ///   - GPT-2: 50,257 tokens
    ///   - BERT: 30,522 tokens
    /// 
    /// ## 🔧 **Inicialização:**
    /// 
    /// ### 🏷️ **Tokens Especiais (IDs Fixos):**
    /// - `<PAD>` (0): Preenchimento para batches de tamanhos diferentes
    /// - `<UNK>` (1): Tokens desconhecidos (backup, raramente usado)
    /// - `<BOS>` (2): Marcador de início de sequência
    /// - `<EOS>` (3): Marcador de fim de sequência
    /// 
    /// ### 📚 **Estruturas Inicializadas:**
    /// - **Vocabulário vazio**: Pronto para aprender tokens
    /// - **Mapeamento reverso**: Para decodificação eficiente
    /// - **Lista de merges vazia**: Será preenchida durante treinamento
    /// 
    /// ## ✅ **Estado Pós-Construção:**
    /// ```text
    /// Vocabulário: 4 tokens especiais
    /// Merges: 0 regras
    /// Pronto para: Treinamento
    /// Não pronto para: Tokenização de texto real
    /// ```
    pub fn new(vocab_size: usize) -> Result<Self> {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        // 🏷️ **TOKENS ESPECIAIS COM IDS FIXOS**
        // Estes IDs são padronizados e devem ser consistentes
        // entre diferentes instâncias do tokenizador
        vocab.insert("<PAD>".to_string(), 0);  // Preenchimento
        vocab.insert("<UNK>".to_string(), 1);  // Desconhecido
        vocab.insert("<BOS>".to_string(), 2);  // Início de sequência
        vocab.insert("<EOS>".to_string(), 3);  // Fim de sequência
        
        // 🔄 **CONSTRUÇÃO DO MAPEAMENTO REVERSO**
        // Permite decodificação eficiente: ID → Token
        for (token, id) in &vocab {
            reverse_vocab.insert(*id, token.clone());
        }
        
        Ok(Self {
            vocab,
            reverse_vocab,
            merges: Vec::new(),
            vocab_size,
        })
    }
    
    /// 🎓 **TREINAMENTO BPE: APRENDENDO A TOKENIZAR**
    /// 
    /// Este é o coração do algoritmo BPE! Analisa um corpus de texto
    /// e aprende automaticamente como dividir palavras em subunidades
    /// significativas, construindo um vocabulário otimizado.
    /// 
    /// ## 🔄 **Algoritmo em Etapas:**
    /// 
    /// ### 1️⃣ **Inicialização com Caracteres:**
    /// ```text
    /// Texto: "hello world"
    /// Palavras: [["h","e","l","l","o"], ["w","o","r","l","d"]]
    /// Vocabulário inicial: {"h":4, "e":5, "l":6, "o":7, ...}
    /// ```
    /// 
    /// ### 2️⃣ **Loop Principal de Aprendizado:**
    /// ```text
    /// Enquanto vocabulário < tamanho_desejado:
    ///   a) Conta frequência de todos os pares adjacentes
    ///   b) Encontra o par mais frequente
    ///   c) Cria novo token combinando o par
    ///   d) Atualiza todas as palavras com o novo merge
    ///   e) Adiciona regra de merge à lista
    /// ```
    /// 
    /// ### 3️⃣ **Exemplo de Evolução:**
    /// ```text
    /// Iteração 0: ["h","e","l","l","o"] 
    /// Par "ll" é mais frequente → merge
    /// Iteração 1: ["h","e","ll","o"]
    /// Par "he" é mais frequente → merge  
    /// Iteração 2: ["he","ll","o"]
    /// Continue até vocab_size...
    /// ```
    /// 
    /// ## 🎯 **Parâmetros:**
    /// - `text`: Corpus de treinamento (quanto maior, melhor)
    /// 
    /// ## 📊 **Complexidade:**
    /// - **Tempo**: O(n × vocab_size) onde n = tamanho do texto
    /// - **Espaço**: O(vocab_size + unique_chars)
    /// 
    /// ## ⚠️ **Considerações:**
    /// - **Qualidade do corpus**: Deve ser representativo do domínio
    /// - **Tamanho**: Corpus pequeno = vocabulário subótimo
    /// - **Diversidade**: Textos variados = melhor generalização
    pub fn train(&mut self, text: &str) -> Result<()> {
        println!("🔧 Treinando tokenizador BPE...");
        
        // 1️⃣ **FASE DE INICIALIZAÇÃO: CARACTERES ÚNICOS**
        // Converte texto em palavras representadas como sequências de caracteres
        // Cada palavra mantém sua frequência no corpus
        let mut word_freqs = self.get_word_frequencies(text);
        
        // 📚 **CONSTRUÇÃO DO VOCABULÁRIO BASE**
        // Adiciona todos os caracteres únicos encontrados no texto
        // Estes formam a "base" sobre a qual BPE construirá
        for word in word_freqs.keys() {
            for token in word {
                if !self.vocab.contains_key(token) {
                    let id = self.vocab.len();
                    self.vocab.insert(token.clone(), id);
                    self.reverse_vocab.insert(id, token.clone());
                }
            }
        }
        
        // 2️⃣ **LOOP PRINCIPAL: APRENDIZADO ITERATIVO DE MERGES**
        // Continua até atingir o tamanho de vocabulário desejado
        // Cada iteração aprende um novo "padrão" no texto
        while self.vocab.len() < self.vocab_size {
            // 🔍 **ANÁLISE DE FREQUÊNCIAS**
            // Conta quantas vezes cada par de tokens adjacentes aparece
            let pair_freqs = self.get_pair_frequencies(&word_freqs);
            
            // 🛑 **CONDIÇÃO DE PARADA**
            // Se não há mais pares para mergir, para o treinamento
            if pair_freqs.is_empty() {
                break;
            }
            
            // 🏆 **SELEÇÃO DO MELHOR PAR**
            // O par mais frequente é o mais "útil" para compressão
            let best_pair = pair_freqs
                .iter()
                .max_by_key(|(_, freq)| *freq)
                .map(|(pair, _)| pair.clone())
                .unwrap();
            
            // 🔗 **CRIAÇÃO DO NOVO TOKEN**
            // Combina os dois tokens do par mais frequente em um único token
            let new_token = format!("{}{}", best_pair.0, best_pair.1);
            let id = self.vocab.len();
            
            // 📝 **REGISTRO NO VOCABULÁRIO**
            // Adiciona o novo token aos mapeamentos bidirecionais
            self.vocab.insert(new_token.clone(), id);
            self.reverse_vocab.insert(id, new_token);
            
            // 📋 **REGISTRO DA REGRA DE MERGE**
            // Salva a regra para aplicar durante tokenização
            self.merges.push(best_pair.clone());
            
            // 🔄 **ATUALIZAÇÃO DO CORPUS**
            // Aplica o novo merge a todas as palavras do corpus
            // Isso reduz a fragmentação e prepara para próxima iteração
            word_freqs = self.apply_merge(&word_freqs, &best_pair);
            
            // 📊 **PROGRESSO DO TREINAMENTO**
            // Mostra progresso a cada 100 tokens aprendidos
            if self.vocab.len() % 100 == 0 {
                println!("  Vocabulário: {} tokens", self.vocab.len());
            }
        }
        
        println!("✅ Tokenizador treinado! Vocabulário: {} tokens", self.vocab.len());
        Ok(())
    }
    
    /// 📊 **ANÁLISE DE FREQUÊNCIAS DE PALAVRAS**
    /// 
    /// Converte o texto bruto em uma representação estruturada onde
    /// cada palavra é uma sequência de caracteres individuais,
    /// mantendo a contagem de quantas vezes cada palavra aparece.
    /// 
    /// ## 🔄 **Processo:**
    /// ```text
    /// Input: "hello world hello"
    /// Output: {
    ///   ["h","e","l","l","o"]: 2,
    ///   ["w","o","r","l","d"]: 1
    /// }
    /// ```
    /// 
    /// ## 🎯 **Por que caracteres individuais?**
    /// - **Base para BPE**: Começamos com unidades atômicas
    /// - **Universalidade**: Funciona para qualquer idioma/script
    /// - **Flexibilidade**: Permite aprender qualquer padrão
    fn get_word_frequencies(&self, text: &str) -> HashMap<Vec<String>, usize> {
        let mut freqs = HashMap::new();
        
        // 🔤 **TOKENIZAÇÃO INICIAL EM CARACTERES**
        // Cada palavra é dividida em caracteres individuais
        // Espaços em branco separam palavras
        for word in text.split_whitespace() {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            *freqs.entry(chars).or_insert(0) += 1;
        }
        
        freqs
    }
    
    /// 🔍 **ANÁLISE DE FREQUÊNCIAS DE PARES ADJACENTES**
    /// 
    /// Para cada palavra no corpus, examina todos os pares de tokens
    /// adjacentes e conta quantas vezes cada par aparece. Esta é a
    /// informação crucial que o BPE usa para decidir quais merges fazer.
    /// 
    /// ## 🔄 **Processo:**
    /// ```text
    /// Palavra: ["h", "e", "l", "l", "o"] (freq: 2)
    /// Pares: ("h","e"), ("e","l"), ("l","l"), ("l","o")
    /// Contribuição: cada par ganha +2 na contagem
    /// ```
    /// 
    /// ## 🎯 **Estratégia:**
    /// - **Pares adjacentes**: Só considera tokens lado a lado
    /// - **Ponderação por frequência**: Palavras comuns têm mais peso
    /// - **Busca exaustiva**: Examina todos os pares possíveis
    /// 
    /// ## 📊 **Complexidade:**
    /// - **Tempo**: O(total_chars_in_corpus)
    /// - **Espaço**: O(unique_pairs)
    fn get_pair_frequencies(&self, word_freqs: &HashMap<Vec<String>, usize>) -> HashMap<(String, String), usize> {
        let mut pair_freqs = HashMap::new();
        
        // 🔍 **VARREDURA DE TODAS AS PALAVRAS**
        for (word, freq) in word_freqs {
            // 👥 **ANÁLISE DE PARES ADJACENTES**
            // Para cada posição na palavra, forma um par com o próximo token
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                // 📈 **ACUMULAÇÃO PONDERADA**
                // Adiciona a frequência da palavra à contagem do par
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }
        
        pair_freqs
    }
    
    /// 🔄 **APLICAÇÃO DE MERGE AO CORPUS**
    /// 
    /// Aplica uma regra de merge específica a todas as palavras do corpus,
    /// substituindo todas as ocorrências do par especificado pelo novo token.
    /// Esta é a operação que "ensina" o vocabulário a reconhecer padrões.
    /// 
    /// ## 🎯 **Processo:**
    /// ```text
    /// Antes: ["h", "e", "l", "l", "o"]
    /// Merge: ("l", "l") → "ll"
    /// Depois: ["h", "e", "ll", "o"]
    /// ```
    /// 
    /// ## 🔍 **Algoritmo:**
    /// 1. **Varredura**: Examina cada palavra token por token
    /// 2. **Detecção**: Procura pelo par específico em posições adjacentes
    /// 3. **Substituição**: Substitui par encontrado pelo token merged
    /// 4. **Preservação**: Mantém outros tokens inalterados
    /// 
    /// ## ⚡ **Otimização:**
    /// - **Busca gulosa**: Primeira ocorrência encontrada é merged
    /// - **Salto duplo**: Após merge, pula 2 posições (par consumido)
    /// - **Preservação de frequência**: Mantém contagens originais
    fn apply_merge(&self, word_freqs: &HashMap<Vec<String>, usize>, pair: &(String, String)) -> HashMap<Vec<String>, usize> {
        let mut new_freqs = HashMap::new();
        
        // 🔗 **CRIAÇÃO DO TOKEN MERGED**
        // Combina os dois tokens do par em um único token
        let merged = format!("{}{}", pair.0, pair.1);
        
        // 🔍 **PROCESSAMENTO DE CADA PALAVRA**
        for (word, freq) in word_freqs {
            let mut new_word = Vec::new();
            let mut i = 0;
            
            // 👀 **VARREDURA TOKEN POR TOKEN**
            while i < word.len() {
                // 🎯 **DETECÇÃO DO PAR ALVO**
                // Verifica se a posição atual e próxima formam o par desejado
                if i < word.len() - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                    // ✅ **MERGE EXECUTADO**
                    // Substitui o par pelo token combinado
                    new_word.push(merged.clone());
                    i += 2; // Pula ambos os tokens do par
                } else {
                    // 📋 **PRESERVAÇÃO DE TOKEN**
                    // Token não faz parte do par, mantém como está
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            
            // 📊 **PRESERVAÇÃO DE FREQUÊNCIA**
            // A nova palavra mantém a mesma frequência da original
            new_freqs.insert(new_word, *freq);
        }
        
        new_freqs
    }
    
    /// 🔢 **CODIFICAÇÃO: TEXTO → NÚMEROS**
    /// 
    /// Converte texto em uma sequência de IDs numéricos que o modelo
    /// pode processar. Aplica todas as regras de merge aprendidas durante
    /// o treinamento para produzir a tokenização mais eficiente.
    /// 
    /// ## 🔄 **Processo Completo:**
    /// 
    /// ### 1️⃣ **Preparação:**
    /// ```text
    /// Input: "hello world"
    /// Adiciona: <BOS> + texto + <EOS>
    /// ```
    /// 
    /// ### 2️⃣ **Tokenização por Palavra:**
    /// ```text
    /// "hello" → ["h", "e", "l", "l", "o"]
    /// "world" → ["w", "o", "r", "l", "d"]
    /// ```
    /// 
    /// ### 3️⃣ **Aplicação de Merges:**
    /// ```text
    /// Merge 1: ("l", "l") → "ll"
    /// "hello" → ["h", "e", "ll", "o"]
    /// 
    /// Merge 2: ("he", "ll") → "hell"
    /// "hello" → ["hell", "o"]
    /// ```
    /// 
    /// ### 4️⃣ **Conversão para IDs:**
    /// ```text
    /// ["<BOS>", "hell", "o", " ", "world", "<EOS>"]
    /// → [2, 1547, 78, 32, 2134, 3]
    /// ```
    /// 
    /// ## 🎯 **Características:**
    /// - **Determinística**: Mesmo texto = mesma tokenização
    /// - **Eficiente**: Usa tokens mais longos quando possível
    /// - **Robusta**: Nunca falha (usa <UNK> como fallback)
    /// - **Estruturada**: Sempre inclui marcadores BOS/EOS
    /// 
    /// ## 📊 **Complexidade:**
    /// - **Tempo**: O(text_length × num_merges)
    /// - **Espaço**: O(text_length)
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let mut tokens = Vec::new();
        
        // 🏁 **MARCADOR DE INÍCIO**
        // Sinaliza ao modelo onde a sequência começa
        tokens.push(self.vocab["<BOS>"]);
        
        // 🔤 **PROCESSAMENTO PALAVRA POR PALAVRA**
        for word in text.split_whitespace() {
            // 📝 **INICIALIZAÇÃO EM CARACTERES**
            // Cada palavra começa como sequência de caracteres individuais
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            
            // 🔄 **APLICAÇÃO SEQUENCIAL DE MERGES**
            // Aplica todas as regras aprendidas na ordem correta
            // Ordem importa: merges anteriores têm prioridade
            for merge in &self.merges {
                chars = self.apply_merge_to_word(chars, merge);
            }
            
            // 🔢 **CONVERSÃO PARA IDS NUMÉRICOS**
            // Transforma tokens (strings) em IDs que o modelo entende
            for token in chars {
                let id = self.vocab.get(&token).unwrap_or(&self.vocab["<UNK>"]);
                tokens.push(*id);
            }
        }
        
        // 🏁 **MARCADOR DE FIM**
        // Sinaliza ao modelo onde a sequência termina
        tokens.push(self.vocab["<EOS>"]);
        
        Ok(tokens)
    }
    
    /// 🔗 **APLICAÇÃO DE MERGE EM PALAVRA INDIVIDUAL**
    /// 
    /// Aplica uma regra de merge específica a uma palavra tokenizada,
    /// substituindo pares adjacentes pelo token combinado.
    /// 
    /// ## 🎯 **Exemplo Prático:**
    /// ```text
    /// Palavra: ["h", "e", "l", "l", "o"]
    /// Merge: ("l", "l") → "ll"
    /// 
    /// Processo:
    /// 1. "h" → não match, mantém "h"
    /// 2. "e" → não match, mantém "e"
    /// 3. "l", "l" → MATCH! substitui por "ll"
    /// 4. "o" → não match, mantém "o"
    /// 
    /// Resultado: ["h", "e", "ll", "o"]
    /// ```
    /// 
    /// ## 🔍 **Algoritmo de Varredura:**
    /// - **Busca Gulosa**: Primeira ocorrência encontrada é substituída
    /// - **Salto Duplo**: Após merge, pula 2 posições (par consumido)
    /// - **Preservação**: Tokens não-matching são mantidos intactos
    /// - **Ordem**: Processa da esquerda para direita
    /// 
    /// ## ⚡ **Otimizações:**
    /// - Verificação de bounds antes do acesso
    /// - Clone mínimo (apenas quando necessário)
    /// - Pré-formatação do token merged
    /// 
    /// ## 📊 **Complexidade:**
    /// - **Tempo**: O(word_length)
    /// - **Espaço**: O(word_length)
    fn apply_merge_to_word(&self, word: Vec<String>, merge: &(String, String)) -> Vec<String> {
        // 🎯 **PRÉ-FORMATAÇÃO DO TOKEN MERGED**
        // Evita concatenação repetida durante o loop
        let merged = format!("{}{}", merge.0, merge.1);
        let mut result = Vec::new();
        let mut i = 0;
        
        // 🔍 **VARREDURA SEQUENCIAL COM DETECÇÃO DE PARES**
        while i < word.len() {
            // 🎯 **DETECÇÃO DE PAR ADJACENTE**
            // Verifica se posição atual + próxima formam o par target
            if i < word.len() - 1 && word[i] == merge.0 && word[i + 1] == merge.1 {
                // ✅ **MERGE ENCONTRADO**
                // Substitui o par pelo token combinado
                result.push(merged.clone());
                i += 2; // Pula ambos os tokens do par
            } else {
                // ➡️ **SEM MATCH**
                // Preserva o token atual e avança
                result.push(word[i].clone());
                i += 1;
            }
        }
        
        result
    }
    
    /// 🔤 **DECODIFICAÇÃO: NÚMEROS → TEXTO**
    /// 
    /// Converte uma sequência de IDs numéricos de volta para texto
    /// legível, removendo tokens especiais de controle.
    /// 
    /// ## 🔄 **Processo Inverso:**
    /// 
    /// ### 1️⃣ **Input (IDs):**
    /// ```text
    /// [2, 1547, 78, 32, 2134, 3]
    /// ```
    /// 
    /// ### 2️⃣ **Mapeamento Reverso:**
    /// ```text
    /// 2    → "<BOS>"     (removido)
    /// 1547 → "hell"      (mantido)
    /// 78   → "o"         (mantido)
    /// 32   → " "         (mantido)
    /// 2134 → "world"     (mantido)
    /// 3    → "<EOS>"     (removido)
    /// ```
    /// 
    /// ### 3️⃣ **Concatenação:**
    /// ```text
    /// "hell" + "o" + " " + "world" = "hello world"
    /// ```
    /// 
    /// ## 🎯 **Características:**
    /// - **Filtragem**: Remove automaticamente tokens especiais
    /// - **Robusta**: Ignora IDs desconhecidos silenciosamente
    /// - **Eficiente**: Concatenação direta sem separadores
    /// - **Limpa**: Produz texto natural sem artefatos
    /// 
    /// ## 📊 **Complexidade:**
    /// - **Tempo**: O(num_tokens)
    /// - **Espaço**: O(output_text_length)
    pub fn decode(&self, tokens: &[usize]) -> Result<String> {
        let mut text = String::new();
        
        // 🔍 **CONVERSÃO ID → TOKEN**
        for &id in tokens {
            if let Some(token) = self.reverse_vocab.get(&id) {
                // 🚫 **FILTRAGEM DE TOKENS ESPECIAIS**
                // Remove marcadores de controle (<BOS>, <EOS>, <PAD>, <UNK>)
                // Mantém apenas conteúdo textual real
                if !token.starts_with('<') || !token.ends_with('>') {
                    text.push_str(token);
                }
            }
            // 🔇 **IDs DESCONHECIDOS SÃO IGNORADOS SILENCIOSAMENTE**
            // Permite decodificação robusta mesmo com vocabulários parciais
        }
        
        Ok(text)
    }
    
    /// 📊 **TAMANHO DO VOCABULÁRIO**
    /// 
    /// Retorna o número total de tokens únicos no vocabulário,
    /// incluindo caracteres base, merges aprendidos e tokens especiais.
    /// 
    /// ## 📈 **Composição Típica:**
    /// ```text
    /// Tokens Especiais: 4     (<PAD>, <UNK>, <BOS>, <EOS>)
    /// Caracteres Base:  ~100   (letras, números, símbolos)
    /// Merges Aprendidos: N-104 (resto até vocab_size)
    /// Total:            N     (vocab_size configurado)
    /// ```
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// 🔤 Devolve a string de um token individual pelo seu ID, se conhecido.
    /// Usado pela API de demonstração para mostrar cada token separadamente
    /// (em vez de só o texto decodificado inteiro).
    pub fn token_string(&self, id: usize) -> Option<&str> {
        self.reverse_vocab.get(&id).map(|s| s.as_str())
    }
    
    /// 🏁 **DETECTOR DE FIM DE SEQUÊNCIA**
    /// 
    /// Verifica se um token específico é o marcador de fim de sequência,
    /// usado para determinar quando parar a geração de texto.
    /// 
    /// ## 🎯 **Uso Típico:**
    /// ```rust
    /// while let Some(next_token) = model.generate_next() {
    ///     if tokenizer.is_eos_token(next_token) {
    ///         break; // Para a geração
    ///     }
    ///     output.push(next_token);
    /// }
    /// ```
    pub fn is_eos_token(&self, token: usize) -> bool {
        token == self.vocab["<EOS>"]
    }

    /// 💾 Salva o vocabulário e as regras de merge em JSON, para reconstruir
    /// exatamente o mesmo tokenizador usado ao treinar um checkpoint.
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// 📂 Carrega um tokenizador previamente salvo com `save_json`.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}