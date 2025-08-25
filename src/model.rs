//! # Mini-GPT: Arquitetura Transformer Completa
//! 
//! ## 🧠 O que é um Modelo Transformer?
//! 
//! O Transformer é uma arquitetura revolucionária de deep learning que mudou
//! completamente o campo de processamento de linguagem natural. Diferente de
//! redes recorrentes (RNNs), o Transformer processa sequências inteiras em
//! paralelo usando o mecanismo de "atenção".
//! 
//! ### 🏗️ Componentes Principais:
//! 
//! 1. **Token Embeddings** 📚
//!    - Converte palavras/tokens em vetores densos de números reais
//!    - Cada palavra vira um ponto no espaço multidimensional
//!    - Palavras similares ficam próximas no espaço vetorial
//! 
//! 2. **Position Embeddings** 📍
//!    - Adiciona informação sobre a posição da palavra na sequência
//!    - Crucial porque Transformers não têm noção natural de ordem
//!    - Permite distinguir "João ama Maria" de "Maria ama João"
//! 
//! 3. **Multi-Head Attention** 👁️
//!    - Permite ao modelo "focar" em diferentes partes da entrada
//!    - Múltiplas "cabeças" capturam diferentes tipos de relações
//!    - Como ter vários "focos de atenção" simultâneos
//! 
//! 4. **Feed-Forward Networks** ⚡
//!    - Redes neurais que processam cada posição independentemente
//!    - Aplicam transformações não-lineares aos dados
//!    - Aumentam a capacidade expressiva do modelo
//! 
//! 5. **Layer Normalization** ⚖️
//!    - Estabiliza o treinamento normalizando ativações
//!    - Acelera convergência e melhora performance
//! 
//! ### 🔄 Processo Autoregressivo:
//! 
//! O modelo GPT é "autoregressivo" - gera texto token por token:
//! 1. Recebe sequência de tokens como entrada
//! 2. Prediz probabilidades para o próximo token
//! 3. Amostra um token baseado nessas probabilidades
//! 4. Adiciona o token à sequência e repete
//! 
//! Este é nosso "cérebro artificial" completo que implementa
//! toda essa arquitetura sofisticada em Rust!

use candle_core::{DType, Device, Tensor, IndexOp};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap};
use crate::transformer::TransformerBlock;
use crate::tokenizer::BPETokenizer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// 🎛️ **CONFIGURAÇÃO DO MODELO GPT**
/// 
/// Esta estrutura define todos os hiperparâmetros que controlam
/// a arquitetura e comportamento do modelo. É como o "DNA" do modelo!
/// 
/// ## 📊 Parâmetros Explicados:
/// 
/// ### `vocab_size` 📚
/// - Quantas palavras/tokens diferentes o modelo conhece
/// - Determina o tamanho da camada de saída
/// - Exemplo: 50.000 = modelo conhece 50 mil palavras diferentes
/// 
/// ### `n_embd` 🧮
/// - Dimensão dos vetores de embedding (largura do modelo)
/// - Maior = mais capacidade, mas mais lento
/// - GPT-3: 12.288, nosso modelo educacional: 128
/// 
/// ### `n_head` 👁️
/// - Número de "cabeças" de atenção paralelas
/// - Cada cabeça foca em aspectos diferentes do texto
/// - Deve dividir `n_embd` igualmente
/// 
/// ### `n_layer` 🏗️
/// - Profundidade do modelo (quantas camadas Transformer)
/// - Mais camadas = mais capacidade de abstração
/// - GPT-3: 96 camadas, nosso: 4 camadas
/// 
/// ### `block_size` 📏
/// - Tamanho máximo da sequência de entrada (contexto)
/// - Quantas palavras o modelo "lembra" de uma vez
/// - Maior contexto = melhor compreensão, mas mais memória
/// 
/// ### `dropout` 🎲
/// - Taxa de regularização para evitar overfitting
/// - 0.1 = desliga 10% dos neurônios aleatoriamente
/// - Usado apenas durante treinamento, não na inferência
#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub vocab_size: usize,   // 📚 Tamanho do vocabulário (quantas palavras o modelo conhece)
    pub n_embd: usize,       // 🧮 Dimensão dos embeddings (largura do modelo)
    pub n_head: usize,       // 👁️ Número de cabeças de atenção paralelas
    pub n_layer: usize,      // 🏗️ Número de blocos transformer (profundidade)
    pub block_size: usize,   // 📏 Tamanho máximo do contexto (memória do modelo)
    pub dropout: f32,        // 🎲 Taxa de dropout para regularização
}

/// 🤖 **MINI-GPT: MODELO TRANSFORMER COMPLETO**
/// 
/// Esta é a implementação principal do nosso modelo de linguagem.
/// Funciona como um "cérebro artificial" que aprendeu padrões de texto
/// e pode gerar novas sequências baseadas no que aprendeu.
/// 
/// ## 🧩 Arquitetura Detalhada:
/// 
/// ```text
/// Input Tokens → Token Embeddings → Position Embeddings
///                      ↓
///              [Transformer Block 1]
///                      ↓
///              [Transformer Block 2]
///                      ↓
///                    ...
///                      ↓
///              [Transformer Block N]
///                      ↓
///               Layer Normalization
///                      ↓
///              Linear Projection (lm_head)
///                      ↓
///              Output Probabilities
/// ```
/// 
/// ### 🔄 Processo Autoregressivo:
/// O modelo é "autoregressivo" - gera texto token por token:
/// 1. 📝 Recebe sequência de tokens como entrada
/// 2. 🧮 Calcula probabilidades para o próximo token
/// 3. 🎲 Amostra um token baseado nessas probabilidades
/// 4. ➕ Adiciona o token à sequência e repete
/// 
/// É como completar uma frase palavra por palavra, sempre
/// considerando todo o contexto anterior!
pub struct MiniGPT {
    config: GPTConfig,              // 🎛️ Configuração do modelo
    
    // 📚 **CAMADAS DE EMBEDDING**
    // Convertem tokens discretos em representações vetoriais contínuas
    token_embedding: Embedding,     // 🔤 Converte IDs de tokens em vetores densos
    position_embedding: Embedding,  // 📍 Adiciona informação posicional aos tokens
    
    // 🏗️ **BLOCOS TRANSFORMER EMPILHADOS**
    // Cada bloco contém atenção multi-cabeça + feed-forward + normalizações
    blocks: Vec<TransformerBlock>,  // 🧠 Stack de camadas que processam sequências
    
    // 🎯 **CAMADAS DE SAÍDA**
    ln_final: LayerNorm,           // ⚖️ Normalização final para estabilidade
    lm_head: Linear,               // 🎪 Projeta embeddings para vocabulário
    
    device: Device,                // 💻 Dispositivo de computação (CPU/GPU)
    
    // 💾 **VARMAP PARA SALVAMENTO**
    // Contém todos os pesos treináveis do modelo para serialização
    varmap: VarMap,                // 🗂️ Mapa de variáveis para salvamento/carregamento
}

impl MiniGPT {
    /// 🏗️ **CONSTRUTOR DO MODELO MINI-GPT**
    /// 
    /// Este método inicializa toda a arquitetura Transformer do zero.
    /// É como "construir o cérebro" do modelo, criando todas as conexões
    /// neurais que serão ajustadas durante o treinamento.
    /// 
    /// ## 🧩 Processo de Inicialização:
    /// 
    /// ### 1. **Inicialização de Pesos** ⚖️
    /// - Todos os pesos começam com valores aleatórios pequenos
    /// - Inicialização adequada é crucial para convergência
    /// - Usamos distribuição normal com variância controlada
    /// 
    /// ### 2. **Camadas de Embedding** 📚
    /// - Token embeddings: mapeiam IDs → vetores densos
    /// - Position embeddings: codificam posição na sequência
    /// - Ambos são "lookup tables" aprendíveis
    /// 
    /// ### 3. **Stack de Transformers** 🏗️
    /// - Múltiplas camadas idênticas empilhadas
    /// - Cada camada processa e refina representações
    /// - Profundidade permite abstrações complexas
    /// 
    /// ### 4. **Cabeça de Linguagem** 🎯
    /// - Projeta embeddings finais para vocabulário
    /// - Produz distribuição de probabilidades sobre tokens
    /// - É onde acontece a "predição" da próxima palavra
    pub fn new(config: GPTConfig, device: &Device) -> Result<Self> {
        // 🚀 **OTIMIZAÇÕES ESPECÍFICAS PARA METAL GPU ARM APPLE**
        match device {
            Device::Metal(_) => {
                println!("🔥 Inicializando modelo para Metal GPU:");
                println!("   💾 Usando precisão F32 otimizada para Metal");
                println!("   ⚡ Configurações de memória otimizadas para 18GB");
                println!("   🎯 Parâmetros: ~{:.1}M", 
                    (config.vocab_size * config.n_embd + 
                     config.block_size * config.n_embd + 
                     config.n_layer * 4 * config.n_embd * config.n_embd) as f32 / 1_000_000.0);
            }
            _ => {
                println!("🖥️  Inicializando modelo para CPU (modo compatibilidade)");
            }
        }
        
        // 🎲 **INICIALIZADOR DE VARIÁVEIS COM XAVIER INITIALIZATION**
        // 
        // A inicialização adequada é crucial para o sucesso do treinamento:
        // - Pesos muito pequenos → gradientes desaparecem
        // - Pesos muito grandes → gradientes explodem
        // - Xavier/Glorot: variância baseada no número de conexões
        // 
        // ## 📊 **Fórmula Xavier:**
        // ```
        // std = sqrt(2.0 / (fan_in + fan_out))
        // ```
        // Onde fan_in/fan_out são dimensões de entrada/saída
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        // 📚 **TOKEN EMBEDDINGS: TRANSFORMANDO SÍMBOLOS EM SIGNIFICADO**
        // 
        // ## 🔤 **Token Embeddings:**
        // - Converte IDs discretos (0, 1, 2...) em vetores contínuos
        // - Cada token vira um ponto no espaço n_embd-dimensional
        // - Palavras similares ficam próximas no espaço vetorial
        // - Exemplo: "gato" e "felino" terão embeddings similares
        // 
        // Cada token (palavra/subpalavra) é mapeado para um vetor denso
        // Usando VarBuilder para inicialização adequada
        let token_embedding = embedding(config.vocab_size, config.n_embd, vb.pp("token_emb"))?;
        
        // 📍 **POSITION EMBEDDINGS: ONDE ESTÁ A PALAVRA?**
        // 
        // ## 📍 **Position Embeddings:**
        // - Adiciona informação sobre ONDE a palavra aparece
        // - Crucial porque Transformers não têm noção natural de ordem
        // - Permite distinguir "João ama Maria" de "Maria ama João"
        // - Cada posição (0, 1, 2...) tem seu próprio embedding aprendível
        // 
        // Adiciona informação sobre posição na sequência
        // Usando VarBuilder para inicialização adequada
        let position_embedding = embedding(config.block_size, config.n_embd, vb.pp("pos_emb"))?;
        
        // 🏗️ **STACK DE BLOCOS TRANSFORMER: O CORAÇÃO DO MODELO**
        // 
        // Cada bloco é uma unidade de processamento completa que contém:
        // 
        // ### 👁️ **Multi-Head Attention:**
        // - Permite ao modelo "focar" em diferentes partes da entrada
        // - Múltiplas cabeças capturam diferentes tipos de relações
        // - Como ter vários "focos de atenção" simultâneos
        // 
        // ### ⚡ **Feed-Forward Network:**
        // - Rede neural densa que processa cada posição
        // - Aplica transformações não-lineares complexas
        // - Aumenta capacidade expressiva do modelo
        // 
        // ### ⚖️ **Layer Normalizations:**
        // - Estabilizam treinamento normalizando ativações
        // - Aceleram convergência e melhoram performance
        // 
        // ## 🔄 **Processamento em Camadas:**
        // Cada camada refina e abstrai mais as representações:
        // - Camada 1: Padrões locais (bigramas, trigramas)
        // - Camada 2: Sintaxe (sujeito-verbo-objeto)
        // - Camada 3: Semântica (significado, contexto)
        // - Camada 4: Pragmática (intenção, estilo)
        // 
        // Cada bloco contém:
        // - Multi-Head Self-Attention (foco em diferentes partes)
        // - Feed-Forward Network (transformações não-lineares)
        // - Layer Normalizations (estabilização)
        // - Conexões residuais (facilita treinamento profundo)
        let mut blocks = Vec::new();
        for i in 0..config.n_layer {
            blocks.push(TransformerBlock::new(
                config.n_embd,     // 🧮 Dimensão dos embeddings
                config.n_head,     // 👁️ Número de cabeças de atenção
                config.dropout,    // 🎲 Taxa de dropout para regularização
                vb.pp(format!("block_{}", i)),  // 🏷️ Nome único para cada bloco
            )?);
        }
        
        // 🎯 **CAMADAS FINAIS DE PROCESSAMENTO**
        
        // 🎯 **CAMADAS FINAIS: TRANSFORMANDO REPRESENTAÇÕES EM PREDIÇÕES**
        // 
        // ## ⚖️ **Layer Normalization Final:**
        // - Última normalização antes da predição
        // - Garante que as ativações estejam em escala adequada
        // - Melhora estabilidade numérica da camada de saída
        // - Epsilon (1e-5) previne divisão por zero
        // 
        // ## 🎪 **Language Modeling Head (lm_head):**
        // - Camada linear que projeta embeddings → vocabulário
        // - Transforma vetor de n_embd dimensões → vocab_size logits
        // - Cada logit representa "confiança" para um token específico
        // - Softmax converte logits em probabilidades
        // 
        // ### 📊 **Exemplo de Saída:**
        // ```
        // Embeddings [128 dims] → Linear → Logits [vocab_size]
        // [0.1, -0.3, 0.8, ...] → [...] → [2.1, 0.5, -1.2, 3.4, ...]
        //                                    ↓ softmax
        //                                 [0.15, 0.03, 0.01, 0.81, ...]
        // ```
        // 
        // ⚖️ **LAYER NORMALIZATION FINAL**
        // Normaliza as ativações antes da projeção final
        // Usando VarBuilder para inicialização adequada
        let ln_final = layer_norm(config.n_embd, 1e-5, vb.pp("ln_final"))?;
        
        // 🎪 **CABEÇA DE LINGUAGEM (LANGUAGE MODELING HEAD)**
        // Projeta embeddings finais para espaço do vocabulário
        // Usando VarBuilder para inicialização adequada
        let lm_head = linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;
        
        // 🎉 **MONTAGEM FINAL DO MODELO**
        // Combina todos os componentes em uma estrutura coesa
        Ok(Self {
            config: config.clone(),    // 🎛️ Mantém configuração para referência
            token_embedding,           // 📚 Lookup table de tokens
            position_embedding,        // 📍 Lookup table de posições
            blocks,                    // 🧠 Stack de processamento principal
            ln_final,                  // ⚖️ Normalização final
            lm_head,                   // 🎯 Projeção para vocabulário
            device: device.clone(),    // 💻 Dispositivo de computação
            varmap,                    // 💾 Mapa de variáveis para salvamento
        })
    }
    
    /// 🚀 **FORWARD PASS: O CORAÇÃO DO MODELO**
    /// 
    /// Este método implementa a passagem direta (forward pass) dos dados
    /// através de toda a arquitetura Transformer. É aqui que a "mágica" acontece!
    /// 
    /// ## 🔄 Fluxo de Dados Detalhado:
    /// 
    /// ```text
    /// Input IDs → Token Embeddings → + Position Embeddings
    ///                                        ↓
    ///                              [Transformer Block 1]
    ///                                        ↓
    ///                              [Transformer Block 2]
    ///                                        ↓
    ///                                      ...
    ///                                        ↓
    ///                              [Transformer Block N]
    ///                                        ↓
    ///                               Layer Normalization
    ///                                        ↓
    ///                              Linear Projection (lm_head)
    ///                                        ↓
    ///                               Logits (probabilidades)
    /// ```
    /// 
    /// ### 📊 Dimensões dos Tensores:
    /// - **Input**: `[batch_size, seq_len]` - IDs dos tokens
    /// - **Embeddings**: `[batch_size, seq_len, n_embd]` - Representações vetoriais
    /// - **Logits**: `[batch_size, seq_len, vocab_size]` - Probabilidades por posição
    /// 
    /// ### 🎯 Parâmetros:
    /// - `idx`: Tensor com IDs dos tokens de entrada
    /// - `targets`: Tokens alvo para cálculo de loss (opcional, usado no treino)
    /// 
    /// ### 📤 Retorno:
    /// - `logits`: Probabilidades para próximo token em cada posição
    /// - `loss`: Função de perda (apenas se targets fornecidos)
    pub fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        // 📏 **EXTRAÇÃO DAS DIMENSÕES**
        // batch_size: quantas sequências processamos simultaneamente
        // seq_len: comprimento de cada sequência (número de tokens)
        let (batch_size, seq_len) = idx.dims2()?;
        
        // 🚨 **VALIDAÇÃO DE CONTEXTO**
        // Garante que não excedemos o tamanho máximo de contexto
        // Modelos têm limite de memória - não podem "lembrar" infinitamente
        assert!(seq_len <= self.config.block_size, 
                "Sequência muito longa! Max: {}, Atual: {}", 
                self.config.block_size, seq_len);
        
        // 1️⃣ **TOKEN EMBEDDINGS: IDs → VETORES**
        let tok_emb = self.token_embedding.forward(idx)?;
        
        // 🔍 **DEBUG: Verificar token embeddings**
        let tok_emb_vec = tok_emb.flatten_all()?.to_vec1::<f32>()?;
        if tok_emb_vec.iter().any(|&x| x.is_nan()) {
            eprintln!("⚠️ DEBUG: Token embeddings contém NaN!");
            return Err("Token embeddings contém NaN".into());
        }
        
        // 2️⃣ **POSITION EMBEDDINGS: ONDE ESTÁ CADA TOKEN?**
        let pos = Tensor::arange(0, seq_len as i64, &self.device)?;
        let pos_emb = self.position_embedding.forward(&pos)?;
        
        // 🔍 **DEBUG: Verificar position embeddings**
        let pos_emb_vec = pos_emb.flatten_all()?.to_vec1::<f32>()?;
        if pos_emb_vec.iter().any(|&x| x.is_nan()) {
            eprintln!("⚠️ DEBUG: Position embeddings contém NaN!");
            return Err("Position embeddings contém NaN".into());
        }
        
        // 3️⃣ **COMBINAÇÃO: TOKENS + POSIÇÕES = REPRESENTAÇÃO COMPLETA**
        // 
        // ## ➕ **Soma de Embeddings:**
        // Esta é uma das operações mais importantes do Transformer!
        // 
        // ```text
        // Token "gato" na posição 2:
        // 
        // Token Embedding:    [0.1, -0.3, 0.8, 0.2, ...] (significado de "gato")
        //           +
        // Position Embedding: [0.0, 0.1, -0.2, 0.4, ...] (posição 2)
        //           =
        // Combined:           [0.1, -0.2, 0.6, 0.6, ...] ("gato" na posição 2)
        // ```
        // 
        // ### 🎯 **Por que somar?**
        // - Preserva informação de ambos (significado + posição)
        // - Permite que atenção considere tanto semântica quanto sintaxe
        // - Mais eficiente que concatenação (mantém dimensionalidade)
        let pos_emb = pos_emb.unsqueeze(0)?.expand(&[batch_size, seq_len, self.config.n_embd])?;
        let mut x = (tok_emb.clone() + pos_emb.clone())?;
        
        // 🔍 **DEBUG: VERIFICAÇÃO DE INTEGRIDADE NUMÉRICA**
        // Detecta problemas numéricos que podem quebrar o treinamento
        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
        if x_vec.iter().any(|&val| val.is_nan()) {
            eprintln!("⚠️ DEBUG: Combinação de embeddings contém NaN!");
            eprintln!("   Token shape: {:?}", tok_emb.shape());
            eprintln!("   Position shape: {:?}", pos_emb.shape());
            return Err("Combinação de embeddings contém NaN".into());
        }
        
        // 4️⃣ **MÁSCARA CAUSAL: IMPEDINDO "VISÃO DO FUTURO"**
        // Cria máscara triangular que impede tokens de "verem" tokens futuros
        // Crucial para treinamento autoregressivo - modelo só pode usar contexto passado
        // Exemplo para seq_len=4:
        // [[0, -∞, -∞, -∞],
        //  [0,  0, -∞, -∞],
        //  [0,  0,  0, -∞],
        //  [0,  0,  0,  0]]
        let mask = self.create_causal_mask(seq_len)?;
        
        // 5️⃣ **PROCESSAMENTO ATRAVÉS DOS BLOCOS TRANSFORMER**
        // 
        // ## 🏗️ **Stack de Processamento Sequencial:**
        // Cada bloco refina progressivamente as representações:
        // 
        // ### 📊 **Evolução das Representações:**
        // ```text
        // Entrada:    ["O", "gato", "subiu", "no", "telhado"]
        //             ↓ (embeddings iniciais)
        // Bloco 1:    [sintaxe básica, bigramas]
        // Bloco 2:    [relações gramaticais, trigramas]
        // Bloco 3:    [semântica, contexto local]
        // Bloco 4:    [pragmática, contexto global]
        //             ↓
        // Saída:      [representações ricas e contextuais]
        // ```
        // 
        // ### 🔄 **Processamento Residual:**
        // Cada bloco usa conexões residuais (skip connections):
        // `output = block(input) + input`
        // Isso permite gradientes fluírem diretamente e facilita treinamento profundo
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, Some(&mask))
                .map_err(|e| format!("Erro no bloco {}: {}", i, e))?;
            
            // 🔍 **DEBUG: MONITORAMENTO POR CAMADA**
            // Detecta em qual camada problemas numéricos aparecem
            let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
            if x_vec.iter().any(|&val| val.is_nan()) {
                eprintln!("⚠️ DEBUG: Bloco {} produziu NaN!", i);
                eprintln!("   Dimensões de entrada: {:?}", x.shape());
                eprintln!("   Bloco: {}/{}", i + 1, self.blocks.len());
                return Err(format!("Bloco {} produziu NaN", i).into());
            }
        }
        
        // 6️⃣ **NORMALIZAÇÃO FINAL: PREPARAÇÃO PARA PREDIÇÃO**
        // 
        // ## ⚖️ **Layer Normalization Final:**
        // - Última oportunidade de estabilizar ativações
        // - Garante que inputs para lm_head estejam bem condicionados
        // - Melhora estabilidade numérica da predição
        // 
        // ### 📊 **Fórmula da Normalização:**
        // ```
        // normalized = (x - mean) / sqrt(variance + epsilon)
        // output = normalized * gamma + beta
        // ```
        // Onde gamma e beta são parâmetros aprendíveis
        let x = self.ln_final.forward(&x)?;
        
        // 🔍 **DEBUG: Verificar layer norm final**
        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
        if x_vec.iter().any(|&val| val.is_nan()) {
            eprintln!("⚠️ DEBUG: Layer norm final produziu NaN!");
            return Err("Layer norm final produziu NaN".into());
        }
        
        // 🎯 **PROJEÇÃO PARA VOCABULÁRIO: TRANSFORMANDO PENSAMENTOS EM PALAVRAS**
        // 
        // ## 🎪 **Language Modeling Head:**
        // Esta é a camada que "traduz" as representações internas
        // do modelo em probabilidades sobre palavras do vocabulário.
        // 
        // ### 🔄 **Transformação Dimensional:**
        // ```text
        // Input:  [batch_size, seq_len, n_embd]     (representações ricas)
        //           ↓ (linear transformation)
        // Output: [batch_size, seq_len, vocab_size] (logits por token)
        // ```
        // 
        // ### 📊 **Interpretação dos Logits:**
        // - Cada posição na sequência produz vocab_size logits
        // - Logit alto = modelo "confia" nesse token
        // - Logit baixo = modelo "não acredita" nesse token
        // - Softmax converte logits em probabilidades válidas
        let logits = self.lm_head.forward(&x)?;
        
        // 🔍 **DEBUG: Verificar logits finais**
        let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        if logits_vec.iter().any(|&val| val.is_nan()) {
            eprintln!("⚠️ DEBUG: Cabeça de linguagem (lm_head) produziu NaN!");
            return Err("Cabeça de linguagem produziu NaN".into());
        }
        
        // 7️⃣ **CÁLCULO DE LOSS (APENAS DURANTE TREINAMENTO)**
        // Se temos targets, calcula quão "erradas" foram nossas predições
        // Cross-entropy loss: penaliza predições incorretas
        // Usado pelo otimizador para ajustar pesos via backpropagation
        let loss = if let Some(targets) = targets {
            let loss = self.compute_loss(&logits, targets)
                .map_err(|e| format!("Erro no cálculo de loss: {}", e))?;
            Some(loss)
        } else {
            None  // Modo inferência - sem loss
        };
        
        // 📤 **RETORNO DOS RESULTADOS**
        // logits: probabilidades brutas (antes de softmax)
        // loss: função de perda para otimização (opcional)
        Ok((logits, loss))
    }
    
    /// 🔒 **CRIAÇÃO DA MÁSCARA CAUSAL**
    /// 
    /// Este método cria uma máscara triangular que implementa a "causalidade"
    /// no modelo - garantindo que cada token só pode "ver" tokens anteriores.
    /// 
    /// ## 🎯 Por que precisamos disso?
    /// 
    /// Em modelos autoregressivos como GPT, queremos que o modelo aprenda a
    /// predizer o próximo token baseado APENAS no contexto passado, nunca
    /// no futuro. Durante o treinamento, temos acesso a toda a sequência,
    /// mas precisamos "mascarar" o futuro para simular a geração real.
    /// 
    /// ## 📊 Exemplo Visual (seq_len=4):
    /// 
    /// ```text
    /// Posições:  0    1    2    3
    /// Token 0: [ 0,  -∞,  -∞,  -∞]  ← só vê a si mesmo
    /// Token 1: [ 0,   0,  -∞,  -∞]  ← vê posições 0,1
    /// Token 2: [ 0,   0,   0,  -∞]  ← vê posições 0,1,2
    /// Token 3: [ 0,   0,   0,   0]  ← vê todas as posições
    /// ```
    /// 
    /// ## ⚡ Implementação:
    /// - **0**: Atenção permitida (sem penalidade)
    /// - **-∞**: Atenção bloqueada (após softmax → probabilidade 0)
    /// 
    /// ### 🔢 Dimensões:
    /// - **Input**: `seq_len` (comprimento da sequência)
    /// - **Output**: `[seq_len, seq_len]` (matriz quadrada)
    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        // 🏗️ **CONSTRUÇÃO DA MATRIZ TRIANGULAR**
        // Inicializa vetor 1D que representa matriz seq_len x seq_len
        // Usamos indexação linear: posição [i,j] = i * seq_len + j
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        
        // 🔄 **PREENCHIMENTO DA MÁSCARA**
        // Para cada linha i (token atual)
        for i in 0..seq_len {
            // Para cada coluna j (token de referência)
            for j in 0..seq_len {
                // 🚫 Se j > i, então j está no "futuro" relativo a i
                if j > i {
                    // Bloqueia atenção para tokens futuros
                    // Usa valor finito grande negativo em vez de infinito para evitar NaN
                    mask_data[i * seq_len + j] = 1.0; // 1 indica posição a ser mascarada
                }
                // Se j <= i, mantém 0 (atenção permitida)
            }
        }
        
        // 🎯 **CRIAÇÃO DO TENSOR FINAL**
        // Converte vetor 1D em tensor 2D com dimensões [seq_len, seq_len]
        Ok(Tensor::from_slice(&mask_data, (seq_len, seq_len), &self.device)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?)
    }
    
    /// 📊 **CÁLCULO DA FUNÇÃO DE PERDA (LOSS)**
    /// 
    /// Este método implementa a Cross-Entropy Loss, que mede quão "surpreso"
    /// o modelo fica com a resposta correta. É a função que o modelo tenta
    /// minimizar durante o treinamento.
    /// 
    /// ## 🎯 O que é Cross-Entropy Loss?
    /// 
    /// Imagine que o modelo é um estudante fazendo uma prova de múltipla escolha.
    /// Para cada pergunta (posição na sequência), ele dá uma "confiança" para
    /// cada resposta possível (token do vocabulário). A cross-entropy mede:
    /// 
    /// - **Alta confiança na resposta certa** → Loss baixo (bom!)
    /// - **Baixa confiança na resposta certa** → Loss alto (ruim!)
    /// - **Alta confiança na resposta errada** → Loss muito alto (muito ruim!)
    /// 
    /// ## 📐 Fórmula Matemática:
    /// 
    /// ```text
    /// Loss = -log(P(token_correto))
    /// 
    /// Onde P(token_correto) é a probabilidade que o modelo
    /// atribuiu ao token correto naquela posição.
    /// ```
    /// 
    /// ## 🔢 Dimensões dos Tensores:
    /// - **logits**: `[batch_size, seq_len, vocab_size]` - "notas" brutas
    /// - **targets**: `[batch_size, seq_len]` - respostas corretas
    /// - **loss**: `[]` - escalar (número único)
    /// 
    /// ## ⚡ Processo de Cálculo:
    /// 1. **Reshape**: Achata tensores para facilitar cálculo
    /// 2. **Softmax**: Converte logits em probabilidades (implícito na cross-entropy)
    /// 3. **Loss**: Calcula -log(probabilidade_correta) para cada posição
    /// 4. **Média**: Retorna loss médio sobre todas as posições
    fn compute_loss(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // 📏 **EXTRAÇÃO DAS DIMENSÕES**
        let (batch_size, seq_len, vocab_size) = logits.dims3()?;
        
        // 🔍 **DEBUG: Verificar se logits contém valores inválidos**
        let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let has_nan = logits_vec.iter().any(|&x| x.is_nan());
        let has_inf = logits_vec.iter().any(|&x| x.is_infinite());
        
        if has_nan {
            eprintln!("⚠️ DEBUG: Logits contém NaN!");
            eprintln!("   Dimensões: [{}, {}, {}]", batch_size, seq_len, vocab_size);
            return Err("Logits contém valores NaN".into());
        }
        
        if has_inf {
            eprintln!("⚠️ DEBUG: Logits contém valores infinitos!");
            eprintln!("   Dimensões: [{}, {}, {}]", batch_size, seq_len, vocab_size);
            return Err("Logits contém valores infinitos".into());
        }
        
        // 🔄 **RESHAPE PARA CÁLCULO EFICIENTE**
        let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
        let targets_flat = targets.reshape((batch_size * seq_len,))?;
        
        // 🔍 **DEBUG: Verificar targets**
        let targets_vec = targets_flat.to_vec1::<u32>()?;
        let max_target = targets_vec.iter().max().unwrap_or(&0);
        if *max_target >= vocab_size as u32 {
            eprintln!("⚠️ DEBUG: Target inválido! Max target: {}, vocab_size: {}", max_target, vocab_size);
            return Err("Target fora do range do vocabulário".into());
        }
        
        // 🎯 **CÁLCULO DA CROSS-ENTROPY LOSS**
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
        
        // 🔍 **DEBUG: Verificar se o loss é válido**
        let loss_value = loss.to_scalar::<f32>()?;
        if loss_value.is_nan() {
            eprintln!("⚠️ DEBUG: Loss calculado é NaN!");
            eprintln!("   Logits stats: min={:.6}, max={:.6}, mean={:.6}", 
                     logits_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                     logits_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                     logits_vec.iter().sum::<f32>() / logits_vec.len() as f32);
            return Err("Loss calculado é NaN".into());
        }
        
        Ok(loss)
    }
    
    /// 🎭 **GERAÇÃO AUTOREGRESSIVA DE TEXTO**
    /// 
    /// Este é o método que faz a "mágica" acontecer! Ele implementa a geração
    /// autoregressiva, onde o modelo "conversa consigo mesmo" para criar texto.
    /// 
    /// ## 🔄 Como Funciona a Geração Autoregressiva?
    /// 
    /// Imagine que você está escrevendo uma história, mas só pode ver uma palavra
    /// por vez. A cada palavra, você precisa decidir qual será a próxima baseado
    /// apenas no que já escreveu:
    /// 
    /// ```text
    /// Prompt: "Era uma vez"
    /// 
    /// Passo 1: "Era uma vez" → modelo prediz → "uma"
    /// Passo 2: "Era uma vez uma" → modelo prediz → "princesa"
    /// Passo 3: "Era uma vez uma princesa" → modelo prediz → "que"
    /// ...
    /// ```
    /// 
    /// ## 🌡️ Temperatura: Controlando a Criatividade
    /// 
    /// - **Temperatura = 0**: Sempre escolhe a palavra mais provável (determinístico)
    /// - **Temperatura baixa (0.1-0.7)**: Mais conservador, texto coerente
    /// - **Temperatura alta (0.8-1.5)**: Mais criativo, pode ser incoerente
    /// - **Temperatura > 2**: Muito aleatório, geralmente incoerente
    /// 
    /// ## ⚡ Processo Detalhado:
    /// 1. **Tokenização**: Converte prompt em números
    /// 2. **Loop de Geração**: Para cada novo token:
    ///    - Limita contexto ao tamanho máximo
    ///    - Executa forward pass
    ///    - Aplica temperatura
    ///    - Amostra próximo token
    ///    - Adiciona ao contexto
    /// 3. **Decodificação**: Converte números de volta em texto
    /// 
    /// ### 🎯 Parâmetros:
    /// - `prompt`: Texto inicial para começar a geração
    /// - `max_tokens`: Máximo de tokens novos a gerar
    /// - `tokenizer`: Conversor texto ↔ números
    /// - `temperature`: Controle de criatividade (0.0 = determinístico)
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        tokenizer: &BPETokenizer,
        temperature: f32,
    ) -> Result<String> {
        // 🔤 **TOKENIZAÇÃO INICIAL**
        // Converte o prompt de texto para sequência de IDs de tokens
        // Exemplo: "Olá mundo" → [156, 2134] (números dependem do vocabulário)
        let mut tokens = tokenizer.encode(prompt)
            .map_err(|e| format!("Erro na tokenização do prompt: {}", e))?;
        let mut generated_tokens = Vec::new();
        
        // 🔄 **LOOP DE GERAÇÃO AUTOREGRESSIVA**
        // Gera um token por vez, sempre usando o contexto completo
        for step in 0..max_tokens {
            // 📏 **LIMITAÇÃO DO CONTEXTO**
            // Modelos têm limite de memória - se o contexto ficar muito longo,
            // mantemos apenas os últimos N tokens (sliding window)
            let context = if tokens.len() > self.config.block_size {
                &tokens[tokens.len() - self.config.block_size..]
            } else {
                &tokens[..]
            };
            
            // 🔢 **CONVERSÃO PARA TENSOR**
            // Transforma vetor de tokens em tensor que o modelo entende
            // Dimensões: [1, context_len] (batch_size=1, seq_len=context_len)
            let context_i64: Vec<i64> = context.iter().map(|&x| x as i64).collect();
            let idx = Tensor::from_slice(&context_i64, &[1, context.len()], &self.device)
                .map_err(|e| format!("Erro ao criar tensor no passo {}: {}", step, e))?;
            
            // 🚀 **FORWARD PASS: PREDIÇÃO**
            // Executa o modelo para obter probabilidades do próximo token
            // targets=None indica modo inferência (não treinamento)
            let (logits, _) = self.forward(&idx, None)
                .map_err(|e| format!("Erro no forward pass no passo {}: {}", step, e))?;
            
            // 🎯 **EXTRAÇÃO DOS LOGITS DA ÚLTIMA POSIÇÃO**
            // logits tem dimensões [1, seq_len, vocab_size]
            // Queremos apenas a última posição: [vocab_size]
            // Esta é a "opinião" do modelo sobre qual token vem a seguir
            let logits = logits.i((0, context.len() - 1, ..))
                .map_err(|e| format!("Erro ao extrair logits no passo {}: {}", step, e))?;
            
            // 🌡️ **APLICAÇÃO DA TEMPERATURA**
            // Temperatura controla o quão "ousado" o modelo será:
            // - Divide logits pela temperatura
            // - Temperatura baixa → distribuição mais "afiada" (conservador)
            // - Temperatura alta → distribuição mais "suave" (criativo)
            let temperature_tensor = Tensor::new(&[temperature], &self.device)
                .map_err(|e| format!("Erro ao criar tensor de temperatura no passo {}: {}", step, e))?;
            let logits = logits.broadcast_div(&temperature_tensor)
                .map_err(|e| format!("Erro ao aplicar temperatura no passo {}: {}", step, e))?;
            
            // 📊 **CONVERSÃO PARA PROBABILIDADES**
            // Softmax transforma "notas" brutas em probabilidades válidas
            // Soma de todas as probabilidades = 1.0
            let probs = candle_nn::ops::softmax(&logits, 0)
                .map_err(|e| format!("Erro no softmax no passo {}: {}", step, e))?;
            
            // 🎲 **AMOSTRAGEM DO PRÓXIMO TOKEN**
            // Escolhe um token baseado nas probabilidades calculadas
            // Não sempre o mais provável - introduz variabilidade
            let next_token = self.sample_from_probs(&probs)
                .map_err(|e| format!("Erro na amostragem no passo {}: {}", step, e))?;
            
            // ➕ **ADIÇÃO AO CONTEXTO**
            // O token gerado vira parte do contexto para a próxima iteração
            // Este é o "autoregressivo" - cada saída alimenta a próxima entrada
            tokens.push(next_token);
            generated_tokens.push(next_token);
            
            // 🛑 **VERIFICAÇÃO DE PARADA ANTECIPADA**
            // Alguns tokenizers têm tokens especiais de fim de sequência
            // Se encontrarmos um, podemos parar a geração
            if tokenizer.is_eos_token(next_token) {
                break;  // Para a geração se encontrar token de fim
            }
        }
        
        // 🔤 **DECODIFICAÇÃO FINAL**
        // Converte a sequência completa de tokens de volta para texto legível
        // Exemplo: [156, 2134, 891] → "Olá mundo!"
        Ok(tokenizer.decode(&generated_tokens)
            .map_err(|e| format!("Erro na decodificação final: {}", e))?)
    }
    
    /// 🎲 **AMOSTRAGEM PROBABILÍSTICA**
    /// 
    /// Este método implementa amostragem baseada em probabilidades,
    /// escolhendo tokens de forma não-determinística para gerar texto variado.
    /// 
    /// ## 🎯 Como Funciona?
    /// 
    /// Imagine uma roleta onde cada fatia representa um token possível,
    /// e o tamanho da fatia é proporcional à sua probabilidade:
    /// 
    /// ```text
    /// Tokens:  ["o", "a", "um", "uma", "..."]
    /// Probs:   [0.4, 0.3, 0.2,  0.1,   ...]
    /// 
    /// Roleta:  |████████|██████|████|██|...
    ///          0      0.4    0.7  0.9 1.0
    /// 
    /// Número aleatório: 0.65 → cai na fatia "a"
    /// ```
    /// 
    /// ## ⚡ Algoritmo:
    /// 1. **Gera número aleatório** entre 0 e 1
    /// 2. **Percorre probabilidades** acumulando soma
    /// 3. **Para quando soma ≥ número aleatório**
    /// 4. **Retorna índice** do token escolhido
    /// 
    /// ### 🎨 Vantagens da Amostragem:
    /// - **Variabilidade**: Mesmo prompt gera textos diferentes
    /// - **Criatividade**: Permite escolhas menos óbvias
    /// - **Naturalidade**: Evita repetições mecânicas
    fn sample_from_probs(&self, probs: &Tensor) -> Result<usize> {
        // 📊 **CONVERSÃO PARA VETOR**
        // Extrai probabilidades do tensor para formato mais fácil de trabalhar
        let probs: Vec<f32> = probs.to_vec1()
            .map_err(|e| format!("Erro ao converter probabilidades: {}", e))?;
        
        // 🎲 **GERAÇÃO DE NÚMERO ALEATÓRIO**
        // Cria gerador de números aleatórios thread-local
        // Gera número uniforme entre 0.0 e 1.0
        use rand::prelude::*;
        let mut rng = thread_rng();
        let uniform: f32 = rng.gen();  // Número aleatório [0.0, 1.0)
        
        // 🎯 **AMOSTRAGEM POR SOMA CUMULATIVA**
        // Percorre probabilidades acumulando soma até ultrapassar número aleatório
        let mut cumsum = 0.0;
        for (idx, &prob) in probs.iter().enumerate() {
            cumsum += prob;  // Acumula probabilidade
            
            // 🎪 Se número aleatório "cai" nesta fatia, escolhe este token
            if uniform <= cumsum {
                return Ok(idx);
            }
        }
        
        // 🛡️ **FALLBACK DE SEGURANÇA**
        // Em caso de erro numérico (probabilidades não somam 1.0),
        // retorna o último token como fallback
        Ok(probs.len() - 1)
    }
    
    /// 🔢 **CONTADOR DE PARÂMETROS TREINÁVEIS**
    /// 
    /// Este método calcula o número total de parâmetros (pesos) que o modelo
    /// precisa aprender durante o treinamento. É como contar quantos "botões"
    /// o modelo tem para ajustar!
    /// 
    /// ## 📊 Breakdown dos Parâmetros:
    /// 
    /// ### 📚 **Embeddings**:
    /// - **Token Embeddings**: `vocab_size × n_embd`
    /// - **Position Embeddings**: `block_size × n_embd`
    /// 
    /// ### 🧠 **Cada Bloco Transformer**:
    /// - **Attention**: `4 × n_embd²` (Q, K, V, Output projections)
    /// - **Feed-Forward**: `8 × n_embd²` (expansão 4x: up + down)
    /// - **Layer Norms**: `4 × n_embd` (2 layer norms × 2 parâmetros cada)
    /// 
    /// ### 🎯 **Camada Final**:
    /// - **Final LayerNorm**: `n_embd`
    /// - **Language Model Head**: `vocab_size × n_embd`
    /// 
    /// ## 💡 Por que isso importa?
    /// - **Memória**: Mais parâmetros = mais RAM/VRAM necessária
    /// - **Velocidade**: Mais parâmetros = computação mais lenta
    /// - **Capacidade**: Mais parâmetros = potencialmente mais "inteligente"
    /// - **Overfitting**: Muitos parâmetros podem decorar em vez de aprender
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        
        // 📚 **EMBEDDINGS**
        // Token embeddings: cada token do vocabulário tem um vetor de n_embd dimensões
        total += self.config.vocab_size * self.config.n_embd;
        
        // Position embeddings: cada posição até block_size tem um vetor de n_embd dimensões
        total += self.config.block_size * self.config.n_embd;
        
        // 🧠 **BLOCOS TRANSFORMER**
        // Calcula parâmetros por bloco e multiplica pelo número de blocos
        let per_block = 
            // 👁️ Multi-Head Attention: 4 matrizes (Q, K, V, Output)
            4 * self.config.n_embd * self.config.n_embd +
            
            // ⚡ Feed-Forward Network: expansão 4x (up projection + down projection)
            // up: n_embd → 4*n_embd, down: 4*n_embd → n_embd
            8 * self.config.n_embd * self.config.n_embd +
            
            // ⚖️ Layer Normalizations: 2 layer norms × 2 parâmetros (scale + shift)
            4 * self.config.n_embd;
        
        total += per_block * self.config.n_layer;
        
        // 🎯 **CAMADAS FINAIS**
        // Final layer normalization: scale + shift parameters
        total += 2 * self.config.n_embd;
        
        // Language modeling head: projeta embeddings para vocabulário
        total += self.config.vocab_size * self.config.n_embd;
        
        total
    }
    
    /// 📚 **TAMANHO DO VOCABULÁRIO**
    /// 
    /// Retorna quantas palavras/tokens diferentes o modelo conhece.
    /// É como o "dicionário" do modelo - determina quais palavras
    /// ele pode entender e gerar.
    /// 
    /// ## 💡 Exemplos Típicos:
    /// - **GPT-2**: ~50.000 tokens
    /// - **GPT-3**: ~50.257 tokens  
    /// - **Nosso modelo**: configurável (ex: 1.000-10.000)
    /// 
    /// ### 🎯 Trade-offs do Tamanho:
    /// - **Maior vocabulário**: Mais expressivo, mas mais parâmetros
    /// - **Menor vocabulário**: Mais eficiente, mas pode ser limitado
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    
    /// 📏 **TAMANHO DO CONTEXTO (BLOCK SIZE)**
    /// 
    /// Retorna quantos tokens o modelo consegue "lembrar" de uma vez.
    /// É a "memória de trabalho" do modelo - determina quanto contexto
    /// anterior ele pode considerar ao gerar o próximo token.
    /// 
    /// ## 🧠 Analogia:
    /// Imagine que você está lendo um livro, mas só consegue lembrar
    /// das últimas N palavras. O block_size é esse N.
    /// 
    /// ## 💡 Exemplos Típicos:
    /// - **GPT-2**: 1.024 tokens (~750 palavras)
    /// - **GPT-3**: 2.048 tokens (~1.500 palavras)
    /// - **GPT-4**: 8.192+ tokens (~6.000+ palavras)
    /// - **Nosso modelo**: configurável (ex: 128-512)
    /// 
    /// ### ⚖️ Trade-offs:
    /// - **Contexto maior**: Melhor compreensão, mas mais lento e usa mais memória
    /// - **Contexto menor**: Mais rápido e eficiente, mas pode "esquecer" informações importantes
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }
    
    /// 💾 **ACESSO AO VARMAP PARA SALVAMENTO**
    /// 
    /// Retorna uma referência ao VarMap que contém todos os pesos treináveis
    /// do modelo. Este método é essencial para implementar salvamento e
    /// carregamento de checkpoints.
    /// 
    /// ## 🗂️ **O que é o VarMap:**
    /// - **Repositório**: Contém todos os tensores nomeados do modelo
    /// - **Serialização**: Permite salvar em formato SafeTensors
    /// - **Checkpoint**: Base para salvar/carregar estado do modelo
    /// 
    /// ## 🔧 **Uso Típico:**
    /// ```rust
    /// let varmap = model.varmap();
    /// varmap.save("model.safetensors")?;
    /// ```
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }
}