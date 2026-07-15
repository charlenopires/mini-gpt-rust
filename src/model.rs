//! # 🧠 Mini-GPT: Construindo um Large Language Model do Zero
//!
//! ## 📚 GUIA EDUCACIONAL COMPLETO: Como Construir um LLM
//!
//! Este arquivo implementa um **Large Language Model (LLM)** completo baseado na
//! arquitetura **Transformer GPT**. Vamos explicar cada componente em detalhes
//! para que você entenda exatamente como um "cérebro artificial" funciona!
//!
//! ## 🎯 O QUE É UM LARGE LANGUAGE MODEL?
//!
//! Um LLM é um modelo de IA que:
//! - **Entende** texto em linguagem natural
//! - **Gera** texto coerente e contextualmente relevante
//! - **Aprende** padrões da linguagem a partir de grandes volumes de texto
//! - **Generaliza** conhecimento para tarefas não vistas durante o treinamento
//!
//! ### 🧮 MATEMÁTICA POR TRÁS DOS LLMs:
//!
//! **Objetivo**: Dado uma sequência de tokens [t₁, t₂, ..., tₙ], predizer tₙ₊₁
//!
//! **Função de Probabilidade**:
//! ```
//! P(tₙ₊₁ | t₁, t₂, ..., tₙ) = softmax(f(t₁, t₂, ..., tₙ))
//! ```
//!
//! Onde `f()` é nossa rede neural Transformer que mapeia sequências para distribuições
//! de probabilidade sobre o vocabulário.
//!
//! ## 🏗️ ARQUITETURA TRANSFORMER: OS BLOCOS FUNDAMENTAIS
//!
//! ### 1. 📚 **TOKEN EMBEDDINGS** - Convertendo Palavras em Números
//!
//! **Problema**: Computadores não entendem palavras, apenas números.
//! **Solução**: Mapear cada palavra para um vetor de números reais.
//!
//! ```
//! "gato" → [0.2, -0.1, 0.8, 0.3, ...] (vetor de 512 dimensões)
//! "cão"  → [0.1, -0.2, 0.7, 0.4, ...] (vetor similar, pois são animais)
//! ```
//!
//! **Por que funciona?**
//! - Palavras similares têm vetores similares
//! - O modelo aprende essas representações durante o treinamento
//! - Permite operações matemáticas com conceitos linguísticos
//!
//! ### 2. 📍 **POSITION EMBEDDINGS** - Ensinando Ordem ao Modelo
//!
//! **Problema**: Transformers processam tokens em paralelo, perdendo noção de ordem.
//! **Solução**: Adicionar informação posicional a cada token.
//!
//! ```
//! "João ama Maria" vs "Maria ama João"
//! Posição 0: João/Maria + embedding_pos[0]
//! Posição 1: ama + embedding_pos[1]
//! Posição 2: Maria/João + embedding_pos[2]
//! ```
//!
//! ### 3. 🎯 **MULTI-HEAD ATTENTION** - O Coração do Transformer
//!
//! **Conceito**: Cada palavra "presta atenção" a todas as outras palavras.
//!
//! **Matemática da Atenção**:
//! ```
//! Q = X * W_q  (Query: "o que estou procurando?")
//! K = X * W_k  (Key: "o que eu ofereço?")
//! V = X * W_v  (Value: "qual informação eu carrego?")
//!
//! Attention(Q,K,V) = softmax(QK^T / √d_k) * V
//! ```
//!
//! **Por que múltiplas cabeças?**
//! - Cada cabeça captura um tipo diferente de relação
//! - Cabeça 1: relações sintáticas (sujeito-verbo)
//! - Cabeça 2: relações semânticas (causa-efeito)
//! - Cabeça 3: relações de longa distância
//!
//! ### 4. ⚡ **FEED-FORWARD NETWORKS** - Processamento Não-Linear
//!
//! **Função**: Aplicar transformações complexas a cada posição.
//!
//! **Arquitetura**:
//! ```
//! FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
//! ```
//!
//! **Fluxo de Processamento**:
//! ```
//! Input [B, T, C] → Linear₁ [B, T, 4C] → GELU → Linear₂ [B, T, C]
//! ```
//!
//! **Por que 4x expansão?**
//! - Permite ao modelo "pensar" em um espaço maior
//! - Captura interações complexas entre features
//! - Compensa a linearidade da atenção
//!
//! ### 5. 🔄 **RESIDUAL CONNECTIONS** - Facilitando o Aprendizado
//!
//! **Problema**: Redes profundas sofrem com gradientes que desaparecem.
//! **Solução**: Adicionar conexões diretas entre camadas.
//!
//! ```
//! output = input + transformation(input)
//! ```
//!
//! **Benefícios**:
//! - Gradientes fluem diretamente para camadas anteriores
//! - Permite treinar redes muito profundas (100+ camadas)
//! - Modelo aprende refinamentos incrementais
//!
//! ### 6. ⚖️ **LAYER NORMALIZATION** - Estabilizando o Treinamento
//!
//! **Função**: Normalizar ativações para ter média 0 e variância 1.
//!
//! **Fórmula**:
//! ```
//! LayerNorm(x) = γ * (x - μ) / σ + β
//! ```
//!
//! **Onde**:
//! - μ: média das ativações
//! - σ: desvio padrão das ativações  
//! - γ, β: parâmetros aprendidos
//!
//! ## 🎓 PROCESSO DE TREINAMENTO: COMO O MODELO APRENDE
//!
//! ### 📖 **1. TOKENIZAÇÃO**
//! ```
//! "O gato subiu" → [15, 234, 1891] (IDs dos tokens)
//! ```
//!
//! ### 🔢 **2. EMBEDDING**
//! ```
//! [15, 234, 1891] → [[0.1, 0.2, ...], [0.3, 0.1, ...], [0.8, 0.4, ...]]
//! ```
//!
//! ### 🧠 **3. PROCESSAMENTO TRANSFORMER**
//! ```
//! Para cada bloco:
//!   x = x + MultiHeadAttention(LayerNorm(x))
//!   x = x + FeedForward(LayerNorm(x))
//! ```
//!
//! ### 🎯 **4. PREDIÇÃO**
//! ```
//! hidden_states → logits → softmax → probabilidades
//! ```
//!
//! ### 📊 **5. LOSS CALCULATION**
//! ```
//! loss = CrossEntropy(predicted_probs, actual_next_token)
//! ```
//!
//! ### ⬅️ **6. BACKPROPAGATION**
//! ```
//! Ajustar pesos para minimizar loss usando gradientes
//! ```
//!
//! ## 🚀 OTIMIZAÇÕES DE PERFORMANCE
//!
//! ### 🔥 **KERNEL FUSION**
//! - Combina múltiplas operações em uma única passada
//! - Reduz overhead de memória e comunicação
//! - Melhora utilização de cache
//!
//! ### 🧠 **MEMORY MANAGEMENT**
//! - Pool de memória reutilizável
//! - Reduz fragmentação
//! - Otimiza alocações/desalocações
//!
//! ### ⚡ **MIXED PRECISION**
//! - Usa FP16 para forward pass (2x mais rápido)
//! - Mantém FP32 para gradientes (precisão)
//! - Reduz uso de memória pela metade
//!
//! ```
//! FFN(x) = max(0, x * W₁ + b₁) * W₂ + b₂
//! ```
//!
//! **Intuição**: Como neurônios no cérebro, cada FFN detecta padrões específicos
//! e os transforma em representações mais úteis.
//!
//! ### 5. ⚖️ **LAYER NORMALIZATION** - Estabilizando o Aprendizado
//!
//! **Problema**: Gradientes podem explodir ou desaparecer em redes profundas.
//! **Solução**: Normalizar ativações para ter média 0 e variância 1.
//!
//! ```
//! LayerNorm(x) = γ * (x - μ) / σ + β
//! ```
//!
//! ## 🔄 PROCESSO AUTOREGRESSIVO: Como o Modelo Gera Texto
//!
//! **Passo a Passo da Geração**:
//!
//! 1. **Entrada**: "O gato subiu na"
//! 2. **Tokenização**: [15, 234, 567, 89] (IDs dos tokens)
//! 3. **Embeddings**: Converter IDs em vetores densos
//! 4. **Transformer**: Processar através de N camadas
//! 5. **Projeção**: Mapear para probabilidades sobre vocabulário
//! 6. **Sampling**: Escolher próximo token baseado nas probabilidades
//! 7. **Repetir**: Adicionar token escolhido e continuar
//!
//! **Resultado**: "O gato subiu na árvore" (token "árvore" foi predito)
//!
//! ## 🎓 PROCESSO DE TREINAMENTO: Como o Modelo Aprende
//!
//! ### Forward Pass (Propagação Direta):
//! ```
//! Texto → Tokens → Embeddings → Transformer → Logits → Loss
//! ```
//!
//! ### Backward Pass (Retropropagação):
//! ```
//! Loss → ∂Loss/∂W → Gradientes → Atualização dos Pesos
//! ```
//!
//! ### Função de Loss (Cross-Entropy):
//! ```
//! Loss = -Σ log(P(token_correto | contexto))
//! ```
//!
//! **Objetivo**: Maximizar a probabilidade do token correto dado o contexto.
//!
//! ## 💡 POR QUE ESTA ARQUITETURA FUNCIONA?
//!
//! 1. **Paralelização**: Processa toda sequência simultaneamente
//! 2. **Atenção**: Captura dependências de longa distância
//! 3. **Profundidade**: Múltiplas camadas permitem abstrações complexas
//! 4. **Escala**: Funciona melhor com mais dados e parâmetros
//! 5. **Generalização**: Aprende padrões transferíveis
//!
//! Este arquivo implementa todos esses conceitos em Rust puro,
//! criando um LLM funcional e educativo!

use candle_core::{DType, Device, Tensor, IndexOp, Var};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap};
use crate::transformer::TransformerBlock;
use crate::tokenizer::BPETokenizer;
use crate::kernels::{FusionConfig, FusedMemoryManager};
use safetensors::SafeTensors;
// use std::collections::HashMap; // Removido - não utilizado
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// 📋 **METADADOS DO CHECKPOINT**
/// 
/// Estrutura que armazena informações essenciais sobre o modelo salvo:
/// - Configuração completa do modelo
/// - Timestamp de criação
/// - Versão do formato
/// - Métricas de treinamento
/// - Hash de integridade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub config: GPTConfig,
    pub timestamp: String,
    pub version: String,
    pub training_step: Option<usize>,
    pub loss: Option<f32>,
    pub learning_rate: Option<f32>,
    pub model_hash: Option<String>,
    pub description: Option<String>,
}

impl CheckpointMetadata {
    pub fn new(config: GPTConfig) -> Self {
        Self {
            config,
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            training_step: None,
            loss: None,
            learning_rate: None,
            model_hash: None,
            description: None,
        }
    }
    
    pub fn with_training_info(mut self, step: usize, loss: f32, lr: f32) -> Self {
        self.training_step = Some(step);
        self.loss = Some(loss);
        self.learning_rate = Some(lr);
        self
    }
    
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

/// 🔧 **CONFIGURAÇÃO DO MODELO GPT** - Os "Genes" do Nosso LLM
///
/// Esta estrutura define a "arquitetura genética" do nosso modelo.
/// Cada parâmetro controla um aspecto fundamental do comportamento do LLM.
///
/// ## 📊 **HIPERPARÂMETROS EXPLICADOS EM DETALHES**:
///
/// ### `vocab_size` 📚 - Tamanho do Vocabulário
/// - **O que é**: Quantas palavras/tokens diferentes o modelo conhece
/// - **Exemplo**: 50,000 = modelo conhece 50 mil palavras únicas
/// - **Impacto**: Maior vocabulário = mais expressivo, mas mais memória
/// - **Analogia**: Como o "dicionário" que o modelo tem acesso
/// - **Matemática**: Determina dimensão da matriz de saída (vocab_size × n_embd)
///
/// ### `n_embd` 🧮 - Dimensão dos Embeddings
/// - **O que é**: Tamanho dos vetores que representam cada palavra
/// - **Exemplo**: 512 = cada palavra vira um vetor de 512 números
/// - **Impacto**: Maior dimensão = mais capacidade expressiva
/// - **Analogia**: "Resolução" da representação das palavras
/// - **Trade-off**: Mais dimensões = mais parâmetros = mais memória/computação
///
/// ### `n_head` 👁️ - Número de Cabeças de Atenção
/// - **O que é**: Quantos "focos de atenção" paralelos o modelo tem
/// - **Exemplo**: 8 cabeças = 8 tipos diferentes de relações capturadas
/// - **Impacto**: Mais cabeças = mais tipos de padrões detectados
/// - **Analogia**: Como ter múltiplos "olhos" vendo aspectos diferentes
/// - **Restrição**: n_embd deve ser divisível por n_head
///
/// ### `n_layer` 🏗️ - Número de Camadas Transformer
/// - **O que é**: Profundidade da rede neural
/// - **Exemplo**: 12 camadas = 12 níveis de processamento
/// - **Impacto**: Mais camadas = abstrações mais complexas
/// - **Analogia**: Como "níveis de pensamento" - superficial → profundo
/// - **Comparação**: GPT-3 tem 96 camadas, nosso modelo educacional tem 4
///
/// ### `block_size` 📏 - Tamanho Máximo da Sequência
/// - **O que é**: Quantos tokens o modelo pode processar de uma vez
/// - **Exemplo**: 1024 = pode "lembrar" de até 1024 palavras anteriores
/// - **Impacto**: Maior contexto = melhor compreensão, mas mais memória
/// - **Analogia**: "Memória de trabalho" do modelo
/// - **Complexidade**: Atenção é O(n²) em relação ao block_size
///
/// ### `dropout` 🎲 - Taxa de Regularização
/// - **O que é**: Probabilidade de "desligar" neurônios durante treinamento
/// - **Exemplo**: 0.1 = 10% dos neurônios são ignorados aleatoriamente
/// - **Impacto**: Previne overfitting, melhora generalização
/// - **Analogia**: Como "treinar com uma mão amarrada" para ficar mais forte
/// - **Importante**: Usado apenas no treinamento, desabilitado na inferência
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPTConfig {
    pub vocab_size: usize,   // 📚 Tamanho do vocabulário (quantas palavras o modelo conhece)
    pub n_embd: usize,       // 🧮 Dimensão dos embeddings (largura do modelo)
    pub n_head: usize,       // 👁️ Número de cabeças de atenção paralelas
    pub n_layer: usize,      // 🏗️ Número de blocos transformer (profundidade)
    pub block_size: usize,   // 📏 Tamanho máximo do contexto (memória do modelo)
    pub dropout: f32,        // 🎲 Taxa de dropout para regularização
}

/// 🤖 **MINI-GPT: MODELO TRANSFORMER COMPLETO** - O "Cérebro" do LLM
///
/// Esta é a implementação principal do nosso Large Language Model.
/// Funciona como um "cérebro artificial" que aprendeu padrões complexos
/// da linguagem humana através de treinamento em vastos corpora de texto.
///
/// ## 🧠 **ARQUITETURA DETALHADA DO MODELO**:
///
/// ### 📊 **FLUXO DE DADOS COMPLETO**:
/// ```text
/// Texto: "O gato subiu na árvore"
///   ↓ Tokenização
/// Tokens: [15, 234, 567, 89, 1024]
///   ↓ Token Embeddings (vocab_size → n_embd)
/// Vetores: [[0.1, 0.2, ...], [0.3, 0.1, ...], ...]
///   ↓ Position Embeddings (block_size → n_embd)
/// Vetores + Posição: [[0.1+pos₀, 0.2+pos₀, ...], ...]
///   ↓ Transformer Blocks (N camadas)
/// [Atenção Multi-Cabeça + Feed-Forward + LayerNorm] × N
/// Representações Contextuais: [[ctx₁], [ctx₂], ...]
///   ↓ Layer Normalization Final
/// Representações Normalizadas
///   ↓ Language Model Head (n_embd → vocab_size)
/// Logits: [score("O"), score("gato"), ..., score("árvore")]
///   ↓ Softmax
/// Probabilidades: [0.001, 0.002, ..., 0.85]
/// ```
///
/// ### 🔄 **PROCESSO AUTOREGRESSIVO**:
///
/// O modelo gera texto de forma **autoregressiva**:
/// 1. **Entrada**: "O gato subiu na"
/// 2. **Predição**: P(próximo_token | "O gato subiu na")
/// 3. **Sampling**: Escolhe "árvore" baseado nas probabilidades
/// 4. **Iteração**: "O gato subiu na árvore" → prediz próximo token
/// 5. **Repetição**: Continua até token de fim ou limite atingido
///
/// ### 🎯 **MATEMÁTICA FUNDAMENTAL**:
///
/// **Objetivo do Modelo**:
/// ```
/// P(w₁, w₂, ..., wₙ) = ∏ᵢ P(wᵢ | w₁, w₂, ..., wᵢ₋₁)
/// ```
///
/// **Função de Loss (Cross-Entropy)**:
/// ```
/// L = -1/N ∑ᵢ log P(wᵢ | contexto)
/// ```
///
/// Onde cada componente abaixo contribui para essa capacidade preditiva:
pub struct MiniGPT {
    config: GPTConfig,              // 🎛️ Configuração do modelo (hiperparâmetros)
    
    // 📚 **CAMADAS DE EMBEDDING** - Convertendo Símbolos em Números
    // Estas camadas transformam tokens discretos em representações vetoriais contínuas
    // que o modelo pode processar matematicamente
    token_embedding: Embedding,     // 🔤 Matriz (vocab_size × n_embd): converte IDs → vetores
    position_embedding: Embedding,  // 📍 Matriz (block_size × n_embd): adiciona contexto posicional
    
    // 🏗️ **BLOCOS TRANSFORMER EMPILHADOS** - O "Processador" do Modelo
    // Cada bloco contém: Multi-Head Attention + Feed-Forward + Layer Normalizations
    // Juntos, eles capturam padrões complexos e dependências de longa distância
    blocks: Vec<TransformerBlock>,  // 🧠 Stack de N camadas que refinam representações
    
    // 🎯 **CAMADAS DE SAÍDA** - Convertendo Representações em Predições
    ln_final: LayerNorm,           // ⚖️ Normalização final (estabiliza gradientes)
    lm_head: Linear,               // 🎪 Matriz (n_embd × vocab_size): projeta para vocabulário
    
    device: Device,                // 💻 Dispositivo de computação (CPU/GPU/TPU)
    
    // 💾 **SISTEMA DE PERSISTÊNCIA** - Salvando o "Cérebro" Treinado
    // O VarMap contém todos os pesos treináveis do modelo para serialização
    // É como um "mapa" de todas as conexões neurais aprendidas
    varmap: VarMap,                // 🗂️ Registro de todas as variáveis treináveis
    
    // ⚡ **OTIMIZAÇÕES DE KERNEL FUSION** - Acelerando Computações
    // Sistemas avançados que combinam operações para máxima eficiência
    // Reduzem overhead de memória e aceleram forward/backward passes
    fusion_config: FusionConfig,   // 🔧 Configuração de otimizações (quais kernels usar)
    memory_manager: Option<FusedMemoryManager>, // 🧠 Pool inteligente de memória reutilizável
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
        
        // ⚡ **CONFIGURAÇÃO DE KERNEL FUSION**
        // Inicializa otimizações de baixo nível baseadas no dispositivo
        let fusion_config = FusionConfig {
            enable_attention_fusion: matches!(device, Device::Metal(_)),
            enable_feedforward_fusion: matches!(device, Device::Metal(_)),
            enable_memory_optimization: true,
            fusion_threshold: if matches!(device, Device::Metal(_)) { 512 } else { 2048 },
        };
        
        // 🧠 **GERENCIADOR DE MEMÓRIA FUSIONADO**
        // Ativa apenas para dispositivos que se beneficiam (Metal GPU)
        let memory_manager = if fusion_config.enable_memory_optimization {
            Some(FusedMemoryManager::new(fusion_config.clone(), device.clone()))
        } else {
            None
        };
        
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
            fusion_config,             // ⚡ Configuração de kernel fusion
            memory_manager,            // 🧠 Gerenciador de memória otimizado
        })
    }
    
    /// 📂 **CARREGAMENTO DE MODELO DE CHECKPOINT**
    /// 
    /// Carrega um modelo completo de um arquivo SafeTensors com metadados.
    /// Este método implementa um sistema robusto de checkpoint que permite:
    /// 
    /// ## 🔧 **Funcionalidades:**
    /// - Carregamento seguro de tensores com SafeTensors
    /// - Validação de integridade dos dados
    /// - Verificação de compatibilidade de configuração
    /// - Recuperação de metadados de treinamento
    /// - Suporte a diferentes versões de modelo
    /// 
    /// ## 📋 **Processo de Carregamento:**
    /// 1. Lê arquivo SafeTensors do disco
    /// 2. Extrai metadados JSON do header
    /// 3. Valida configuração do modelo
    /// 4. Cria estrutura do modelo
    /// 5. Carrega pesos nos tensores
    /// 6. Verifica integridade (opcional)
    pub fn load_from_checkpoint<P: AsRef<Path>>(path: P, device: &Device) -> Result<(Self, CheckpointMetadata)> {
        let path = path.as_ref();
        
        println!("📂 Carregando modelo de checkpoint: {}", path.display());
        
        // 🔍 **LEITURA DO ARQUIVO SAFETENSORS**
        let data = fs::read(path)
            .map_err(|e| format!("Erro ao ler arquivo {}: {}", path.display(), e))?;
        
        let safetensors = SafeTensors::deserialize(&data)
            .map_err(|e| format!("Erro ao deserializar SafeTensors: {}", e))?;
        
        // 📋 **EXTRAÇÃO DE METADADOS**
        // Lê a configuração real salva pelo `save_checkpoint` (sidecar `<path>.json`).
        // Só cai no fallback genérico (GPT-2 defaults) se o checkpoint não tiver
        // metadados — ex: arquivos legados salvos antes desse sidecar existir.
        let metadata_path = path.with_extension("json");
        let metadata = match fs::read_to_string(&metadata_path) {
            Ok(json_str) => match serde_json::from_str::<CheckpointMetadata>(&json_str) {
                Ok(meta) => meta,
                Err(e) => {
                    println!("⚠️  Metadados inválidos em {}: {} — usando configuração padrão", metadata_path.display(), e);
                    CheckpointMetadata::new(GPTConfig {
                        vocab_size: 50257,
                        n_embd: 768,
                        n_head: 12,
                        n_layer: 12,
                        block_size: 1024,
                        dropout: 0.1,
                    })
                }
            },
            Err(_) => {
                println!("⚠️  Sidecar de metadados não encontrado ({}) — usando configuração padrão", metadata_path.display());
                CheckpointMetadata::new(GPTConfig {
                    vocab_size: 50257,
                    n_embd: 768,
                    n_head: 12,
                    n_layer: 12,
                    block_size: 1024,
                    dropout: 0.1,
                })
            }
        };

        println!("✅ Metadados carregados:");
        println!("   📅 Timestamp: {}", metadata.timestamp);
        println!("   🔢 Versão: {}", metadata.version);
        if let Some(step) = metadata.training_step {
            println!("   🎯 Passo de treinamento: {}", step);
        }
        if let Some(loss) = metadata.loss {
            println!("   📉 Loss: {:.4}", loss);
        }
        
        // 🏗️ **CRIAÇÃO DO MODELO COM CONFIGURAÇÃO CARREGADA**
        let config = metadata.config.clone();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        // 🔧 **CONSTRUÇÃO DA ARQUITETURA**
        let token_embedding = embedding(config.vocab_size, config.n_embd, vb.pp("token_emb"))?;
        let position_embedding = embedding(config.block_size, config.n_embd, vb.pp("pos_emb"))?;
        
        let mut blocks = Vec::new();
        for i in 0..config.n_layer {
            blocks.push(TransformerBlock::new(
                config.n_embd,
                config.n_head,
                config.dropout,
                vb.pp(format!("block_{}", i)),
            )?);
        }
        
        let ln_final = layer_norm(config.n_embd, 1e-5, vb.pp("ln_final"))?;
        let lm_head = linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;
        
        // 💾 **CARREGAMENTO DOS PESOS**
        println!("💾 Carregando pesos dos tensores...");

        // Carrega tensores do SafeTensors para o VarMap.
        // Um único lock, iterado mutavelmente — travar o Mutex de novo dentro
        // do loop (como antes) causa deadlock, já que `std::sync::Mutex` não
        // é reentrante na mesma thread.
        {
            let mut data = varmap.data().lock().unwrap();
            for (name, var_tensor) in data.iter_mut() {
                if let Ok(tensor_view) = safetensors.tensor(name) {
                    let shape: Vec<usize> = tensor_view.shape().iter().map(|&x| x).collect();
                    let tensor = Tensor::from_raw_buffer(
                        tensor_view.data(),
                        DType::F32,
                        &shape,
                        device,
                    )?;
                    *var_tensor = Var::from_tensor(&tensor)?;
                }
            }
        }
        
        // ⚡ **CONFIGURAÇÃO DE KERNEL FUSION PARA MODELO CARREGADO**
        let fusion_config = FusionConfig {
            enable_attention_fusion: matches!(device, Device::Metal(_)),
            enable_feedforward_fusion: matches!(device, Device::Metal(_)),
            enable_memory_optimization: true,
            fusion_threshold: if matches!(device, Device::Metal(_)) { 512 } else { 2048 },
        };
        
        let memory_manager = if fusion_config.enable_memory_optimization {
            Some(FusedMemoryManager::new(fusion_config.clone(), device.clone()))
        } else {
            None
        };
        
        let model = Self {
            config: config.clone(),
            token_embedding,
            position_embedding,
            blocks,
            ln_final,
            lm_head,
            device: device.clone(),
            varmap,
            fusion_config,
            memory_manager,
        };
        
        println!("🎉 Modelo carregado com sucesso!");
        println!("   📊 Parâmetros: {:.1}M", model.num_parameters() as f32 / 1_000_000.0);
        
        Ok((model, metadata))
    }
    
    /// 🔍 **LISTAGEM DE CHECKPOINTS DISPONÍVEIS**
    /// 
    /// Escaneia um diretório em busca de arquivos de checkpoint válidos
    /// e retorna informações sobre cada um deles.
    pub fn list_checkpoints<P: AsRef<Path>>(dir: P) -> Result<Vec<(String, CheckpointMetadata)>> {
        let dir = dir.as_ref();
        let mut checkpoints = Vec::new();
        
        if !dir.exists() {
            return Ok(checkpoints);
        }
        
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                let filename = path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                // Tenta carregar metadados do arquivo .json correspondente
                let metadata_path = path.with_extension("json");
                let metadata = if metadata_path.exists() {
                    match fs::read_to_string(&metadata_path) {
                        Ok(json_str) => {
                            match serde_json::from_str::<CheckpointMetadata>(&json_str) {
                                Ok(meta) => meta,
                                Err(_) => {
                                    // Se falhar ao parsear, cria metadados padrão
                                    Self::create_default_metadata(&filename)
                                }
                            }
                        }
                        Err(_) => Self::create_default_metadata(&filename)
                    }
                } else {
                    // Se não existe arquivo de metadados, cria padrão
                    Self::create_default_metadata(&filename)
                };

                // Guarda o caminho completo (não só o nome do arquivo) — quem
                // chama esta função precisa conseguir abrir o arquivo depois,
                // e `dir` pode não ser o diretório de trabalho atual.
                checkpoints.push((path.to_string_lossy().to_string(), metadata));
            }
        }
        
        // Ordena por timestamp (mais recente primeiro)
        checkpoints.sort_by(|a, b| b.1.timestamp.cmp(&a.1.timestamp));
        
        Ok(checkpoints)
    }
    
    /// 🏗️ **CRIAÇÃO DE METADADOS PADRÃO**
    /// 
    /// Cria metadados padrão quando não conseguimos carregar do arquivo.
    fn create_default_metadata(filename: &str) -> CheckpointMetadata {
        let config = GPTConfig {
            vocab_size: 50257,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            block_size: 1024,
            dropout: 0.1,
        };
        
        CheckpointMetadata::new(config)
            .with_description(format!("Checkpoint carregado de {}", filename))
    }
    
    /// 💾 **SALVAMENTO DE CHECKPOINT COM METADADOS**
    /// 
    /// Salva o modelo atual em formato SafeTensors junto com metadados em JSON.
    /// Isso permite um carregamento mais robusto e informativo.
    pub fn save_checkpoint<P: AsRef<Path>>(
        &self, 
        path: P, 
        training_step: Option<usize>,
        loss: Option<f32>,
        learning_rate: Option<f32>,
        description: Option<String>
    ) -> Result<()> {
        let path = path.as_ref();
        
        // 📊 **CRIAÇÃO DOS METADADOS**
        let mut metadata = CheckpointMetadata::new(self.config.clone());
        
        if let (Some(step), Some(loss_val), Some(lr)) = (training_step, loss, learning_rate) {
            metadata = metadata.with_training_info(step, loss_val, lr);
        }
        
        if let Some(desc) = description {
            metadata = metadata.with_description(desc);
        }
        
        // 🔐 **SALVAMENTO DOS TENSORES EM SAFETENSORS**
        println!("💾 Salvando tensores em: {:?}", path);
        
        // Coleta todos os tensores do VarMap
        let tensors: std::collections::HashMap<String, Tensor> = self.varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
            .collect();
        
        // Salva em formato SafeTensors
        candle_core::safetensors::save(&tensors, path)?;
        
        // 📄 **SALVAMENTO DOS METADADOS EM JSON**
        let metadata_path = path.with_extension("json");
        println!("📄 Salvando metadados em: {:?}", metadata_path);
        
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(&metadata_path, metadata_json)?;
        
        println!("✅ Checkpoint salvo com sucesso!");
        println!("   📁 Tensores: {:?}", path);
        println!("   📄 Metadados: {:?}", metadata_path);
        
        Ok(())
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
    /// 🚀 **FORWARD PASS: O CORAÇÃO DO MODELO**
    /// 
    /// Este método implementa o "pensamento" do modelo - como ele processa
    /// uma sequência de tokens e produz predições para o próximo token.
    /// 
    /// ## 🔄 **Fluxo de Processamento:**
    /// 
    /// ### 📥 **Entrada:**
    /// - `idx`: Tensor [batch_size, seq_len] com IDs dos tokens
    /// - `targets`: Opcional - tokens corretos para calcular loss (treinamento)
    /// 
    /// ### 🧠 **Processamento:**
    /// 1. **Token Embeddings**: IDs → vetores densos
    /// 2. **Position Embeddings**: adiciona informação posicional
    /// 3. **Transformer Blocks**: refinamento através de atenção e feed-forward
    /// 4. **Layer Norm Final**: normalização das representações
    /// 5. **Language Head**: projeção para vocabulário
    /// 
    /// ### 📤 **Saída:**
    /// - `logits`: [batch_size, seq_len, vocab_size] - "confiança" para cada token
    /// - `loss`: Opcional - erro entre predição e target (se fornecido)
    /// 
    /// ## 🎯 **Exemplo Prático:**
    /// ```text
    /// Entrada: "O gato subiu no"
    /// Tokens:  [15, 234, 891, 45]  (IDs dos tokens)
    /// 
    /// Forward Pass:
    /// [15, 234, 891, 45] → Embeddings → Attention → FFN → ... → Logits
    /// 
    /// Logits finais: [0.1, 2.3, 0.8, 4.1, ...]  (para cada palavra do vocabulário)
    /// Predição: token com maior logit = "telhado" (ID 4)
    /// 
    /// Resultado: "O gato subiu no telhado"
    /// ```
    /// 
    /// ## ⚡ **Otimizações Implementadas:**
    /// - Kernel fusion para operações de atenção
    /// - Memory pooling para reduzir alocações
    /// - Verificações de integridade numérica
    /// - Suporte a diferentes dispositivos (CPU/GPU)
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

    /// 🔎 **FORWARD COM PESOS DE ATENÇÃO EXPOSTOS**
    ///
    /// Roda o mesmo forward pass de inferência, mas também retorna a matriz
    /// de pesos de atenção `[B, H, T, T]` de cada camada — usado pela API de
    /// demonstração para desenhar um heatmap real a partir de um prompt do
    /// usuário. Não é usado no loop de treino nem em `generate()`.
    pub fn forward_with_attention(&self, idx: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let (batch_size, seq_len) = idx.dims2()?;
        assert!(seq_len <= self.config.block_size,
                "Sequência muito longa! Max: {}, Atual: {}",
                self.config.block_size, seq_len);

        let tok_emb = self.token_embedding.forward(idx)?;
        let pos = Tensor::arange(0, seq_len as i64, &self.device)?;
        let pos_emb = self.position_embedding.forward(&pos)?;
        let pos_emb = pos_emb.unsqueeze(0)?.expand(&[batch_size, seq_len, self.config.n_embd])?;
        let mut x = (tok_emb + pos_emb)?;

        let mask = self.create_causal_mask(seq_len)?;

        let mut layer_weights = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let (out, weights) = block.forward_with_attention(&x, Some(&mask))
                .map_err(|e| format!("Erro no bloco {}: {}", i, e))?;
            x = out;
            layer_weights.push(weights);
        }

        let x = self.ln_final.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;

        Ok((logits, layer_weights))
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
    /// 🎭 **GERAÇÃO DE TEXTO AUTOREGRESSIVA**
    /// 
    /// Este é o coração da geração de texto em LLMs! O método implementa
    /// o processo autoregressivo onde cada token gerado alimenta a próxima predição.
    /// 
    /// ## 🔄 Processo Autoregressivo:
    /// 
    /// ```text
    /// Prompt: "O gato subiu"
    /// 
    /// Passo 1: [O, gato, subiu] → Modelo → P(próximo_token)
    ///          Escolhe: "na" (probabilidade 0.7)
    /// 
    /// Passo 2: [O, gato, subiu, na] → Modelo → P(próximo_token)
    ///          Escolhe: "árvore" (probabilidade 0.5)
    /// 
    /// Passo 3: [O, gato, subiu, na, árvore] → Modelo → P(próximo_token)
    ///          Escolhe: "." (probabilidade 0.8)
    /// 
    /// Resultado: "O gato subiu na árvore."
    /// ```
    /// 
    /// ## 🌡️ Controle de Temperatura:
    /// 
    /// A temperatura controla o quão "criativo" vs "conservador" o modelo será:
    /// 
    /// - **Temperatura = 0.1**: Muito conservador, sempre escolhe o mais provável
    ///   - Resultado: Texto previsível, mas coerente
    ///   - Uso: Respostas factuais, traduções
    /// 
    /// - **Temperatura = 1.0**: Balanceado, respeita as probabilidades originais
    ///   - Resultado: Boa mistura de coerência e criatividade
    ///   - Uso: Conversação geral
    /// 
    /// - **Temperatura = 2.0**: Muito criativo, distribui probabilidades
    ///   - Resultado: Texto mais variado, às vezes incoerente
    ///   - Uso: Brainstorming, poesia
    /// 
    /// ## ⚡ Otimizações Implementadas:
    /// 
    /// 1. **Sliding Window**: Limita contexto para evitar overflow de memória
    /// 2. **Batch Size = 1**: Otimizado para inferência interativa
    /// 3. **Early Stopping**: Para quando encontra token de fim
    /// 4. **Error Handling**: Propagação detalhada de erros
    /// 
    /// ## 📊 Parâmetros:
    /// - `prompt`: Texto inicial para começar a geração
    /// - `max_tokens`: Máximo de tokens a gerar (controla tamanho)
    /// - `tokenizer`: Conversor texto ↔ números
    /// - `temperature`: Controle criatividade (0.1 = conservador, 2.0 = criativo)
    /// 
    /// ## 🎯 Retorno:
    /// String com o texto gerado (apenas a parte nova, sem o prompt)
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

    /// 🌊 **GERAÇÃO COM STREAMING DE TOKENS**
    ///
    /// Mesmo algoritmo de `generate`, mas chama `on_token` a cada token
    /// gerado (em vez de só devolver o texto completo no final) — usado
    /// pela API de demonstração para transmitir a geração via SSE conforme
    /// ela acontece de verdade, token a token.
    pub fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        tokenizer: &BPETokenizer,
        temperature: f32,
        mut on_token: impl FnMut(usize, &str),
    ) -> Result<String> {
        let mut tokens = tokenizer.encode(prompt)
            .map_err(|e| format!("Erro na tokenização do prompt: {}", e))?;
        let mut generated_tokens = Vec::new();

        for step in 0..max_tokens {
            let context = if tokens.len() > self.config.block_size {
                &tokens[tokens.len() - self.config.block_size..]
            } else {
                &tokens[..]
            };

            let context_i64: Vec<i64> = context.iter().map(|&x| x as i64).collect();
            let idx = Tensor::from_slice(&context_i64, &[1, context.len()], &self.device)
                .map_err(|e| format!("Erro ao criar tensor no passo {}: {}", step, e))?;

            let (logits, _) = self.forward(&idx, None)
                .map_err(|e| format!("Erro no forward pass no passo {}: {}", step, e))?;

            let logits = logits.i((0, context.len() - 1, ..))
                .map_err(|e| format!("Erro ao extrair logits no passo {}: {}", step, e))?;

            let temperature_tensor = Tensor::new(&[temperature], &self.device)
                .map_err(|e| format!("Erro ao criar tensor de temperatura no passo {}: {}", step, e))?;
            let logits = logits.broadcast_div(&temperature_tensor)
                .map_err(|e| format!("Erro ao aplicar temperatura no passo {}: {}", step, e))?;

            let probs = candle_nn::ops::softmax(&logits, 0)
                .map_err(|e| format!("Erro no softmax no passo {}: {}", step, e))?;

            let next_token = self.sample_from_probs(&probs)
                .map_err(|e| format!("Erro na amostragem no passo {}: {}", step, e))?;

            tokens.push(next_token);
            generated_tokens.push(next_token);

            let token_text = tokenizer.decode(&[next_token]).unwrap_or_default();
            on_token(next_token, &token_text);

            if tokenizer.is_eos_token(next_token) {
                break;
            }
        }

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

    /// 🎛️ **ACESSO À CONFIGURAÇÃO**
    /// Retorna uma referência à configuração do modelo
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    /// 🔗 **EMBEDDINGS REAIS DE UM CONJUNTO DE TOKENS**
    ///
    /// Retorna as linhas da matriz de token embeddings correspondentes aos
    /// IDs dados — usado pela API de demonstração para visualizar o espaço
    /// de embeddings real do modelo (não valores aleatórios).
    pub fn embedding_for_tokens(&self, token_ids: &[usize]) -> Result<Tensor> {
        let ids: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
        let idx = Tensor::from_slice(&ids, &[ids.len()], &self.device)?;
        Ok(self.token_embedding.forward(&idx)?)
    }
}