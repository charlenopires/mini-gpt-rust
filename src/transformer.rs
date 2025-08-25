//! # Transformer Block: A Unidade Fundamental
//! 
//! 🏗️ Analogia: Como um prédio é feito de andares, nosso modelo
//! é feito de blocos Transformer empilhados. Cada bloco processa
//! e refina a informação antes de passar para o próximo.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};
use crate::attention::MultiHeadAttention;

/// 🍽️ **FEED-FORWARD NETWORK: PROCESSAMENTO INDIVIDUAL**
/// 
/// Após a "conversa coletiva" da atenção, cada token passa por
/// um processamento individual para refinar sua representação.
/// 
/// ## 🎭 Analogia do Restaurante:
/// Imagine um restaurante onde:
/// - **Atenção**: Garçons conversam entre si sobre pedidos
/// - **Feed-Forward**: Cada chef prepara seu prato individualmente
/// 
/// ## 🧮 Arquitetura Matemática:
/// ```text
/// Input:  [B, T, C] = [batch, seq_len, n_embd]
///         ↓ Linear 1 (expansão 4x)
/// Hidden: [B, T, 4C] = [batch, seq_len, 4*n_embd]
///         ↓ GELU (ativação)
/// Activated: [B, T, 4C]
///         ↓ Dropout (regularização)
/// Regularized: [B, T, 4C]
///         ↓ Linear 2 (contração)
/// Output: [B, T, C] = [batch, seq_len, n_embd]
/// ```
/// 
/// ## 🎯 Por que Expansão 4x?
/// - **Capacidade**: Mais neurônios = mais expressividade
/// - **Padrão**: Estabelecido pelo paper "Attention is All You Need"
/// - **Trade-off**: Balança performance vs. eficiência computacional
/// 
/// ## ⚡ Complexidade Computacional:
/// - **Parâmetros**: 8 × C² (duas matrizes: C→4C e 4C→C)
/// - **FLOPs**: O(8 × B × T × C²) por forward pass
pub struct FeedForward {
    fc1: Linear,        // 🚀 Primeira camada: expansão (C → 4C)
    fc2: Linear,        // 🎯 Segunda camada: contração (4C → C)
    dropout: f32,       // 🎲 Taxa de dropout para regularização
}

impl FeedForward {
    /// 🏗️ **CONSTRUTOR: INICIALIZANDO A REDE FEED-FORWARD**
    /// 
    /// Cria uma rede de duas camadas com expansão intermediária,
    /// seguindo o padrão estabelecido pelos Transformers originais.
    /// 
    /// ## 📊 Dimensões das Camadas:
    /// ```text
    /// fc1: [n_embd] → [4 * n_embd]  (expansão)
    /// fc2: [4 * n_embd] → [n_embd]  (contração)
    /// ```
    /// 
    /// ## 🎯 Parâmetros:
    /// - `n_embd`: Dimensão dos embeddings
    /// - `dropout`: Taxa de dropout (0.0 a 1.0)
    /// - `vb`: Variable builder para inicialização de pesos
    pub fn new(n_embd: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        // 📈 **EXPANSÃO 4X: PADRÃO DOS TRANSFORMERS**
        // Aumenta a capacidade representacional da rede
        // Permite capturar padrões mais complexos
        let hidden_dim = 4 * n_embd;
        
        // 🏗️ **CONSTRUÇÃO DAS CAMADAS LINEARES**
        Ok(Self {
            // 🚀 Primeira camada: expande o espaço de características
            fc1: linear(n_embd, hidden_dim, vb.pp("fc1"))?,
            
            // 🎯 Segunda camada: projeta de volta ao espaço original
            fc2: linear(hidden_dim, n_embd, vb.pp("fc2"))?,
            
            // 🎲 Taxa de dropout para regularização
            dropout,
        })
    }
    
    /// 🚀 **FORWARD PASS: PROCESSAMENTO NÃO-LINEAR**
    /// 
    /// Implementa o fluxo completo da rede feed-forward:
    /// Linear → GELU → Dropout → Linear
    /// 
    /// ## 🧮 Fórmula Matemática:
    /// ```text
    /// FFN(x) = Linear₂(Dropout(GELU(Linear₁(x))))
    /// ```
    /// 
    /// ## 🎭 Por que GELU em vez de ReLU?
    /// 
    /// ### 📊 **GELU vs ReLU:**
    /// ```text
    /// ReLU(x) = max(0, x)           # Função degrau
    /// GELU(x) = x * Φ(x)            # Função suave
    /// 
    /// Onde Φ(x) é a CDF da distribuição normal padrão
    /// ```
    /// 
    /// ### 🎯 **Vantagens do GELU:**
    /// - **Suavidade**: Gradientes mais estáveis
    /// - **Não-monotônico**: Permite valores negativos pequenos
    /// - **Probabilístico**: Baseado em distribuições estatísticas
    /// - **Performance**: Melhor em modelos de linguagem
    /// 
    /// ## 🔄 **Fluxo de Processamento:**
    /// ```text
    /// Input: "O gato subiu"  [B, T, C]
    ///        ↓ fc1 (expansão)
    /// Hidden: representações expandidas [B, T, 4C]
    ///        ↓ GELU (ativação)
    /// Active: ativações não-lineares [B, T, 4C]
    ///        ↓ dropout (regularização)
    /// Regularized: [B, T, 4C]
    ///        ↓ fc2 (contração)
    /// Output: representações refinadas [B, T, C]
    /// ```
    /// 
    /// ## 🎯 Parâmetros:
    /// - `x`: Tensor de entrada [batch_size, seq_len, n_embd]
    /// 
    /// ## 📤 Retorna:
    /// - Tensor processado [batch_size, seq_len, n_embd]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 🚀 **PRIMEIRA CAMADA LINEAR: EXPANSÃO 4X**
        // Transforma [B, T, C] → [B, T, 4C]
        // Aumenta a capacidade representacional para capturar padrões complexos
        let x = self.fc1.forward(x)?;
        
        // ⚡ **ATIVAÇÃO GELU: INTRODUÇÃO DE NÃO-LINEARIDADE**
        // GELU(x) = x * Φ(x) onde Φ é a CDF da normal padrão
        // Permite que a rede aprenda funções não-lineares complexas
        // Mais suave que ReLU, melhor para gradientes
        let x = x.gelu()?;
        
        // 🎲 **DROPOUT: REGULARIZAÇÃO ESTOCÁSTICA**
        // Durante treinamento: zera aleatoriamente alguns neurônios
        // Previne overfitting e melhora generalização
        // Durante inferência: mantém todos os neurônios ativos
        let x = if self.dropout > 0.0 {
            candle_nn::ops::dropout(&x, self.dropout)?
        } else {
            x  // Sem dropout durante inferência
        };
        
        // 🎯 **SEGUNDA CAMADA LINEAR: CONTRAÇÃO DE VOLTA**
        // Transforma [B, T, 4C] → [B, T, C]
        // Projeta as representações expandidas de volta ao espaço original
        // Mantém compatibilidade dimensional com conexões residuais
        self.fc2.forward(&x)
    }
}

/// 🏗️ **TRANSFORMER BLOCK: ARQUITETURA FUNDAMENTAL**
/// 
/// O bloco Transformer é a unidade básica que combina:
/// 1. **Multi-Head Attention**: Comunicação entre tokens
/// 2. **Feed-Forward Network**: Processamento individual
/// 3. **Layer Normalization**: Estabilização do treinamento
/// 4. **Residual Connections**: Fluxo de gradientes
/// 
/// ## 🎭 Analogia da Orquestra:
/// Imagine uma orquestra onde:
/// - **Atenção**: Músicos se coordenam entre si
/// - **Feed-Forward**: Cada músico aprimora sua parte
/// - **Layer Norm**: Maestro ajusta o volume geral
/// - **Residual**: Memória da melodia original
/// 
/// ## 🧮 Arquitetura Matemática:
/// ```text
/// Input: x [B, T, C]
///   ↓
/// x₁ = x + MultiHeadAttention(LayerNorm(x))
///   ↓
/// x₂ = x₁ + FeedForward(LayerNorm(x₁))
///   ↓
/// Output: x₂ [B, T, C]
/// ```
/// 
/// ## 🔄 **PRE-LN vs POST-LN:**
/// 
/// ### 📊 **Pre-LN (usado aqui):**
/// ```text
/// x = x + Attention(LayerNorm(x))
/// x = x + FFN(LayerNorm(x))
/// ```
/// 
/// ### 📊 **Post-LN (original):**
/// ```text
/// x = LayerNorm(x + Attention(x))
/// x = LayerNorm(x + FFN(x))
/// ```
/// 
/// ### ✅ **Vantagens do Pre-LN:**
/// - **Treinamento mais estável**: Gradientes mais suaves
/// - **Convergência mais rápida**: Menos explosão/desaparecimento
/// - **Menos sensível à inicialização**: Mais robusto
/// 
/// ## ⚡ Complexidade Computacional:
/// - **Parâmetros**: ~12 × C² (atenção + FFN + layer norms)
/// - **FLOPs**: O(12 × B × T × C² + 4 × B × T² × C) por bloco
pub struct TransformerBlock {
    attention: MultiHeadAttention,    // 🎯 Mecanismo de atenção multi-cabeça
    feed_forward: FeedForward,        // 🍽️ Rede feed-forward para processamento
    ln1: LayerNorm,                   // 📏 Layer norm antes da atenção
    ln2: LayerNorm,                   // 📏 Layer norm antes do feed-forward
}

impl TransformerBlock {
    /// 🏗️ **CONSTRUTOR: INICIALIZANDO O BLOCO TRANSFORMER**
    /// 
    /// Cria um bloco completo com todos os componentes necessários,
    /// seguindo a arquitetura Pre-LN para melhor estabilidade.
    /// 
    /// ## 🎯 Parâmetros:
    /// - `n_embd`: Dimensão dos embeddings (tipicamente 512, 768, 1024...)
    /// - `n_head`: Número de cabeças de atenção (tipicamente 8, 12, 16...)
    /// - `dropout`: Taxa de dropout para regularização (0.0 a 1.0)
    /// - `vb`: Variable builder para inicialização de pesos
    /// 
    /// ## 📊 Distribuição de Parâmetros:
    /// ```text
    /// MultiHeadAttention: ~4 × C² parâmetros (Q, K, V, Out)
    /// FeedForward:        ~8 × C² parâmetros (fc1, fc2)
    /// LayerNorm (2x):     ~4 × C parâmetros (scale, bias)
    /// Total:              ~12 × C² + 4 × C parâmetros
    /// ```
    pub fn new(n_embd: usize, n_head: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            // 🎯 **ATENÇÃO MULTI-CABEÇA**
            // Permite que o modelo "preste atenção" a diferentes
            // aspectos da sequência simultaneamente
            attention: MultiHeadAttention::new(n_embd, n_head, dropout, vb.pp("attention"))?,
            
            // 🍽️ **REDE FEED-FORWARD**
            // Processamento não-linear individual para cada posição
            // Expansão 4x seguida de contração para aumentar expressividade
            feed_forward: FeedForward::new(n_embd, dropout, vb.pp("feed_forward"))?,
            
            // 📏 **LAYER NORMALIZATIONS**
            // Pre-LN: normalização ANTES das operações principais
            // Estabiliza gradientes e acelera convergência
            // eps=1e-5 é o padrão para estabilidade numérica
            ln1: layer_norm(n_embd, 1e-5, vb.pp("ln1"))?,  // Antes da atenção
            ln2: layer_norm(n_embd, 1e-5, vb.pp("ln2"))?,  // Antes do feed-forward
        })
    }
    
    /// 🚀 **FORWARD PASS: PROCESSAMENTO COMPLETO DO BLOCO**
    /// 
    /// Implementa a arquitetura Pre-LN Transformer com conexões residuais:
    /// 
    /// ## 🔄 **Fluxo de Processamento:**
    /// ```text
    /// Input: x [B, T, C]
    ///   ↓
    /// 1️⃣ Atenção com Residual:
    ///    norm1 = LayerNorm(x)
    ///    attn_out = MultiHeadAttention(norm1, mask)
    ///    x = x + attn_out  # Conexão residual
    ///   ↓
    /// 2️⃣ Feed-Forward com Residual:
    ///    norm2 = LayerNorm(x)
    ///    ffn_out = FeedForward(norm2)
    ///    x = x + ffn_out   # Conexão residual
    ///   ↓
    /// Output: x [B, T, C]
    /// ```
    /// 
    /// ## 🎯 **Por que Conexões Residuais?**
    /// 
    /// ### 📊 **Problema dos Gradientes:**
    /// Em redes profundas, gradientes podem:
    /// - **Desaparecer**: Ficam muito pequenos (vanishing gradients)
    /// - **Explodir**: Ficam muito grandes (exploding gradients)
    /// 
    /// ### ✅ **Solução Residual:**
    /// ```text
    /// Sem residual:  y = F(x)
    /// Com residual:  y = x + F(x)
    /// ```
    /// 
    /// **Vantagens:**
    /// - **Gradiente direto**: ∂y/∂x = 1 + ∂F(x)/∂x
    /// - **Identidade preservada**: Se F(x) = 0, y = x
    /// - **Aprendizado incremental**: F(x) aprende apenas o "delta"
    /// 
    /// ## 🎭 **Analogia da Edição:**
    /// Imagine editar um documento:
    /// - **Sem residual**: Reescrever tudo do zero
    /// - **Com residual**: Fazer apenas correções/melhorias
    /// 
    /// ## 🔍 **Parâmetros:**
    /// - `x`: Tensor de entrada [batch_size, seq_len, n_embd]
    /// - `mask`: Máscara causal opcional para atenção
    /// 
    /// ## 📤 **Retorno:**
    /// - Tensor processado [batch_size, seq_len, n_embd]
            ln1: layer_norm(n_embd, 1e-5, vb.pp("ln1"))?,  // Antes da atenção
            ln2: layer_norm(n_embd, 1e-5, vb.pp("ln2"))?,  // Antes do feed-forward
        })
    }
    
    /// 🚀 **FORWARD PASS: PROCESSAMENTO COMPLETO DO BLOCO**
    /// 
    /// Implementa a arquitetura Pre-LN com conexões residuais:
    /// 1. **Atenção com Residual**: x + Attention(LayerNorm(x))
    /// 2. **Feed-Forward com Residual**: x + FFN(LayerNorm(x))
    /// 
    /// ## 🔄 **Por que Conexões Residuais?**
    /// 
    /// ### 🎯 **Problema do Gradiente Desaparecendo:**
    /// Em redes profundas, gradientes podem "desaparecer" durante
    /// o backpropagation, tornando o treinamento impossível.
    /// 
    /// ### ✅ **Solução das Residuais:**
    /// ```text
    /// ∂L/∂x = ∂L/∂output × (1 + ∂F(x)/∂x)
    /// ```
    /// O termo "1" garante que sempre há um caminho direto
    /// para os gradientes fluírem, mesmo se ∂F(x)/∂x ≈ 0.
    /// 
    /// ## 📏 **Layer Normalization:**
    /// 
    /// ### 🧮 **Fórmula:**
    /// ```text
    /// LayerNorm(x) = γ × (x - μ) / σ + β
    /// 
    /// Onde:
    /// μ = mean(x)     # Média da camada
    /// σ = std(x)      # Desvio padrão da camada
    /// γ, β            # Parâmetros aprendíveis
    /// ```
    /// 
    /// ### ✅ **Benefícios:**
    /// - **Estabilidade**: Normaliza ativações para média 0, std 1
    /// - **Velocidade**: Acelera convergência do treinamento
    /// - **Robustez**: Menos sensível à inicialização de pesos
    /// 
    /// ## 🎯 Parâmetros:
    /// - `x`: Tensor de entrada [batch_size, seq_len, n_embd]
    /// - `mask`: Máscara causal opcional para atenção
    /// 
    /// ## 📤 Retorna:
    /// - Tensor processado [batch_size, seq_len, n_embd]
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // 🎯 **ETAPA 1: ATENÇÃO COM CONEXÃO RESIDUAL**
        // 
        // Pre-LN: Normaliza ANTES da atenção (mais estável que Post-LN)
        // Isso estabiliza os gradientes e acelera a convergência
        let normalized_x = self.ln1.forward(x)?;
        
        // 🔍 **MULTI-HEAD ATTENTION**
        // Permite que o modelo "olhe" para diferentes posições simultaneamente
        // A máscara causal garante que só vemos tokens anteriores (autoregressive)
        let attn_out = self.attention.forward(&normalized_x, mask)?;
        
        // 🔄 **PRIMEIRA CONEXÃO RESIDUAL**
        // x_new = x_original + attention_output
        // Preserva a informação original e permite gradientes diretos
        let x = (x + attn_out)?;
        
        // 🍽️ **ETAPA 2: FEED-FORWARD COM CONEXÃO RESIDUAL**
        // 
        // Normalização antes do processamento feed-forward
        let normalized_x2 = self.ln2.forward(&x)?;
        
        // 🧠 **PROCESSAMENTO FEED-FORWARD**
        // Cada posição é processada independentemente
        // Expansão 4x → GELU → Contração → Dropout
        let ff_out = self.feed_forward.forward(&normalized_x2)?;
        
        // 🔄 **SEGUNDA CONEXÃO RESIDUAL**
        // x_final = x_after_attention + feedforward_output
        let x = (x + ff_out)?;
        
        // 📤 **SAÍDA FINAL**
        // Tensor com representações refinadas de cada token
        // Cada token agora "conhece" o contexto e foi processado individualmente
        // Dimensões preservadas: [batch_size, seq_len, n_embd]
        Ok(x)
    }
}