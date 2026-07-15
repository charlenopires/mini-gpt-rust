//! # 🧠 SELF-ATTENTION: O CORAÇÃO PULSANTE DO TRANSFORMER
//! 
//! Este módulo implementa o mecanismo de atenção que revolucionou o NLP.
//! É aqui que a "mágica" dos Transformers realmente acontece!
//! 
//! ## 🎯 ANALOGIA INTUITIVA: A SALA DE AULA INTELIGENTE
//! 
//! Imagine uma sala de aula onde cada aluno (token) pode "prestar atenção"
//! em todos os outros alunos simultaneamente para entender melhor o contexto:
//! 
//! ```text
//! Frase: "O gato subiu no telhado"
//! 
//! Token "gato":
//!   - Presta muita atenção em "subiu" (verbo relacionado)
//!   - Presta pouca atenção em "no" (preposição menos relevante)
//!   - Presta atenção média em "telhado" (objeto da ação)
//! 
//! Token "subiu":
//!   - Presta muita atenção em "gato" (sujeito da ação)
//!   - Presta muita atenção em "telhado" (destino da ação)
//!   - Presta pouca atenção em "O" (artigo menos relevante)
//! ```
//! 
//! ## 🔬 FUNDAMENTOS MATEMÁTICOS
//! 
//! O mecanismo de atenção implementa a fórmula:
//! 
//! ```text
//! Attention(Q,K,V) = softmax(QK^T / √d_k)V
//! 
//! Onde:
//! - Q (Query): "O que eu quero saber?"
//! - K (Key): "O que eu tenho para oferecer?"
//! - V (Value): "Qual é minha informação real?"
//! - d_k: Dimensão das chaves (para normalização)
//! ```
//! 
//! ## 🎭 MULTI-HEAD ATTENTION: MÚLTIPLAS PERSPECTIVAS
//! 
//! Como ter vários "tipos de atenção" simultaneamente:
//! - **Cabeça 1**: Foca em relações sintáticas (sujeito-verbo)
//! - **Cabeça 2**: Foca em relações semânticas (palavras relacionadas)
//! - **Cabeça 3**: Foca em dependências de longo alcance
//! - **Cabeça N**: Cada uma aprende padrões diferentes!

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// 🎯 **SELF-ATTENTION: IMPLEMENTAÇÃO DO MECANISMO REVOLUCIONÁRIO**
/// 
/// Esta estrutura implementa o Scaled Dot-Product Attention, o coração
/// dos modelos Transformer que permitiu avanços como GPT, BERT, e ChatGPT.
/// 
/// ## 🧮 COMPONENTES PRINCIPAIS:
/// 
/// ### 🔍 **Query (Q)**: "O que eu quero saber?"
/// - Representa a "pergunta" que cada token faz
/// - Determina que tipo de informação o token está buscando
/// - Exemplo: token "ele" busca informação sobre a quem se refere
/// 
/// ### 🗝️ **Key (K)**: "O que eu tenho para oferecer?"
/// - Representa o "índice" ou "etiqueta" de cada token
/// - Determina que tipo de informação cada token pode fornecer
/// - Exemplo: token "João" oferece informação sobre uma pessoa
/// 
/// ### 💎 **Value (V)**: "Qual é minha informação real?"
/// - Contém a informação semântica real do token
/// - É o que será "misturado" baseado nos scores de atenção
/// - Exemplo: representação rica do significado de "João"
/// 
/// ## ⚡ PROCESSO DE ATENÇÃO:
/// 
/// 1. **Projeção**: X → Q, K, V (através de matrizes aprendidas)
/// 2. **Scores**: Q × K^T (mede compatibilidade entre tokens)
/// 3. **Escala**: Divide por √d_k (evita gradientes muito pequenos)
/// 4. **Máscara**: Aplica máscara causal (impede visão do futuro)
/// 5. **Softmax**: Converte scores em probabilidades
/// 6. **Agregação**: Combina Values baseado nas probabilidades
/// 7. **Projeção final**: Transforma resultado para dimensão original
pub struct SelfAttention {
    // 🎯 **MATRIZES DE PROJEÇÃO APRENDIDAS**
    // Estas são as "lentes" que transformam embeddings em Q, K, V
    w_query: Linear,    // 🔍 Projeta X → Query ("o que eu quero saber?")
    w_key: Linear,      // 🗝️ Projeta X → Key ("o que eu ofereço?")
    w_value: Linear,    // 💎 Projeta X → Value ("minha informação real")
    w_out: Linear,      // 🎪 Projeta resultado final de volta ao espaço original
    
    // 📐 **HIPERPARÂMETROS DE ARQUITETURA**
    n_embd: usize,      // 🧮 Dimensão dos embeddings (largura do modelo)
    n_head: usize,      // 👁️ Número de cabeças de atenção paralelas
    head_dim: usize,    // 📏 Dimensão de cada cabeça (n_embd / n_head)
    dropout: f32,       // 🎲 Taxa de dropout para regularização
}

impl SelfAttention {
    /// 🏗️ **CONSTRUTOR: INICIALIZANDO O MECANISMO DE ATENÇÃO**
    /// 
    /// Este método cria uma nova instância de Self-Attention, configurando
    /// todas as matrizes de projeção e hiperparâmetros necessários.
    /// 
    /// ## 🎯 **Parâmetros de Entrada:**
    /// 
    /// ### 📐 **n_embd**: Dimensão dos Embeddings
    /// - Largura do modelo (ex: 768 para GPT-2 small, 1024 para medium)
    /// - Determina a "capacidade" de representação do modelo
    /// - Deve ser divisível por n_head para distribuição uniforme
    /// 
    /// ### 👁️ **n_head**: Número de Cabeças de Atenção
    /// - Permite múltiplas "perspectivas" de atenção simultâneas
    /// - Cada cabeça aprende padrões diferentes (sintático, semântico, etc.)
    /// - Valores típicos: 8, 12, 16 (potências de 2 para eficiência)
    /// 
    /// ### 🎲 **dropout**: Taxa de Regularização
    /// - Previne overfitting zerando aleatoriamente algumas conexões
    /// - Valores típicos: 0.1 (10%) para modelos pequenos, 0.0 para inferência
    /// 
    /// ### 🏗️ **vb**: VarBuilder para Inicialização de Pesos
    /// - Gerencia a criação e inicialização das matrizes de peso
    /// - Garante inicialização adequada (Xavier/Kaiming) para convergência
    /// 
    /// ## 🧮 **Cálculos de Dimensão:**
    /// ```text
    /// head_dim = n_embd / n_head
    /// 
    /// Exemplo com n_embd=768, n_head=12:
    /// head_dim = 768 / 12 = 64
    /// 
    /// Cada cabeça processa vetores de 64 dimensões
    /// 12 cabeças × 64 dim = 768 dim total (preserva dimensionalidade)
    /// ```
    pub fn new(n_embd: usize, n_head: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        // 📏 **CÁLCULO DA DIMENSÃO POR CABEÇA**
        // Divide o espaço de embeddings igualmente entre as cabeças
        // Garante que n_embd seja divisível por n_head
        assert_eq!(n_embd % n_head, 0, 
                   "n_embd ({}) deve ser divisível por n_head ({})", n_embd, n_head);
        let head_dim = n_embd / n_head;
        
        // 🎯 **CRIAÇÃO DAS MATRIZES DE PROJEÇÃO**
        // Cada matriz é uma transformação linear aprendida:
        // 
        // 🔍 Query: "Que tipo de informação eu estou procurando?"
        // Transforma embedding em vetor de "busca"
        let w_query = linear(n_embd, n_embd, vb.pp("w_query"))?;
        
        // 🗝️ Key: "Que tipo de informação eu posso oferecer?"
        // Transforma embedding em vetor de "índice/etiqueta"
        let w_key = linear(n_embd, n_embd, vb.pp("w_key"))?;
        
        // 💎 Value: "Qual é minha informação semântica real?"
        // Transforma embedding no conteúdo que será misturado
        let w_value = linear(n_embd, n_embd, vb.pp("w_value"))?;
        
        // 🎪 Output: "Como combino informações de todas as cabeças?"
        // Projeta resultado concatenado de volta ao espaço original
        let w_out = linear(n_embd, n_embd, vb.pp("w_out"))?;
        
        // 🏗️ **CONSTRUÇÃO DA ESTRUTURA FINAL**
        Ok(Self {
            w_query,
            w_key,
            w_value,
            w_out,
            n_embd,
            n_head,
            head_dim,
            dropout,
        })
    }
    
    /// 🚀 **FORWARD PASS: O CORAÇÃO DO MECANISMO DE ATENÇÃO**
    /// 
    /// Este método implementa o algoritmo completo de Self-Attention,
    /// permitindo que cada token "converse" com todos os outros tokens
    /// na sequência para entender o contexto.
    /// 
    /// ## 🎭 Analogia da Sala de Aula:
    /// Imagine uma sala de aula onde cada aluno (token) pode fazer
    /// perguntas para todos os outros alunos simultaneamente:
    /// - **Query**: "Que informação eu preciso?"
    /// - **Key**: "Que informação eu tenho?"
    /// - **Value**: "Aqui está minha informação!"
    /// 
    /// ## 📊 Dimensões dos Tensores:
    /// ```text
    /// Input:  [B, T, C] = [batch, sequence_length, embedding_dim]
    /// Q,K,V:  [B, H, T, D] = [batch, heads, seq_len, head_dim]
    /// Scores: [B, H, T, T] = [batch, heads, seq_len, seq_len]
    /// Output: [B, T, C] = [batch, sequence_length, embedding_dim]
    /// ```
    /// 
    /// ## ⚡ Complexidade Computacional:
    /// - **Tempo**: O(T² × C) - quadrática no comprimento da sequência
    /// - **Memória**: O(T²) - para armazenar matriz de atenção
    /// 
    /// ## 🎯 Parâmetros:
    /// - `x`: Tensor de entrada [batch_size, seq_len, n_embd]
    /// - `mask`: Máscara causal opcional para bloquear tokens futuros
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // 📏 **EXTRAÇÃO DAS DIMENSÕES**
        // Obtém as dimensões do tensor de entrada para reshaping posterior
        let (batch_size, seq_len, _) = x.dims3()?;
        
        // 🎯 **ETAPA 1: PROJEÇÕES LINEARES Q, K, V**
        // Transforma cada embedding em três "visões" diferentes:
        // 
        // 🔍 Query: "Que tipo de informação eu estou procurando?"
        // Exemplo: Para "gato", query pode focar em "animal", "doméstico"
        let q = self.w_query.forward(x)?;  // [B, T, C]
        
        // 🗝️ Key: "Que informação eu posso oferecer?"
        // Exemplo: Para "ronrona", key oferece "som", "comportamento"
        let k = self.w_key.forward(x)?;    // [B, T, C]
        
        // 💎 Value: "Qual é minha informação semântica real?"
        // Exemplo: Para "felino", value contém características específicas
        let v = self.w_value.forward(x)?;  // [B, T, C]
        
        // 🎭 **ETAPA 2: DIVISÃO EM MÚLTIPLAS CABEÇAS**
        // Analogia: Dividir a turma em grupos especializados
        // Cada cabeça foca em um aspecto diferente da linguagem:
        // - Cabeça 1: Relações sintáticas (sujeito-verbo)
        // - Cabeça 2: Semântica (significado)
        // - Cabeça 3: Dependências de longo alcance
        let q = self.split_heads(&q, batch_size, seq_len)?; // [B, H, T, D]
        let k = self.split_heads(&k, batch_size, seq_len)?; // [B, H, T, D]
        let v = self.split_heads(&v, batch_size, seq_len)?; // [B, H, T, D]
        
        // 🧮 **ETAPA 3: CÁLCULO DOS SCORES DE ATENÇÃO**
        let scores = self.attention_scores(&q, &k)?; // [B, H, T, T]
        
        // 🔍 **DEBUG: Verificar scores de atenção**
        let scores_vec = scores.flatten_all()?.to_vec1::<f32>()?;
        if scores_vec.iter().any(|&x| x.is_nan()) {
            eprintln!("⚠️ DEBUG: Attention scores contém NaN!");
            return Err(candle_core::Error::Msg("Attention scores contém NaN".to_string()));
        }
        if scores_vec.iter().any(|&x| x.is_infinite()) {
            eprintln!("⚠️ DEBUG: Attention scores contém infinito!");
            return Err(candle_core::Error::Msg("Attention scores contém infinito".to_string()));
        }
        
        // 🔍 **DEBUG: Verificar scores antes da máscara**
        let scores_before_mask = scores.flatten_all()?.to_vec1::<f32>()?;
        if scores_before_mask.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            eprintln!("⚠️ DEBUG: Scores contém valores inválidos ANTES da máscara!");
            return Err(candle_core::Error::Msg("Scores inválidos antes da máscara".to_string()));
        }
        
        // 🚫 **ETAPA 4: APLICAÇÃO DA MÁSCARA CAUSAL**
        let scores = if let Some(mask) = mask {
            // Expande máscara de [T, T] para [B, H, T, T]
            let expanded_mask = mask.unsqueeze(0)?.unsqueeze(0)?
                .expand(&[batch_size, self.n_head, seq_len, seq_len])?;
            
            // 🔍 **DEBUG: Verificar máscara**
            let mask_vec = expanded_mask.flatten_all()?.to_vec1::<f32>()?;
            if mask_vec.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                eprintln!("⚠️ DEBUG: Máscara contém valores inválidos!");
                return Err(candle_core::Error::Msg("Máscara inválida".to_string()));
            }
            
            // Usar valor menor para evitar overflow
            let mask_value = -1e4; // Menor que -1e9 para evitar overflow
            let masked_scores = (scores + expanded_mask * mask_value)?;
            
            // 🔍 **DEBUG: Verificar scores após máscara**
            let masked_vec = masked_scores.flatten_all()?.to_vec1::<f32>()?;
            if masked_vec.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                eprintln!("⚠️ DEBUG: Scores contém valores inválidos APÓS a máscara!");
                let max_before = scores_before_mask.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
                let max_mask = mask_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
                eprintln!("⚠️ DEBUG: Max scores antes: {}, Max mask: {}", max_before, max_mask);
                return Err(candle_core::Error::Msg("Scores inválidos após máscara".to_string()));
            }
            
            masked_scores
        } else {
            scores
        };
        
        // 🔍 **DEBUG: Verificar scores antes do softmax**
        let scores_vec = scores.flatten_all()?.to_vec1::<f32>()?;
        if scores_vec.iter().any(|&x| x.is_nan()) {
            eprintln!("⚠️ DEBUG: Scores contém NaN antes do softmax!");
            return Err(candle_core::Error::Msg("Scores contém NaN antes do softmax".to_string()));
        }
        if scores_vec.iter().any(|&x| x.is_infinite()) {
            eprintln!("⚠️ DEBUG: Scores contém infinito antes do softmax!");
            return Err(candle_core::Error::Msg("Scores contém infinito antes do softmax".to_string()));
        }
        
        // 🎲 **ETAPA 5: SOFTMAX - CONVERSÃO EM PROBABILIDADES**
        // Transforma scores em probabilidades que somam 1
        // 
        // 📊 Interpretação: "Quanto de atenção dar para cada token?"
        // Exemplo: [0.1, 0.7, 0.2] = 10% token1, 70% token2, 20% token3
        let weights = candle_nn::ops::softmax(&scores, 3)?; // [B, H, T, T]
        
        // 🎯 **ETAPA 6: DROPOUT PARA REGULARIZAÇÃO**
        // Durante o treinamento, "desliga" aleatoriamente algumas conexões
        // Previne overfitting e melhora generalização
        let weights = if self.dropout > 0.0 {
            candle_nn::ops::dropout(&weights, self.dropout)?
        } else {
            weights
        };
        
        // 💫 **ETAPA 7: AGREGAÇÃO PONDERADA DOS VALUES**
        // Combina informações de todos os tokens baseado nos pesos
        // 
        // 🎭 Analogia: Cada aluno contribui com sua informação
        // proporcionalmente ao quanto é "relevante" para a pergunta
        let attention = weights.matmul(&v)?; // [B, H, T, D]
        
        // 🔗 **ETAPA 8: CONCATENAÇÃO DAS CABEÇAS**
        // Junta as informações de todas as cabeças especializadas
        // De [B, H, T, D] volta para [B, T, C] onde C = H × D
        let attention = self.merge_heads(&attention, batch_size, seq_len)?; // [B, T, C]
        
        // 🎪 **ETAPA 9: PROJEÇÃO FINAL DE SAÍDA**
        // Última transformação linear para refinar o resultado
        // Permite que o modelo "misture" informações das diferentes cabeças
        self.w_out.forward(&attention) // [B, T, C]
    }

    /// 🔎 **FORWARD COM PESOS DE ATENÇÃO EXPOSTOS**
    ///
    /// Mesmo cálculo de `forward`, mas também retorna a matriz de pesos de
    /// atenção `[B, H, T, T]` (pós-softmax) para visualização — usado pela
    /// API de demonstração para desenhar heatmaps reais a partir de um
    /// prompt, nunca no loop de treino (que continua chamando `forward`
    /// normalmente, sem nenhum custo extra).
    pub fn forward_with_attention(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.w_query.forward(x)?;
        let k = self.w_key.forward(x)?;
        let v = self.w_value.forward(x)?;

        let q = self.split_heads(&q, batch_size, seq_len)?;
        let k = self.split_heads(&k, batch_size, seq_len)?;
        let v = self.split_heads(&v, batch_size, seq_len)?;

        let scores = self.attention_scores(&q, &k)?;

        let scores = if let Some(mask) = mask {
            let expanded_mask = mask.unsqueeze(0)?.unsqueeze(0)?
                .expand(&[batch_size, self.n_head, seq_len, seq_len])?;
            let mask_value = -1e4;
            (scores + expanded_mask * mask_value)?
        } else {
            scores
        };

        let weights = candle_nn::ops::softmax(&scores, 3)?; // [B, H, T, T]
        let attention = weights.matmul(&v)?;
        let attention = self.merge_heads(&attention, batch_size, seq_len)?;
        let output = self.w_out.forward(&attention)?;

        Ok((output, weights))
    }

    /// 🔧 **EXTRAÇÃO DE TENSORES Q, K, V PARA KERNELS FUSIONADOS**
    /// 
    /// Extrai e processa os tensores Query, Key e Value sem aplicar
    /// o mecanismo de atenção completo. Usado por kernels fusionados
    /// para otimizações de performance.
    /// 
    /// ## 📊 **Pipeline de Processamento:**
    /// 1. **Projeções Lineares**: X → Q, K, V
    /// 2. **Divisão em Cabeças**: Reshape para multi-head
    /// 3. **Retorno**: Tensores prontos para atenção fusionada
    /// 
    /// ## 🎯 **Formato de Saída:**
    /// - Q: [batch_size, n_head, seq_len, head_dim]
    /// - K: [batch_size, n_head, seq_len, head_dim]
    /// - V: [batch_size, n_head, seq_len, head_dim]
    pub fn get_qkv_tensors(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // 📏 **EXTRAÇÃO DAS DIMENSÕES**
        let (batch_size, seq_len, _) = x.dims3()?;
        
        // 🎯 **PROJEÇÕES LINEARES Q, K, V**
        let q = self.w_query.forward(x)?;  // [B, T, C]
        let k = self.w_key.forward(x)?;    // [B, T, C]
        let v = self.w_value.forward(x)?;  // [B, T, C]
        
        // 🎭 **DIVISÃO EM MÚLTIPLAS CABEÇAS**
        let q = self.split_heads(&q, batch_size, seq_len)?; // [B, H, T, D]
        let k = self.split_heads(&k, batch_size, seq_len)?; // [B, H, T, D]
        let v = self.split_heads(&v, batch_size, seq_len)?; // [B, H, T, D]
        
        Ok((q, k, v))
    }
    
    /// 🔀 **SPLIT HEADS: DIVIDINDO EM MÚLTIPLAS PERSPECTIVAS**
    /// 
    /// Transforma um tensor "monolítico" em múltiplas cabeças de atenção,
    /// permitindo que o modelo processe diferentes aspectos da informação
    /// simultaneamente.
    /// 
    /// ## 🎭 Analogia da Orquestra:
    /// É como dividir uma orquestra em seções (cordas, sopros, percussão).
    /// Cada seção toca sua parte, mas todas contribuem para a harmonia final.
    /// 
    /// ## 📐 Transformação de Dimensões:
    /// ```text
    /// Input:  [B, T, C] = [batch, seq_len, n_embd]
    ///         ↓ reshape
    /// Step 1: [B, T, H, D] = [batch, seq_len, n_head, head_dim]
    ///         ↓ transpose(1,2)
    /// Output: [B, H, T, D] = [batch, n_head, seq_len, head_dim]
    /// 
    /// Onde: C = H × D (n_embd = n_head × head_dim)
    /// ```
    /// 
    /// ## 🧮 Exemplo Numérico:
    /// ```text
    /// n_embd=768, n_head=12 → head_dim=64
    /// [2, 10, 768] → [2, 10, 12, 64] → [2, 12, 10, 64]
    /// ```
    fn split_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        // 📏 **PASSO 1: RESHAPE PARA EXPOR AS CABEÇAS**
        // Divide a dimensão de embedding em (n_head, head_dim)
        // [B, T, C] → [B, T, H, D]
        let reshaped = x.reshape(&[batch_size, seq_len, self.n_head, self.head_dim])?;
        
        // 🔄 **PASSO 2: TRANSPOSE PARA AGRUPAR POR CABEÇA**
        // Move a dimensão das cabeças para a segunda posição
        // [B, T, H, D] → [B, H, T, D]
        // Isso permite processamento paralelo de cada cabeça
        // Agora cada cabeça pode processar independentemente!
        reshaped.transpose(1, 2)?.contiguous()
    }
    
    /// 🔗 **MERGE HEADS: REUNINDO AS PERSPECTIVAS**
    /// 
    /// Reconstrói o tensor original concatenando todas as cabeças,
    /// combinando as diferentes perspectivas em uma representação unificada.
    /// 
    /// ## 🎭 Analogia da Orquestra:
    /// É como mixar todas as seções da orquestra em uma gravação final.
    /// Cada instrumento contribuiu sua parte, agora temos a música completa.
    /// 
    /// ## 📐 Transformação de Dimensões:
    /// ```text
    /// Input:  [B, H, T, D] = [batch, n_head, seq_len, head_dim]
    ///         ↓ transpose(1,2)
    /// Step 1: [B, T, H, D] = [batch, seq_len, n_head, head_dim]
    ///         ↓ reshape
    /// Output: [B, T, C] = [batch, seq_len, n_embd]
    /// 
    /// Onde: C = H × D (concatenação das cabeças)
    /// ```
    fn merge_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        // 🔄 **PASSO 1: TRANSPOSE DE VOLTA**
        // [B, H, T, D] → [B, T, H, D]
        // Reorganiza para que as cabeças fiquem na última dimensão
        let transposed = x.transpose(1, 2)?.contiguous()?;
        
        // 🔗 **PASSO 2: CONCATENAÇÃO DAS CABEÇAS**
        // [B, T, H, D] → [B, T, C] onde C = H × D
        // Achata as dimensões H e D em uma única dimensão
        transposed.reshape(&[batch_size, seq_len, self.n_embd])
    }
    
    /// 🧮 **ATTENTION SCORES: O CORAÇÃO DO MECANISMO**
    /// 
    /// Calcula a "compatibilidade" entre queries e keys usando o produto
    /// escalar, implementando a fórmula fundamental da atenção.
    /// 
    /// ## 📊 Fórmula Matemática:
    /// ```text
    /// Attention(Q,K,V) = softmax(QK^T / √d_k)V
    ///                              ↑
    ///                    Esta função calcula esta parte
    /// ```
    /// 
    /// ## 🎯 Por que dividir por √d_k?
    /// 
    /// ### 📈 **Problema dos Gradientes Explosivos:**
    /// - Sem escalonamento: valores muito grandes → softmax saturado
    /// - Com √d_k: valores controlados → gradientes estáveis
    /// 
    /// ### 🧮 **Intuição Matemática:**
    /// ```text
    /// Se Q e K têm variância 1, então QK^T tem variância d_k
    /// Dividindo por √d_k, restauramos variância ≈ 1
    /// ```
    /// 
    /// ### 📊 **Exemplo Prático:**
    /// ```text
    /// d_k = 64 → √d_k = 8
    /// Scores antes: [-50, 30, 80] (muito extremos)
    /// Scores depois: [-6.25, 3.75, 10] (mais balanceados)
    /// ```
    /// 
    /// ## 🎭 Analogia:
    /// É como ajustar o volume de um amplificador - muito alto
    /// e você não consegue distinguir os detalhes, muito baixo
    /// e você não ouve nada!
    fn attention_scores(&self, q: &Tensor, k: &Tensor) -> Result<Tensor> {
        // 🔍 **DEBUG: Verificar Q e K**
        let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
        let k_vec = k.flatten_all()?.to_vec1::<f32>()?;
        
        if q_vec.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            eprintln!("⚠️ DEBUG: Q contém NaN ou infinito!");
            return Err(candle_core::Error::Msg("Q contém valores inválidos".to_string()));
        }
        
        if k_vec.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            eprintln!("⚠️ DEBUG: K contém NaN ou infinito!");
            return Err(candle_core::Error::Msg("K contém valores inválidos".to_string()));
        }
        
        let q_max = q_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let k_max = k_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        
        if q_max > 100.0 || k_max > 100.0 {
            eprintln!("⚠️ DEBUG: Valores muito grandes - Q_max: {}, K_max: {}", q_max, k_max);
        }
        
        // 📐 **CÁLCULO DO FATOR DE ESCALA**
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        
        // 🔄 **PRODUTO MATRICIAL Q @ K^T**
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        
        // 🔍 **DEBUG: Verificar scores após matmul**
        let scores_vec = scores.flatten_all()?.to_vec1::<f32>()?;
        if scores_vec.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            eprintln!("⚠️ DEBUG: Scores contém valores inválidos após matmul!");
            let max_score = scores_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
            eprintln!("⚠️ DEBUG: Max score absoluto: {}", max_score);
            return Err(candle_core::Error::Msg("Scores inválidos após matmul".to_string()));
        }
        
        // ⚖️ **APLICAÇÃO DO ESCALONAMENTO**
        let scaled_scores = (scores * scale)?;
        
        // 🔍 **DEBUG: Verificar scores após escalonamento**
        let scaled_vec = scaled_scores.flatten_all()?.to_vec1::<f32>()?;
        if scaled_vec.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            eprintln!("⚠️ DEBUG: Scores contém valores inválidos após escalonamento!");
            eprintln!("⚠️ DEBUG: Scale factor: {}", scale);
            return Err(candle_core::Error::Msg("Scores inválidos após escalonamento".to_string()));
        }
        
        Ok(scaled_scores)
    }
}

/// 🎭 **MULTI-HEAD ATTENTION: MÚLTIPLAS PERSPECTIVAS SIMULTÂNEAS**
/// 
/// Esta estrutura é um wrapper elegante em torno do SelfAttention,
/// fornecendo uma interface limpa para o mecanismo de atenção multi-cabeça.
/// 
/// ## 🎬 **Analogia do Cinema:**
/// Imagine assistir a um filme com múltiplas câmeras simultaneamente:
/// - **Câmera 1**: Foco nos rostos (expressões emocionais)
/// - **Câmera 2**: Foco nas ações (movimentos corporais)
/// - **Câmera 3**: Foco no cenário (contexto ambiental)
/// - **Câmera N**: Cada uma captura aspectos únicos!
/// 
/// ## 🧠 **Por que Multi-Head?**
/// 
/// ### 🎯 **Especialização de Cabeças:**
/// Cada cabeça pode aprender a focar em diferentes aspectos:
/// - **Sintático**: Relações gramaticais (sujeito-verbo-objeto)
/// - **Semântico**: Significados e conceitos relacionados
/// - **Posicional**: Dependências de curta e longa distância
/// - **Contextual**: Nuances e ambiguidades
/// 
/// ### 📊 **Vantagens Computacionais:**
/// - **Paralelização**: Todas as cabeças processam simultaneamente
/// - **Diversidade**: Múltiplas representações do mesmo input
/// - **Robustez**: Se uma cabeça falha, outras compensam
/// 
/// ## 🔄 **Fluxo de Processamento:**
/// ```text
/// Input: [B, T, C]
///   ↓
/// SelfAttention (com N cabeças internas)
///   ↓
/// Output: [B, T, C]
/// ```
pub struct MultiHeadAttention {
    /// 🧠 Instância do mecanismo de self-attention
    /// Contém todas as N cabeças e lógica de processamento
    attention: SelfAttention,
}

impl MultiHeadAttention {
    /// 🏗️ **CONSTRUTOR: CRIANDO O MECANISMO MULTI-CABEÇA**
    /// 
    /// Inicializa uma nova instância de Multi-Head Attention,
    /// delegando toda a complexidade para a estrutura SelfAttention.
    /// 
    /// ## 🎯 **Parâmetros:**
    /// - `n_embd`: Dimensão dos embeddings (largura do modelo)
    /// - `n_head`: Número de cabeças de atenção paralelas
    /// - `dropout`: Taxa de regularização durante treinamento
    /// - `vb`: VarBuilder para inicialização de pesos
    /// 
    /// ## 📐 **Validação Automática:**
    /// - Verifica se n_embd é divisível por n_head
    /// - Garante dimensões consistentes para todas as cabeças
    pub fn new(n_embd: usize, n_head: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention: SelfAttention::new(n_embd, n_head, dropout, vb)?,
        })
    }
    
    /// 🚀 **FORWARD PASS: PROCESSAMENTO MULTI-CABEÇA**
    /// 
    /// Executa o mecanismo completo de atenção multi-cabeça,
    /// permitindo que o modelo "olhe" para a sequência através
    /// de múltiplas perspectivas especializadas simultaneamente.
    /// 
    /// ## 🔄 **Processo Interno:**
    /// 1. **Projeção**: X → Q, K, V (para todas as cabeças)
    /// 2. **Divisão**: Separa em N cabeças independentes
    /// 3. **Atenção**: Cada cabeça calcula sua própria atenção
    /// 4. **Concatenação**: Junta resultados de todas as cabeças
    /// 5. **Projeção Final**: Transforma resultado concatenado
    /// 
    /// ## 🎯 **Parâmetros:**
    /// - `x`: Tensor de entrada [batch_size, seq_len, n_embd]
    /// - `mask`: Máscara causal opcional (para modelos autoregressivos)
    /// 
    /// ## 📤 **Retorno:**
    /// - Tensor processado [batch_size, seq_len, n_embd]
    ///   com informações contextuais enriquecidas
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        self.attention.forward(x, mask)
    }

    /// 🔎 Mesmo que `forward`, mas também retorna os pesos de atenção
    /// `[B, H, T, T]` — ver `SelfAttention::forward_with_attention`.
    pub fn forward_with_attention(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        self.attention.forward_with_attention(x, mask)
    }

    /// 🔧 **EXTRAÇÃO DE TENSORES Q, K, V PARA KERNELS FUSIONADOS**
    /// 
    /// Extrai os tensores Query, Key e Value processados para uso
    /// em kernels fusionados de atenção otimizados.
    /// 
    /// ## 📊 **Formato de Saída:**
    /// - Q: [batch_size, n_head, seq_len, head_dim]
    /// - K: [batch_size, n_head, seq_len, head_dim] 
    /// - V: [batch_size, n_head, seq_len, head_dim]
    /// 
    /// ## ⚡ **Uso em Kernels Fusionados:**
    /// ```rust
    /// let (q, k, v) = attention.get_qkv_tensors(&input)?;
    /// let output = fused_attention_kernel.forward(&q, &k, &v, mask, dropout)?;
    /// ```
    pub fn get_qkv_tensors(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        self.attention.get_qkv_tensors(x)
    }
}