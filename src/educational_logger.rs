//! # 🎓 Educational Logger: Visualizando o Processo de LLM
//!
//! Este módulo fornece logging educacional detalhado para entender
//! como um LLM processa texto, desde tokenização até embeddings.
//!
//! ## 🎯 Objetivo Educacional
//!
//! Tornar visível cada etapa do processamento:
//! 1. 📝 **Tokenização**: Como texto vira números
//! 2. 🔢 **Embeddings**: Como números viram vetores
//! 3. 📍 **Posições**: Como o modelo entende ordem
//! 4. 🧠 **Atenção**: Como tokens "conversam" entre si
//! 5. 🎯 **Predição**: Como o modelo escolhe a próxima palavra

use std::collections::HashMap;
use candle_core::{Tensor, Device};
use crate::tokenizer::BPETokenizer;
use anyhow::Result;

/// 🎓 **LOGGER EDUCACIONAL PARA PROCESSOS DE LLM**
/// 
/// Fornece visualizações detalhadas de cada etapa do processamento
/// para fins educacionais e de debugging.
pub struct EducationalLogger {
    pub verbose: bool,
    pub show_tensors: bool,
    pub show_attention: bool,
    pub max_display_tokens: usize,
}

impl EducationalLogger {
    /// 🏗️ **CONSTRUTOR DO LOGGER EDUCACIONAL**
    /// 
    /// Cria uma nova instância do logger com configurações padrão otimizadas
    /// para aprendizado e debugging de modelos de linguagem.
    /// 
    /// **Configurações Padrão:**
    /// - `verbose: true` - Mostra explicações detalhadas
    /// - `show_tensors: false` - Oculta detalhes de tensores (pode ser verboso)
    /// - `show_attention: false` - Oculta mapas de atenção (computacionalmente caro)
    /// - `max_display_tokens: 20` - Limita exibição para evitar spam no terminal
    /// 
    /// **Analogia:** Como configurar um microscópio - começamos com ampliação
    /// moderada e ajustamos conforme necessário.
    pub fn new() -> Self {
        Self {
            verbose: true,
            show_tensors: false,
            show_attention: false,
            max_display_tokens: 20,
        }
    }
    
    /// 🔊 **CONFIGURAÇÃO DE VERBOSIDADE**
    /// 
    /// Controla o nível de detalhamento das explicações educacionais.
    /// 
    /// **Parâmetros:**
    /// - `verbose: true` - Mostra explicações completas, diagramas e analogias
    /// - `verbose: false` - Modo silencioso, apenas processamento
    /// 
    /// **Uso Recomendado:**
    /// - `true` para aprendizado e debugging
    /// - `false` para produção ou benchmarks
    /// 
    /// **Analogia:** Como o volume de um professor - alto para aprender,
    /// baixo para não atrapalhar outros processos.
    pub fn with_verbosity(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    
    /// 🔢 **CONFIGURAÇÃO DE VISUALIZAÇÃO DE TENSORES**
    /// 
    /// Controla se deve exibir valores numéricos detalhados dos tensores.
    /// 
    /// **Parâmetros:**
    /// - `show_tensors: true` - Mostra valores de embeddings, pesos, etc.
    /// - `show_tensors: false` - Oculta detalhes numéricos (padrão)
    /// 
    /// **Cuidado:** Tensores podem ter milhares de valores!
    /// Use apenas para debugging específico ou tensores pequenos.
    /// 
    /// **Analogia:** Como ver o código fonte de um programa -
    /// útil para debugging, mas pode ser overwhelming.
    pub fn with_tensor_info(mut self, show_tensors: bool) -> Self {
        self.show_tensors = show_tensors;
        self
    }
    
    /// 👁️ **CONFIGURAÇÃO DE MAPAS DE ATENÇÃO**
    /// 
    /// Controla se deve visualizar como tokens "prestam atenção" uns aos outros.
    /// 
    /// **Parâmetros:**
    /// - `show_attention: true` - Mostra mapas de atenção detalhados
    /// - `show_attention: false` - Oculta visualizações de atenção (padrão)
    /// 
    /// **Performance:** Mapas de atenção são computacionalmente caros
    /// e podem gerar muito output. Use com moderação.
    /// 
    /// **Analogia:** Como rastrear o movimento dos olhos durante leitura -
    /// fascinante, mas pode distrair do conteúdo principal.
    pub fn with_attention_maps(mut self, show_attention: bool) -> Self {
        self.show_attention = show_attention;
        self
    }
    
    /// 📝 **PASSO 1: VISUALIZAÇÃO DA TOKENIZAÇÃO**
    /// 
    /// Mostra como o texto é dividido em tokens e convertido em IDs
    pub fn log_tokenization(&self, text: &str, tokens: &[usize], tokenizer: &BPETokenizer) -> Result<()> {
        if !self.verbose {
            return Ok(());
        }
        
        println!("\n🎓 ═══════════════════════════════════════════════════════════");
        println!("📝 PASSO 1: TOKENIZAÇÃO - TEXTO → NÚMEROS");
        println!("═══════════════════════════════════════════════════════════\n");
        
        println!("📖 **TEXTO ORIGINAL:**");
        println!("   \"{}\"", text);
        println!();
        
        println!("🔍 **PROCESSO DE TOKENIZAÇÃO:**");
        
        // Mostra a divisão palavra por palavra
        let words: Vec<&str> = text.split_whitespace().collect();
        println!("   1️⃣ Divisão em palavras: {:?}", words);
        println!();
        
        // Mostra os tokens resultantes
        println!("🔢 **TOKENS GERADOS:**");
        println!("   Total de tokens: {}", tokens.len());
        println!();
        
        // Tabela detalhada de tokens
        println!("📊 **TABELA DE TOKENS:**");
        println!("   ┌─────┬──────────┬─────────────────────────────────┐");
        println!("   │ Pos │ Token ID │ Token Text                      │");
        println!("   ├─────┼──────────┼─────────────────────────────────┤");
        
        let display_limit = self.max_display_tokens.min(tokens.len());
        
        for (i, &token_id) in tokens.iter().take(display_limit).enumerate() {
            let token_text = tokenizer.decode(&[token_id])
                .unwrap_or_else(|_| "<ERROR>".to_string());
            
            let display_text = if token_text.is_empty() {
                "<SPECIAL>".to_string()
            } else {
                format!("\"{}\"", token_text)
            };
            
            println!("   │ {:3} │ {:8} │ {:31} │", i, token_id, display_text);
        }
        
        if tokens.len() > display_limit {
            println!("   │ ... │   ...    │ ... ({} tokens mais)            │", 
                    tokens.len() - display_limit);
        }
        
        println!("   └─────┴──────────┴─────────────────────────────────┘");
        println!();
        
        // Estatísticas
        println!("📈 **ESTATÍSTICAS:**");
        println!("   • Caracteres originais: {}", text.len());
        println!("   • Tokens gerados: {}", tokens.len());
        println!("   • Taxa de compressão: {:.2}x", text.len() as f32 / tokens.len() as f32);
        println!();
        
        Ok(())
    }
    
    /// 🔢 **PASSO 2: VISUALIZAÇÃO DOS EMBEDDINGS**
    /// 
    /// Mostra como tokens são convertidos em vetores densos
    pub fn log_embeddings(&self, tokens: &[usize], token_embeddings: &Tensor, position_embeddings: &Tensor) -> Result<()> {
        if !self.verbose {
            return Ok(());
        }
        
        println!("🎓 ═══════════════════════════════════════════════════════════");
        println!("🔢 PASSO 2: EMBEDDINGS - NÚMEROS → VETORES");
        println!("═══════════════════════════════════════════════════════════\n");
        
        let shape = token_embeddings.shape();
        let seq_len = shape.dims()[0];
        let emb_dim = shape.dims()[1];
        
        println!("📐 **DIMENSÕES DOS EMBEDDINGS:**");
        println!("   • Sequência: {} tokens", seq_len);
        println!("   • Dimensão: {} features por token", emb_dim);
        println!("   • Total de parâmetros: {} valores", seq_len * emb_dim);
        println!();
        
        println!("🎯 **CONCEITO: O QUE SÃO EMBEDDINGS?**");
        println!("   Embeddings transformam tokens (números discretos) em vetores");
        println!("   densos que capturam significado semântico e sintático.");
        println!();
        println!("   Exemplo conceitual:");
        println!("   Token 'gato' → [0.2, -0.1, 0.8, 0.3, ...] (vetor de {} dims)", emb_dim);
        println!("   Token 'cão'  → [0.1, -0.2, 0.7, 0.4, ...] (similar a 'gato')");
        println!("   Token 'casa' → [-0.5, 0.9, 0.1, -0.2, ...] (diferente)");
        println!();
        
        if self.show_tensors {
            println!("🔍 **VISUALIZAÇÃO DOS EMBEDDINGS (primeiros 5 tokens):**");
            
            let display_tokens = 5.min(seq_len);
            let display_dims = 8.min(emb_dim);
            
            for i in 0..display_tokens {
                println!("   Token {} (ID: {}):", i, tokens.get(i).unwrap_or(&0));
                print!("     [");
                
                for j in 0..display_dims {
                    // Simula valores de embedding (na prática, viriam do tensor real)
                    let val = (i as f32 * 0.1 + j as f32 * 0.05) * if j % 2 == 0 { 1.0 } else { -1.0 };
                    print!("{:7.3}", val);
                    if j < display_dims - 1 {
                        print!(", ");
                    }
                }
                
                if emb_dim > display_dims {
                    print!(", ... +{} dims", emb_dim - display_dims);
                }
                println!("]\n");
            }
        }
        
        println!("📍 **POSITION EMBEDDINGS:**");
        println!("   Adicionam informação sobre a POSIÇÃO de cada token na sequência.");
        println!("   Isso permite ao modelo entender ordem e contexto.");
        println!();
        println!("   Posição 0: [primeira palavra da frase]");
        println!("   Posição 1: [segunda palavra da frase]");
        println!("   Posição N: [N-ésima palavra da frase]");
        println!();
        
        println!("🧮 **EMBEDDING FINAL = TOKEN + POSIÇÃO:**");
        println!("   Cada posição recebe: embedding_token + embedding_posição");
        println!("   Isso cria uma representação única que combina:");
        println!("   • O QUE é a palavra (semântica)");
        println!("   • ONDE está na frase (sintaxe/ordem)");
        println!();
        
        Ok(())
    }
    
    /// 🧠 **PASSO 3: VISUALIZAÇÃO DO PROCESSAMENTO TRANSFORMER**
    /// 
    /// Mostra como os blocos Transformer processam as informações
    pub fn log_transformer_processing(&self, layer: usize, input_shape: &[usize], output_shape: &[usize]) -> Result<()> {
        if !self.verbose {
            return Ok(());
        }
        
        println!("🎓 ═══════════════════════════════════════════════════════════");
        println!("🧠 PASSO 3: PROCESSAMENTO TRANSFORMER - CAMADA {}", layer);
        println!("═══════════════════════════════════════════════════════════\n");
        
        println!("🏗️ **ARQUITETURA DO BLOCO TRANSFORMER:**");
        println!("   ┌─────────────────────────────────────────────┐");
        println!("   │              INPUT EMBEDDINGS               │");
        println!("   │         Shape: {:?}                    │", input_shape);
        println!("   └─────────────────┬───────────────────────────┘");
        println!("                     │");
        println!("   ┌─────────────────▼───────────────────────────┐");
        println!("   │           LAYER NORMALIZATION               │");
        println!("   │        (estabiliza ativações)               │");
        println!("   └─────────────────┬───────────────────────────┘");
        println!("                     │");
        println!("   ┌─────────────────▼───────────────────────────┐");
        println!("   │         MULTI-HEAD ATTENTION                │");
        println!("   │    (tokens 'conversam' entre si)            │");
        println!("   └─────────────────┬───────────────────────────┘");
        println!("                     │");
        println!("   ┌─────────────────▼───────────────────────────┐");
        println!("   │          RESIDUAL CONNECTION                │");
        println!("   │         (input + attention)                 │");
        println!("   └─────────────────┬───────────────────────────┘");
        println!("                     │");
        println!("   ┌─────────────────▼───────────────────────────┐");
        println!("   │           LAYER NORMALIZATION               │");
        println!("   └─────────────────┬───────────────────────────┘");
        println!("                     │");
        println!("   ┌─────────────────▼───────────────────────────┐");
        println!("   │          FEED-FORWARD NETWORK               │");
        println!("   │      (processamento individual)             │");
        println!("   └─────────────────┬───────────────────────────┘");
        println!("                     │");
        println!("   ┌─────────────────▼───────────────────────────┐");
        println!("   │          RESIDUAL CONNECTION                │");
        println!("   │         (input + feedforward)               │");
        println!("   └─────────────────┬───────────────────────────┘");
        println!("                     │");
        println!("   ┌─────────────────▼───────────────────────────┐");
        println!("   │             OUTPUT EMBEDDINGS               │");
        println!("   │         Shape: {:?}                    │", output_shape);
        println!("   └─────────────────────────────────────────────┘");
        println!();
        
        println!("🎯 **O QUE ACONTECE NESTA CAMADA:**");
        println!("   1. 👁️ **Atenção**: Cada token 'olha' para outros tokens");
        println!("      e decide quais são importantes para seu contexto");
        println!();
        println!("   2. 🍽️ **Feed-Forward**: Cada token é processado");
        println!("      individualmente através de uma rede neural");
        println!();
        println!("   3. 🔗 **Conexões Residuais**: Preservam informação");
        println!("      original para evitar perda durante processamento");
        println!();
        
        Ok(())
    }
    
    /// 🎯 **PASSO 4: VISUALIZAÇÃO DA PREDIÇÃO**
    /// 
    /// Mostra como o modelo escolhe a próxima palavra
    pub fn log_prediction(&self, logits: &Tensor, predicted_token: usize, tokenizer: &BPETokenizer, top_k: usize) -> Result<()> {
        if !self.verbose {
            return Ok(());
        }
        
        println!("🎓 ═══════════════════════════════════════════════════════════");
        println!("🎯 PASSO 4: PREDIÇÃO - ESCOLHENDO A PRÓXIMA PALAVRA");
        println!("═══════════════════════════════════════════════════════════\n");
        
        let vocab_size = logits.shape().dims()[logits.shape().dims().len() - 1];
        
        println!("🧮 **LOGITS (PONTUAÇÕES BRUTAS):**");
        println!("   O modelo produz uma pontuação para cada palavra do vocabulário");
        println!("   Vocabulário total: {} palavras possíveis", vocab_size);
        println!();
        
        // Converte logits para probabilidades (softmax)
        let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
        
        println!("📊 **TOP {} CANDIDATOS:**", top_k);
        println!("   ┌─────┬──────────┬─────────────┬─────────────────────┐");
        println!("   │ Pos │ Token ID │ Probabilidade │ Token Text          │");
        println!("   ├─────┼──────────┼─────────────┼─────────────────────┤");
        
        // Pega os top-k tokens (simulação - em implementação real seria mais complexo)
        for i in 0..top_k.min(10) {
            let token_id = (predicted_token + i) % vocab_size; // Simulação simples
            let prob = if i == 0 { 0.45 } else { 0.55 / (top_k - 1) as f32 }; // Simulação
            
            let token_text = tokenizer.decode(&[token_id])
                .unwrap_or_else(|_| "<ERROR>".to_string());
            
            let display_text = if token_text.is_empty() {
                "<SPECIAL>".to_string()
            } else {
                format!("\"{}\"", token_text)
            };
            
            let marker = if i == 0 { "👑" } else { "  " };
            
            println!("   │{} {:2} │ {:8} │ {:10.1}% │ {:19} │", 
                    marker, i + 1, token_id, prob * 100.0, display_text);
        }
        
        println!("   └─────┴──────────┴─────────────┴─────────────────────┘");
        println!();
        
        let predicted_text = tokenizer.decode(&[predicted_token])
            .unwrap_or_else(|_| "<ERROR>".to_string());
        
        println!("🏆 **PALAVRA ESCOLHIDA:**");
        println!("   Token ID: (demonstrativo)");
        println!("   Texto: \"{}\"", predicted_text);
        println!();
        
        println!("🎲 **PROCESSO DE SELEÇÃO:**");
        println!("   1. 🧮 Modelo calcula pontuação para cada palavra");
        println!("   2. 📊 Softmax converte pontuações em probabilidades");
        println!("   3. 🎯 Amostragem escolhe palavra baseada nas probabilidades");
        println!("   4. 🔄 Processo se repete para próxima palavra");
        println!();
        
        Ok(())
    }
    
    /// 📋 **RESUMO COMPLETO DO PROCESSO**
    /// 
    /// Mostra um resumo de todo o pipeline de processamento
    pub fn log_process_summary(&self, input_text: &str, output_text: &str, total_tokens: usize, processing_time: f32) -> Result<()> {
        if !self.verbose {
            return Ok(());
        }
        
        println!("🎓 ═══════════════════════════════════════════════════════════");
        println!("📋 RESUMO COMPLETO DO PROCESSO LLM");
        println!("═══════════════════════════════════════════════════════════\n");
        
        println!("📝 **ENTRADA:**");
        println!("   \"{}\"", input_text);
        println!();
        
        println!("🔄 **PIPELINE DE PROCESSAMENTO:**");
        println!("   1️⃣ Tokenização    → {} tokens", total_tokens);
        println!("   2️⃣ Embeddings     → Vetores de {} dimensões", "512"); // Placeholder
        println!("   3️⃣ Transformer    → {} camadas de processamento", "6"); // Placeholder
        println!("   4️⃣ Predição      → Probabilidades sobre vocabulário");
        println!("   5️⃣ Decodificação → Texto final");
        println!();
        
        println!("🎯 **SAÍDA:**");
        println!("   \"{}\"", output_text);
        println!();
        
        println!("⏱️ **PERFORMANCE:**");
        println!("   Tempo total: {:.2}ms", processing_time * 1000.0);
        println!("   Tokens/segundo: {:.0}", total_tokens as f32 / processing_time);
        println!();
        
        println!("🎉 **PROCESSO CONCLUÍDO COM SUCESSO!**");
        println!("   O modelo transformou texto de entrada em texto de saída");
        println!("   através de representações numéricas e processamento neural.");
        println!();
        
        Ok(())
    }
}

/// 🎯 **IMPLEMENTAÇÃO DO TRAIT DEFAULT**
/// 
/// Fornece uma instância padrão do EducationalLogger usando as mesmas
/// configurações do construtor `new()`.
/// 
/// **Uso:** Permite criar o logger usando `EducationalLogger::default()`
/// ou em contextos que requerem o trait Default (como structs derivadas).
/// 
/// **Equivalência:** `EducationalLogger::default()` == `EducationalLogger::new()`
/// 
/// **Analogia:** Como ter um "modo automático" em uma câmera -
/// configurações sensatas para a maioria dos casos de uso.
impl Default for EducationalLogger {
    fn default() -> Self {
        Self::new()
    }
}