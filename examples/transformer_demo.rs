//! # 🏗️ Demonstração da Arquitetura Transformer
//!
//! Este exemplo demonstra como funcionam os blocos Transformer,
//! os componentes fundamentais dos modelos de linguagem modernos.
//!
//! ## 🎯 O que você aprenderá:
//! - Como funciona um bloco Transformer completo
//! - Interação entre Multi-Head Attention e Feed-Forward
//! - Papel das Residual Connections e Layer Normalization
//! - Análise de performance e otimizações
//! - Comparação entre diferentes configurações

use std::time::Instant;
use std::collections::HashMap;

// Estruturas simplificadas para demonstração
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub n_embd: usize,     // Dimensão dos embeddings
    pub n_head: usize,     // Número de cabeças de atenção
    pub n_layer: usize,    // Número de camadas
    pub dropout: f32,      // Taxa de dropout
    pub vocab_size: usize, // Tamanho do vocabulário
    pub max_seq_len: usize, // Comprimento máximo da sequência
}

#[derive(Debug)]
pub struct Matrix {
    pub data: Vec<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![0.0; cols]; rows],
            rows,
            cols,
        }
    }
    
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut matrix = Self::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                matrix.data[i][j] = (i as f32 * 0.1 + j as f32 * 0.01) % 1.0;
            }
        }
        matrix
    }
    
    pub fn add(&self, other: &Matrix) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }
    
    pub fn layer_norm(&self) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            let mean: f32 = self.data[i].iter().sum::<f32>() / self.cols as f32;
            let variance: f32 = self.data[i].iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / self.cols as f32;
            let std_dev = (variance + 1e-5).sqrt();
            
            for j in 0..self.cols {
                result.data[i][j] = (self.data[i][j] - mean) / std_dev;
            }
        }
        result
    }
}

#[derive(Debug)]
pub struct MultiHeadAttention {
    pub n_head: usize,
    pub head_dim: usize,
    pub n_embd: usize,
}

impl MultiHeadAttention {
    pub fn new(n_embd: usize, n_head: usize) -> Self {
        Self {
            n_head,
            head_dim: n_embd / n_head,
            n_embd,
        }
    }
    
    pub fn forward(&self, input: &Matrix) -> Matrix {
        // Simulação simplificada do Multi-Head Attention
        let mut output = Matrix::new(input.rows, input.cols);
        
        // Para cada cabeça de atenção
        for head in 0..self.n_head {
            let start_dim = head * self.head_dim;
            let end_dim = start_dim + self.head_dim;
            
            // Simula Q, K, V projections e attention
            for i in 0..input.rows {
                for j in start_dim..end_dim {
                    if j < input.cols {
                        // Simulação de atenção: combina informações de diferentes posições
                        let mut attention_sum = 0.0;
                        for k in 0..input.rows {
                            let attention_weight = (1.0 / (1.0 + (i as f32 - k as f32).abs()));
                            attention_sum += input.data[k][j] * attention_weight;
                        }
                        output.data[i][j] += attention_sum / input.rows as f32;
                    }
                }
            }
        }
        
        output
    }
}

#[derive(Debug)]
pub struct FeedForward {
    pub n_embd: usize,
    pub hidden_dim: usize,
}

impl FeedForward {
    pub fn new(n_embd: usize) -> Self {
        Self {
            n_embd,
            hidden_dim: n_embd * 4, // Expansão típica 4x
        }
    }
    
    pub fn forward(&self, input: &Matrix) -> Matrix {
        let mut output = Matrix::new(input.rows, input.cols);
        
        // Simulação de Feed-Forward: Linear -> GELU -> Linear
        for i in 0..input.rows {
            for j in 0..input.cols {
                // Primeira camada linear (expansão)
                let expanded = input.data[i][j] * 2.0; // Simulação
                
                // GELU activation (aproximação)
                let gelu_output = expanded * 0.5 * (1.0 + (expanded * 0.7978845608).tanh());
                
                // Segunda camada linear (contração)
                output.data[i][j] = gelu_output * 0.5; // Simulação
            }
        }
        
        output
    }
}

#[derive(Debug)]
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub feed_forward: FeedForward,
    pub config: TransformerConfig,
}

impl TransformerBlock {
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            attention: MultiHeadAttention::new(config.n_embd, config.n_head),
            feed_forward: FeedForward::new(config.n_embd),
            config,
        }
    }
    
    pub fn forward(&self, input: &Matrix) -> Matrix {
        // 1. Layer Norm + Multi-Head Attention + Residual
        let normed1 = input.layer_norm();
        let attention_out = self.attention.forward(&normed1);
        let residual1 = input.add(&attention_out);
        
        // 2. Layer Norm + Feed-Forward + Residual
        let normed2 = residual1.layer_norm();
        let ff_out = self.feed_forward.forward(&normed2);
        let residual2 = residual1.add(&ff_out);
        
        residual2
    }
}

// === DEMONSTRAÇÕES ===

fn demo_transformer_basics() {
    println!("\n🏗️ === DEMONSTRAÇÃO: FUNDAMENTOS DO TRANSFORMER ===");
    
    let config = TransformerConfig {
        n_embd: 64,
        n_head: 8,
        n_layer: 6,
        dropout: 0.1,
        vocab_size: 1000,
        max_seq_len: 128,
    };
    
    println!("📊 Configuração do Transformer:");
    println!("   • Dimensão dos embeddings: {}", config.n_embd);
    println!("   • Número de cabeças: {}", config.n_head);
    println!("   • Dimensão por cabeça: {}", config.n_embd / config.n_head);
    println!("   • Número de camadas: {}", config.n_layer);
    println!("   • Tamanho do vocabulário: {}", config.vocab_size);
    
    // Simula uma sequência de entrada
    let seq_len = 10;
    let input = Matrix::random(seq_len, config.n_embd);
    
    println!("\n🔤 Entrada: sequência de {} tokens, {} dimensões", seq_len, config.n_embd);
    println!("   Formato da matriz: {}x{}", input.rows, input.cols);
    
    // Cria um bloco Transformer
    let transformer_block = TransformerBlock::new(config);
    
    // Processa através do bloco
    let start = Instant::now();
    let output = transformer_block.forward(&input);
    let duration = start.elapsed();
    
    println!("\n⚡ Processamento concluído em {:?}", duration);
    println!("   Saída: {}x{} (mesma forma da entrada)", output.rows, output.cols);
    
    // Mostra algumas estatísticas
    let input_sum: f32 = input.data.iter().flatten().sum();
    let output_sum: f32 = output.data.iter().flatten().sum();
    
    println!("\n📈 Estatísticas:");
    println!("   • Soma da entrada: {:.4}", input_sum);
    println!("   • Soma da saída: {:.4}", output_sum);
    println!("   • Diferença: {:.4}", (output_sum - input_sum).abs());
}

fn demo_attention_vs_feedforward() {
    println!("\n🎯 === DEMONSTRAÇÃO: ATENÇÃO vs FEED-FORWARD ===");
    
    let config = TransformerConfig {
        n_embd: 32,
        n_head: 4,
        n_layer: 1,
        dropout: 0.0,
        vocab_size: 100,
        max_seq_len: 64,
    };
    
    let seq_len = 8;
    let input = Matrix::random(seq_len, config.n_embd);
    
    let attention = MultiHeadAttention::new(config.n_embd, config.n_head);
    let feed_forward = FeedForward::new(config.n_embd);
    
    println!("🔍 Comparando componentes do Transformer:");
    
    // Testa Multi-Head Attention
    let start = Instant::now();
    let attention_out = attention.forward(&input);
    let attention_time = start.elapsed();
    
    // Testa Feed-Forward
    let start = Instant::now();
    let ff_out = feed_forward.forward(&input);
    let ff_time = start.elapsed();
    
    println!("\n⏱️ Performance:");
    println!("   • Multi-Head Attention: {:?}", attention_time);
    println!("   • Feed-Forward Network: {:?}", ff_time);
    
    // Analisa as saídas
    let attention_variance = calculate_variance(&attention_out);
    let ff_variance = calculate_variance(&ff_out);
    
    println!("\n📊 Análise das saídas:");
    println!("   • Variância da Atenção: {:.6}", attention_variance);
    println!("   • Variância do Feed-Forward: {:.6}", ff_variance);
    
    println!("\n🧠 Interpretação:");
    println!("   • Atenção: Mistura informações entre posições");
    println!("   • Feed-Forward: Processa cada posição independentemente");
    println!("   • Juntos: Capturam padrões locais e globais");
}

fn demo_residual_connections() {
    println!("\n🔗 === DEMONSTRAÇÃO: RESIDUAL CONNECTIONS ===");
    
    let config = TransformerConfig {
        n_embd: 16,
        n_head: 2,
        n_layer: 1,
        dropout: 0.0,
        vocab_size: 50,
        max_seq_len: 32,
    };
    
    let seq_len = 5;
    let input = Matrix::random(seq_len, config.n_embd);
    
    println!("🔄 Demonstrando o papel das Residual Connections:");
    
    // Sem residual connections
    let attention = MultiHeadAttention::new(config.n_embd, config.n_head);
    let attention_only = attention.forward(&input);
    
    // Com residual connections
    let with_residual = input.add(&attention_only);
    
    // Calcula magnitudes
    let input_magnitude = calculate_magnitude(&input);
    let attention_magnitude = calculate_magnitude(&attention_only);
    let residual_magnitude = calculate_magnitude(&with_residual);
    
    println!("\n📏 Magnitudes dos vetores:");
    println!("   • Entrada original: {:.4}", input_magnitude);
    println!("   • Apenas atenção: {:.4}", attention_magnitude);
    println!("   • Com residual: {:.4}", residual_magnitude);
    
    println!("\n✅ Benefícios das Residual Connections:");
    println!("   • Preservam informação original");
    println!("   • Facilitam o fluxo de gradientes");
    println!("   • Permitem redes mais profundas");
    println!("   • Estabilizam o treinamento");
}

fn demo_layer_normalization() {
    println!("\n📏 === DEMONSTRAÇÃO: LAYER NORMALIZATION ===");
    
    let input = Matrix::random(4, 8);
    
    println!("🔢 Demonstrando Layer Normalization:");
    
    // Antes da normalização
    println!("\n📊 Antes da normalização:");
    for i in 0..input.rows {
        let mean: f32 = input.data[i].iter().sum::<f32>() / input.cols as f32;
        let variance: f32 = input.data[i].iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / input.cols as f32;
        println!("   Token {}: média={:.4}, variância={:.4}", i, mean, variance);
    }
    
    // Após a normalização
    let normalized = input.layer_norm();
    
    println!("\n📊 Após a normalização:");
    for i in 0..normalized.rows {
        let mean: f32 = normalized.data[i].iter().sum::<f32>() / normalized.cols as f32;
        let variance: f32 = normalized.data[i].iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / normalized.cols as f32;
        println!("   Token {}: média={:.4}, variância={:.4}", i, mean, variance);
    }
    
    println!("\n✅ Efeitos da Layer Normalization:");
    println!("   • Média ≈ 0 para cada token");
    println!("   • Variância ≈ 1 para cada token");
    println!("   • Estabiliza o treinamento");
    println!("   • Acelera a convergência");
}

fn demo_scaling_analysis() {
    println!("\n📈 === DEMONSTRAÇÃO: ANÁLISE DE ESCALABILIDADE ===");
    
    let configs = vec![
        ("Pequeno", 64, 4, 4),
        ("Médio", 128, 8, 6),
        ("Grande", 256, 16, 12),
    ];
    
    println!("🔍 Analisando diferentes tamanhos de modelo:");
    
    for (name, n_embd, n_head, n_layer) in configs {
        let config = TransformerConfig {
            n_embd,
            n_head,
            n_layer,
            dropout: 0.1,
            vocab_size: 1000,
            max_seq_len: 128,
        };
        
        // Calcula parâmetros aproximados
        let attention_params = n_embd * n_embd * 4; // Q, K, V, O projections
        let ff_params = n_embd * (n_embd * 4) * 2; // Duas camadas lineares
        let layer_params = attention_params + ff_params;
        let total_params = layer_params * n_layer;
        
        // Calcula complexidade computacional (aproximada)
        let seq_len = 128;
        let attention_flops = seq_len * seq_len * n_embd * n_head;
        let ff_flops = seq_len * n_embd * (n_embd * 4) * 2;
        let layer_flops = attention_flops + ff_flops;
        let total_flops = layer_flops * n_layer;
        
        println!("\n🏗️ Modelo {}:", name);
        println!("   • Dimensão: {}, Cabeças: {}, Camadas: {}", n_embd, n_head, n_layer);
        println!("   • Parâmetros: ~{:.1}M", total_params as f32 / 1_000_000.0);
        println!("   • FLOPs (seq={}): ~{:.1}M", seq_len, total_flops as f32 / 1_000_000.0);
        
        // Testa performance
        let input = Matrix::random(seq_len, n_embd);
        let block = TransformerBlock::new(config);
        
        let start = Instant::now();
        let _output = block.forward(&input);
        let duration = start.elapsed();
        
        println!("   • Tempo de execução: {:?}", duration);
    }
}

fn demo_optimization_techniques() {
    println!("\n⚡ === DEMONSTRAÇÃO: TÉCNICAS DE OTIMIZAÇÃO ===");
    
    println!("🚀 Principais otimizações em Transformers:");
    
    println!("\n1. 🔥 **Kernel Fusion**:");
    println!("   • Combina operações em um único kernel");
    println!("   • Reduz transferências de memória");
    println!("   • Exemplo: LayerNorm + Attention fusionados");
    
    println!("\n2. 💾 **Memory Optimization**:");
    println!("   • Gradient Checkpointing");
    println!("   • Mixed Precision (FP16/BF16)");
    println!("   • Activation Recomputation");
    
    println!("\n3. 🎯 **Attention Optimization**:");
    println!("   • Flash Attention (reduz complexidade de memória)");
    println!("   • Sparse Attention (padrões de atenção limitados)");
    println!("   • Multi-Query Attention (compartilha K,V)");
    
    println!("\n4. 🔄 **Parallelization**:");
    println!("   • Data Parallelism (batch splitting)");
    println!("   • Model Parallelism (layer splitting)");
    println!("   • Pipeline Parallelism (stage splitting)");
    
    // Simula diferentes estratégias
    let configs = vec![
        ("Padrão", 128, 8, false),
        ("Multi-Query", 128, 8, true),
    ];
    
    println!("\n📊 Comparação de estratégias:");
    
    for (name, n_embd, n_head, multi_query) in configs {
        let kv_heads = if multi_query { 1 } else { n_head };
        let memory_reduction = if multi_query {
            1.0 - (kv_heads as f32 / n_head as f32)
        } else {
            0.0
        };
        
        println!("\n🔧 Estratégia {}:", name);
        println!("   • Cabeças Q: {}", n_head);
        println!("   • Cabeças K,V: {}", kv_heads);
        println!("   • Redução de memória: {:.1}%", memory_reduction * 100.0);
    }
}

// === EXERCÍCIOS PRÁTICOS ===

fn practical_exercises() {
    println!("\n🎓 === EXERCÍCIOS PRÁTICOS ===");
    
    println!("\n📝 **Exercício 1: Análise de Atenção**");
    println!("   Implemente uma função que visualiza os padrões de atenção");
    println!("   entre diferentes tokens em uma sequência.");
    
    println!("\n📝 **Exercício 2: Otimização de Memória**");
    println!("   Compare o uso de memória entre diferentes configurações");
    println!("   de cabeças de atenção e dimensões de embedding.");
    
    println!("\n📝 **Exercício 3: Análise de Gradientes**");
    println!("   Implemente gradient clipping e analise como as");
    println!("   residual connections afetam o fluxo de gradientes.");
    
    println!("\n📝 **Exercício 4: Comparação de Arquiteturas**");
    println!("   Compare Transformer com RNN/LSTM em termos de:");
    println!("   - Paralelização");
    println!("   - Memória de longo prazo");
    println!("   - Complexidade computacional");
    
    println!("\n📝 **Exercício 5: Implementação Avançada**");
    println!("   Implemente Flash Attention ou outra otimização");
    println!("   e meça o impacto na performance.");
}

// === FUNÇÕES AUXILIARES ===

fn calculate_variance(matrix: &Matrix) -> f32 {
    let values: Vec<f32> = matrix.data.iter().flatten().cloned().collect();
    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32
}

fn calculate_magnitude(matrix: &Matrix) -> f32 {
    matrix.data.iter()
        .flatten()
        .map(|x| x.powi(2))
        .sum::<f32>()
        .sqrt()
}

fn benchmark_transformer() {
    println!("\n⚡ === BENCHMARK DE PERFORMANCE ===");
    
    let configs = vec![
        ("Mini", 64, 4, 2),
        ("Pequeno", 128, 8, 4),
        ("Médio", 256, 16, 6),
    ];
    
    let sequence_lengths = vec![32, 64, 128];
    
    println!("🏃 Testando performance com diferentes configurações:");
    
    for (name, n_embd, n_head, n_layer) in configs {
        println!("\n🔧 Modelo {}: {}d, {}h, {}l", name, n_embd, n_head, n_layer);
        
        for &seq_len in &sequence_lengths {
            let config = TransformerConfig {
                n_embd,
                n_head,
                n_layer,
                dropout: 0.0,
                vocab_size: 1000,
                max_seq_len: seq_len,
            };
            
            let input = Matrix::random(seq_len, n_embd);
            let block = TransformerBlock::new(config);
            
            // Aquecimento
            let _ = block.forward(&input);
            
            // Benchmark
            let iterations = 10;
            let start = Instant::now();
            
            for _ in 0..iterations {
                let _ = block.forward(&input);
            }
            
            let total_time = start.elapsed();
            let avg_time = total_time / iterations;
            
            println!("   📏 Seq={}: {:?}/iter", seq_len, avg_time);
        }
    }
}

fn main() {
    println!("🏗️ === DEMONSTRAÇÃO DA ARQUITETURA TRANSFORMER ===");
    println!("\nEste exemplo explora os componentes fundamentais dos Transformers,");
    println!("a arquitetura que revolucionou a IA e possibilitou modelos como GPT.");
    
    demo_transformer_basics();
    demo_attention_vs_feedforward();
    demo_residual_connections();
    demo_layer_normalization();
    demo_scaling_analysis();
    demo_optimization_techniques();
    benchmark_transformer();
    practical_exercises();
    
    println!("\n🎉 === DEMONSTRAÇÃO CONCLUÍDA ===");
    println!("\n🚀 **Próximos Passos:**");
    println!("   • Experimente com diferentes configurações");
    println!("   • Implemente otimizações avançadas");
    println!("   • Analise padrões de atenção reais");
    println!("   • Compare com outras arquiteturas");
    
    println!("\n📚 **Recursos Adicionais:**");
    println!("   • Paper original: 'Attention is All You Need'");
    println!("   • Implementação completa em src/transformer.rs");
    println!("   • Exemplos educacionais em examples/educational/");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transformer_block_forward() {
        let config = TransformerConfig {
            n_embd: 32,
            n_head: 4,
            n_layer: 1,
            dropout: 0.0,
            vocab_size: 100,
            max_seq_len: 64,
        };
        
        let input = Matrix::random(8, 32);
        let block = TransformerBlock::new(config);
        let output = block.forward(&input);
        
        assert_eq!(output.rows, input.rows);
        assert_eq!(output.cols, input.cols);
    }
    
    #[test]
    fn test_layer_normalization() {
        let input = Matrix::random(4, 8);
        let normalized = input.layer_norm();
        
        // Verifica que a normalização preserva as dimensões
        assert_eq!(normalized.rows, input.rows);
        assert_eq!(normalized.cols, input.cols);
        
        // Verifica que a média está próxima de zero
        for i in 0..normalized.rows {
            let mean: f32 = normalized.data[i].iter().sum::<f32>() / normalized.cols as f32;
            assert!((mean.abs()) < 1e-5);
        }
    }
    
    #[test]
    fn test_residual_connections() {
        let input = Matrix::random(4, 16);
        let attention = MultiHeadAttention::new(16, 4);
        let attention_out = attention.forward(&input);
        let with_residual = input.add(&attention_out);
        
        // Verifica que as dimensões são preservadas
        assert_eq!(with_residual.rows, input.rows);
        assert_eq!(with_residual.cols, input.cols);
    }
}