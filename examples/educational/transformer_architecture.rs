//! # Exemplo Didático: Arquitetura Transformer
//!
//! Este exemplo demonstra os componentes fundamentais da arquitetura Transformer:
//! - Mecanismo de Atenção Multi-Head
//! - Feed-Forward Networks
//! - Normalização de Camadas
//! - Conexões Residuais
//!
//! ## Como executar:
//! ```bash
//! cargo run --example transformer_architecture
//! ```

use std::collections::HashMap;

/// Representa um tensor simplificado para fins didáticos
#[derive(Debug, Clone)]
struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    /// Cria um novo tensor com valores aleatórios
    fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = (0..size).map(|i| (i as f32 * 0.1) % 1.0).collect();
        Self { data, shape }
    }

    /// Cria um tensor com valores específicos
    fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product());
        Self { data, shape }
    }

    /// Multiplicação de matrizes simplificada
    fn matmul(&self, other: &Tensor) -> Tensor {
        // Implementação simplificada para fins didáticos
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape[0]);
        
        let rows = self.shape[0];
        let cols = other.shape[1];
        let inner = self.shape[1];
        
        let mut result = vec![0.0; rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..inner {
                    result[i * cols + j] += self.data[i * inner + k] * other.data[k * cols + j];
                }
            }
        }
        
        Tensor::from_data(result, vec![rows, cols])
    }

    /// Aplicar função softmax
    fn softmax(&self) -> Tensor {
        let mut result = self.data.clone();
        let seq_len = self.shape[1];
        
        for i in 0..self.shape[0] {
            let start = i * seq_len;
            let end = start + seq_len;
            let slice = &mut result[start..end];
            
            // Encontrar o máximo para estabilidade numérica
            let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            // Aplicar exponencial
            for val in slice.iter_mut() {
                *val = (*val - max_val).exp();
            }
            
            // Normalizar
            let sum: f32 = slice.iter().sum();
            for val in slice.iter_mut() {
                *val /= sum;
            }
        }
        
        Tensor::from_data(result, self.shape.clone())
    }

    /// Adicionar dois tensores (conexão residual)
    fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let result: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::from_data(result, self.shape.clone())
    }

    /// Normalização de camada simplificada
    fn layer_norm(&self) -> Tensor {
        let mut result = self.data.clone();
        let feature_size = self.shape[1];
        
        for i in 0..self.shape[0] {
            let start = i * feature_size;
            let end = start + feature_size;
            let slice = &mut result[start..end];
            
            // Calcular média
            let mean: f32 = slice.iter().sum::<f32>() / feature_size as f32;
            
            // Calcular variância
            let variance: f32 = slice.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / feature_size as f32;
            
            // Normalizar
            let std_dev = (variance + 1e-6).sqrt();
            for val in slice.iter_mut() {
                *val = (*val - mean) / std_dev;
            }
        }
        
        Tensor::from_data(result, self.shape.clone())
    }
}

/// ## 1. Mecanismo de Atenção Multi-Head
/// 
/// A atenção é o coração do Transformer. Permite que o modelo "preste atenção"
/// a diferentes partes da sequência de entrada simultaneamente.
#[derive(Debug)]
struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    model_dim: usize,
    // Matrizes de projeção para Query, Key, Value
    w_q: Tensor,
    w_k: Tensor, 
    w_v: Tensor,
    w_o: Tensor, // Projeção de saída
}

impl MultiHeadAttention {
    fn new(model_dim: usize, num_heads: usize) -> Self {
        assert_eq!(model_dim % num_heads, 0, "model_dim deve ser divisível por num_heads");
        
        let head_dim = model_dim / num_heads;
        
        Self {
            num_heads,
            head_dim,
            model_dim,
            // Inicializar matrizes de peso (simplificado)
            w_q: Tensor::new(vec![model_dim, model_dim]),
            w_k: Tensor::new(vec![model_dim, model_dim]),
            w_v: Tensor::new(vec![model_dim, model_dim]),
            w_o: Tensor::new(vec![model_dim, model_dim]),
        }
    }

    /// Calcula a atenção scaled dot-product
    /// 
    /// Fórmula: Attention(Q,K,V) = softmax(QK^T / √d_k)V
    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        println!("🔍 Calculando Atenção Scaled Dot-Product:");
        println!("   - Query shape: {:?}", q.shape);
        println!("   - Key shape: {:?}", k.shape);
        println!("   - Value shape: {:?}", v.shape);
        
        // 1. Transpor K para calcular QK^T
        let k_transposed = self.transpose(k);
        println!("   - Key transposta shape: {:?}", k_transposed.shape);
        
        // 2. Calcular QK^T
        let scores = q.matmul(&k_transposed);
        println!("   - Scores (QK^T) calculados, shape: {:?}", scores.shape);
        
        // 3. Escalar por √d_k para estabilidade
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scaled_scores = Tensor::from_data(
            scores.data.iter().map(|x| x * scale).collect(),
            scores.shape.clone()
        );
        println!("   - Scores escalados por √d_k = {:.3}", scale);
        
        // 4. Aplicar softmax para obter pesos de atenção
        let attention_weights = scaled_scores.softmax();
        println!("   - Pesos de atenção calculados via softmax");
        
        // 5. Aplicar pesos aos valores
        let output = attention_weights.matmul(v);
        println!("   - Saída final: pesos × valores, shape: {:?}", output.shape);
        
        output
    }
    
    /// Transpõe uma matriz 2D
    fn transpose(&self, tensor: &Tensor) -> Tensor {
        assert_eq!(tensor.shape.len(), 2, "Transpose só funciona para matrizes 2D");
        
        let rows = tensor.shape[0];
        let cols = tensor.shape[1];
        let mut transposed_data = vec![0.0; rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = tensor.data[i * cols + j];
            }
        }
        
        Tensor::from_data(transposed_data, vec![cols, rows])
    }

    /// Processa a entrada através de múltiplas cabeças de atenção
    fn forward(&self, input: &Tensor) -> Tensor {
        println!("\n🧠 === MULTI-HEAD ATTENTION ===");
        println!("Entrada shape: {:?}", input.shape);
        println!("Número de cabeças: {}", self.num_heads);
        println!("Dimensão por cabeça: {}", self.head_dim);
        
        // 1. Projetar entrada para Q, K, V
        let q = input.matmul(&self.w_q);
        let k = input.matmul(&self.w_k);
        let v = input.matmul(&self.w_v);
        
        println!("\n📊 Projeções lineares criadas:");
        println!("   - Q (Query): {:?}", q.shape);
        println!("   - K (Key): {:?}", k.shape);
        println!("   - V (Value): {:?}", v.shape);
        
        // 2. Para cada cabeça, calcular atenção
        // (Simplificado: usando apenas uma cabeça para demonstração)
        let attention_output = self.scaled_dot_product_attention(&q, &k, &v);
        
        // 3. Projeção final
        let output = attention_output.matmul(&self.w_o);
        
        println!("\n✅ Multi-Head Attention concluída!");
        println!("Saída shape: {:?}", output.shape);
        
        output
    }
}

/// ## 2. Feed-Forward Network
/// 
/// Rede neural simples que processa cada posição independentemente.
/// Estrutura: Linear -> ReLU -> Linear
#[derive(Debug)]
struct FeedForwardNetwork {
    w1: Tensor, // Primeira camada linear
    w2: Tensor, // Segunda camada linear
    hidden_dim: usize,
}

impl FeedForwardNetwork {
    fn new(model_dim: usize, hidden_dim: usize) -> Self {
        Self {
            w1: Tensor::new(vec![model_dim, hidden_dim]),
            w2: Tensor::new(vec![hidden_dim, model_dim]),
            hidden_dim,
        }
    }

    /// Função de ativação ReLU
    fn relu(&self, tensor: &Tensor) -> Tensor {
        let data: Vec<f32> = tensor.data.iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();
        Tensor::from_data(data, tensor.shape.clone())
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        println!("\n🔄 === FEED-FORWARD NETWORK ===");
        println!("Entrada shape: {:?}", input.shape);
        println!("Dimensão oculta: {}", self.hidden_dim);
        
        // 1. Primeira transformação linear
        let hidden = input.matmul(&self.w1);
        println!("Após primeira linear: {:?}", hidden.shape);
        
        // 2. Ativação ReLU
        let activated = self.relu(&hidden);
        println!("Após ReLU: {:?}", activated.shape);
        
        // 3. Segunda transformação linear
        let output = activated.matmul(&self.w2);
        println!("Saída final: {:?}", output.shape);
        
        println!("✅ Feed-Forward concluída!");
        output
    }
}

/// ## 3. Bloco Transformer Completo
/// 
/// Combina atenção multi-head, feed-forward e conexões residuais
#[derive(Debug)]
struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForwardNetwork,
}

impl TransformerBlock {
    fn new(model_dim: usize, num_heads: usize, ff_hidden_dim: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(model_dim, num_heads),
            feed_forward: FeedForwardNetwork::new(model_dim, ff_hidden_dim),
        }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        println!("\n🏗️  === TRANSFORMER BLOCK ===");
        
        // 1. Multi-Head Attention + Conexão Residual + Layer Norm
        println!("\n🔗 Aplicando atenção com conexão residual...");
        let attention_output = self.attention.forward(input);
        let residual1 = input.add(&attention_output); // Conexão residual
        let norm1 = residual1.layer_norm(); // Normalização
        
        println!("✅ Primeira sub-camada (Atenção + Residual + Norm) concluída");
        
        // 2. Feed-Forward + Conexão Residual + Layer Norm
        println!("\n🔗 Aplicando feed-forward com conexão residual...");
        let ff_output = self.feed_forward.forward(&norm1);
        let residual2 = norm1.add(&ff_output); // Conexão residual
        let norm2 = residual2.layer_norm(); // Normalização
        
        println!("✅ Segunda sub-camada (FF + Residual + Norm) concluída");
        println!("\n🎉 TRANSFORMER BLOCK COMPLETO!");
        
        norm2
    }
}

/// Função principal para demonstrar o funcionamento
fn main() {
    println!("🚀 === DEMONSTRAÇÃO DA ARQUITETURA TRANSFORMER ===");
    println!("\nEste exemplo mostra como funciona um bloco Transformer básico.");
    println!("Vamos processar uma sequência de exemplo através de todas as camadas.\n");
    
    // Configurações do modelo
    let seq_len = 4;      // Comprimento da sequência
    let model_dim = 8;    // Dimensão do modelo
    let num_heads = 2;    // Número de cabeças de atenção
    let ff_hidden = 16;   // Dimensão oculta do feed-forward
    
    println!("📋 Configurações:");
    println!("   - Comprimento da sequência: {}", seq_len);
    println!("   - Dimensão do modelo: {}", model_dim);
    println!("   - Número de cabeças: {}", num_heads);
    println!("   - Dimensão FF oculta: {}", ff_hidden);
    
    // Criar entrada de exemplo (representando embeddings de tokens)
    let input = Tensor::new(vec![seq_len, model_dim]);
    println!("\n📥 Entrada criada: shape {:?}", input.shape);
    println!("   Representa {} tokens, cada um com {} dimensões", seq_len, model_dim);
    
    // Criar e executar o bloco Transformer
    let transformer_block = TransformerBlock::new(model_dim, num_heads, ff_hidden);
    let output = transformer_block.forward(&input);
    
    println!("\n📤 Saída final: shape {:?}", output.shape);
    println!("   Primeiros valores: {:?}", &output.data[0..4]);
    
    // Explicação dos conceitos
    println!("\n\n📚 === CONCEITOS FUNDAMENTAIS ===");
    
    println!("\n🔍 ATENÇÃO MULTI-HEAD:");
    println!("   • Permite ao modelo focar em diferentes aspectos da entrada");
    println!("   • Cada 'cabeça' aprende padrões diferentes");
    println!("   • Query: 'o que estou procurando?'");
    println!("   • Key: 'o que está disponível?'");
    println!("   • Value: 'qual informação carregar?'");
    
    println!("\n🔄 FEED-FORWARD:");
    println!("   • Processa cada posição independentemente");
    println!("   • Adiciona capacidade de representação não-linear");
    println!("   • Estrutura: Linear → ReLU → Linear");
    
    println!("\n🔗 CONEXÕES RESIDUAIS:");
    println!("   • Facilitam o treinamento de redes profundas");
    println!("   • Permitem que gradientes fluam diretamente");
    println!("   • Fórmula: output = LayerNorm(input + SubLayer(input))");
    
    println!("\n📏 NORMALIZAÇÃO DE CAMADAS:");
    println!("   • Estabiliza o treinamento");
    println!("   • Normaliza ativações para média 0 e variância 1");
    println!("   • Aplicada após cada sub-camada");
    
    // EXERCÍCIOS PRÁTICOS
    println!("\n\n🎓 === EXERCÍCIOS PRÁTICOS ===");
    exercicio_1_analise_atencao();
    exercicio_2_comparacao_dimensoes();
    exercicio_3_visualizacao_pesos();
    exercicio_4_benchmark_performance();
    exercicio_5_implementacao_posicional();
    
    println!("\n✨ Transformer concluído com sucesso! ✨");
}

/// EXERCÍCIO 1: Análise de Padrões de Atenção
fn exercicio_1_analise_atencao() {
    println!("\n--- Exercício 1: Análise de Padrões de Atenção ---");
    
    let model_dim = 64; // Menor para visualização
    let num_heads = 4;
    let seq_len = 8;
    
    let attention = MultiHeadAttention::new(model_dim, num_heads);
    
    // Cria sequência com padrão específico
    let mut input_data = vec![0.0; seq_len * model_dim];
    for i in 0..seq_len {
        for j in 0..model_dim {
            input_data[i * model_dim + j] = if j < 10 { (i + j) as f32 * 0.1 } else { 0.1 };
        }
    }
    
    let input = Tensor::from_data(input_data, vec![seq_len, model_dim]);
    let output = attention.forward(&input);
    
    println!("Entrada: {} tokens, {} dimensões", seq_len, model_dim);
    println!("Saída: {:?}", output.shape);
    
    // Simula análise de atenção
    println!("Padrões de atenção detectados:");
    for head in 0..num_heads {
        let attention_strength = (head as f32 + 1.0) / num_heads as f32;
        println!("  Cabeça {}: Força de atenção {:.2}", head, attention_strength);
    }
    
    println!("💡 Dica: Em modelos reais, você pode extrair e visualizar os pesos de atenção!");
}

/// EXERCÍCIO 2: Comparação de Diferentes Dimensões
fn exercicio_2_comparacao_dimensoes() {
    println!("\n--- Exercício 2: Comparação de Diferentes Dimensões ---");
    
    let configuracoes = vec![
        ("Pequeno", 128, 4, 512),
        ("Médio", 256, 8, 1024),
        ("Grande", 512, 16, 2048),
    ];
    
    for (nome, model_dim, num_heads, ff_dim) in configuracoes {
        println!("\nConfiguracao {}: {}d, {}h, {}ff", nome, model_dim, num_heads, ff_dim);
        
        let transformer = TransformerBlock::new(model_dim, num_heads, ff_dim);
        let input = Tensor::new(vec![10, model_dim]);
        
        let start = std::time::Instant::now();
        let _output = transformer.forward(&input);
        let duration = start.elapsed();
        
        let params = calcular_parametros(model_dim, num_heads, ff_dim);
        
        println!("  Parâmetros: ~{:.1}K", params as f32 / 1000.0);
        println!("  Tempo: {:?}", duration);
        println!("  Memória estimada: ~{:.1}MB", (params * 4) as f32 / (1024.0 * 1024.0));
    }
    
    println!("\n💡 Dica: Observe como o número de parâmetros cresce quadraticamente!");
}

/// EXERCÍCIO 3: Visualização de Pesos
fn exercicio_3_visualizacao_pesos() {
    println!("\n--- Exercício 3: Visualização de Pesos ---");
    
    let model_dim = 64;
    let num_heads = 4;
    
    let attention = MultiHeadAttention::new(model_dim, num_heads);
    
    println!("Análise dos pesos de atenção:");
    println!("  W_Q shape: {:?}", attention.w_q.shape);
    println!("  W_K shape: {:?}", attention.w_k.shape);
    println!("  W_V shape: {:?}", attention.w_v.shape);
    println!("  W_O shape: {:?}", attention.w_o.shape);
    
    // Simula análise de distribuição de pesos
    let sample_weights = &attention.w_q.data[0..10];
    let mean = sample_weights.iter().sum::<f32>() / sample_weights.len() as f32;
    let variance = sample_weights.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / sample_weights.len() as f32;
    
    println!("\nEstatísticas dos pesos W_Q (amostra):");
    println!("  Média: {:.4}", mean);
    println!("  Variância: {:.4}", variance);
    println!("  Desvio padrão: {:.4}", variance.sqrt());
    
    println!("\n💡 Dica: Inicialização adequada dos pesos é crucial para convergência!");
}

/// EXERCÍCIO 4: Benchmark de Performance
fn exercicio_4_benchmark_performance() {
    println!("\n--- Exercício 4: Benchmark de Performance ---");
    
    let model_dim = 256;
    let num_heads = 8;
    let ff_dim = 1024;
    
    let transformer = TransformerBlock::new(model_dim, num_heads, ff_dim);
    
    let sequencias = vec![10, 50, 100, 200];
    
    println!("Testando diferentes comprimentos de sequência:");
    
    for seq_len in sequencias {
        let input = Tensor::new(vec![seq_len, model_dim]);
        
        // Aquecimento
        let _warmup = transformer.forward(&input);
        
        // Benchmark
        let num_runs = 10;
        let start = std::time::Instant::now();
        
        for _ in 0..num_runs {
            let _output = transformer.forward(&input);
        }
        
        let total_time = start.elapsed();
        let avg_time = total_time / num_runs;
        let tokens_per_sec = (seq_len as f64 / avg_time.as_secs_f64()) as u32;
        
        println!("  Seq len {}: {:?}/forward, ~{} tokens/s", 
                seq_len, avg_time, tokens_per_sec);
    }
    
    println!("\n💡 Dica: A complexidade da atenção é O(n²) com o comprimento da sequência!");
}

/// EXERCÍCIO 5: Implementação de Encoding Posicional
fn exercicio_5_implementacao_posicional() {
    println!("\n--- Exercício 5: Encoding Posicional ---");
    
    let model_dim = 128;
    let max_seq_len = 100;
    
    // Implementa encoding posicional sinusoidal
    let pos_encoding = criar_encoding_posicional(max_seq_len, model_dim);
    
    println!("Encoding posicional criado: {:?}", pos_encoding.shape);
    
    // Demonstra como aplicar
    let seq_len = 20;
    let input = Tensor::new(vec![seq_len, model_dim]);
    let input_com_posicao = adicionar_encoding_posicional(&input, &pos_encoding);
    
    println!("Entrada original: {:?}", input.shape);
    println!("Com encoding posicional: {:?}", input_com_posicao.shape);
    
    // Analisa padrões
    println!("\nAnálise do encoding posicional:");
    for pos in [0, 5, 10, 15] {
        if pos < seq_len {
            let sample = &pos_encoding.data[pos * model_dim..pos * model_dim + 5];
            println!("  Posição {}: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}...]", 
                    pos, sample[0], sample[1], sample[2], sample[3], sample[4]);
        }
    }
    
    println!("\n💡 Dica: O encoding posicional permite ao modelo entender a ordem das palavras!");
}

/// Calcula número aproximado de parâmetros
fn calcular_parametros(model_dim: usize, num_heads: usize, ff_dim: usize) -> usize {
    // Atenção: 4 matrizes de peso (Q, K, V, O)
    let attention_params = 4 * model_dim * model_dim;
    
    // Feed-forward: 2 matrizes de peso
    let ff_params = model_dim * ff_dim + ff_dim * model_dim;
    
    // Layer norm: 2 * model_dim (gamma e beta) para cada uma
    let ln_params = 4 * model_dim;
    
    attention_params + ff_params + ln_params
}

/// Cria encoding posicional sinusoidal
fn criar_encoding_posicional(max_len: usize, model_dim: usize) -> Tensor {
    let mut data = vec![0.0; max_len * model_dim];
    
    for pos in 0..max_len {
        for i in 0..model_dim {
            let angle = pos as f32 / 10000.0_f32.powf(2.0 * (i / 2) as f32 / model_dim as f32);
            
            if i % 2 == 0 {
                data[pos * model_dim + i] = angle.sin();
            } else {
                data[pos * model_dim + i] = angle.cos();
            }
        }
    }
    
    Tensor::from_data(data, vec![max_len, model_dim])
}

/// Adiciona encoding posicional à entrada
fn adicionar_encoding_posicional(input: &Tensor, pos_encoding: &Tensor) -> Tensor {
    // Simplificado: apenas retorna a entrada (em implementação real, somaria)
    let mut result = input.clone();
    
    // Simula adição do encoding posicional
    for i in 0..result.data.len().min(pos_encoding.data.len()) {
        result.data[i] += pos_encoding.data[i] * 0.1; // Fator de escala
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_operations() {
        let t1 = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        
        let result = t1.matmul(&t2);
        assert_eq!(result.shape, vec![2, 2]);
    }

    #[test]
    fn test_multi_head_attention() {
        let attention = MultiHeadAttention::new(8, 2);
        let input = Tensor::new(vec![4, 8]);
        
        let output = attention.forward(&input);
        assert_eq!(output.shape, input.shape);
    }

    #[test]
    fn test_transformer_block() {
        let block = TransformerBlock::new(8, 2, 16);
        let input = Tensor::new(vec![4, 8]);
        
        let output = block.forward(&input);
        assert_eq!(output.shape, input.shape);
    }
}