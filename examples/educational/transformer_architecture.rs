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
    
    println!("\n\n🎓 === EXERCÍCIOS SUGERIDOS ===");
    println!("1. Modifique o número de cabeças e observe o comportamento");
    println!("2. Altere as dimensões do modelo e feed-forward");
    println!("3. Adicione mais blocos Transformer em sequência");
    println!("4. Implemente diferentes funções de ativação");
    println!("5. Adicione dropout para regularização");
    
    println!("\n✨ Transformer concluído com sucesso! ✨");
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