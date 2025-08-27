//! # 🧠 Demonstração do Mecanismo de Atenção
//!
//! Este exemplo demonstra como o mecanismo de Self-Attention funciona na prática,
//! mostrando desde conceitos básicos até implementações avançadas com múltiplas cabeças.
//!
//! ## 🎯 O que você vai aprender:
//! - Como funciona o Scaled Dot-Product Attention
//! - Visualização de matrizes Q, K, V
//! - Multi-Head Attention em ação
//! - Padrões de atenção em sequências reais
//! - Análise de performance e complexidade

use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::time::Instant;

// Importa os módulos do projeto principal
// Ajuste os caminhos conforme a estrutura do seu projeto
use std::path::PathBuf;

// Para compilar este exemplo, você precisa ter acesso aos módulos
// ou implementar versões simplificadas das estruturas

// Estruturas simplificadas para demonstração
// Em um projeto real, estas viriam de mini_gpt_rust::attention
struct SelfAttention {
    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    dropout: f32,
}

struct MultiHeadAttention {
    attention: SelfAttention,
}

impl SelfAttention {
    fn new(n_embd: usize, n_head: usize, dropout: f32, _vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            n_embd,
            n_head,
            head_dim: n_embd / n_head,
            dropout,
        })
    }
    
    fn forward(&self, x: &Tensor, _mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        // Implementação simplificada para demonstração
        Ok(x.clone())
    }
    
    fn get_qkv_tensors(&self, x: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        // Simula projeções Q, K, V
        let q = x.clone();
        let k = x.clone();
        let v = x.clone();
        Ok((q, k, v))
    }
}

impl MultiHeadAttention {
    fn new(n_embd: usize, n_head: usize, dropout: f32, vb: VarBuilder) -> candle_core::Result<Self> {
        let attention = SelfAttention::new(n_embd, n_head, dropout, vb)?;
        Ok(Self { attention })
    }
    
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        self.attention.forward(x, mask)
    }
    
    fn get_qkv_tensors(&self, x: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        self.attention.get_qkv_tensors(x)
    }
}

/// 🎭 Estrutura para demonstrar conceitos de atenção
struct AttentionDemo {
    device: Device,
    vocab_size: usize,
    seq_len: usize,
    n_embd: usize,
    n_head: usize,
}

impl AttentionDemo {
    /// 🏗️ Cria uma nova instância de demonstração
    fn new() -> candle_core::Result<Self> {
        let device = Device::Cpu; // Use Device::cuda(0)? para GPU
        
        Ok(Self {
            device,
            vocab_size: 1000,
            seq_len: 8,
            n_embd: 64,
            n_head: 4,
        })
    }
    
    /// 🔍 Demonstra o conceito básico de atenção com exemplo simples
    fn demo_basic_attention_concept(&self) -> candle_core::Result<()> {
        println!("\n🔍 === CONCEITO BÁSICO DE ATENÇÃO ===");
        println!("Vamos simular como a atenção funciona com uma frase simples:");
        println!("Frase: 'O gato subiu no telhado'");
        
        // Simula tokens da frase
        let tokens = vec!["O", "gato", "subiu", "no", "telhado"];
        let seq_len = tokens.len();
        
        println!("\n📊 Tokens indexados:");
        for (i, token) in tokens.iter().enumerate() {
            println!("  {}: {}", i, token);
        }
        
        // Cria embeddings simulados (normalmente seriam aprendidos)
        let embeddings = Tensor::randn(0f32, 1f32, (seq_len, self.n_embd), &self.device)?;
        
        println!("\n🧮 Dimensões dos embeddings: {:?}", embeddings.shape());
        println!("Cada token é representado por um vetor de {} dimensões", self.n_embd);
        
        // Demonstra o cálculo de similaridade básico
        self.demo_similarity_calculation(&embeddings, &tokens)?;
        
        Ok(())
    }
    
    /// 📐 Demonstra cálculo de similaridade entre tokens
    fn demo_similarity_calculation(&self, embeddings: &Tensor, tokens: &[&str]) -> candle_core::Result<()> {
        println!("\n📐 === CÁLCULO DE SIMILARIDADE ===");
        
        // Calcula produto escalar entre todos os pares de tokens
        let similarity_matrix = embeddings.matmul(&embeddings.t()?)?;
        let similarity_data = similarity_matrix.to_vec2::<f32>()?;
        
        println!("\n🔢 Matriz de Similaridade (produto escalar):");
        print!("     ");
        for token in tokens {
            print!("{:>8}", token);
        }
        println!();
        
        for (i, row) in similarity_data.iter().enumerate() {
            print!("{:>4} ", tokens[i]);
            for &value in row {
                print!("{:>8.2}", value);
            }
            println!();
        }
        
        println!("\n💡 Interpretação:");
        println!("  - Valores maiores = tokens mais 'similares' ou relacionados");
        println!("  - Diagonal = auto-similaridade (sempre alta)");
        println!("  - Off-diagonal = relações entre diferentes tokens");
        
        Ok(())
    }
    
    /// 🎯 Demonstra Self-Attention completo com Q, K, V
    fn demo_self_attention(&self) -> candle_core::Result<()> {
        println!("\n🎯 === SELF-ATTENTION COMPLETO ===");
        
        // Cria VarBuilder para inicializar pesos
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
        
        // Inicializa camada de Self-Attention
        let attention = SelfAttention::new(self.n_embd, self.n_head, 0.1, vb.pp("attention"))?;
        
        // Cria sequência de entrada simulada
        let batch_size = 1;
        let input = Tensor::randn(0f32, 1f32, (batch_size, self.seq_len, self.n_embd), &self.device)?;
        
        println!("📊 Entrada: shape = {:?}", input.shape());
        println!("  - Batch size: {}", batch_size);
        println!("  - Sequence length: {}", self.seq_len);
        println!("  - Embedding dimension: {}", self.n_embd);
        
        // Extrai matrizes Q, K, V
        let (q, k, v) = attention.get_qkv_tensors(&input)?;
        
        println!("\n🔍 Matrizes Q, K, V:");
        println!("  - Query (Q): {:?} - 'O que eu quero saber?'", q.shape());
        println!("  - Key (K):   {:?} - 'O que eu tenho para oferecer?'", k.shape());
        println!("  - Value (V): {:?} - 'Minha informação real'", v.shape());
        
        // Aplica Self-Attention
        let start_time = Instant::now();
        let output = attention.forward(&input, None)?;
        let duration = start_time.elapsed();
        
        println!("\n✅ Saída da Self-Attention:");
        println!("  - Shape: {:?}", output.shape());
        println!("  - Tempo de execução: {:?}", duration);
        
        // Analisa mudanças na representação
        self.analyze_attention_output(&input, &output)?;
        
        Ok(())
    }
    
    /// 📊 Analisa como a atenção modifica as representações
    fn analyze_attention_output(&self, input: &Tensor, output: &Tensor) -> candle_core::Result<()> {
        println!("\n📊 === ANÁLISE DA TRANSFORMAÇÃO ===");
        
        // Calcula normas dos vetores antes e depois
        let input_norms = input.sqr()?.sum_keepdim(2)?.sqrt()?;
        let output_norms = output.sqr()?.sum_keepdim(2)?.sqrt()?;
        
        let input_norms_data = input_norms.squeeze(0)?.to_vec1::<f32>()?;
        let output_norms_data = output_norms.squeeze(0)?.to_vec1::<f32>()?;
        
        println!("🔢 Normas dos vetores por posição:");
        println!("Pos  | Antes    | Depois   | Mudança");
        println!("-----|----------|----------|----------");
        
        for i in 0..self.seq_len {
            let before = input_norms_data[i];
            let after = output_norms_data[i];
            let change = ((after - before) / before * 100.0);
            println!("{:3}  | {:8.3} | {:8.3} | {:+7.2}%", i, before, after, change);
        }
        
        // Calcula similaridade entre entrada e saída
        let similarity = self.calculate_cosine_similarity(input, output)?;
        println!("\n🎯 Similaridade cosseno média: {:.4}", similarity);
        println!("  - 1.0 = idêntico, 0.0 = ortogonal, -1.0 = oposto");
        
        if similarity > 0.8 {
            println!("  ✅ Alta preservação da informação original");
        } else if similarity > 0.5 {
            println!("  ⚠️  Transformação moderada da informação");
        } else {
            println!("  🔄 Transformação significativa da informação");
        }
        
        Ok(())
    }
    
    /// 🧮 Calcula similaridade cosseno entre tensores
    fn calculate_cosine_similarity(&self, a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
        let a_flat = a.flatten_all()?;
        let b_flat = b.flatten_all()?;
        
        let dot_product = (&a_flat * &b_flat)?.sum_all()?.to_scalar::<f32>()?;
        let norm_a = a_flat.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let norm_b = b_flat.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        
        Ok(dot_product / (norm_a * norm_b))
    }
    
    /// 👁️ Demonstra Multi-Head Attention
    fn demo_multi_head_attention(&self) -> candle_core::Result<()> {
        println!("\n👁️ === MULTI-HEAD ATTENTION ===");
        println!("Múltiplas 'perspectivas' de atenção trabalhando em paralelo");
        
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
        
        let mha = MultiHeadAttention::new(self.n_embd, self.n_head, 0.1, vb.pp("mha"))?;
        
        let batch_size = 2;
        let input = Tensor::randn(0f32, 1f32, (batch_size, self.seq_len, self.n_embd), &self.device)?;
        
        println!("\n🏗️ Configuração Multi-Head:");
        println!("  - Número de cabeças: {}", self.n_head);
        println!("  - Dimensão por cabeça: {}", self.n_embd / self.n_head);
        println!("  - Dimensão total: {}", self.n_embd);
        
        // Mede performance
        let start_time = Instant::now();
        let output = mha.forward(&input, None)?;
        let duration = start_time.elapsed();
        
        println!("\n⚡ Performance:");
        println!("  - Tempo de execução: {:?}", duration);
        println!("  - Throughput: {:.2} tokens/ms", 
                (batch_size * self.seq_len) as f64 / duration.as_millis() as f64);
        
        // Analisa complexidade
        self.analyze_complexity()?;
        
        Ok(())
    }
    
    /// 📈 Analisa complexidade computacional
    fn analyze_complexity(&self) -> candle_core::Result<()> {
        println!("\n📈 === ANÁLISE DE COMPLEXIDADE ===");
        
        let seq_len = self.seq_len;
        let d_model = self.n_embd;
        
        // Complexidade do Self-Attention: O(n²d)
        let attention_ops = seq_len * seq_len * d_model;
        
        // Complexidade das projeções lineares: O(nd²)
        let linear_ops = 4 * seq_len * d_model * d_model; // Q, K, V, Output
        
        println!("🧮 Operações computacionais:");
        println!("  - Self-Attention: {} ops (O(n²d))", attention_ops);
        println!("  - Projeções lineares: {} ops (O(nd²))", linear_ops);
        println!("  - Total: {} ops", attention_ops + linear_ops);
        
        println!("\n📊 Escalabilidade:");
        for &n in &[16, 32, 64, 128, 256, 512] {
            let attention_cost = n * n * d_model;
            let linear_cost = 4 * n * d_model * d_model;
            let total_cost = attention_cost + linear_cost;
            
            println!("  seq_len={:3}: {:>10} ops ({:.1}x)", 
                    n, total_cost, total_cost as f32 / (attention_ops + linear_ops) as f32);
        }
        
        Ok(())
    }
    
    /// 🎨 Demonstra padrões de atenção com máscara causal
    fn demo_causal_attention(&self) -> candle_core::Result<()> {
        println!("\n🎨 === ATENÇÃO CAUSAL (AUTOREGRESSIVE) ===");
        println!("Impede que tokens 'vejam o futuro' durante o treinamento");
        
        // Cria máscara causal (triangular inferior)
        let mask = self.create_causal_mask(self.seq_len)?;
        
        println!("\n🔒 Máscara Causal (1=permitido, 0=bloqueado):");
        let mask_data = mask.to_vec2::<f32>()?;
        
        print!("     ");
        for i in 0..self.seq_len {
            print!("{:3}", i);
        }
        println!();
        
        for (i, row) in mask_data.iter().enumerate() {
            print!("{:3}: ", i);
            for &value in row {
                if value > 0.5 {
                    print!(" ✓ ");
                } else {
                    print!(" ✗ ");
                }
            }
            println!();
        }
        
        println!("\n💡 Interpretação:");
        println!("  - Token na posição i só pode 'ver' tokens nas posições 0..=i");
        println!("  - Isso simula geração sequencial durante inferência");
        println!("  - Previne vazamento de informação do futuro");
        
        Ok(())
    }
    
    /// 🔒 Cria máscara causal triangular
    fn create_causal_mask(&self, size: usize) -> candle_core::Result<Tensor> {
        let mut mask_data = vec![vec![0.0f32; size]; size];
        
        for i in 0..size {
            for j in 0..=i {
                mask_data[i][j] = 1.0;
            }
        }
        
        let flat_data: Vec<f32> = mask_data.into_iter().flatten().collect();
        Tensor::from_vec(flat_data, (size, size), &self.device)
    }
    
    /// 🏃‍♂️ Executa benchmark de performance
    fn benchmark_attention(&self) -> candle_core::Result<()> {
        println!("\n🏃‍♂️ === BENCHMARK DE PERFORMANCE ===");
        
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
        
        let configs = vec![
            ("Pequeno", 32, 4, 64),
            ("Médio", 64, 8, 128),
            ("Grande", 128, 12, 256),
        ];
        
        println!("\n⏱️  Resultados do Benchmark:");
        println!("Config   | Seq Len | Heads | Embd | Tempo (ms) | Throughput");
        println!("---------|---------|-------|------|------------|------------");
        
        for (name, seq_len, n_head, n_embd) in configs {
            let attention = SelfAttention::new(n_embd, n_head, 0.0, vb.pp(&format!("bench_{}", name)))?;
            let input = Tensor::randn(0f32, 1f32, (1, seq_len, n_embd), &self.device)?;
            
            // Aquecimento
            for _ in 0..3 {
                let _ = attention.forward(&input, None)?;
            }
            
            // Benchmark real
            let start = Instant::now();
            let iterations = 10;
            
            for _ in 0..iterations {
                let _ = attention.forward(&input, None)?;
            }
            
            let duration = start.elapsed();
            let avg_time = duration.as_millis() as f32 / iterations as f32;
            let throughput = seq_len as f32 / avg_time * 1000.0;
            
            println!("{:8} | {:7} | {:5} | {:4} | {:10.2} | {:8.0} tok/s", 
                    name, seq_len, n_head, n_embd, avg_time, throughput);
        }
        
        Ok(())
    }
}

/// 🎯 Exercícios práticos para aprofundar o entendimento
struct AttentionExercises;

impl AttentionExercises {
    /// 📝 Exercício 1: Análise de padrões de atenção
    fn exercise_attention_patterns() {
        println!("\n📝 === EXERCÍCIO 1: PADRÕES DE ATENÇÃO ===");
        println!("\n🎯 Objetivo: Entender como diferentes tipos de texto geram padrões distintos");
        
        let patterns = vec![
            ("Narrativa", "Era uma vez um rei que vivia em um castelo"),
            ("Técnico", "A função retorna um valor booleano verdadeiro"),
            ("Poesia", "Lua cheia brilha sobre o mar sereno"),
        ];
        
        println!("\n📊 Padrões esperados por tipo de texto:");
        for (tipo, exemplo) in patterns {
            println!("\n🔸 {}: '{}'", tipo, exemplo);
            match tipo {
                "Narrativa" => {
                    println!("  - Atenção forte entre sujeito-verbo-objeto");
                    println!("  - Referências pronominais conectam a entidades");
                    println!("  - Sequência temporal de eventos");
                },
                "Técnico" => {
                    println!("  - Atenção em termos técnicos específicos");
                    println!("  - Relações funcionais entre conceitos");
                    println!("  - Estrutura lógica mais que temporal");
                },
                "Poesia" => {
                    println!("  - Atenção em palavras com carga emocional");
                    println!("  - Conexões semânticas e sonoras");
                    println!("  - Padrões rítmicos e métricos");
                },
                _ => {}
            }
        }
        
        println!("\n💡 Experimento sugerido:");
        println!("  1. Implemente visualização de matrizes de atenção");
        println!("  2. Compare padrões entre diferentes tipos de texto");
        println!("  3. Identifique cabeças especializadas em diferentes aspectos");
    }
    
    /// 🔬 Exercício 2: Otimização de performance
    fn exercise_performance_optimization() {
        println!("\n🔬 === EXERCÍCIO 2: OTIMIZAÇÃO DE PERFORMANCE ===");
        println!("\n🎯 Objetivo: Implementar otimizações para acelerar a atenção");
        
        println!("\n⚡ Técnicas de otimização:");
        println!("\n1. 🧮 Flash Attention:");
        println!("   - Reduz uso de memória de O(n²) para O(n)");
        println!("   - Usa tiling e recomputação inteligente");
        println!("   - Speedup de 2-4x em sequências longas");
        
        println!("\n2. 🎯 Sparse Attention:");
        println!("   - Só computa atenção para subconjunto de posições");
        println!("   - Padrões: local, strided, random");
        println!("   - Reduz complexidade de O(n²) para O(n√n)");
        
        println!("\n3. 📦 Quantização:");
        println!("   - FP16 ou INT8 em vez de FP32");
        println!("   - Reduz uso de memória e aumenta throughput");
        println!("   - Cuidado com precisão numérica");
        
        println!("\n4. 🔄 Kernel Fusion:");
        println!("   - Combina operações em um único kernel GPU");
        println!("   - Reduz transferências de memória");
        println!("   - Implementação específica para hardware");
        
        println!("\n💡 Implementação sugerida:");
        println!("  1. Benchmark baseline com implementação atual");
        println!("  2. Implemente uma das técnicas acima");
        println!("  3. Meça speedup e uso de memória");
        println!("  4. Analise trade-offs entre velocidade e precisão");
    }
    
    /// 🎭 Exercício 3: Visualização de atenção
    fn exercise_attention_visualization() {
        println!("\n🎭 === EXERCÍCIO 3: VISUALIZAÇÃO DE ATENÇÃO ===");
        println!("\n🎯 Objetivo: Criar visualizações para entender padrões de atenção");
        
        println!("\n🎨 Tipos de visualização:");
        println!("\n1. 🔥 Heatmap de Atenção:");
        println!("   - Matriz colorida mostrando pesos de atenção");
        println!("   - Eixo X: tokens de origem, Eixo Y: tokens de destino");
        println!("   - Cores quentes = alta atenção, frias = baixa atenção");
        
        println!("\n2. 🕸️ Grafo de Atenção:");
        println!("   - Nós = tokens, arestas = pesos de atenção");
        println!("   - Espessura da aresta proporcional ao peso");
        println!("   - Layout que preserva ordem sequencial");
        
        println!("\n3. 📊 Análise por Cabeça:");
        println!("   - Visualização separada para cada cabeça de atenção");
        println!("   - Identificação de especializações");
        println!("   - Comparação entre diferentes cabeças");
        
        println!("\n4. 🎬 Animação Temporal:");
        println!("   - Mostra evolução da atenção durante geração");
        println!("   - Útil para entender processo autoregressivo");
        println!("   - Revela como contexto influencia próximas palavras");
        
        println!("\n💡 Ferramentas sugeridas:");
        println!("  - plotters (Rust) para gráficos estáticos");
        println!("  - egui (Rust) para interface interativa");
        println!("  - Export para Python/matplotlib para análise avançada");
    }
}

/// 🚀 Função principal que executa todas as demonstrações
fn main() -> candle_core::Result<()> {
    println!("🧠 === DEMONSTRAÇÃO DO MECANISMO DE ATENÇÃO ===");
    println!("Explorando o coração dos modelos Transformer");
    
    let demo = AttentionDemo::new()?;
    
    // Executa demonstrações básicas
    demo.demo_basic_attention_concept()?;
    demo.demo_self_attention()?;
    demo.demo_multi_head_attention()?;
    demo.demo_causal_attention()?;
    
    // Benchmark de performance
    demo.benchmark_attention()?;
    
    // Exercícios educacionais
    println!("\n\n🎓 === EXERCÍCIOS PRÁTICOS ===");
    AttentionExercises::exercise_attention_patterns();
    AttentionExercises::exercise_performance_optimization();
    AttentionExercises::exercise_attention_visualization();
    
    println!("\n\n✅ === DEMONSTRAÇÃO CONCLUÍDA ===");
    println!("🎯 Próximos passos:");
    println!("  1. Experimente com diferentes configurações");
    println!("  2. Implemente os exercícios sugeridos");
    println!("  3. Teste com sequências reais de texto");
    println!("  4. Explore otimizações avançadas");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_demo_creation() {
        let demo = AttentionDemo::new().unwrap();
        assert_eq!(demo.n_embd, 64);
        assert_eq!(demo.n_head, 4);
        assert_eq!(demo.seq_len, 8);
    }
    
    #[test]
    fn test_causal_mask_creation() {
        let demo = AttentionDemo::new().unwrap();
        let mask = demo.create_causal_mask(4).unwrap();
        let mask_data = mask.to_vec2::<f32>().unwrap();
        
        // Verifica estrutura triangular inferior
        assert_eq!(mask_data[0][0], 1.0);
        assert_eq!(mask_data[1][0], 1.0);
        assert_eq!(mask_data[1][1], 1.0);
        assert_eq!(mask_data[0][1], 0.0);
        assert_eq!(mask_data[0][2], 0.0);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let demo = AttentionDemo::new().unwrap();
        let device = &demo.device;
        
        // Testa similaridade de tensor consigo mesmo (deve ser 1.0)
        let tensor = Tensor::randn(0f32, 1f32, (2, 3, 4), device).unwrap();
        let similarity = demo.calculate_cosine_similarity(&tensor, &tensor).unwrap();
        
        assert!((similarity - 1.0).abs() < 1e-5);
    }
}