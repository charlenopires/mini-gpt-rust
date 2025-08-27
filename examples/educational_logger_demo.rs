//! # 📚 **DEMONSTRAÇÃO: SISTEMA DE LOGGING EDUCACIONAL AVANÇADO**
//!
//! Este exemplo demonstra o sistema completo de logging educacional que torna
//! visível todo o ciclo de vida de um Large Language Model (LLM), desde a
//! inicialização até a geração de texto.
//!
//! ## 🎯 **O QUE VOCÊ VAI APRENDER:**
//!
//! 1. **Logging de Inicialização** - Como o modelo é construído
//! 2. **Logging de Treinamento** - Como o modelo aprende
//! 3. **Logging de Inferência** - Como o modelo gera texto
//! 4. **Análise de Performance** - Métricas e otimizações
//! 5. **Visualização de Atenção** - Como o modelo "presta atenção"
//! 6. **Debugging Avançado** - Detecção de problemas

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// 🏗️ ESTRUTURAS SIMPLIFICADAS PARA DEMONSTRAÇÃO
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum ModelPhase {
    /// 🏗️ Inicializando arquitetura e pesos
    Initialization,
    /// 📊 Preparando dados de entrada
    Preprocessing,
    /// 🎓 Treinando o modelo
    Training { epoch: u32, batch: u32 },
    /// 🔮 Gerando texto
    Inference { step: u32 },
    /// 📈 Avaliando performance
    Evaluation,
}

#[derive(Debug, Clone)]
pub struct TrainingEpochMetrics {
    /// 📊 Número da época
    pub epoch: u32,
    /// 📉 Loss média da época
    pub avg_loss: f32,
    /// 📉 Menor loss da época
    pub min_loss: f32,
    /// 📈 Maior loss da época
    pub max_loss: f32,
    /// ⏱️ Duração da época em ms
    pub duration_ms: u64,
    /// 🎯 Acurácia (opcional)
    pub accuracy: Option<f32>,
    /// 📐 Norma dos gradientes
    pub gradient_norm: f32,
    /// 📚 Taxa de aprendizado
    pub learning_rate: f32,
    /// 💾 Uso de memória em MB
    pub memory_usage_mb: f32,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    /// ⏰ Timestamp da operação
    pub timestamp: Instant,
    /// 🏷️ Tipo da operação
    pub operation_type: OperationType,
    /// 📝 Descrição da operação
    pub description: String,
    /// 📊 Dados associados
    pub data: LogData,
    /// ⭐ Importância (0-3)
    pub importance: u8,
}

#[derive(Debug, Clone)]
pub enum OperationType {
    /// 🏗️ Inicialização de componentes
    Initialization(String),
    /// ➡️ Forward pass
    Forward(String),
    /// ⬅️ Backward pass
    Backward(String),
    /// 👁️ Operações de atenção
    Attention(String),
    /// 🧮 Operações matemáticas
    Mathematical(String),
    /// 📊 Métricas
    Metric(String),
    /// ⚠️ Avisos
    Warning(String),
}

#[derive(Debug, Clone)]
pub enum LogData {
    /// 📊 Informações sobre tensors
    TensorInfo {
        shape: Vec<usize>,
        dtype: String,
        device: String,
        memory_mb: f32,
    },
    /// 🧮 Resultado de operação matemática
    MathResult {
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
        operation: String,
        flops: u64,
    },
    /// 📈 Métrica
    Metric {
        name: String,
        value: f32,
        unit: String,
    },
    /// ⚡ Performance
    Performance {
        duration_ms: f64,
        memory_delta_mb: f32,
        cpu_usage: f32,
    },
    /// 📝 Texto simples
    Text(String),
}

#[derive(Debug, Clone, Default)]
pub struct LogStats {
    /// 🔢 Total de operações
    pub total_operations: u64,
    /// ⏱️ Tempo total em ms
    pub total_time_ms: f64,
    /// 💾 Pico de memória em MB
    pub peak_memory_mb: f32,
    /// 🧮 Total de FLOPs
    pub total_flops: u64,
    /// 📊 Distribuição de operações
    pub operation_distribution: HashMap<String, u64>,
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }
    
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut matrix = Self::new(rows, cols);
        for i in 0..matrix.data.len() {
            matrix.data[i] = (i as f32 * 0.1) % 1.0;
        }
        matrix
    }
    
    pub fn shape(&self) -> Vec<usize> {
        vec![self.rows, self.cols]
    }
    
    pub fn memory_mb(&self) -> f32 {
        (self.data.len() * 4) as f32 / 1024.0 / 1024.0
    }
}

// ============================================================================
// 📚 SISTEMA DE LOGGING EDUCACIONAL
// ============================================================================

#[derive(Debug, Clone)]
pub struct EducationalLogger {
    /// 📚 Histórico completo de todas as operações
    pub operations: Vec<LogEntry>,
    /// ⏱️ Medição de tempos para análise de performance
    pub timestamps: HashMap<String, Instant>,
    /// 📊 Estatísticas acumuladas durante execução
    pub stats: LogStats,
    /// 🎯 Nível de detalhamento (0=mínimo, 3=máximo)
    pub verbosity: u8,
    /// 🏗️ Fase atual do modelo
    pub current_phase: ModelPhase,
    /// 📈 Métricas de treinamento por época
    pub training_metrics: Vec<TrainingEpochMetrics>,
    /// 🧮 Contador de operações matemáticas
    pub operation_counts: HashMap<String, u64>,
    /// 🔧 Configurações de visualização
    pub show_tensors: bool,
    pub show_attention: bool,
}

impl EducationalLogger {
    pub fn new(verbosity: u8) -> Self {
        Self {
            operations: Vec::new(),
            timestamps: HashMap::new(),
            stats: LogStats::default(),
            verbosity,
            current_phase: ModelPhase::Initialization,
            training_metrics: Vec::new(),
            operation_counts: HashMap::new(),
            show_tensors: true,
            show_attention: true,
        }
    }
    
    pub fn new_simple() -> Self {
        Self::new(1)
    }
    
    pub fn with_verbosity(mut self, verbosity: u8) -> Self {
        self.verbosity = verbosity;
        self
    }
    
    pub fn with_tensor_info(mut self, show_tensors: bool) -> Self {
        self.show_tensors = show_tensors;
        self
    }
    
    pub fn with_attention_maps(mut self, show_attention: bool) -> Self {
        self.show_attention = show_attention;
        self
    }
    
    /// 🏗️ Iniciar uma nova fase do modelo
    pub fn start_phase(&mut self, phase: ModelPhase) {
        let phase_name = match &phase {
            ModelPhase::Initialization => "🏗️ INICIALIZAÇÃO",
            ModelPhase::Preprocessing => "📊 PRÉ-PROCESSAMENTO",
            ModelPhase::Training { epoch, batch } => {
                println!("🎓 TREINAMENTO - Época: {}, Batch: {}", epoch, batch);
                "🎓 TREINAMENTO"
            },
            ModelPhase::Inference { step } => {
                println!("🔮 INFERÊNCIA - Passo: {}", step);
                "🔮 INFERÊNCIA"
            },
            ModelPhase::Evaluation => "📈 AVALIAÇÃO",
        };
        
        if self.verbosity >= 1 {
            println!("\n🚀 ============================================");
            println!("📍 MUDANÇA DE FASE: {}", phase_name);
            println!("============================================");
        }
        
        self.current_phase = phase;
        self.log_operation(
            OperationType::Initialization("phase_change".to_string()),
            format!("Iniciando fase: {}", phase_name),
            LogData::Text(format!("Transição para: {:?}", self.current_phase)),
            2,
        );
    }
    
    /// ⏱️ Iniciar cronômetro para uma operação
    pub fn start_timer(&mut self, operation: &str) {
        self.timestamps.insert(operation.to_string(), Instant::now());
        
        if self.verbosity >= 2 {
            println!("⏱️  Iniciando timer: {}", operation);
        }
    }
    
    /// ⏹️ Parar cronômetro e retornar duração
    pub fn end_timer(&mut self, operation: &str) -> f64 {
        if let Some(start_time) = self.timestamps.remove(operation) {
            let duration = start_time.elapsed().as_secs_f64() * 1000.0;
            self.stats.total_time_ms += duration;
            
            if self.verbosity >= 2 {
                println!("⏹️  Timer finalizado: {} ({:.2}ms)", operation, duration);
            }
            
            duration
        } else {
            0.0
        }
    }
    
    /// 🧮 Log de operação matemática
    pub fn log_math_operation(
        &mut self,
        operation_name: &str,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
        explanation: &str,
    ) {
        let flops = self.estimate_flops(operation_name, &input_shapes, &output_shape);
        self.stats.total_flops += flops;
        
        *self.operation_counts.entry(operation_name.to_string()).or_insert(0) += 1;
        
        if self.verbosity >= 2 {
            println!("🧮 Operação: {} → {}", operation_name, explanation);
            println!("   📊 Input shapes: {:?}", input_shapes);
            println!("   📊 Output shape: {:?}", output_shape);
            println!("   ⚡ FLOPs: {}", self.format_flops(flops));
        }
        
        self.log_operation(
            OperationType::Mathematical(operation_name.to_string()),
            explanation.to_string(),
            LogData::MathResult {
                input_shapes,
                output_shape,
                operation: operation_name.to_string(),
                flops,
            },
            1,
        );
    }
    
    /// 👁️ Log de operação de atenção
    pub fn log_attention_operation(
        &mut self,
        layer: usize,
        head: usize,
        seq_len: usize,
        attention_scores: Option<&Matrix>,
    ) {
        if self.verbosity >= 1 {
            println!("👁️  Atenção - Camada: {}, Cabeça: {}, Seq: {}", layer, head, seq_len);
        }
        
        if let Some(scores) = attention_scores {
            if self.show_attention && self.verbosity >= 2 {
                self.visualize_attention_pattern(scores, seq_len);
            }
        }
        
        self.log_operation(
            OperationType::Attention(format!("layer_{}_head_{}", layer, head)),
            format!("Atenção na camada {} cabeça {} (seq_len={})", layer, head, seq_len),
            LogData::Text(format!("Processando atenção {}x{}", seq_len, seq_len)),
            2,
        );
    }
    
    /// 🎓 Log de época de treinamento
    pub fn log_training_epoch(
        &mut self,
        epoch: u32,
        avg_loss: f32,
        min_loss: f32,
        max_loss: f32,
        gradient_norm: f32,
        learning_rate: f32,
        duration_ms: u64,
    ) {
        let metrics = TrainingEpochMetrics {
            epoch,
            avg_loss,
            min_loss,
            max_loss,
            duration_ms,
            accuracy: None,
            gradient_norm,
            learning_rate,
            memory_usage_mb: self.stats.peak_memory_mb,
        };
        
        if self.verbosity >= 1 {
            println!("\n🎓 ÉPOCA {} CONCLUÍDA:", epoch);
            println!("   📉 Loss: {:.4} (min: {:.4}, max: {:.4})", avg_loss, min_loss, max_loss);
            println!("   📐 Gradient Norm: {:.4}", gradient_norm);
            println!("   📚 Learning Rate: {:.6}", learning_rate);
            println!("   ⏱️ Duração: {}ms", duration_ms);
            println!("   💾 Memória: {:.1}MB", metrics.memory_usage_mb);
        }
        
        self.analyze_training_progress(&metrics);
        self.training_metrics.push(metrics);
    }
    
    /// 📊 Analisar progresso do treinamento
    fn analyze_training_progress(&self, current: &TrainingEpochMetrics) {
        if let Some(previous) = self.training_metrics.last() {
            let loss_change = current.avg_loss - previous.avg_loss;
            let loss_change_percent = (loss_change / previous.avg_loss) * 100.0;
            
            if self.verbosity >= 1 {
                if loss_change < 0.0 {
                    println!("   ✅ Melhoria: {:.2}% ({:+.4})", loss_change_percent.abs(), loss_change);
                } else {
                    println!("   ⚠️  Piora: {:.2}% ({:+.4})", loss_change_percent, loss_change);
                }
                
                // Detectar possíveis problemas
                if current.gradient_norm < 1e-6 {
                    println!("   🚨 AVISO: Gradientes muito pequenos (vanishing gradients?)");
                } else if current.gradient_norm > 10.0 {
                    println!("   🚨 AVISO: Gradientes muito grandes (exploding gradients?)");
                }
            }
        }
    }
    
    /// 🧮 Estimar FLOPs de uma operação
    fn estimate_flops(&self, operation: &str, inputs: &[Vec<usize>], output: &[usize]) -> u64 {
        match operation {
            "matmul" | "linear" => {
                if inputs.len() >= 2 {
                    let m = inputs[0][0] as u64;
                    let k = inputs[0][1] as u64;
                    let n = inputs[1][1] as u64;
                    2 * m * k * n // 2 FLOPs por multiplicação-adição
                } else {
                    0
                }
            },
            "attention" => {
                if !output.is_empty() {
                    let seq_len = output[0] as u64;
                    4 * seq_len * seq_len * seq_len // Q*K^T + Softmax + *V
                } else {
                    0
                }
            },
            "softmax" | "gelu" | "layernorm" => {
                output.iter().product::<usize>() as u64 * 5 // ~5 ops por elemento
            },
            _ => output.iter().product::<usize>() as u64,
        }
    }
    
    /// 📊 Formatar FLOPs para exibição
    fn format_flops(&self, flops: u64) -> String {
        if flops >= 1_000_000_000 {
            format!("{:.2}G FLOPs", flops as f64 / 1_000_000_000.0)
        } else if flops >= 1_000_000 {
            format!("{:.2}M FLOPs", flops as f64 / 1_000_000.0)
        } else if flops >= 1_000 {
            format!("{:.2}K FLOPs", flops as f64 / 1_000.0)
        } else {
            format!("{} FLOPs", flops)
        }
    }
    
    /// 👁️ Visualizar padrão de atenção
    fn visualize_attention_pattern(&self, attention_scores: &Matrix, seq_len: usize) {
        if self.verbosity >= 2 {
            println!("\n👁️  MAPA DE ATENÇÃO ({}x{}):", seq_len, seq_len);
            
            // Cabeçalho
            print!("     ");
            for j in 0..seq_len.min(8) {
                print!("{:6}", j);
            }
            println!();
            
            // Linhas da matriz
            for i in 0..seq_len.min(8) {
                print!("{:3}: ", i);
                for j in 0..seq_len.min(8) {
                    let idx = i * attention_scores.cols + j;
                    if idx < attention_scores.data.len() {
                        let value = attention_scores.data[idx];
                        let symbol = if value > 0.5 {
                            "██"
                        } else if value > 0.3 {
                            "▓▓"
                        } else if value > 0.1 {
                            "░░"
                        } else {
                            "  "
                        };
                        print!("{:>4}", symbol);
                    } else {
                        print!("    ");
                    }
                }
                println!();
            }
            
            if seq_len > 8 {
                println!("   ... (mostrando apenas 8x8 primeiros elementos)");
            }
            println!();
        }
    }
    
    /// 📝 Log genérico de operação
    fn log_operation(
        &mut self,
        operation_type: OperationType,
        description: String,
        data: LogData,
        importance: u8,
    ) {
        let entry = LogEntry {
            timestamp: Instant::now(),
            operation_type,
            description,
            data,
            importance,
        };
        
        self.operations.push(entry);
        self.stats.total_operations += 1;
    }
    
    /// 📊 Log de tokenização
    pub fn log_tokenization(&self, text: &str, tokens: &[usize]) {
        if self.verbosity >= 1 {
            println!("\n🔤 TOKENIZAÇÃO:");
            println!("   📝 Texto original: \"{}\"", text);
            println!("   🔢 Tokens: {:?}", tokens);
            println!("   📊 Comprimento: {} → {} tokens", text.len(), tokens.len());
            
            if self.verbosity >= 2 {
                let compression_ratio = text.len() as f32 / tokens.len() as f32;
                println!("   📈 Taxa de compressão: {:.2} chars/token", compression_ratio);
            }
        }
    }
    
    /// 🧮 Log de análise de embeddings
    pub fn log_embedding_analysis(&self, tokens: &[usize], token_embeddings: &Matrix) {
        if self.verbosity >= 1 {
            println!("\n🧮 ANÁLISE DE EMBEDDINGS:");
            println!("   🔢 Tokens: {} elementos", tokens.len());
            println!("   📊 Embedding shape: {:?}", token_embeddings.shape());
            println!("   💾 Memória: {:.2}MB", token_embeddings.memory_mb());
            
            if self.verbosity >= 2 {
                // Calcular estatísticas básicas
                let mean = token_embeddings.data.iter().sum::<f32>() / token_embeddings.data.len() as f32;
                let variance = token_embeddings.data.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f32>() / token_embeddings.data.len() as f32;
                let std_dev = variance.sqrt();
                
                println!("   📈 Estatísticas:");
                println!("      • Média: {:.4}", mean);
                println!("      • Desvio padrão: {:.4}", std_dev);
                println!("      • Variância: {:.4}", variance);
            }
        }
    }
    
    /// 🔄 Log de processamento do transformer
    pub fn log_transformer_processing(&self, layer: usize, input_shape: &[usize], output_shape: &[usize]) {
        if self.verbosity >= 1 {
            println!("\n🔄 TRANSFORMER LAYER {}:", layer);
            println!("   📊 Input shape: {:?}", input_shape);
            println!("   📊 Output shape: {:?}", output_shape);
            
            if self.verbosity >= 2 {
                println!("   🧮 Componentes processados:");
                println!("      1. 👁️  Multi-Head Attention");
                println!("      2. ➕ Residual Connection + LayerNorm");
                println!("      3. 🧠 Feed-Forward Network");
                println!("      4. ➕ Residual Connection + LayerNorm");
            }
        }
    }
    
    /// 🔮 Log de predição
    pub fn log_prediction(&self, logits: &Matrix, predicted_token: usize, top_k: usize) {
        if self.verbosity >= 1 {
            println!("\n🔮 PREDIÇÃO:");
            println!("   📊 Logits shape: {:?}", logits.shape());
            println!("   🎯 Token predito: {}", predicted_token);
            
            if self.verbosity >= 2 && top_k > 0 {
                println!("   📈 Top-{} candidatos:", top_k);
                
                // Simular top-k (em implementação real, seria sorted)
                let mut top_indices: Vec<usize> = (0..logits.data.len().min(top_k)).collect();
                top_indices.sort_by(|&a, &b| {
                    logits.data[b].partial_cmp(&logits.data[a]).unwrap_or(std::cmp::Ordering::Equal)
                });
                
                for (rank, &idx) in top_indices.iter().enumerate() {
                    let prob = logits.data[idx];
                    println!("      {}. Token {}: {:.4}", rank + 1, idx, prob);
                }
            }
        }
    }
    
    /// 📋 Log de resumo do processo
    pub fn log_process_summary(&self, input_text: &str, output_text: &str, total_tokens: usize, processing_time: f32) {
        if self.verbosity >= 1 {
            println!("\n📋 RESUMO DO PROCESSO:");
            println!("   📝 Input: \"{}\"", input_text);
            println!("   📝 Output: \"{}\"", output_text);
            println!("   🔢 Total de tokens processados: {}", total_tokens);
            println!("   ⏱️ Tempo total: {:.2}ms", processing_time);
            println!("   ⚡ Tokens/segundo: {:.1}", total_tokens as f32 / (processing_time / 1000.0));
            
            if self.verbosity >= 2 {
                println!("   📊 Estatísticas detalhadas:");
                println!("      • Total de operações: {}", self.stats.total_operations);
                println!("      • Total de FLOPs: {}", self.format_flops(self.stats.total_flops));
                println!("      • Pico de memória: {:.1}MB", self.stats.peak_memory_mb);
            }
        }
    }
    
    /// 📊 Gerar relatório final
    pub fn generate_final_report(&self) {
        println!("\n📊 ============================================");
        println!("📈 RELATÓRIO FINAL DE PERFORMANCE");
        println!("============================================\n");
        
        // Estatísticas gerais
        println!("🔢 ESTATÍSTICAS GERAIS:");
        println!("   • Total de operações: {}", self.stats.total_operations);
        println!("   • Tempo total: {:.2}ms", self.stats.total_time_ms);
        println!("   • Total de FLOPs: {}", self.format_flops(self.stats.total_flops));
        println!("   • Pico de memória: {:.1}MB", self.stats.peak_memory_mb);
        
        // Distribuição de operações
        if !self.operation_counts.is_empty() {
            println!("\n📊 DISTRIBUIÇÃO DE OPERAÇÕES:");
            let mut ops: Vec<_> = self.operation_counts.iter().collect();
            ops.sort_by(|a, b| b.1.cmp(a.1));
            
            for (op, count) in ops.iter().take(10) {
                let percentage = **count as f64 / self.stats.total_operations as f64 * 100.0;
                println!("   • {}: {} ({:.1}%)", op, count, percentage);
            }
        }
        
        // Métricas de treinamento
        if !self.training_metrics.is_empty() {
            println!("\n🎓 PROGRESSO DO TREINAMENTO:");
            let first = &self.training_metrics[0];
            let last = &self.training_metrics[self.training_metrics.len() - 1];
            
            let loss_improvement = (first.avg_loss - last.avg_loss) / first.avg_loss * 100.0;
            println!("   • Épocas: {}", self.training_metrics.len());
            println!("   • Loss inicial: {:.4}", first.avg_loss);
            println!("   • Loss final: {:.4}", last.avg_loss);
            println!("   • Melhoria: {:.1}%", loss_improvement);
            
            let total_training_time: u64 = self.training_metrics.iter().map(|m| m.duration_ms).sum();
            println!("   • Tempo total de treinamento: {:.2}s", total_training_time as f64 / 1000.0);
        }
        
        // Recomendações
        println!("\n💡 RECOMENDAÇÕES:");
        if self.stats.peak_memory_mb > 1000.0 {
            println!("   • ⚠️  Alto uso de memória - considere batch size menor");
        }
        if self.stats.total_time_ms > 10000.0 {
            println!("   • ⚠️  Processamento lento - considere otimizações");
        }
        if !self.training_metrics.is_empty() {
            let last = &self.training_metrics[self.training_metrics.len() - 1];
            if last.gradient_norm < 1e-6 {
                println!("   • ⚠️  Gradientes muito pequenos - ajuste learning rate");
            }
        }
        println!("   • ✅ Use kernel fusion para melhor performance");
        println!("   • ✅ Monitore métricas regularmente");
        println!("   • ✅ Implemente early stopping se necessário");
    }
}

impl Default for EducationalLogger {
    fn default() -> Self {
        Self::new(1)
    }
}

// ============================================================================
// 🎯 DEMONSTRAÇÕES EDUCACIONAIS
// ============================================================================

fn demonstrate_initialization_logging() {
    println!("\n🎯 ============================================");
    println!("🏗️ DEMONSTRAÇÃO: LOGGING DE INICIALIZAÇÃO");
    println!("============================================\n");
    
    let mut logger = EducationalLogger::new(2)
        .with_tensor_info(true)
        .with_attention_maps(true);
    
    logger.start_phase(ModelPhase::Initialization);
    
    // Simular inicialização de componentes
    println!("🔧 Inicializando componentes do modelo...");
    
    logger.start_timer("model_init");
    
    // Embeddings
    let vocab_size = 50000;
    let d_model = 768;
    let token_embeddings = Matrix::random(vocab_size, d_model);
    
    logger.log_operation(
        OperationType::Initialization("embeddings".to_string()),
        "Inicializando token embeddings".to_string(),
        LogData::TensorInfo {
            shape: token_embeddings.shape(),
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            memory_mb: token_embeddings.memory_mb(),
        },
        3,
    );
    
    // Position embeddings
    let max_seq_len = 2048;
    let pos_embeddings = Matrix::random(max_seq_len, d_model);
    
    logger.log_operation(
        OperationType::Initialization("position_embeddings".to_string()),
        "Inicializando position embeddings".to_string(),
        LogData::TensorInfo {
            shape: pos_embeddings.shape(),
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            memory_mb: pos_embeddings.memory_mb(),
        },
        3,
    );
    
    // Transformer layers
    let num_layers = 12;
    for layer in 0..num_layers {
        let layer_params = d_model * d_model * 4; // Aproximação
        let layer_memory = (layer_params * 4) as f32 / 1024.0 / 1024.0;
        
        logger.log_operation(
            OperationType::Initialization(format!("layer_{}", layer)),
            format!("Inicializando Transformer layer {}", layer),
            LogData::TensorInfo {
                shape: vec![d_model, d_model],
                dtype: "f32".to_string(),
                device: "cpu".to_string(),
                memory_mb: layer_memory,
            },
            2,
        );
        
        // Simular tempo de inicialização
        std::thread::sleep(Duration::from_millis(10));
    }
    
    let init_time = logger.end_timer("model_init");
    
    // Calcular estatísticas do modelo
    let total_params = vocab_size * d_model + max_seq_len * d_model + num_layers * d_model * d_model * 4;
    let total_memory = (total_params * 4) as f32 / 1024.0 / 1024.0;
    
    logger.stats.peak_memory_mb = total_memory;
    
    println!("\n📊 ESTATÍSTICAS DE INICIALIZAÇÃO:");
    println!("   • Total de parâmetros: {:.1}M", total_params as f32 / 1_000_000.0);
    println!("   • Memória total: {:.1}MB", total_memory);
    println!("   • Tempo de inicialização: {:.2}ms", init_time);
    println!("   • Camadas Transformer: {}", num_layers);
    println!("   • Dimensão do modelo: {}", d_model);
    println!("   • Tamanho do vocabulário: {}", vocab_size);
}

fn demonstrate_training_logging() {
    println!("\n🎯 ============================================");
    println!("🎓 DEMONSTRAÇÃO: LOGGING DE TREINAMENTO");
    println!("============================================\n");
    
    let mut logger = EducationalLogger::new(2);
    
    // Simular várias épocas de treinamento
    let num_epochs = 5;
    let mut current_loss = 4.5;
    
    for epoch in 1..=num_epochs {
        logger.start_phase(ModelPhase::Training { epoch, batch: 0 });
        logger.start_timer(&format!("epoch_{}", epoch));
        
        // Simular batches
        let num_batches = 10;
        let mut epoch_losses = Vec::new();
        
        for batch in 1..=num_batches {
            logger.start_phase(ModelPhase::Training { epoch, batch });
            
            // Simular forward pass
            logger.log_math_operation(
                "forward_pass",
                vec![vec![32, 512], vec![512, 768]], // batch_size, seq_len, d_model
                vec![32, 512, 768],
                "Forward pass através do modelo",
            );
            
            // Simular cálculo de loss
            let batch_loss = current_loss + (rand::random::<f32>() - 0.5) * 0.2;
            epoch_losses.push(batch_loss);
            
            logger.log_operation(
                OperationType::Metric("loss".to_string()),
                format!("Loss do batch {}", batch),
                LogData::Metric {
                    name: "cross_entropy_loss".to_string(),
                    value: batch_loss,
                    unit: "nats".to_string(),
                },
                2,
            );
            
            // Simular backward pass
            logger.log_math_operation(
                "backward_pass",
                vec![vec![32, 512, 768]],
                vec![32, 512, 768],
                "Backward pass - cálculo de gradientes",
            );
            
            // Simular tempo de processamento
            std::thread::sleep(Duration::from_millis(20));
        }
        
        let epoch_time = logger.end_timer(&format!("epoch_{}", epoch)) as u64;
        
        // Calcular estatísticas da época
        let avg_loss = epoch_losses.iter().sum::<f32>() / epoch_losses.len() as f32;
        let min_loss = epoch_losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_loss = epoch_losses.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Simular gradiente norm e learning rate
        let gradient_norm = 0.5 + rand::random::<f32>() * 0.3;
        let learning_rate = 0.001 * (0.95_f32).powi(epoch as i32 - 1);
        
        logger.log_training_epoch(
            epoch,
            avg_loss,
            min_loss,
            max_loss,
            gradient_norm,
            learning_rate,
            epoch_time,
        );
        
        // Reduzir loss gradualmente (simulando aprendizado)
        current_loss *= 0.85;
    }
    
    println!("\n🎓 Treinamento concluído! Veja o progresso acima.");
}

fn demonstrate_inference_logging() {
    println!("\n🎯 ============================================");
    println!("🔮 DEMONSTRAÇÃO: LOGGING DE INFERÊNCIA");
    println!("============================================\n");
    
    let mut logger = EducationalLogger::new(2)
        .with_attention_maps(true);
    
    let input_text = "O futuro da inteligência artificial";
    let tokens = vec![15, 2847, 18, 9284, 7834]; // Simulado
    
    logger.start_phase(ModelPhase::Preprocessing);
    
    // Log de tokenização
    logger.log_tokenization(input_text, &tokens);
    
    // Log de embeddings
    let d_model = 768;
    let token_embeddings = Matrix::random(tokens.len(), d_model);
    logger.log_embedding_analysis(&tokens, &token_embeddings);
    
    logger.start_phase(ModelPhase::Inference { step: 0 });
    logger.start_timer("inference");
    
    // Simular processamento através das camadas
    let num_layers = 12;
    for layer in 0..num_layers {
        logger.log_transformer_processing(
            layer,
            &[tokens.len(), d_model],
            &[tokens.len(), d_model],
        );
        
        // Simular atenção
        let attention_scores = Matrix::random(tokens.len(), tokens.len());
        logger.log_attention_operation(
            layer,
            0, // head 0
            tokens.len(),
            Some(&attention_scores),
        );
        
        std::thread::sleep(Duration::from_millis(5));
    }
    
    // Simular predição final
    let vocab_size = 50000;
    let logits = Matrix::random(1, vocab_size);
    let predicted_token = 1234; // Simulado
    
    logger.log_prediction(&logits, predicted_token, 5);
    
    let inference_time = logger.end_timer("inference");
    
    // Log de resumo
    let output_text = "O futuro da inteligência artificial será";
    logger.log_process_summary(
        input_text,
        output_text,
        tokens.len() + 1,
        inference_time as f32,
    );
}

fn demonstrate_performance_analysis() {
    println!("\n🎯 ============================================");
    println!("📊 DEMONSTRAÇÃO: ANÁLISE DE PERFORMANCE");
    println!("============================================\n");
    
    let mut logger = EducationalLogger::new(3);
    
    // Simular várias operações com diferentes complexidades
    let operations = vec![
        ("embedding_lookup", vec![vec![32, 512]], vec![32, 512, 768]),
        ("attention", vec![vec![32, 512, 768]], vec![32, 512, 768]),
        ("feedforward", vec![vec![32, 512, 768]], vec![32, 512, 768]),
        ("layernorm", vec![vec![32, 512, 768]], vec![32, 512, 768]),
        ("linear", vec![vec![32, 512, 768], vec![768, 3072]], vec![32, 512, 3072]),
    ];
    
    for (op_name, inputs, output) in operations {
        logger.start_timer(op_name);
        
        // Simular processamento
        std::thread::sleep(Duration::from_millis(rand::random_u64() % 50 + 10));
        
        logger.log_math_operation(
            op_name,
            inputs,
            output,
            &format!("Executando operação {}", op_name),
        );
        
        logger.end_timer(op_name);
    }
    
    // Simular uso de memória
    logger.stats.peak_memory_mb = 1024.0 + rand::random::<f32>() * 512.0;
    
    println!("\n📈 Análise de performance concluída!");
    println!("Veja as estatísticas detalhadas no relatório final.");
}

fn demonstrate_debugging_features() {
    println!("\n🎯 ============================================");
    println!("🔍 DEMONSTRAÇÃO: RECURSOS DE DEBUGGING");
    println!("============================================\n");
    
    let mut logger = EducationalLogger::new(3);
    
    // Simular problemas comuns e como o logger os detecta
    
    // 1. Gradientes explodindo
    println!("🚨 Simulando gradientes explodindo...");
    logger.log_training_epoch(
        1,
        2.5,
        2.3,
        2.8,
        15.7, // Gradient norm muito alto
        0.001,
        5000,
    );
    
    // 2. Gradientes desaparecendo
    println!("\n🚨 Simulando gradientes desaparecendo...");
    logger.log_training_epoch(
        2,
        2.4,
        2.35,
        2.45,
        1e-8, // Gradient norm muito baixo
        0.001,
        5000,
    );
    
    // 3. Alto uso de memória
    logger.stats.peak_memory_mb = 8192.0; // 8GB
    
    // 4. Operações lentas
    logger.start_timer("slow_operation");
    std::thread::sleep(Duration::from_millis(100));
    logger.end_timer("slow_operation");
    
    println!("\n🔍 Recursos de debugging demonstrados!");
    println!("O logger detectou automaticamente vários problemas potenciais.");
}

// ============================================================================
// 🎓 EXERCÍCIOS PRÁTICOS
// ============================================================================

fn practical_exercises() {
    println!("\n🎯 ============================================");
    println!("🎓 EXERCÍCIOS PRÁTICOS");
    println!("============================================\n");
    
    println!("📚 EXERCÍCIO 1: Logger Customizado");
    println!("   Tarefa: Criar um logger específico para debugging de atenção");
    println!("   Dica: Foque em visualizar padrões de atenção anômalos");
    println!("   Meta: Detectar heads que não aprendem\n");
    
    println!("📚 EXERCÍCIO 2: Métricas Avançadas");
    println!("   Tarefa: Implementar tracking de diversidade de gradientes");
    println!("   Dica: Meça a variância dos gradientes entre camadas");
    println!("   Meta: Detectar layers que não contribuem\n");
    
    println!("📚 EXERCÍCIO 3: Profiling Automático");
    println!("   Tarefa: Criar sistema de profiling que identifica gargalos");
    println!("   Dica: Compare tempos esperados vs reais");
    println!("   Meta: Sugerir otimizações automaticamente\n");
    
    println!("📚 EXERCÍCIO 4: Visualização Interativa");
    println!("   Tarefa: Criar dashboard web para métricas em tempo real");
    println!("   Dica: Use WebSockets para updates em tempo real");
    println!("   Meta: Monitoramento visual durante treinamento\n");
    
    println!("📚 EXERCÍCIO 5: Análise de Convergência");
    println!("   Tarefa: Implementar detecção automática de convergência");
    println!("   Dica: Analise tendências de loss e gradientes");
    println!("   Meta: Early stopping inteligente\n");
    
    println!("🏆 DESAFIO AVANÇADO: Logger Distribuído");
    println!("   Implemente logging para treinamento distribuído");
    println!("   Meta: Sincronizar métricas entre múltiplos workers");
}

// ============================================================================
// 🚀 FUNÇÃO PRINCIPAL
// ============================================================================

fn main() {
    println!("📚 ============================================");
    println!("🚀 MINI GPT RUST - EDUCATIONAL LOGGER DEMO");
    println!("============================================");
    println!("Sistema completo de logging educacional que torna");
    println!("visível todo o ciclo de vida de um LLM! 🦀📊");
    
    // 1. Demonstração de inicialização
    demonstrate_initialization_logging();
    
    // 2. Demonstração de treinamento
    demonstrate_training_logging();
    
    // 3. Demonstração de inferência
    demonstrate_inference_logging();
    
    // 4. Análise de performance
    demonstrate_performance_analysis();
    
    // 5. Recursos de debugging
    demonstrate_debugging_features();
    
    // 6. Relatório final
    let logger = EducationalLogger::new(1);
    logger.generate_final_report();
    
    // 7. Exercícios práticos
    practical_exercises();
    
    println!("\n🎉 ============================================");
    println!("✨ DEMONSTRAÇÃO CONCLUÍDA!");
    println!("============================================");
    println!("Agora você entende como o sistema de logging");
    println!("educacional torna visível cada aspecto do LLM! 📚");
    println!("\n💡 Próximos passos:");
    println!("   • Integre o logger em seus próprios modelos");
    println!("   • Customize os níveis de verbosidade");
    println!("   • Implemente métricas específicas do seu domínio");
    println!("   • Crie visualizações personalizadas");
    println!("   • Contribua com melhorias para o projeto!");
}

// Módulo para simular números aleatórios simples
mod rand {
    use std::cell::Cell;
    
    thread_local! {
        static SEED: Cell<u64> = Cell::new(1);
    }
    
    pub fn random<T>() -> T
    where
        T: From<f32>,
    {
        SEED.with(|seed| {
            let current = seed.get();
            let next = current.wrapping_mul(1103515245).wrapping_add(12345);
            seed.set(next);
            let normalized = (next as f32) / (u64::MAX as f32);
            T::from(normalized)
        })
    }
    
    // Função específica para u64
    pub fn random_u64() -> u64 {
        SEED.with(|seed| {
            let current = seed.get();
            let next = current.wrapping_mul(1103515245).wrapping_add(12345);
            seed.set(next);
            next % 100 // Retorna valor entre 0-99
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_logger_initialization() {
        let logger = EducationalLogger::new(1);
        assert_eq!(logger.verbosity, 1);
        assert_eq!(logger.current_phase, ModelPhase::Initialization);
        assert!(logger.operations.is_empty());
    }
    
    #[test]
    fn test_timer_functionality() {
        let mut logger = EducationalLogger::new(0);
        
        logger.start_timer("test_op");
        std::thread::sleep(Duration::from_millis(10));
        let duration = logger.end_timer("test_op");
        
        assert!(duration >= 10.0); // Pelo menos 10ms
        assert!(duration < 100.0);  // Menos que 100ms (margem de segurança)
    }
    
    #[test]
    fn test_math_operation_logging() {
        let mut logger = EducationalLogger::new(0);
        
        logger.log_math_operation(
            "matmul",
            vec![vec![32, 768], vec![768, 3072]],
            vec![32, 3072],
            "Matrix multiplication test",
        );
        
        assert_eq!(logger.operations.len(), 1);
        assert_eq!(logger.stats.total_operations, 1);
        assert!(logger.stats.total_flops > 0);
    }
    
    #[test]
    fn test_training_metrics() {
        let mut logger = EducationalLogger::new(0);
        
        logger.log_training_epoch(1, 2.5, 2.3, 2.7, 0.5, 0.001, 5000);
        
        assert_eq!(logger.training_metrics.len(), 1);
        assert_eq!(logger.training_metrics[0].epoch, 1);
        assert_eq!(logger.training_metrics[0].avg_loss, 2.5);
    }
    
    #[test]
    fn test_phase_transitions() {
        let mut logger = EducationalLogger::new(0);
        
        logger.start_phase(ModelPhase::Training { epoch: 1, batch: 0 });
        assert!(matches!(logger.current_phase, ModelPhase::Training { epoch: 1, batch: 0 }));
        
        logger.start_phase(ModelPhase::Inference { step: 5 });
        assert!(matches!(logger.current_phase, ModelPhase::Inference { step: 5 }));
    }
    
    #[test]
    fn test_flops_estimation() {
        let logger = EducationalLogger::new(0);
        
        // Test matrix multiplication FLOPs
        let flops = logger.estimate_flops(
            "matmul",
            &[vec![32, 768], vec![768, 3072]],
            &[32, 3072],
        );
        
        // Expected: 2 * 32 * 768 * 3072 = 150,994,944
        let expected = 2 * 32 * 768 * 3072;
        assert_eq!(flops, expected);
    }
}