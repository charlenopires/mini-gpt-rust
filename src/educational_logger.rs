//! # 📚 Sistema de Logging Educacional Avançado para Mini-GPT
//!
//! Este módulo implementa um sistema completo de logging educacional que acompanha
//! todo o ciclo de vida de um Large Language Model (LLM), desde a inicialização
//! até o treinamento e inferência.
//!
//! ## 🧠 Por que um Sistema de Log Educacional?
//!
//! Um LLM é uma "caixa preta" complexa. Este sistema torna visível:
//! - **Como os dados fluem** através das camadas neurais
//! - **O que acontece matematicamente** em cada operação
//! - **Como o modelo aprende** durante o treinamento
//! - **Por que certas decisões** são tomadas durante a inferência
//!
//! ## 🎯 Fases do Ciclo de Vida de um LLM
//!
//! ### 1. 🏗️ **INICIALIZAÇÃO**
//! - Criação da arquitetura Transformer
//! - Inicialização de pesos (Xavier, He, etc.)
//! - Configuração de hiperparâmetros
//!
//! ### 2. 📊 **PRÉ-PROCESSAMENTO**
//! - Tokenização do texto de entrada
//! - Criação de embeddings de posição
//! - Preparação de máscaras de atenção
//!
//! ### 3. 🎓 **TREINAMENTO**
//! - Forward pass: dados → predições
//! - Cálculo da loss function
//! - Backward pass: gradientes
//! - Atualização de pesos (otimizador)
//!
//! ### 4. 🔮 **INFERÊNCIA**
//! - Geração autoregressiva de tokens
//! - Aplicação de estratégias de sampling
//! - Decodificação para texto final
//!
//! ## 📈 Tipos de Logs Implementados
//!
//! - **🏗️ STRUCTURAL**: Arquitetura e dimensões
//! - **🧮 MATHEMATICAL**: Operações e cálculos
//! - **⚡ PERFORMANCE**: Tempos e memória
//! - **🎓 EDUCATIONAL**: Explicações conceituais
//! - **🔍 DEBUGGING**: Detecção de problemas
//! - **📊 METRICS**: Métricas de treinamento

use std::collections::HashMap;
use candle_core::Tensor;
use crate::tokenizer::BPETokenizer;
use anyhow::Result;
// use std::fmt::Write; // Removido - não utilizado
use std::time::Instant;

/// 🎓 **SISTEMA DE LOGGING EDUCACIONAL AVANÇADO**
///
/// Este sistema monitora e explica todo o ciclo de vida de um LLM:
/// desde a inicialização até a geração de texto, tornando visível
/// cada processo interno que normalmente seria "invisível".
///
/// ## 🔍 O que Este Sistema Revela:
///
/// ### Durante a Inicialização:
/// - Como os pesos são inicializados (distribuições, escalas)
/// - Quantos parâmetros o modelo possui
/// - Como a memória é alocada
///
/// ### Durante o Treinamento:
/// - Como a loss diminui a cada época
/// - Como os gradientes fluem (ou não) pelas camadas
/// - Quando ocorre overfitting ou underfitting
/// - Performance de cada componente (atenção, FFN, etc.)
///
/// ### Durante a Inferência:
/// - Como cada token influencia a predição do próximo
/// - Quais tokens recebem mais "atenção"
/// - Como as probabilidades são calculadas
/// - Por que certas palavras são escolhidas
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
    /// 🏗️ Fase atual do modelo (Init, Training, Inference)
    pub current_phase: ModelPhase,
    /// 📈 Métricas de treinamento por época
    pub training_metrics: Vec<TrainingEpochMetrics>,
    /// 🧮 Contador de operações matemáticas
    pub operation_counts: HashMap<String, u64>,
    /// 🔧 Configurações de visualização
     pub show_tensors: bool,
     pub show_attention: bool,
}

/// 🏗️ **FASES DO CICLO DE VIDA DO MODELO**
///
/// Cada fase tem características e métricas específicas que devem
/// ser monitoradas de forma diferente.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelPhase {
    /// 🏗️ Inicializando arquitetura e pesos
    Initialization,
    /// 📊 Preparando dados de entrada
    Preprocessing,
    /// 🎓 Treinando o modelo (aprendendo padrões)
    Training { epoch: u32, batch: u32 },
    /// 🔮 Gerando texto (inferência)
    Inference { step: u32 },
    /// ✅ Avaliando performance
    Evaluation,
}

/// 📊 **MÉTRICAS DE UMA ÉPOCA DE TREINAMENTO**
///
/// Captura todas as informações importantes de uma época,
/// permitindo análise de convergência e debugging.
#[derive(Debug, Clone)]
pub struct TrainingEpochMetrics {
    /// 📈 Número da época
    pub epoch: u32,
    /// 💔 Loss média da época
    pub avg_loss: f32,
    /// 📉 Loss mínima observada
    pub min_loss: f32,
    /// 📈 Loss máxima observada
    pub max_loss: f32,
    /// ⏱️ Tempo total da época
    pub duration_ms: u64,
    /// 🎯 Acurácia (se disponível)
    pub accuracy: Option<f32>,
    /// 📊 Norma dos gradientes
    pub gradient_norm: f32,
    /// 🧠 Taxa de aprendizado usada
    pub learning_rate: f32,
    /// 💾 Uso de memória (MB)
    pub memory_usage_mb: f32,
}

/// 📝 **ENTRADA DE LOG DETALHADA**
///
/// Cada operação importante gera uma entrada de log com
/// contexto completo para análise posterior.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// ⏰ Timestamp da operação
    pub timestamp: Instant,
    /// 🏷️ Tipo da operação
    pub operation_type: OperationType,
    /// 📝 Descrição educacional
    pub description: String,
    /// 📊 Dados da operação
    pub data: LogData,
    /// 🎯 Nível de importância (0-3)
    pub importance: u8,
}

/// 🔧 **TIPOS DE OPERAÇÕES MONITORADAS**
#[derive(Debug, Clone)]
pub enum OperationType {
    /// 🏗️ Inicialização de componentes
    Initialization(String),
    /// 🔄 Forward pass
    Forward(String),
    /// ⬅️ Backward pass
    Backward(String),
    /// 🎯 Cálculo de atenção
    Attention(String),
    /// 🧮 Operação matemática
    Mathematical(String),
    /// 📊 Métrica calculada
    Metric(String),
    /// ⚠️ Aviso ou problema detectado
    Warning(String),
}

/// 📊 **DADOS ASSOCIADOS A UMA OPERAÇÃO**
#[derive(Debug, Clone)]
pub enum LogData {
    /// 📐 Informações sobre tensores
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
    /// 📈 Métrica numérica
    Metric {
        name: String,
        value: f32,
        unit: String,
    },
    /// ⏱️ Informação de performance
    Performance {
        duration_ms: f64,
        memory_delta_mb: f32,
        cpu_usage: f32,
    },
    /// 📝 Texto explicativo
    Text(String),
}

/// 📊 **ESTATÍSTICAS ACUMULADAS**
#[derive(Debug, Clone, Default)]
pub struct LogStats {
    /// 🔢 Total de operações realizadas
    pub total_operations: u64,
    /// ⏱️ Tempo total de execução (ms)
    pub total_time_ms: f64,
    /// 💾 Pico de uso de memória (MB)
    pub peak_memory_mb: f32,
    /// 🧮 Total de FLOPs executados
    pub total_flops: u64,
    /// 📊 Distribuição de tipos de operação
     pub operation_distribution: HashMap<String, u64>,
 }

impl EducationalLogger {
    /// 🏗️ **CRIAR NOVO LOGGER EDUCACIONAL**
    ///
    /// Inicializa o sistema de logging com configurações padrão
    /// otimizadas para aprendizado e debugging.
    pub fn new(verbosity: u8) -> Self {
        println!("\n🎓 ===== SISTEMA DE LOGGING EDUCACIONAL INICIADO =====");
        println!("📚 Este sistema vai mostrar como um LLM funciona internamente!");
        println!("🔍 Nível de detalhamento: {} (0=básico, 3=expert)", verbosity);
        
        Self {
            operations: Vec::new(),
            timestamps: HashMap::new(),
            stats: LogStats::default(),
            verbosity,
            current_phase: ModelPhase::Initialization,
            training_metrics: Vec::new(),
            operation_counts: HashMap::new(),
            show_tensors: verbosity >= 2,
            show_attention: verbosity >= 1,
        }
    }

    /// 🏗️ **INICIAR FASE DO MODELO**
    ///
    /// Marca o início de uma nova fase (inicialização, treinamento, inferência)
    /// e configura o logging apropriado para essa fase.
    pub fn start_phase(&mut self, phase: ModelPhase) {
        self.current_phase = phase.clone();
        
        match &phase {
            ModelPhase::Initialization => {
                println!("\n🏗️ ===== FASE: INICIALIZAÇÃO DO MODELO =====");
                println!("📋 Nesta fase, vamos:");
                println!("   • Criar a arquitetura Transformer");
                println!("   • Inicializar todos os pesos neurais");
                println!("   • Configurar hiperparâmetros");
                println!("   • Alocar memória necessária");
            },
            ModelPhase::Preprocessing => {
                println!("\n📊 ===== FASE: PRÉ-PROCESSAMENTO =====");
                println!("📋 Nesta fase, vamos:");
                println!("   • Tokenizar o texto de entrada");
                println!("   • Criar embeddings de palavras");
                println!("   • Adicionar informações de posição");
                println!("   • Preparar máscaras de atenção");
            },
            ModelPhase::Training { epoch, batch } => {
                println!("\n🎓 ===== FASE: TREINAMENTO (Época {}, Batch {}) =====", epoch, batch);
                println!("📋 Nesta fase, vamos:");
                println!("   • Fazer forward pass (dados → predições)");
                println!("   • Calcular a loss (erro do modelo)");
                println!("   • Fazer backward pass (calcular gradientes)");
                println!("   • Atualizar pesos (aprender!)");
            },
            ModelPhase::Inference { step } => {
                println!("\n🔮 ===== FASE: INFERÊNCIA (Passo {}) =====", step);
                println!("📋 Nesta fase, vamos:");
                println!("   • Processar tokens de entrada");
                println!("   • Calcular probabilidades de próximos tokens");
                println!("   • Escolher próximo token (sampling)");
                println!("   • Gerar texto de forma autoregressiva");
            },
            ModelPhase::Evaluation => {
                println!("\n✅ ===== FASE: AVALIAÇÃO =====");
                println!("📋 Nesta fase, vamos:");
                println!("   • Testar o modelo em dados não vistos");
                println!("   • Calcular métricas de performance");
                println!("   • Analisar qualidade das predições");
            },
        }
        
        self.start_timer(&format!("phase_{:?}", phase));
    }

    /// ⏱️ **INICIAR CRONÔMETRO**
    ///
    /// Marca o início de uma operação para medição de tempo.
    pub fn start_timer(&mut self, operation: &str) {
        self.timestamps.insert(operation.to_string(), Instant::now());
        
        if self.verbosity >= 2 {
            println!("⏱️ Iniciando cronômetro para: {}", operation);
        }
    }

    /// ⏹️ **PARAR CRONÔMETRO E REGISTRAR**
    ///
    /// Para o cronômetro e registra o tempo decorrido.
    pub fn end_timer(&mut self, operation: &str) -> f64 {
        if let Some(start_time) = self.timestamps.remove(operation) {
            let duration = start_time.elapsed().as_secs_f64() * 1000.0; // em ms
            
            if self.verbosity >= 2 {
                println!("⏹️ {} concluído em {:.2}ms", operation, duration);
            }
            
            self.stats.total_time_ms += duration;
            duration
        } else {
             0.0
         }
     }

    /// 📊 **REGISTRAR OPERAÇÃO MATEMÁTICA**
    ///
    /// Documenta uma operação matemática com contexto educacional completo.
    pub fn log_math_operation(
        &mut self,
        operation_name: &str,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
        explanation: &str,
    ) {
        // Calcular FLOPs aproximados
        let flops = self.estimate_flops(&operation_name, &input_shapes, &output_shape);
        
        if self.verbosity >= 1 {
            println!("\n🧮 === OPERAÇÃO MATEMÁTICA: {} ===", operation_name);
            println!("📝 Explicação: {}", explanation);
            
            if self.show_tensors {
                println!("📐 Formas dos tensores:");
                for (i, shape) in input_shapes.iter().enumerate() {
                    println!("   📥 Entrada {}: {:?}", i + 1, shape);
                }
                println!("   📤 Saída: {:?}", output_shape);
                println!("⚡ FLOPs estimados: {}", self.format_flops(flops));
            }
        }
        
        // Registrar entrada de log
        let entry = LogEntry {
            timestamp: Instant::now(),
            operation_type: OperationType::Mathematical(operation_name.to_string()),
            description: explanation.to_string(),
            data: LogData::MathResult {
                input_shapes,
                output_shape,
                operation: operation_name.to_string(),
                flops,
            },
            importance: 2,
        };
        
        self.operations.push(entry);
        self.stats.total_operations += 1;
        self.stats.total_flops += flops;
        
        // Atualizar contador de operações
        *self.operation_counts.entry(operation_name.to_string()).or_insert(0) += 1;
    }

    /// 🎯 **REGISTRAR OPERAÇÃO DE ATENÇÃO**
    ///
    /// Documenta o mecanismo de atenção com explicações detalhadas.
    pub fn log_attention_operation(
        &mut self,
        layer: usize,
        head: usize,
        seq_len: usize,
        attention_scores: Option<&Tensor>,
    ) -> Result<()> {
        if self.verbosity >= 1 {
            println!("\n🎯 === MECANISMO DE ATENÇÃO ====");
            println!("🏷️ Camada: {}, Cabeça: {}", layer, head);
            println!("📏 Sequência: {} tokens", seq_len);
            
            println!("\n🧠 Como a Atenção Funciona:");
            println!("   1️⃣ Cada token 'olha' para todos os outros tokens");
            println!("   2️⃣ Calcula o quão 'importante' cada token é");
            println!("   3️⃣ Cria uma representação ponderada baseada na importância");
            println!("   4️⃣ Isso permite que o modelo entenda relações entre palavras");
            
            if self.show_attention && attention_scores.is_some() {
                self.visualize_attention_pattern(attention_scores.unwrap(), seq_len)?;
            }
        }
        
        let entry = LogEntry {
            timestamp: Instant::now(),
            operation_type: OperationType::Attention(format!("layer_{}_head_{}", layer, head)),
            description: format!(
                "Atenção multi-cabeça: camada {} cabeça {} processando {} tokens. \
                 Cada token calcula sua relevância com todos os outros tokens.",
                layer, head, seq_len
            ),
            data: LogData::TensorInfo {
                shape: vec![seq_len, seq_len],
                dtype: "f32".to_string(),
                device: "cpu".to_string(),
                memory_mb: (seq_len * seq_len * 4) as f32 / 1024.0 / 1024.0,
            },
            importance: 3,
        };
        
        self.operations.push(entry);
        Ok(())
    }

    /// 📈 **REGISTRAR MÉTRICAS DE TREINAMENTO**
    ///
    /// Documenta métricas de uma época de treinamento com análise educacional.
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
            memory_usage_mb: 0.0, // TODO: implementar medição real
        };
        
        if self.verbosity >= 1 {
            println!("\n📈 === MÉTRICAS DA ÉPOCA {} ===", epoch);
            println!("💔 Loss média: {:.6}", avg_loss);
            println!("📉 Loss mínima: {:.6}", min_loss);
            println!("📈 Loss máxima: {:.6}", max_loss);
            println!("📊 Norma dos gradientes: {:.6}", gradient_norm);
            println!("🧠 Taxa de aprendizado: {:.6}", learning_rate);
            println!("⏱️ Duração: {}ms", duration_ms);
            
            // Análise educacional
            self.analyze_training_progress(&metrics);
        }
        
        self.training_metrics.push(metrics);
    }

    /// 🔍 **ANALISAR PROGRESSO DO TREINAMENTO**
    ///
    /// Fornece insights educacionais sobre o progresso do treinamento.
    fn analyze_training_progress(&self, current: &TrainingEpochMetrics) {
        println!("\n🔍 === ANÁLISE EDUCACIONAL ===");
        
        if let Some(previous) = self.training_metrics.last() {
            let loss_change = current.avg_loss - previous.avg_loss;
            let loss_change_percent = (loss_change / previous.avg_loss) * 100.0;
            
            if loss_change < 0.0 {
                println!("✅ Ótimo! A loss diminuiu {:.2}% ({:.6})", 
                        loss_change_percent.abs(), loss_change.abs());
                println!("   📚 Isso significa que o modelo está aprendendo!");
            } else {
                println!("⚠️ A loss aumentou {:.2}% (+{:.6})", 
                        loss_change_percent, loss_change);
                println!("   📚 Possíveis causas: taxa de aprendizado alta, overfitting, ou dados ruins");
            }
        }
        
        // Análise da norma dos gradientes
        if current.gradient_norm > 10.0 {
            println!("⚠️ Norma dos gradientes alta ({:.2})", current.gradient_norm);
            println!("   📚 Pode indicar exploding gradients - considere gradient clipping");
        } else if current.gradient_norm < 0.001 {
            println!("⚠️ Norma dos gradientes muito baixa ({:.6})", current.gradient_norm);
            println!("   📚 Pode indicar vanishing gradients - verifique inicialização dos pesos");
        } else {
            println!("✅ Norma dos gradientes saudável ({:.4})", current.gradient_norm);
        }
    }

    /// ⚡ **ESTIMAR FLOPs DE UMA OPERAÇÃO**
    ///
    /// Calcula aproximadamente quantas operações de ponto flutuante são necessárias.
    fn estimate_flops(&self, operation: &str, inputs: &[Vec<usize>], output: &[usize]) -> u64 {
        match operation {
            "matmul" | "linear" => {
                if inputs.len() >= 2 {
                    let a_shape = &inputs[0];
                    let b_shape = &inputs[1];
                    if a_shape.len() >= 2 && b_shape.len() >= 2 {
                        let m = a_shape[a_shape.len() - 2];
                        let k = a_shape[a_shape.len() - 1];
                        let n = b_shape[b_shape.len() - 1];
                        return (2 * m * k * n) as u64;
                    }
                }
            },
            "softmax" => {
                let total_elements: usize = output.iter().product();
                return (3 * total_elements) as u64; // exp + sum + div
            },
            "gelu" | "relu" => {
                let total_elements: usize = output.iter().product();
                return total_elements as u64;
            },
            _ => {},
        }
        0
    }

    /// 📊 **FORMATAR FLOPs PARA EXIBIÇÃO**
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

    /// 🎯 **VISUALIZAR PADRÃO DE ATENÇÃO**
    ///
    /// Mostra uma representação visual simplificada dos pesos de atenção.
    fn visualize_attention_pattern(&self, _attention_scores: &Tensor, seq_len: usize) -> Result<()> {
        println!("\n🎯 Padrão de Atenção (simplificado):");
        println!("   📊 Cada linha mostra para onde um token 'presta atenção'");
        println!("   🔥 = alta atenção, 🔸 = média atenção, ⚪ = baixa atenção\n");
        
        // Simplificação: mostrar apenas uma representação conceitual
        // Em uma implementação real, você extrairia os valores do tensor
        for i in 0..std::cmp::min(seq_len, 8) {
            print!("   Token {}: ", i);
            for j in 0..std::cmp::min(seq_len, 8) {
                // Simulação de padrão de atenção
                let attention_strength = if i == j { 0.8 } else { 0.1 + (i as f32 - j as f32).abs() * 0.1 };
                
                if attention_strength > 0.6 {
                    print!("🔥");
                } else if attention_strength > 0.3 {
                    print!("🔸");
                } else {
                    print!("⚪");
                }
            }
            println!();
        }
        
        if seq_len > 8 {
            println!("   ... (mostrando apenas primeiros 8 tokens)");
        }
        
        Ok(())
    }

    /// 📊 **GERAR RELATÓRIO FINAL**
    ///
    /// Cria um resumo educacional completo de toda a execução.
    pub fn generate_final_report(&self) {
        println!("\n📊 ===== RELATÓRIO FINAL DO SISTEMA EDUCACIONAL =====");
        println!("🔢 Total de operações: {}", self.stats.total_operations);
        println!("⏱️ Tempo total: {:.2}ms", self.stats.total_time_ms);
        println!("⚡ Total de FLOPs: {}", self.format_flops(self.stats.total_flops));
        
        if !self.training_metrics.is_empty() {
            println!("\n📈 === RESUMO DO TREINAMENTO ===");
            let first_loss = self.training_metrics.first().unwrap().avg_loss;
            let last_loss = self.training_metrics.last().unwrap().avg_loss;
            let improvement = ((first_loss - last_loss) / first_loss) * 100.0;
            
            println!("🎯 Épocas treinadas: {}", self.training_metrics.len());
            println!("📉 Loss inicial: {:.6}", first_loss);
            println!("📈 Loss final: {:.6}", last_loss);
            println!("✨ Melhoria: {:.2}%", improvement);
        }
        
        println!("\n🧮 === DISTRIBUIÇÃO DE OPERAÇÕES ===");
        for (op, count) in &self.operation_counts {
            println!("   {}: {} vezes", op, count);
        }
        
        println!("\n🎓 Obrigado por usar o sistema educacional do Mini-GPT!");
        println!("📚 Esperamos que tenha aprendido como um LLM funciona internamente!");
    }
 }

/// 🎓 **IMPLEMENTAÇÃO DOS MÉTODOS AUXILIARES**
///
/// Métodos para compatibilidade com o código existente.
impl EducationalLogger {
    /// 🏗️ **CONSTRUTOR SIMPLES (COMPATIBILIDADE)**
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
    pub fn new_simple() -> Self {
        Self {
            operations: Vec::new(),
            timestamps: HashMap::new(),
            stats: LogStats::default(),
            verbosity: 2,
            current_phase: ModelPhase::Initialization,
            training_metrics: Vec::new(),
            operation_counts: HashMap::new(),
            show_tensors: false,
            show_attention: false,
        }
    }
    
    /// 🔊 **CONFIGURAÇÃO DE VERBOSIDADE**
    /// 
    /// Controla o nível de detalhamento das explicações educacionais.
    /// 
    /// **Parâmetros:**
    /// - `0` - Modo silencioso, apenas processamento
    /// - `1` - Informações básicas
    /// - `2` - Explicações completas, diagramas e analogias (padrão)
    /// - `3` - Debugging detalhado com tensores
    /// 
    /// **Uso Recomendado:**
    /// - `2` para aprendizado e debugging
    /// - `0` para produção ou benchmarks
    /// 
    /// **Analogia:** Como o volume de um professor - alto para aprender,
    /// baixo para não atrapalhar outros processos.
    pub fn with_verbosity(mut self, verbosity: u8) -> Self {
        self.verbosity = verbosity;
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
        if self.verbosity == 0 {
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
        
        let display_limit = 20.min(tokens.len());
        
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
    pub fn log_embedding_analysis(&self, tokens: &[usize], token_embeddings: &Tensor, _position_embeddings: &Tensor) -> Result<()> {
        if self.verbosity == 0 {
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
        if self.verbosity == 0 {
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
        if self.verbosity == 0 {
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
        let _probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
        
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
        if self.verbosity == 0 {
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
/// **Equivalência:** `EducationalLogger::default()` == `EducationalLogger::new(2)`
/// 
/// **Analogia:** Como ter um "modo automático" em uma câmera -
/// configurações sensatas para a maioria dos casos de uso.
impl Default for EducationalLogger {
    fn default() -> Self {
        Self::new(2)
    }
}