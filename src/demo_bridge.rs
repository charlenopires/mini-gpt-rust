//! 🌉 Ponte entre Sistema Demo CLI e Interface Web
//! 
//! Este módulo implementa a integração entre as demonstrações educacionais
//! do CLI e a interface web, permitindo execução e visualização em tempo real.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// 📊 Dados de demonstração para visualização
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoData {
    pub module: String,
    pub step: String,
    pub data: serde_json::Value,
    pub timestamp: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// 🎯 Resultado de execução de demo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoResult {
    pub success: bool,
    pub module: String,
    pub execution_time: u64,
    pub output: String,
    pub visualizations: Vec<VisualizationData>,
    pub metrics: PerformanceMetrics,
}

/// 📈 Dados de visualização
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub viz_type: String,
    pub title: String,
    pub data: serde_json::Value,
    pub config: HashMap<String, serde_json::Value>,
}

/// ⚡ Métricas de performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub latency: f64,
    pub memory_usage: f64,
    pub cpu_usage: f64,
    pub execution_time: u64,
    pub throughput: f64,
}

/// 🎛️ Parâmetros de demonstração
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoParameters {
    pub module: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub config: DemoConfig,
}

/// ⚙️ Configuração de demonstração
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoConfig {
    pub educational: bool,
    pub show_tensors: bool,
    pub show_attention: bool,
    pub benchmark: bool,
    pub interactive: bool,
    pub real_time: bool,
}

/// 🌉 Ponte principal entre CLI e Web
pub struct DemoBridge {
    /// 📡 Canal de broadcast para dados de demo
    demo_sender: broadcast::Sender<DemoData>,
    
    /// 📊 Canal de broadcast para resultados
    result_sender: broadcast::Sender<DemoResult>,
    
    /// 🎛️ Parâmetros ativos por módulo
    active_parameters: Arc<RwLock<HashMap<String, DemoParameters>>>,
    
    /// 📈 Histórico de métricas
    metrics_history: Arc<RwLock<Vec<PerformanceMetrics>>>,
    
    /// 🔄 Estado de execução
    execution_state: Arc<RwLock<HashMap<String, bool>>>,
}

impl DemoBridge {
    /// 🏗️ Cria nova ponte
    pub fn new() -> Self {
        let (demo_sender, _) = broadcast::channel(1000);
        let (result_sender, _) = broadcast::channel(1000);
        
        Self {
            demo_sender,
            result_sender,
            active_parameters: Arc::new(RwLock::new(HashMap::new())),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            execution_state: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 📡 Obtém receiver para dados de demo
    pub fn subscribe_demo_data(&self) -> broadcast::Receiver<DemoData> {
        self.demo_sender.subscribe()
    }
    
    /// 📊 Obtém receiver para resultados
    pub fn subscribe_results(&self) -> broadcast::Receiver<DemoResult> {
        self.result_sender.subscribe()
    }
    
    /// 📤 Envia dados de demonstração
    pub async fn send_demo_data(&self, data: DemoData) -> Result<()> {
        self.demo_sender.send(data).map_err(|e| anyhow::anyhow!("Erro ao enviar dados: {}", e))?;
        Ok(())
    }
    
    /// 📤 Envia resultado de demonstração
    pub async fn send_demo_result(&self, result: DemoResult) -> Result<()> {
        // Atualiza histórico de métricas
        let mut history = self.metrics_history.write().await;
        history.push(result.metrics.clone());
        
        // Mantém apenas os últimos 100 registros
        if history.len() > 100 {
            history.remove(0);
        }
        
        // Atualiza estado de execução
        let mut state = self.execution_state.write().await;
        state.insert(result.module.clone(), false);
        
        self.result_sender.send(result).map_err(|e| anyhow::anyhow!("Erro ao enviar resultado: {}", e))?;
        Ok(())
    }
    
    /// 🎛️ Atualiza parâmetros de módulo
    pub async fn update_parameters(&self, module: String, parameters: DemoParameters) -> Result<()> {
        let mut params = self.active_parameters.write().await;
        params.insert(module, parameters);
        Ok(())
    }
    
    /// 🎛️ Obtém parâmetros de módulo
    pub async fn get_parameters(&self, module: &str) -> Option<DemoParameters> {
        let params = self.active_parameters.read().await;
        params.get(module).cloned()
    }
    
    /// 📊 Obtém histórico de métricas
    pub async fn get_metrics_history(&self) -> Vec<PerformanceMetrics> {
        let history = self.metrics_history.read().await;
        history.clone()
    }
    
    /// ▶️ Executa demonstração de módulo
    pub async fn run_demo(&self, module: &str) -> Result<DemoResult> {
        // Marca módulo como em execução
        {
            let mut state = self.execution_state.write().await;
            state.insert(module.to_string(), true);
        }
        
        let start_time = std::time::Instant::now();
        
        // Obtém parâmetros do módulo
        let parameters = self.get_parameters(module).await
            .unwrap_or_else(|| self.get_default_parameters(module));
        
        // Executa demonstração baseada no módulo
        let result = match module {
            "tokenizer" => self.run_tokenizer_demo(parameters).await?,
            "attention" => self.run_attention_demo(parameters).await?,
            "embeddings" => self.run_embeddings_demo(parameters).await?,
            "transformer" => self.run_transformer_demo(parameters).await?,
            "training" => self.run_training_demo(parameters).await?,
            "inference" => self.run_inference_demo(parameters).await?,
            "chunking" => self.run_chunking_demo(parameters).await?,
            _ => return Err(anyhow::anyhow!("Módulo desconhecido: {}", module)),
        };
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        let demo_result = DemoResult {
            success: true,
            module: module.to_string(),
            execution_time,
            output: result.output,
            visualizations: result.visualizations,
            metrics: PerformanceMetrics {
                latency: execution_time as f64,
                memory_usage: self.get_memory_usage(),
                cpu_usage: self.get_cpu_usage(),
                execution_time,
                throughput: result.throughput,
            },
        };
        
        // Envia resultado
        self.send_demo_result(demo_result.clone()).await?;
        
        Ok(demo_result)
    }
    
    /// 🎛️ Obtém parâmetros padrão para módulo
    fn get_default_parameters(&self, module: &str) -> DemoParameters {
        let mut parameters = HashMap::new();
        
        match module {
            "tokenizer" => {
                parameters.insert("text".to_string(), serde_json::Value::String("Hello, world!".to_string()));
                parameters.insert("show_tokens".to_string(), serde_json::Value::Bool(true));
            },
            "attention" => {
                parameters.insert("seq_len".to_string(), serde_json::Value::Number(serde_json::Number::from(16)));
                parameters.insert("num_heads".to_string(), serde_json::Value::Number(serde_json::Number::from(8)));
                parameters.insert("show_heatmap".to_string(), serde_json::Value::Bool(true));
            },
            "embeddings" => {
                parameters.insert("vocab_size".to_string(), serde_json::Value::Number(serde_json::Number::from(1000)));
                parameters.insert("embed_dim".to_string(), serde_json::Value::Number(serde_json::Number::from(512)));
            },
            "transformer" => {
                parameters.insert("num_layers".to_string(), serde_json::Value::Number(serde_json::Number::from(6)));
                parameters.insert("d_model".to_string(), serde_json::Value::Number(serde_json::Number::from(512)));
            },
            "training" => {
                parameters.insert("batch_size".to_string(), serde_json::Value::Number(serde_json::Number::from(32)));
                parameters.insert("learning_rate".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.001).unwrap()));
            },
            "inference" => {
                parameters.insert("prompt".to_string(), serde_json::Value::String("Once upon a time".to_string()));
                parameters.insert("max_tokens".to_string(), serde_json::Value::Number(serde_json::Number::from(50)));
            },
            "chunking" => {
                parameters.insert("chunk_size".to_string(), serde_json::Value::Number(serde_json::Number::from(512)));
                parameters.insert("overlap".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.1).unwrap()));
            },
            _ => {}
        }
        
        DemoParameters {
            module: module.to_string(),
            parameters,
            config: DemoConfig {
                educational: true,
                show_tensors: false,
                show_attention: true,
                benchmark: false,
                interactive: true,
                real_time: true,
            },
        }
    }
    
    /// 🔤 Executa demo de tokenização
    async fn run_tokenizer_demo(&self, params: DemoParameters) -> Result<DemoExecutionResult> {
        let text = params.parameters.get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("Hello, world!");
        
        // Simula tokenização
        let tokens = text.split_whitespace().collect::<Vec<_>>();
        
        let visualization = VisualizationData {
            viz_type: "tokenizer".to_string(),
            title: "Tokenização".to_string(),
            data: serde_json::json!({
                "text": text,
                "tokens": tokens,
                "token_count": tokens.len()
            }),
            config: HashMap::new(),
        };
        
        Ok(DemoExecutionResult {
            output: format!("Texto tokenizado: {} tokens", tokens.len()),
            visualizations: vec![visualization],
            throughput: tokens.len() as f64,
        })
    }
    
    /// 🎯 Executa demo de atenção
    async fn run_attention_demo(&self, params: DemoParameters) -> Result<DemoExecutionResult> {
        let seq_len = params.parameters.get("seq_len")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
        
        let num_heads = params.parameters.get("num_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        
        // Simula matriz de atenção
        let mut attention_matrix = Vec::new();
        for i in 0..seq_len {
            let mut row = Vec::new();
            for j in 0..seq_len {
                // Simula padrão de atenção
                let attention = if i == j { 1.0 } else { 0.1 / (i as f64 - j as f64).abs() };
                row.push(attention);
            }
            attention_matrix.push(row);
        }
        
        let visualization = VisualizationData {
            viz_type: "attention_heatmap".to_string(),
            title: "Mapa de Atenção".to_string(),
            data: serde_json::json!({
                "matrix": attention_matrix,
                "seq_len": seq_len,
                "num_heads": num_heads
            }),
            config: HashMap::new(),
        };
        
        Ok(DemoExecutionResult {
            output: format!("Atenção calculada: {}x{} com {} cabeças", seq_len, seq_len, num_heads),
            visualizations: vec![visualization],
            throughput: (seq_len * seq_len) as f64,
        })
    }
    
    /// 📊 Executa demos restantes (implementação simplificada)
    async fn run_embeddings_demo(&self, _params: DemoParameters) -> Result<DemoExecutionResult> {
        Ok(DemoExecutionResult {
            output: "Embeddings gerados com sucesso".to_string(),
            visualizations: vec![],
            throughput: 1000.0,
        })
    }
    
    async fn run_transformer_demo(&self, _params: DemoParameters) -> Result<DemoExecutionResult> {
        Ok(DemoExecutionResult {
            output: "Transformer executado com sucesso".to_string(),
            visualizations: vec![],
            throughput: 500.0,
        })
    }
    
    async fn run_training_demo(&self, _params: DemoParameters) -> Result<DemoExecutionResult> {
        Ok(DemoExecutionResult {
            output: "Treinamento simulado com sucesso".to_string(),
            visualizations: vec![],
            throughput: 100.0,
        })
    }
    
    async fn run_inference_demo(&self, _params: DemoParameters) -> Result<DemoExecutionResult> {
        Ok(DemoExecutionResult {
            output: "Inferência executada com sucesso".to_string(),
            visualizations: vec![],
            throughput: 200.0,
        })
    }
    
    async fn run_chunking_demo(&self, _params: DemoParameters) -> Result<DemoExecutionResult> {
        Ok(DemoExecutionResult {
            output: "Chunking executado com sucesso".to_string(),
            visualizations: vec![],
            throughput: 300.0,
        })
    }
    
    /// 💾 Obtém uso de memória (simulado)
    fn get_memory_usage(&self) -> f64 {
        // Em uma implementação real, usaria bibliotecas como `sysinfo`
        rand::random::<f64>() * 100.0 + 50.0
    }
    
    /// 🖥️ Obtém uso de CPU (simulado)
    fn get_cpu_usage(&self) -> f64 {
        // Em uma implementação real, usaria bibliotecas como `sysinfo`
        rand::random::<f64>() * 50.0 + 10.0
    }
}

/// 📊 Resultado de execução de demo
struct DemoExecutionResult {
    output: String,
    visualizations: Vec<VisualizationData>,
    throughput: f64,
}

impl Default for DemoBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// 🌉 Instância global da ponte
static DEMO_BRIDGE: std::sync::OnceLock<Arc<DemoBridge>> = std::sync::OnceLock::new();

/// 🌉 Obtém instância global da ponte
pub fn get_demo_bridge() -> Arc<DemoBridge> {
    DEMO_BRIDGE.get_or_init(|| Arc::new(DemoBridge::new())).clone()
}