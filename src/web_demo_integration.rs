//! Integração Web-Demo: Sistema de comunicação em tempo real entre interface web e CLI
//! 
//! Este módulo implementa:
//! - WebSocket server para comunicação bidirecional
//! - API REST para controle de parâmetros dinâmicos
//! - Sistema de sincronização em tempo real
//! - Gerenciamento de estado compartilhado

use axum::{
    extract::{ws::WebSocket, ws::WebSocketUpgrade, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use dashmap::DashMap;
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{broadcast, mpsc},
    time::interval,
};
use uuid::Uuid;
use tower_http::cors::CorsLayer;
// use crate::demo_bridge::{get_demo_bridge, DemoData, DemoResult, DemoParameters as BridgeParameters, PerformanceMetrics as BridgeMetrics};

// Tipos temporários para evitar dependência do demo_bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeParameters {
    pub module: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeMetrics {
    pub latency: f64,
    pub memory_usage: f64,
    pub cpu_usage: f64,
    pub execution_time: u64,
    pub throughput: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoResult {
    pub execution_time: u64,
    pub output: String,
    pub visualizations: Vec<serde_json::Value>,
    pub metrics: BridgeMetrics,
}

// Eventos de sincronização
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncEvent {
    ParameterUpdate {
        module: String,
        parameter: String,
        value: serde_json::Value,
        source: String, // "cli" ou "web"
        timestamp: u64,
    },
    DemoExecution {
        module: String,
        result: DemoResult,
        source: String,
        timestamp: u64,
    },
    StateChange {
        module: String,
        state: serde_json::Value,
        source: String,
        timestamp: u64,
    },
    MetricsUpdate {
        metrics: BridgeMetrics,
        timestamp: u64,
    },
}

// Canal de eventos para sincronização
pub type SyncEventSender = broadcast::Sender<SyncEvent>;
pub type SyncEventReceiver = broadcast::Receiver<SyncEvent>;

/// Tipos de mensagens WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WebSocketMessage {
    /// Conectar cliente
    Connect { client_id: String },
    /// Desconectar cliente
    Disconnect { client_id: String },
    /// Atualização de parâmetros
    ParameterUpdate {
        module: String,
        parameter: String,
        value: serde_json::Value,
    },
    /// Dados de demonstração
    DemoData {
        module: String,
        data: serde_json::Value,
        timestamp: u64,
    },
    /// Métricas de performance
    PerformanceMetrics {
        module: String,
        metrics: PerformanceData,
    },
    /// Visualização de dados
    Visualization {
        module: String,
        viz_type: String,
        data: serde_json::Value,
    },
    /// Status do sistema
    SystemStatus {
        status: String,
        modules: Vec<ModuleStatus>,
    },
    /// Erro
    Error { message: String },
}

/// Dados de performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub execution_time: f64,
    pub throughput: f64,
    pub timestamp: u64,
}

/// Status de módulo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleStatus {
    pub name: String,
    pub status: String, // "running", "idle", "error"
    pub last_update: u64,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Parâmetros de demonstração
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoParameters {
    pub module: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub updated_at: u64,
}

/// Cliente WebSocket conectado
#[derive(Debug, Clone)]
pub struct WebSocketClient {
    pub id: String,
    pub connected_at: u64,
    pub last_ping: u64,
    pub subscribed_modules: Vec<String>,
}

/// Estado compartilhado da integração
#[derive(Debug, Clone)]
pub struct IntegrationState {
    /// Clientes WebSocket conectados
    pub clients: Arc<DashMap<String, WebSocketClient>>,
    /// Parâmetros atuais dos módulos
    pub module_parameters: Arc<DashMap<String, DemoParameters>>,
    /// Dados de demonstração em cache
    pub demo_data_cache: Arc<DashMap<String, serde_json::Value>>,
    /// Métricas de performance
    pub performance_metrics: Arc<DashMap<String, PerformanceData>>,
    /// Canal de broadcast para mensagens
    pub broadcast_tx: broadcast::Sender<WebSocketMessage>,
    /// Status dos módulos
    pub module_status: Arc<DashMap<String, ModuleStatus>>,
    /// Canal de eventos de sincronização
    pub sync_tx: SyncEventSender,
}

impl IntegrationState {
    /// Criar novo estado de integração
    pub fn new() -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);
        let (sync_tx, _) = broadcast::channel(1000);
        
        Self {
            clients: Arc::new(DashMap::new()),
            module_parameters: Arc::new(DashMap::new()),
            demo_data_cache: Arc::new(DashMap::new()),
            performance_metrics: Arc::new(DashMap::new()),
            broadcast_tx,
            module_status: Arc::new(DashMap::new()),
            sync_tx,
        }
    }
    
    /// Envia evento de sincronização
    pub fn send_sync_event(&self, event: SyncEvent) {
        if let Err(e) = self.sync_tx.send(event) {
            eprintln!("Erro ao enviar evento de sincronização: {}", e);
        }
    }

    /// Obtém receptor de eventos de sincronização
    pub fn subscribe_sync_events(&self) -> SyncEventReceiver {
        self.sync_tx.subscribe()
    }

    /// Atualiza parâmetros e envia evento de sincronização
    pub fn sync_parameter_update(&self, module: &str, parameter: &str, value: serde_json::Value, source: &str) {
        let timestamp = current_timestamp();

        let event = SyncEvent::ParameterUpdate {
            module: module.to_string(),
            parameter: parameter.to_string(),
            value: value.clone(),
            source: source.to_string(),
            timestamp,
        };

        self.send_sync_event(event);
    }

    /// Envia evento de execução de demo
    pub fn sync_demo_execution(&self, module: &str, source: &str) {
        let result = DemoResult {
            execution_time: 100,
            output: format!("Demo {} executado com sucesso", module),
            visualizations: vec![],
            metrics: BridgeMetrics {
                latency: 0.1,
                memory_usage: 50.0,
                cpu_usage: 25.0,
                execution_time: 100,
                throughput: 1000.0,
            },
        };
        
        let timestamp = current_timestamp();

        let event = SyncEvent::DemoExecution {
            module: module.to_string(),
            result,
            source: source.to_string(),
            timestamp,
        };

        self.send_sync_event(event);
    }

    /// Envia evento de atualização de métricas
    pub fn sync_metrics_update(&self, metrics: BridgeMetrics) {
        let timestamp = current_timestamp();

        let event = SyncEvent::MetricsUpdate {
            metrics: metrics.clone(),
            timestamp,
        };

        self.send_sync_event(event);
    }
    
    /// Adicionar cliente WebSocket
    pub fn add_client(&self, client_id: String) {
        let client = WebSocketClient {
            id: client_id.clone(),
            connected_at: current_timestamp(),
            last_ping: current_timestamp(),
            subscribed_modules: Vec::new(),
        };
        
        self.clients.insert(client_id.clone(), client);
        
        // Notificar conexão
        let _ = self.broadcast_tx.send(WebSocketMessage::Connect { client_id });
    }
    
    /// Remover cliente WebSocket
    pub fn remove_client(&self, client_id: &str) {
        self.clients.remove(client_id);
        
        // Notificar desconexão
        let _ = self.broadcast_tx.send(WebSocketMessage::Disconnect {
            client_id: client_id.to_string(),
        });
    }
    
    /// Atualizar parâmetros de módulo
    pub fn update_parameters(&self, module: String, parameters: HashMap<String, serde_json::Value>) {
        let demo_params = DemoParameters {
            module: module.clone(),
            parameters: parameters.clone(),
            updated_at: current_timestamp(),
        };
        
        self.module_parameters.insert(module.clone(), demo_params);
        
        // Broadcast para todos os clientes
        for (param_name, value) in parameters {
            let _ = self.broadcast_tx.send(WebSocketMessage::ParameterUpdate {
                module: module.clone(),
                parameter: param_name,
                value,
            });
        }
    }
    
    /// Atualizar dados de demonstração
    pub fn update_demo_data(&self, module: String, data: serde_json::Value) {
        self.demo_data_cache.insert(module.clone(), data.clone());
        
        let _ = self.broadcast_tx.send(WebSocketMessage::DemoData {
            module,
            data,
            timestamp: current_timestamp(),
        });
    }
    
    /// Atualizar métricas de performance
    pub fn update_performance_metrics(&self, module: String, metrics: PerformanceData) {
        self.performance_metrics.insert(module.clone(), metrics.clone());
        
        let _ = self.broadcast_tx.send(WebSocketMessage::PerformanceMetrics {
            module,
            metrics,
        });
    }
    
    /// Obter status do sistema
    pub fn get_system_status(&self) -> WebSocketMessage {
        let modules: Vec<ModuleStatus> = self.module_status
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        
        WebSocketMessage::SystemStatus {
            status: "running".to_string(),
            modules,
        }
    }
}

/// Obter timestamp atual
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Configuração da integração
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    pub websocket_path: String,
    pub api_prefix: String,
    pub heartbeat_interval: Duration,
    pub max_clients: usize,
    pub enable_performance_monitoring: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            websocket_path: "/ws".to_string(),
            api_prefix: "/api/v1".to_string(),
            heartbeat_interval: Duration::from_secs(30),
            max_clients: 100,
            enable_performance_monitoring: true,
        }
    }
}

/// Handler para upgrade de WebSocket
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<IntegrationState>,
) -> Response {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

/// Gerenciar conexão WebSocket
async fn handle_websocket(socket: WebSocket, state: IntegrationState) {
    let client_id = Uuid::new_v4().to_string();
    state.add_client(client_id.clone());
    
    let (mut sender, mut receiver) = socket.split();
    let mut broadcast_rx = state.broadcast_tx.subscribe();
    
    // Task para enviar mensagens broadcast
    let send_state = state.clone();
    let send_client_id = client_id.clone();
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = broadcast_rx.recv().await {
            let json_msg = serde_json::to_string(&msg).unwrap_or_default();
            if sender.send(axum::extract::ws::Message::Text(json_msg)).await.is_err() {
                break;
            }
        }
    });
    
    // Task para receber mensagens do cliente
    let recv_state = state.clone();
    let recv_client_id = client_id.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            if let Ok(msg) = msg {
                if let axum::extract::ws::Message::Text(text) = msg {
                    if let Ok(ws_msg) = serde_json::from_str::<WebSocketMessage>(&text) {
                        handle_websocket_message(ws_msg, &recv_state, &recv_client_id).await;
                    }
                }
            } else {
                break;
            }
        }
    });
    
    // Aguardar até uma das tasks terminar
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
    
    // Limpar cliente
    state.remove_client(&client_id);
}

/// Processar mensagem WebSocket recebida
async fn handle_websocket_message(
    msg: WebSocketMessage,
    state: &IntegrationState,
    client_id: &str,
) {
    match msg {
        WebSocketMessage::ParameterUpdate { module, parameter, value } => {
            let mut params = HashMap::new();
            params.insert(parameter, value);
            state.update_parameters(module, params);
        }
        WebSocketMessage::Connect { .. } => {
            // Enviar status atual do sistema
            let status = state.get_system_status();
            let _ = state.broadcast_tx.send(status);
        }
        _ => {
            // Outros tipos de mensagem podem ser processados aqui
        }
    }
}

/// Handler para obter parâmetros de um módulo
pub async fn get_module_parameters(
    Path(module): Path<String>,
    State(state): State<IntegrationState>,
) -> Result<Json<DemoParameters>, StatusCode> {
    if let Some(params) = state.module_parameters.get(&module) {
        Ok(Json(params.clone()))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

/// Handler para atualizar parâmetros de um módulo
pub async fn update_module_parameters(
    Path(module): Path<String>,
    State(state): State<IntegrationState>,
    Json(parameters): Json<HashMap<String, serde_json::Value>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Atualizar parâmetros no estado
    let demo_params = DemoParameters {
        module: module.clone(),
        parameters: parameters.clone(),
        updated_at: current_timestamp(),
    };
    
    state.module_parameters.insert(module.clone(), demo_params.clone());
    
    // Enviar eventos de sincronização para cada parâmetro atualizado
    for (param_name, param_value) in &parameters {
        state.sync_parameter_update(&module, param_name, param_value.clone(), "web");
    }
    
    // Broadcast para clientes WebSocket
    for (param_name, value) in &parameters {
        let ws_message = WebSocketMessage::ParameterUpdate {
            module: module.clone(),
            parameter: param_name.clone(),
            value: value.clone(),
        };
        
        if let Err(e) = state.broadcast_tx.send(ws_message) {
            eprintln!("Erro ao enviar mensagem WebSocket: {}", e);
        }
    }
    
    Ok(Json(serde_json::json!({
        "success": true,
        "module": module,
        "parameters": parameters,
        "message": format!("Parâmetros do módulo {} atualizados e sincronizados", module)
    })))
}

/// Handler para obter dados de demonstração
pub async fn get_demo_data(
    Path(module): Path<String>,
    State(state): State<IntegrationState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(data) = state.demo_data_cache.get(&module) {
        Ok(Json(data.clone()))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

/// Handler para obter métricas de performance
pub async fn get_performance_metrics(
    Path(module): Path<String>,
    State(state): State<IntegrationState>,
) -> Result<Json<PerformanceData>, StatusCode> {
    if let Some(metrics) = state.performance_metrics.get(&module) {
        Ok(Json(metrics.clone()))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

/// Handler para obter status do sistema
pub async fn get_system_status(
    State(state): State<IntegrationState>,
) -> Json<WebSocketMessage> {
    Json(state.get_system_status())
}

/// Handler para executar demonstração de módulo
pub async fn run_demo_module(
    State(state): State<IntegrationState>,
    Path(module): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Simular execução de demo (temporário)
    let result_json = serde_json::json!({
        "success": true,
        "module": module,
        "output": format!("Demo {} executado com sucesso", module),
        "execution_time": 100,
        "visualizations": [],
        "metrics": {
            "latency": 0.1,
            "memory_usage": 50.0,
            "cpu_usage": 25.0,
            "execution_time": 100,
            "throughput": 1000.0
        }
    });
    
    state.update_demo_data(module.clone(), result_json.clone());
    
    // Atualizar métricas de performance
    let perf_data = PerformanceData {
        cpu_usage: 25.0,
        memory_usage: 50.0,
        execution_time: 100.0,
        throughput: 1000.0,
        timestamp: current_timestamp(),
    };
    state.update_performance_metrics(module.clone(), perf_data);
    
    // Broadcast para clientes WebSocket
    let ws_message = WebSocketMessage::DemoData {
        module: module.clone(),
        data: result_json.clone(),
        timestamp: current_timestamp(),
    };
    
    if let Err(e) = state.broadcast_tx.send(ws_message) {
        eprintln!("Erro ao enviar mensagem WebSocket: {}", e);
    }
    
    Ok(Json(result_json))
}

/// Handler para obter histórico de demonstrações
pub async fn get_demo_history(
    State(_state): State<IntegrationState>,
) -> Json<serde_json::Value> {
    // Simular histórico de métricas (temporário)
    let history = vec![
        serde_json::json!({
            "latency": 0.1,
            "memory_usage": 45.0,
            "cpu_usage": 20.0,
            "execution_time": 95,
            "throughput": 950.0
        }),
        serde_json::json!({
            "latency": 0.12,
            "memory_usage": 50.0,
            "cpu_usage": 25.0,
            "execution_time": 100,
            "throughput": 1000.0
        })
    ];
    
    Json(serde_json::json!({
        "success": true,
        "history": history
    }))
}

/// Handler para listar todos os módulos
pub async fn list_modules(
    State(state): State<IntegrationState>,
) -> Json<Vec<String>> {
    let modules: Vec<String> = state.module_status
        .iter()
        .map(|entry| entry.key().clone())
        .collect();
    Json(modules)
}

/// Handler para eventos de sincronização via SSE
pub async fn sync_events_handler(
    State(state): State<IntegrationState>,
) -> impl IntoResponse {
    let mut rx = state.subscribe_sync_events();
    
    let stream = async_stream::stream! {
        while let Ok(event) = rx.recv().await {
            let json_data = serde_json::to_string(&event).unwrap_or_default();
            let sse_data = format!("data: {}\n\n", json_data);
            yield Ok::<_, axum::Error>(sse_data);
        }
    };
    
    axum::response::Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

/// Handler para conectar CLI e receber eventos
pub async fn cli_connect_handler(
    State(state): State<IntegrationState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let cli_id = payload.get("cli_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    
    // Registrar CLI como cliente especial
    let client = WebSocketClient {
        id: format!("cli_{}", cli_id),
        connected_at: current_timestamp(),
        last_ping: current_timestamp(),
        subscribed_modules: vec!["all".to_string()],
    };
    
    state.clients.insert(client.id.clone(), client);
    
    Ok(Json(serde_json::json!({
        "success": true,
        "client_id": format!("cli_{}", cli_id),
        "message": "CLI conectado com sucesso",
        "sync_endpoint": "/api/v1/sync/events"
    })))
}

/// Criar router com todas as rotas da integração
pub fn create_integration_router(state: IntegrationState) -> Router {
    Router::new()
        // WebSocket
        .route("/ws", get(websocket_handler))
        // API REST - versão v1
        .route("/api/v1/modules", get(list_modules))
        .route("/api/v1/modules/:module/parameters", 
               get(get_module_parameters).put(update_module_parameters))
        .route("/api/v1/modules/:module/data", get(get_demo_data))
        .route("/api/v1/modules/:module/metrics", get(get_performance_metrics))
        .route("/api/v1/system/status", get(get_system_status))
        .route("/api/v1/demo/:module/run", post(run_demo_module))
        .route("/api/v1/demo/history", get(get_demo_history))
        .route("/api/v1/sync/events", get(sync_events_handler))
        .route("/api/v1/sync/cli/connect", post(cli_connect_handler))
        // API REST - compatibilidade (sem versão)
        .route("/api/modules", get(list_modules))
        .route("/api/modules/:module/parameters", 
               get(get_module_parameters).put(update_module_parameters))
        .route("/api/modules/:module/data", get(get_demo_data))
        .route("/api/modules/:module/metrics", get(get_performance_metrics))
        .route("/api/system/status", get(get_system_status))
        .route("/api/demo/:module/run", post(run_demo_module))
        .route("/api/demo/history", get(get_demo_history))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Inicializar módulos padrão
pub fn initialize_default_modules(state: &IntegrationState) {
    let modules = vec![
        "tokenization", "attention", "embeddings", 
        "transformer", "training", "inference", "chunking"
    ];
    
    for module in modules {
        let status = ModuleStatus {
            name: module.to_string(),
            status: "idle".to_string(),
            last_update: current_timestamp(),
            parameters: HashMap::new(),
        };
        
        state.module_status.insert(module.to_string(), status);
        
        // Parâmetros padrão para cada módulo
        let default_params = get_default_parameters(module);
        state.update_parameters(module.to_string(), default_params);
    }
}

/// Obter parâmetros padrão para um módulo
fn get_default_parameters(module: &str) -> HashMap<String, serde_json::Value> {
    let mut params = HashMap::new();
    
    match module {
        "tokenization" => {
            params.insert("vocab_size".to_string(), serde_json::Value::Number(serde_json::Number::from(50000)));
            params.insert("max_length".to_string(), serde_json::Value::Number(serde_json::Number::from(512)));
            params.insert("padding".to_string(), serde_json::Value::Bool(true));
        }
        "attention" => {
            params.insert("num_heads".to_string(), serde_json::Value::Number(serde_json::Number::from(8)));
            params.insert("head_dim".to_string(), serde_json::Value::Number(serde_json::Number::from(64)));
            params.insert("dropout".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.1).unwrap()));
        }
        "embeddings" => {
            params.insert("embedding_dim".to_string(), serde_json::Value::Number(serde_json::Number::from(512)));
            params.insert("max_position".to_string(), serde_json::Value::Number(serde_json::Number::from(1024)));
        }
        "transformer" => {
            params.insert("num_layers".to_string(), serde_json::Value::Number(serde_json::Number::from(6)));
            params.insert("hidden_size".to_string(), serde_json::Value::Number(serde_json::Number::from(512)));
            params.insert("intermediate_size".to_string(), serde_json::Value::Number(serde_json::Number::from(2048)));
        }
        "training" => {
            params.insert("learning_rate".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.001).unwrap()));
            params.insert("batch_size".to_string(), serde_json::Value::Number(serde_json::Number::from(32)));
            params.insert("epochs".to_string(), serde_json::Value::Number(serde_json::Number::from(10)));
        }
        "inference" => {
            params.insert("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.8).unwrap()));
            params.insert("top_k".to_string(), serde_json::Value::Number(serde_json::Number::from(50)));
            params.insert("top_p".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.9).unwrap()));
        }
        "chunking" => {
            params.insert("chunk_size".to_string(), serde_json::Value::Number(serde_json::Number::from(1000)));
            params.insert("overlap".to_string(), serde_json::Value::Number(serde_json::Number::from(200)));
            params.insert("strategy".to_string(), serde_json::Value::String("semantic".to_string()));
        }
        _ => {}
    }
    
    params
}