//! Integração Web-Demo: Sistema de comunicação em tempo real entre interface web e CLI
//! 
//! Este módulo implementa:
//! - WebSocket server para comunicação bidirecional
//! - API REST para controle de parâmetros dinâmicos
//! - Sistema de sincronização em tempo real
//! - Gerenciamento de estado compartilhado

use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use serde_json;
use tokio::time::Duration;
use tokio::sync::broadcast;
use axum::{
    extract::{ws::WebSocket, ws::WebSocketUpgrade, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use dashmap::DashMap;
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
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
    let _send_state = state.clone();
    let _send_client_id = client_id.clone();
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
    _client_id: &str,
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

/// Criar router com todas as rotas da integração
pub fn create_integration_router(state: IntegrationState) -> Router {
    Router::new()
        .route("/ws", get(websocket_handler))
        .route("/parameters/:module", get(get_module_parameters))
        .route("/parameters", put(update_module_parameters))
        .route("/demo_data/:module", get(get_demo_data))
        .route("/metrics", get(get_performance_metrics))
        .route("/status", get(get_system_status_handler))
        .route("/run_demo", post(run_demo_module))
        .route("/history", get(get_demo_history))
        .route("/modules", get(list_modules))
        .with_state(state)
}

// Handlers da API REST (implementações dummy)

async fn get_module_parameters(
    State(state): State<IntegrationState>,
    Path(module): Path<String>,
) -> impl IntoResponse {
    if let Some(params) = state.module_parameters.get(&module) {
        Json(params.value().clone()).into_response()
    } else {
        (StatusCode::NOT_FOUND, format!("Módulo '{}' não encontrado", module)).into_response()
    }
}

async fn update_module_parameters(
    State(state): State<IntegrationState>,
    Json(payload): Json<DemoParameters>,
) -> impl IntoResponse {
    state.update_parameters(payload.module.clone(), payload.parameters.clone());
    (StatusCode::OK, Json(payload)).into_response()
}

async fn get_demo_data(
    State(state): State<IntegrationState>,
    Path(module): Path<String>,
) -> impl IntoResponse {
    if let Some(data) = state.demo_data_cache.get(&module) {
        Json(data.value().clone()).into_response()
    } else {
        (StatusCode::NOT_FOUND, format!("Dados de demonstração para '{}' não encontrados", module)).into_response()
    }
}

async fn get_performance_metrics(State(state): State<IntegrationState>) -> impl IntoResponse {
    let metrics: HashMap<_, _> = state.performance_metrics.iter().map(|m| (m.key().clone(), m.value().clone())).collect();
    Json(metrics).into_response()
}

async fn get_system_status_handler(State(state): State<IntegrationState>) -> impl IntoResponse {
    let status = state.get_system_status();
    Json(status).into_response()
}

async fn run_demo_module(
    State(state): State<IntegrationState>,
    Json(payload): Json<HashMap<String, String>>,
) -> impl IntoResponse {
    let module = payload.get("module").cloned().unwrap_or_default();
    state.sync_demo_execution(&module, "web");
    (StatusCode::OK, format!("Execução do módulo '{}' iniciada", module)).into_response()
}

async fn get_demo_history() -> impl IntoResponse {
    (StatusCode::OK, Json(vec!["histórico de demonstração"])).into_response()
}

async fn list_modules(State(state): State<IntegrationState>) -> impl IntoResponse {
    let modules: Vec<String> = state.module_status.iter().map(|m| m.key().clone()).collect();
    Json(modules).into_response()
}

/// Inicializa módulos padrão do sistema
pub fn initialize_default_modules(state: &IntegrationState) {
    // Módulo de inferência
    state.module_status.insert(
        "inference".to_string(),
        ModuleStatus {
            name: "inference".to_string(),
            status: "ready".to_string(),
            last_update: current_timestamp(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("model_path".to_string(), serde_json::Value::String("models/mini_gpt.safetensors".to_string()));
                params.insert("max_tokens".to_string(), serde_json::Value::Number(serde_json::Number::from(100)));
                params.insert("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.8).unwrap()));
                params.insert("top_p".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.9).unwrap()));
                params
            },
        },
    );
    
    // Módulo de tokenização
    state.module_status.insert(
        "tokenization".to_string(),
        ModuleStatus {
            name: "tokenization".to_string(),
            status: "ready".to_string(),
            last_update: current_timestamp(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("model_path".to_string(), serde_json::Value::String("models/tokenizer.json".to_string()));
                params.insert("max_tokens".to_string(), serde_json::Value::Number(serde_json::Number::from(512)));
                params.insert("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(1.0).unwrap()));
                params.insert("top_p".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(1.0).unwrap()));
                params
            },
        },
    );
    
    // Módulo de atenção
    state.module_status.insert(
        "attention".to_string(),
        ModuleStatus {
            name: "attention".to_string(),
            status: "ready".to_string(),
            last_update: current_timestamp(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("model_path".to_string(), serde_json::Value::String("models/mini_gpt.safetensors".to_string()));
                params.insert("max_tokens".to_string(), serde_json::Value::Number(serde_json::Number::from(50)));
                params.insert("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.7).unwrap()));
                params.insert("top_p".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.95).unwrap()));
                params
            },
        },
    );
}