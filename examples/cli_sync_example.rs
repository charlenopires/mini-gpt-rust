//! Exemplo de como o CLI pode se conectar ao sistema de sincronização web
//! 
//! Este exemplo demonstra como o CLI pode:
//! 1. Conectar-se ao servidor web
//! 2. Receber eventos de sincronização da interface web
//! 3. Enviar atualizações de volta para a web

use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug, serde::Deserialize)]
struct SyncEvent {
    #[serde(flatten)]
    event_type: EventType,
}

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type")]
enum EventType {
    ParameterUpdate {
        module: String,
        parameter: String,
        value: Value,
        source: String,
        timestamp: u64,
    },
    DemoExecution {
        module: String,
        result: Value,
        source: String,
        timestamp: u64,
    },
    StateChange {
        module: String,
        state: Value,
        source: String,
        timestamp: u64,
    },
    MetricsUpdate {
        metrics: Value,
        timestamp: u64,
    },
}

/// Cliente CLI para sincronização com a interface web
pub struct CliSyncClient {
    client: Client,
    base_url: String,
    cli_id: String,
}

impl CliSyncClient {
    pub fn new(base_url: String, cli_id: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            cli_id,
        }
    }

    /// Conecta o CLI ao servidor de sincronização
    pub async fn connect(&self) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}/api/v1/sync/cli/connect", self.base_url);
        
        let payload = json!({
            "cli_id": self.cli_id,
            "capabilities": ["parameter_sync", "demo_execution", "metrics_reporting"]
        });

        let response = self.client
            .post(&url)
            .json(&payload)
            .send()
            .await?
            .json::<Value>()
            .await?;

        if response["success"].as_bool().unwrap_or(false) {
            println!("✅ CLI conectado com sucesso: {}", response["client_id"]);
            Ok(response["client_id"].as_str().unwrap_or("").to_string())
        } else {
            Err("Falha ao conectar CLI".into())
        }
    }

    /// Escuta eventos de sincronização via Server-Sent Events
    pub async fn listen_sync_events(&self) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{}/api/v1/sync/events", self.base_url);
        
        println!("🔄 Iniciando escuta de eventos de sincronização...");
        
        // Simulação de escuta de eventos SSE
        // Em uma implementação real, usaríamos uma biblioteca como eventsource-stream
        loop {
            match self.client.get(&url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        println!("📡 Conectado ao stream de eventos");
                        // Aqui processaríamos os eventos SSE
                        self.simulate_event_processing().await;
                    }
                }
                Err(e) => {
                    eprintln!("❌ Erro ao conectar ao stream: {}", e);
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }

    /// Simula o processamento de eventos recebidos
    async fn simulate_event_processing(&self) {
        println!("🎯 Processando eventos de sincronização...");
        
        // Simular recebimento de eventos
        let sample_events = vec![
            "ParameterUpdate: tokenizer.max_length = 1024 (fonte: web)",
            "DemoExecution: attention executado (fonte: web)",
            "MetricsUpdate: CPU 45%, Memory 250MB",
        ];

        for event in sample_events {
            println!("📨 Evento recebido: {}", event);
            sleep(Duration::from_secs(2)).await;
            
            // Simular resposta do CLI
            self.send_cli_update().await.ok();
        }
    }

    /// Envia atualização do CLI para a web
    pub async fn send_cli_update(&self) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{}/api/v1/modules/tokenizer/parameters", self.base_url);
        
        let cli_params = json!({
            "learning_rate": 0.002,
            "batch_size": 64,
            "source": "cli",
            "timestamp": chrono::Utc::now().timestamp()
        });

        let response = self.client
            .post(&url)
            .json(&cli_params)
            .send()
            .await?
            .json::<Value>()
            .await?;

        if response["success"].as_bool().unwrap_or(false) {
            println!("✅ Parâmetros CLI sincronizados com a web");
        }

        Ok(())
    }

    /// Executa demo via CLI e sincroniza resultado
    pub async fn run_demo_and_sync(&self, module: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Executando demo '{}' via CLI...", module);
        
        // Simular execução de demo
        sleep(Duration::from_millis(500)).await;
        
        let url = format!("{}/api/v1/demo/{}/run", self.base_url, module);
        
        let response = self.client
            .post(&url)
            .send()
            .await?
            .json::<Value>()
            .await?;

        if response["success"].as_bool().unwrap_or(false) {
            println!("✅ Demo '{}' executado e sincronizado", module);
            println!("📊 Resultado: {:?}", response["result"]);
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Exemplo de Sincronização CLI ↔ Web");
    println!("=====================================\n");

    let client = CliSyncClient::new(
        "http://127.0.0.1:3001".to_string(),
        "example_cli".to_string(),
    );

    // 1. Conectar CLI
    match client.connect().await {
        Ok(client_id) => println!("🎯 Cliente conectado: {}", client_id),
        Err(e) => {
            eprintln!("❌ Erro ao conectar: {}", e);
            return Ok(());
        }
    }

    // 2. Demonstrar sincronização de parâmetros
    println!("\n📝 Testando sincronização de parâmetros...");
    client.send_cli_update().await?;

    // 3. Demonstrar execução de demo
    println!("\n🚀 Testando execução de demo...");
    client.run_demo_and_sync("tokenizer").await?;
    
    // 4. Simular escuta de eventos (em loop)
    println!("\n🔄 Iniciando escuta de eventos (Ctrl+C para parar)...");
    client.listen_sync_events().await?;

    Ok(())
}