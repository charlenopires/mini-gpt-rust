//! 🌐 **SERVIDOR WEB**
//!
//! Serve os arquivos estáticos de `interativos/` (as páginas de
//! demonstração) e monta a API real em `/api` (ver `crate::api`). Não gera
//! mais HTML inline em Rust — as páginas vivem em `interativos/*.html`.

use std::{net::SocketAddr, path::PathBuf};

use anyhow::Result;
use axum::Router;
use candle_core::Device;
use tower_http::{cors::CorsLayer, services::ServeDir};

use crate::api;

/// Configuração do servidor web
#[derive(Debug, Clone)]
pub struct WebServerConfig {
    pub host: String,
    pub port: u16,
    pub interativos_dir: PathBuf,
    pub corpus_path: PathBuf,
    pub models_dir: PathBuf,
}

impl Default for WebServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            interativos_dir: PathBuf::from("interativos"),
            corpus_path: PathBuf::from("data/corpus_pt_br.txt"),
            models_dir: PathBuf::from("models"),
        }
    }
}

/// Inicia o servidor web: monta a API real em `/api` e serve
/// `interativos/` como arquivos estáticos em tudo mais (incluindo `/`,
/// que resolve para `interativos/index.html`).
pub async fn start_web_server(config: WebServerConfig, device: Device) -> Result<()> {
    let state = api::AppState::new(device, config.corpus_path.clone(), config.models_dir.clone());

    let app = Router::new()
        .nest("/api", api::router(state))
        .fallback_service(ServeDir::new(&config.interativos_dir))
        .layer(CorsLayer::permissive());

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;

    println!("🌐 Servidor web em http://{}", addr);
    println!("📁 Interativos: {:?}", config.interativos_dir);
    println!("📡 API real em: http://{}/api", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
