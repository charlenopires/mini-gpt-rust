//! 🌐 **API REAL DA DEMONSTRAÇÃO WEB**
//!
//! Substitui `web_demo_integration.rs`/`demo_bridge.rs` (que só devolviam
//! dados fixos/simulados). Toda rota aqui chama o tokenizer, o modelo ou o
//! treinador de verdade — nada de `Math.random()` ou métricas fabricadas.

use std::{
    collections::VecDeque,
    convert::Infallible,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use anyhow::Result as AnyResult;
use async_stream::stream;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::model::{CheckpointMetadata, GPTConfig, MiniGPT};
use crate::tokenizer::BPETokenizer;
use crate::training::{Trainer, TrainingEvent};

const DEMO_VOCAB_SIZE: usize = 1000;
const DEFAULT_N_EMBD: usize = 128;
const DEFAULT_N_HEAD: usize = 4;
const DEFAULT_N_LAYER: usize = 4;
const DEFAULT_BLOCK_SIZE: usize = 64;
const DEFAULT_DROPOUT: f32 = 0.1;
const MAX_RECENT_EVENTS: usize = 200;
/// Limite de tamanho para qualquer campo de texto enviado pelo cliente
/// (tokenização, atenção, embeddings) — evita que uma requisição gigante
/// prenda a thread do servidor processando texto arbitrariamente grande.
const MAX_INPUT_TEXT_LEN: usize = 20_000;

/// Um modelo + tokenizer efetivamente carregados na memória, prontos para
/// atenção/embeddings/geração real.
struct LoadedModel {
    model: MiniGPT,
    tokenizer: BPETokenizer,
    checkpoint_path: String,
    metadata: CheckpointMetadata,
}

#[derive(Clone)]
pub struct AppState {
    device: Device,
    corpus_path: PathBuf,
    models_dir: PathBuf,
    /// Tokenizer treinado uma vez no boot a partir do corpus real — permite
    /// que a página de tokenização funcione mesmo antes de qualquer
    /// treinamento/checkpoint carregado.
    demo_tokenizer: Option<Arc<BPETokenizer>>,
    model: Arc<Mutex<Option<LoadedModel>>>,
    training_tx: broadcast::Sender<TrainingEvent>,
    training_running: Arc<AtomicBool>,
    recent_events: Arc<Mutex<VecDeque<TrainingEvent>>>,
}

impl AppState {
    pub fn new(device: Device, corpus_path: PathBuf, models_dir: PathBuf) -> Self {
        let demo_tokenizer = match std::fs::read_to_string(&corpus_path) {
            Ok(text) => match BPETokenizer::new(DEMO_VOCAB_SIZE).and_then(|mut t| {
                t.train(&text)?;
                Ok(t)
            }) {
                Ok(t) => {
                    println!(
                        "🔤 Tokenizer de demonstração treinado ({} tokens) a partir de {}",
                        t.vocab_size(),
                        corpus_path.display()
                    );
                    Some(Arc::new(t))
                }
                Err(e) => {
                    println!("⚠️  Não foi possível treinar o tokenizer de demonstração: {}", e);
                    None
                }
            },
            Err(e) => {
                println!("⚠️  Corpus não encontrado em {:?}: {}", corpus_path, e);
                None
            }
        };

        let model = MiniGPT::list_checkpoints(&models_dir)
            .ok()
            .and_then(|checkpoints| checkpoints.into_iter().next())
            .and_then(|(path, _)| load_checkpoint(&path, &device).ok());

        if let Some(ref loaded) = model {
            println!("✅ Checkpoint carregado automaticamente: {}", loaded.checkpoint_path);
        }

        let (training_tx, _) = broadcast::channel(256);

        Self {
            device,
            corpus_path,
            models_dir,
            demo_tokenizer,
            model: Arc::new(Mutex::new(model)),
            training_tx,
            training_running: Arc::new(AtomicBool::new(false)),
            recent_events: Arc::new(Mutex::new(VecDeque::with_capacity(MAX_RECENT_EVENTS))),
        }
    }
}

fn load_checkpoint(path: &str, device: &Device) -> AnyResult<LoadedModel> {
    let (model, metadata) =
        MiniGPT::load_from_checkpoint(path, device).map_err(|e| anyhow::anyhow!("{}", e))?;
    let tokenizer_path = format!("{}.tokenizer.json", path);
    let tokenizer =
        BPETokenizer::load_json(&tokenizer_path).map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(LoadedModel {
        model,
        tokenizer,
        checkpoint_path: path.to_string(),
        metadata,
    })
}

fn model_status_json(loaded: &LoadedModel) -> serde_json::Value {
    serde_json::json!({
        "loaded": true,
        "checkpoint_path": loaded.checkpoint_path,
        "config": loaded.model.config(),
        "num_parameters": loaded.model.num_parameters(),
        "vocab_size": loaded.tokenizer.vocab_size(),
        "training_step": loaded.metadata.training_step,
        "loss": loaded.metadata.loss,
    })
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/corpus/stats", get(corpus_stats))
        .route("/tokenize", post(tokenize))
        .route("/checkpoints", get(list_checkpoints_handler))
        .route("/model/load", post(load_model))
        .route("/model/status", get(model_status))
        .route("/predict", post(predict))
        .route("/attention", post(attention))
        .route("/embeddings", post(embeddings))
        .route("/generate", get(generate_sse))
        .route("/train/start", post(train_start))
        .route("/train/ws", get(train_ws))
        .route("/train/status", get(train_status))
        .with_state(state)
}

// ── Corpus & Tokenização ────────────────────────────────────────────────

#[derive(Serialize)]
struct CorpusStatsResponse {
    path: String,
    chars: usize,
    lines: usize,
    words: usize,
    tokens: Option<usize>,
}

async fn corpus_stats(State(state): State<AppState>) -> Response {
    let text = match std::fs::read_to_string(&state.corpus_path) {
        Ok(t) => t,
        Err(e) => return err_response(404, format!("Corpus não encontrado em {:?}: {}", state.corpus_path, e)),
    };
    let tokens = state
        .demo_tokenizer
        .as_ref()
        .and_then(|t| t.encode(&text).ok())
        .map(|t| t.len());

    Json(CorpusStatsResponse {
        path: state.corpus_path.display().to_string(),
        chars: text.chars().count(),
        lines: text.lines().count(),
        words: text.split_whitespace().count(),
        tokens,
    })
    .into_response()
}

#[derive(Deserialize)]
struct TokenizeRequest {
    text: String,
}

#[derive(Serialize)]
struct TokenInfo {
    id: usize,
    text: String,
}

async fn tokenize(State(state): State<AppState>, Json(req): Json<TokenizeRequest>) -> Response {
    if req.text.len() > MAX_INPUT_TEXT_LEN {
        return err_response(413, format!("Texto muito longo (máximo {} caracteres)", MAX_INPUT_TEXT_LEN));
    }
    let Some(tokenizer) = state.demo_tokenizer.as_ref() else {
        return err_response(503, "Tokenizer de demonstração indisponível (corpus não encontrado no boot)");
    };
    let ids = match tokenizer.encode(&req.text) {
        Ok(ids) => ids,
        Err(e) => return err_response(400, format!("Erro ao tokenizar: {}", e)),
    };
    let tokens: Vec<TokenInfo> = ids
        .iter()
        .map(|&id| TokenInfo {
            id,
            text: tokenizer.token_string(id).unwrap_or("?").to_string(),
        })
        .collect();

    Json(serde_json::json!({ "count": tokens.len(), "tokens": tokens })).into_response()
}

// ── Checkpoints & modelo carregado ──────────────────────────────────────

async fn list_checkpoints_handler(State(state): State<AppState>) -> Response {
    match MiniGPT::list_checkpoints(&state.models_dir) {
        Ok(checkpoints) => {
            let items: Vec<_> = checkpoints
                .into_iter()
                .map(|(path, meta)| {
                    serde_json::json!({
                        "path": path,
                        "timestamp": meta.timestamp,
                        "training_step": meta.training_step,
                        "loss": meta.loss,
                        "description": meta.description,
                    })
                })
                .collect();
            Json(items).into_response()
        }
        Err(e) => err_response(500, format!("{}", e)),
    }
}

#[derive(Deserialize, Default)]
struct LoadModelRequest {
    #[serde(default)]
    checkpoint: Option<String>,
}

async fn load_model(
    State(state): State<AppState>,
    body: Option<Json<LoadModelRequest>>,
) -> Response {
    let requested = body.and_then(|b| b.0.checkpoint);

    // Nunca usa um caminho vindo do cliente diretamente — só aceita um valor
    // que já esteja na lista de checkpoints que o próprio servidor enumerou
    // em `models_dir`. Isso fecha a porta para path traversal (ex.:
    // `{"checkpoint": "../../../../etc/passwd"}`).
    let checkpoints = match MiniGPT::list_checkpoints(&state.models_dir) {
        Ok(cps) => cps,
        Err(e) => return err_response(500, format!("{}", e)),
    };
    if checkpoints.is_empty() {
        return err_response(404, "Nenhum checkpoint encontrado");
    }

    let path = match requested {
        None => checkpoints[0].0.clone(),
        Some(requested) => match checkpoints.iter().find(|(p, _)| *p == requested) {
            Some((p, _)) => p.clone(),
            None => {
                return err_response(
                    400,
                    "Checkpoint desconhecido — use um caminho retornado por GET /api/checkpoints",
                )
            }
        },
    };

    match load_checkpoint(&path, &state.device) {
        Ok(loaded) => {
            let response = model_status_json(&loaded);
            *state.model.lock().unwrap() = Some(loaded);
            Json(response).into_response()
        }
        Err(e) => err_response(400, format!("Erro ao carregar checkpoint: {}", e)),
    }
}

async fn model_status(State(state): State<AppState>) -> Response {
    let guard = state.model.lock().unwrap();
    match guard.as_ref() {
        Some(loaded) => Json(model_status_json(loaded)).into_response(),
        None => Json(serde_json::json!({ "loaded": false })).into_response(),
    }
}

// ── Atenção real ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct AttentionRequest {
    prompt: String,
}

async fn attention(State(state): State<AppState>, Json(req): Json<AttentionRequest>) -> Response {
    if req.prompt.len() > MAX_INPUT_TEXT_LEN {
        return err_response(413, format!("Prompt muito longo (máximo {} caracteres)", MAX_INPUT_TEXT_LEN));
    }
    let guard = state.model.lock().unwrap();
    let Some(loaded) = guard.as_ref() else {
        return err_response(503, "Nenhum modelo carregado — treine ou carregue um checkpoint primeiro");
    };

    let ids = match loaded.tokenizer.encode(&req.prompt) {
        Ok(ids) if !ids.is_empty() => ids,
        Ok(_) => return err_response(400, "Prompt vazio"),
        Err(e) => return err_response(400, format!("Erro ao tokenizar: {}", e)),
    };

    let block_size = loaded.model.block_size();
    let ids: Vec<usize> = if ids.len() > block_size {
        ids[ids.len() - block_size..].to_vec()
    } else {
        ids
    };
    let ids_i64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();

    let idx = match Tensor::from_slice(&ids_i64, &[1, ids.len()], &state.device) {
        Ok(t) => t,
        Err(e) => return err_response(500, format!("{}", e)),
    };

    let layer_weights = match loaded.model.forward_with_attention(&idx) {
        Ok((_, w)) => w,
        Err(e) => return err_response(500, format!("Erro no forward pass: {}", e)),
    };

    // Devolve TODAS as camadas × cabeças numa resposta só — o cliente faz o
    // scrub de camada/cabeça (e a média entre cabeças) sem novos round-trips.
    // Cada tensor de camada é [1, H, T, T] pós-softmax → [H][T][T].
    let mut layers: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(layer_weights.len());
    for w in &layer_weights {
        match w.squeeze(0).and_then(|t| t.to_vec3::<f32>()) {
            Ok(heads) => layers.push(heads),
            Err(e) => return err_response(500, format!("{}", e)),
        }
    }
    let num_layers = layers.len();
    let num_heads = layers.first().map(|h| h.len()).unwrap_or(0);

    let tokens: Vec<String> = ids
        .iter()
        .map(|&id| loaded.tokenizer.token_string(id).unwrap_or("?").to_string())
        .collect();

    Json(serde_json::json!({
        "tokens": tokens,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "layers": layers,
    }))
    .into_response()
}

// ── Previsão do próximo token (forward pass, logits crus) ────────────────

#[derive(Deserialize)]
struct PredictRequest {
    prompt: String,
    #[serde(default = "default_predict_top_k")]
    top_k: usize,
}
fn default_predict_top_k() -> usize {
    40
}
const MAX_PREDICT_TOP_K: usize = 100;

/// Roda um forward pass no prompt e devolve os **logits crus** do próximo
/// token na última posição, mais os `top_k` candidatos rotulados. O cliente
/// reaplica a temperatura no navegador (softmax sobre estes logits), então o
/// slider remodela a distribuição sem novo round-trip. O conjunto top-k é
/// invariante à temperatura (que é monotônica), então rotular por logit já
/// cobre qualquer posição do slider.
async fn predict(State(state): State<AppState>, Json(req): Json<PredictRequest>) -> Response {
    if req.prompt.len() > MAX_INPUT_TEXT_LEN {
        return err_response(413, format!("Prompt muito longo (máximo {} caracteres)", MAX_INPUT_TEXT_LEN));
    }
    let guard = state.model.lock().unwrap();
    let Some(loaded) = guard.as_ref() else {
        return err_response(503, "Nenhum modelo carregado — treine ou carregue um checkpoint primeiro");
    };

    let ids = match loaded.tokenizer.encode(&req.prompt) {
        Ok(ids) if !ids.is_empty() => ids,
        Ok(_) => return err_response(400, "Prompt vazio"),
        Err(e) => return err_response(400, format!("Erro ao tokenizar: {}", e)),
    };

    let block_size = loaded.model.block_size();
    let ids: Vec<usize> = if ids.len() > block_size {
        ids[ids.len() - block_size..].to_vec()
    } else {
        ids
    };
    let ids_i64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();

    let idx = match Tensor::from_slice(&ids_i64, &[1, ids.len()], &state.device) {
        Ok(t) => t,
        Err(e) => return err_response(500, format!("{}", e)),
    };

    // logits [1, T, vocab] → última posição = a "opinião" sobre o próximo token.
    let logits3: Vec<Vec<Vec<f32>>> = match loaded
        .model
        .forward(&idx, None)
        .map_err(|e| e.to_string())
        .and_then(|(l, _)| l.to_vec3::<f32>().map_err(|e| e.to_string()))
    {
        Ok(l) => l,
        Err(e) => return err_response(500, format!("Erro no forward pass: {}", e)),
    };
    let last_pos = ids.len() - 1;
    let logits: Vec<f32> = match logits3.first().and_then(|b| b.get(last_pos)) {
        Some(row) => row.clone(),
        None => return err_response(500, "Forma inesperada dos logits"),
    };
    let vocab_size = logits.len();

    let top_k = req.top_k.clamp(1, MAX_PREDICT_TOP_K).min(vocab_size.max(1));
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);
    let top: Vec<_> = indexed
        .into_iter()
        .map(|(id, logit)| {
            serde_json::json!({
                "id": id,
                "token": loaded.tokenizer.token_string(id).unwrap_or("?").to_string(),
                "logit": logit,
            })
        })
        .collect();

    let tokens: Vec<TokenInfo> = ids
        .iter()
        .map(|&id| TokenInfo {
            id,
            text: loaded.tokenizer.token_string(id).unwrap_or("?").to_string(),
        })
        .collect();

    Json(serde_json::json!({
        "tokens": tokens,
        "vocab_size": vocab_size,
        "position": last_pos,
        "logits": logits,
        "top": top,
    }))
    .into_response()
}

// ── Embeddings reais + projeção 2D (PCA por power iteration) ─────────────

#[derive(Deserialize)]
struct EmbeddingsRequest {
    text: String,
}

async fn embeddings(State(state): State<AppState>, Json(req): Json<EmbeddingsRequest>) -> Response {
    if req.text.len() > MAX_INPUT_TEXT_LEN {
        return err_response(413, format!("Texto muito longo (máximo {} caracteres)", MAX_INPUT_TEXT_LEN));
    }
    let guard = state.model.lock().unwrap();
    let Some(loaded) = guard.as_ref() else {
        return err_response(503, "Nenhum modelo carregado — treine ou carregue um checkpoint primeiro");
    };

    let ids = match loaded.tokenizer.encode(&req.text) {
        Ok(ids) if !ids.is_empty() => ids,
        Ok(_) => return err_response(400, "Texto vazio"),
        Err(e) => return err_response(400, format!("{}", e)),
    };

    let vectors: Vec<Vec<f32>> = match loaded
        .model
        .embedding_for_tokens(&ids)
        .map_err(|e| e.to_string())
        .and_then(|t| t.to_vec2().map_err(|e| e.to_string()))
    {
        Ok(v) => v,
        Err(e) => return err_response(500, format!("{}", e)),
    };

    let points = pca_2d(&vectors);
    let tokens: Vec<String> = ids
        .iter()
        .map(|&id| loaded.tokenizer.token_string(id).unwrap_or("?").to_string())
        .collect();

    let items: Vec<_> = tokens
        .iter()
        .zip(&points)
        .map(|(token, [x, y])| serde_json::json!({ "token": token, "x": x, "y": y }))
        .collect();

    // Similaridade de cosseno entre os tokens do prompt (T×T) — habilita o
    // hover-para-destacar-vizinhos no scatter.
    let normed: Vec<Vec<f32>> = vectors.iter().map(|v| l2_normalize(v)).collect();
    let similarity: Vec<Vec<f32>> = normed
        .iter()
        .map(|a| normed.iter().map(|b| dot(a, b)).collect())
        .collect();

    // Vizinhos mais próximos no vocabulário inteiro ("o que o modelo acha
    // parecido"). Calcula a matriz de embedding do vocab uma vez; se falhar,
    // devolve lista vazia (o resto da resposta ainda é útil).
    let neighbors = match loaded
        .model
        .embedding_for_tokens(&(0..loaded.tokenizer.vocab_size()).collect::<Vec<_>>())
        .map_err(|e| e.to_string())
        .and_then(|t| t.to_vec2::<f32>().map_err(|e| e.to_string()))
    {
        Ok(vocab_vecs) => {
            let vocab_normed: Vec<Vec<f32>> = vocab_vecs.iter().map(|v| l2_normalize(v)).collect();
            ids.iter()
                .zip(&normed)
                .map(|(&tid, tvec)| {
                    let mut sims: Vec<(usize, f32)> = vocab_normed
                        .iter()
                        .enumerate()
                        .filter(|(vid, _)| *vid != tid)
                        .map(|(vid, vvec)| (vid, dot(tvec, vvec)))
                        .collect();
                    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    sims.truncate(8);
                    let nearest: Vec<_> = sims
                        .into_iter()
                        .map(|(vid, sim)| {
                            serde_json::json!({
                                "id": vid,
                                "token": loaded.tokenizer.token_string(vid).unwrap_or("?").to_string(),
                                "sim": sim,
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "token": loaded.tokenizer.token_string(tid).unwrap_or("?").to_string(),
                        "nearest": nearest,
                    })
                })
                .collect::<Vec<_>>()
        }
        Err(_) => Vec::new(),
    };

    Json(serde_json::json!({
        "points": items,
        "similarity": similarity,
        "neighbors": neighbors,
    }))
    .into_response()
}

/// Projeção 2D via PCA (power iteration sobre a matriz de covariância) —
/// sem nenhuma dependência de álgebra linear extra e sem números aleatórios:
/// os dois eixos são os componentes de maior variância dos embeddings reais.
fn pca_2d(vectors: &[Vec<f32>]) -> Vec<[f32; 2]> {
    let n = vectors.len();
    if n == 0 {
        return vec![];
    }
    let d = vectors[0].len();
    if d == 0 {
        return vec![[0.0, 0.0]; n];
    }

    let mut mean = vec![0f32; d];
    for v in vectors {
        for i in 0..d {
            mean[i] += v[i];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f32;
    }
    let centered: Vec<Vec<f32>> = vectors
        .iter()
        .map(|v| v.iter().zip(&mean).map(|(a, b)| a - b).collect())
        .collect();

    let power_iterate = |exclude: Option<&[f32]>| -> Vec<f32> {
        let mut v = vec![1.0f32; d];
        for _ in 0..64 {
            let mut result = vec![0f32; d];
            for row in &centered {
                let dot: f32 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
                for i in 0..d {
                    result[i] += row[i] * dot;
                }
            }
            if let Some(prev) = exclude {
                let overlap: f32 = result.iter().zip(prev).map(|(a, b)| a * b).sum();
                for i in 0..d {
                    result[i] -= overlap * prev[i];
                }
            }
            let norm = result.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for r in result.iter_mut() {
                *r /= norm;
            }
            v = result;
        }
        v
    };

    let pc1 = power_iterate(None);
    let pc2 = power_iterate(Some(&pc1));

    centered
        .iter()
        .map(|row| {
            let x: f32 = row.iter().zip(&pc1).map(|(a, b)| a * b).sum();
            let y: f32 = row.iter().zip(&pc2).map(|(a, b)| a * b).sum();
            [x, y]
        })
        .collect()
}

/// Normaliza um vetor para norma L2 = 1 (base da similaridade de cosseno).
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / norm).collect()
}

/// Produto escalar entre dois vetores (cosseno, quando ambos normalizados).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ── Geração real via SSE ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct GenerateParams {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_generate_top_k")]
    top_k: usize,
}
fn default_max_tokens() -> usize {
    60
}
fn default_temperature() -> f32 {
    0.8
}
fn default_generate_top_k() -> usize {
    10
}
const MAX_GENERATE_TOP_K: usize = 20;

const MAX_GENERATE_TOKENS: usize = 500;
const MAX_TRAIN_EPOCHS: usize = 1000;

async fn generate_sse(State(state): State<AppState>, Query(params): Query<GenerateParams>) -> Response {
    if params.prompt.len() > MAX_INPUT_TEXT_LEN {
        return err_response(413, format!("Prompt muito longo (máximo {} caracteres)", MAX_INPUT_TEXT_LEN));
    }
    if state.model.lock().unwrap().is_none() {
        return err_response(503, "Nenhum modelo carregado — treine ou carregue um checkpoint primeiro");
    }
    let max_tokens = params.max_tokens.min(MAX_GENERATE_TOKENS);
    let top_k = params.top_k.min(MAX_GENERATE_TOP_K);

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let model_arc = state.model.clone();

    tokio::task::spawn_blocking(move || {
        let guard = model_arc.lock().unwrap();
        if let Some(loaded) = guard.as_ref() {
            let _ = loaded.model.generate_stream_with_probs(
                &params.prompt,
                max_tokens,
                &loaded.tokenizer,
                params.temperature,
                top_k,
                |id, text, top| {
                    // `top`: candidatos (id, prob) deste passo, já ordenados.
                    // `chosen` = id amostrado, mesmo quando não é o mais provável
                    // (mostra que amostragem ≠ argmax).
                    let top_json: Vec<_> = top
                        .iter()
                        .map(|(tid, p)| {
                            serde_json::json!({
                                "id": *tid,
                                "token": loaded.tokenizer.token_string(*tid).unwrap_or("?").to_string(),
                                "prob": *p,
                            })
                        })
                        .collect();
                    let _ = tx.send(
                        serde_json::json!({
                            "id": id,
                            "text": text,
                            "chosen": id,
                            "top": top_json,
                        })
                        .to_string(),
                    );
                },
            );
        }
        let _ = tx.send("[DONE]".to_string());
    });

    let event_stream = stream! {
        while let Some(msg) = rx.recv().await {
            if msg == "[DONE]" {
                yield Ok::<_, Infallible>(Event::default().event("done").data("done"));
                break;
            }
            yield Ok::<_, Infallible>(Event::default().data(msg));
        }
    };

    Sse::new(event_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ── Treinamento real, transmitido ao vivo via WebSocket ──────────────────

#[derive(Deserialize, Default)]
struct TrainStartRequest {
    #[serde(default = "default_epochs")]
    epochs: usize,
}
fn default_epochs() -> usize {
    30
}

async fn train_start(
    State(state): State<AppState>,
    body: Option<Json<TrainStartRequest>>,
) -> Response {
    if state.training_running.swap(true, Ordering::SeqCst) {
        return err_response(409, "Já existe um treinamento em andamento");
    }

    let epochs = body.map(|b| b.0.epochs).unwrap_or_else(default_epochs).min(MAX_TRAIN_EPOCHS).max(1);
    let corpus_path = state.corpus_path.clone();
    let device = state.device.clone();
    let models_dir = state.models_dir.clone();
    let tx = state.training_tx.clone();
    let running = state.training_running.clone();
    let recent = state.recent_events.clone();
    let model_slot = state.model.clone();

    tokio::task::spawn_blocking(move || {
        let recent_for_events = recent.clone();
        let tx_for_events = tx.clone();
        let result = run_training(&corpus_path, epochs, &device, &models_dir, move |ev| {
            let mut buf = recent_for_events.lock().unwrap();
            if buf.len() >= MAX_RECENT_EVENTS {
                buf.pop_front();
            }
            buf.push_back(ev.clone());
            drop(buf);
            let _ = tx_for_events.send(ev);
        });

        match result {
            Ok(checkpoint_path) => {
                if let Ok(loaded) = load_checkpoint(&checkpoint_path, &device) {
                    *model_slot.lock().unwrap() = Some(loaded);
                }
            }
            Err(e) => {
                eprintln!("⚠️  Treinamento via web falhou: {}", e);
            }
        }

        running.store(false, Ordering::SeqCst);
    });

    (axum::http::StatusCode::ACCEPTED, "Treinamento iniciado").into_response()
}

fn run_training(
    corpus_path: &PathBuf,
    epochs: usize,
    device: &Device,
    models_dir: &PathBuf,
    on_event: impl FnMut(TrainingEvent),
) -> AnyResult<String> {
    let text = std::fs::read_to_string(corpus_path)
        .map_err(|e| anyhow::anyhow!("Erro ao ler corpus: {}", e))?;

    let mut tokenizer = BPETokenizer::new(DEMO_VOCAB_SIZE).map_err(|e| anyhow::anyhow!("{}", e))?;
    tokenizer.train(&text).map_err(|e| anyhow::anyhow!("{}", e))?;
    let tokens = tokenizer.encode(&text).map_err(|e| anyhow::anyhow!("{}", e))?;

    let config = GPTConfig {
        vocab_size: tokenizer.vocab_size(),
        n_embd: DEFAULT_N_EMBD,
        n_head: DEFAULT_N_HEAD,
        n_layer: DEFAULT_N_LAYER,
        block_size: DEFAULT_BLOCK_SIZE,
        dropout: DEFAULT_DROPOUT,
    };
    let model = MiniGPT::new(config, device).map_err(|e| anyhow::anyhow!("{}", e))?;
    let mut trainer =
        Trainer::new(model, tokenizer, device.clone()).map_err(|e| anyhow::anyhow!("{}", e))?;

    trainer
        .train_with_callback(&tokens, epochs, on_event)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    std::fs::create_dir_all(models_dir)?;
    let checkpoint_path = models_dir.join("mini_gpt.safetensors");
    let checkpoint_path_str = checkpoint_path.to_string_lossy().to_string();
    trainer
        .save(&checkpoint_path_str)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(checkpoint_path_str)
}

async fn train_ws(State(state): State<AppState>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| handle_train_ws(socket, state))
}

async fn handle_train_ws(mut socket: WebSocket, state: AppState) {
    let backlog: Vec<TrainingEvent> = {
        let buf = state.recent_events.lock().unwrap();
        buf.iter().cloned().collect()
    };
    for ev in backlog {
        if let Ok(json) = serde_json::to_string(&ev) {
            if socket.send(Message::Text(json)).await.is_err() {
                return;
            }
        }
    }

    let mut rx = state.training_tx.subscribe();
    loop {
        tokio::select! {
            ev = rx.recv() => {
                match ev {
                    Ok(ev) => {
                        if let Ok(json) = serde_json::to_string(&ev) {
                            if socket.send(Message::Text(json)).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            msg = socket.recv() => {
                if msg.is_none() {
                    break;
                }
            }
        }
    }
}

async fn train_status(State(state): State<AppState>) -> Response {
    let running = state.training_running.load(Ordering::SeqCst);
    let recent: Vec<TrainingEvent> = state.recent_events.lock().unwrap().iter().cloned().collect();
    Json(serde_json::json!({ "running": running, "recent_events": recent })).into_response()
}

// ── Utilidades ─────────────────────────────────────────────────────────

fn err_response(status: u16, message: impl Into<String>) -> Response {
    let status = axum::http::StatusCode::from_u16(status).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(serde_json::json!({ "error": message.into() }))).into_response()
}
