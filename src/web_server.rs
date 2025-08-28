//! Servidor web para hospedar interativos educacionais
//! 
//! Este módulo implementa um servidor web usando Axum que serve:
//! - Página de índice com todos os interativos disponíveis
//! - Interativos HTML individuais para cada conceito do GPT
//! - Arquivos estáticos (CSS, JS, imagens)

use axum::{
    extract::Path,
    http::{StatusCode, Uri},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};
use crate::web_demo_integration::{
    IntegrationState, IntegrationConfig, 
    create_integration_router, initialize_default_modules
};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
};
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    services::ServeDir,
};
use tokio::fs;
use anyhow::Result;

/// Configuração do servidor web
#[derive(Debug, Clone)]
pub struct WebServerConfig {
    pub host: String,
    pub port: u16,
    pub interativos_dir: PathBuf,
    pub enable_integration: bool,
}

impl Default for WebServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            interativos_dir: PathBuf::from("interativos"),
            enable_integration: true,
        }
    }
}

/// Metadados de um interativo
#[derive(Debug, Clone)]
pub struct InteractiveMetadata {
    pub id: String,
    pub title: String,
    pub description: String,
    pub difficulty: String,
    pub category: String,
    pub file_path: String,
}

/// Servidor web para interativos educacionais
pub struct WebServer {
    config: WebServerConfig,
    interactives: HashMap<String, InteractiveMetadata>,
}

impl WebServer {
    /// Cria uma nova instância do servidor web
    pub fn new(config: WebServerConfig) -> Self {
        let mut interactives = HashMap::new();
        
        // Registra todos os interativos disponíveis
        interactives.insert("chunking".to_string(), InteractiveMetadata {
            id: "chunking".to_string(),
            title: "Text Chunking".to_string(),
            description: "Aprenda como dividir texto em pedaços menores para processamento".to_string(),
            difficulty: "Iniciante".to_string(),
            category: "Pré-processamento".to_string(),
            file_path: "chunking.html".to_string(),
        });
        
        interactives.insert("attention".to_string(), InteractiveMetadata {
            id: "attention".to_string(),
            title: "Mecanismo de Atenção".to_string(),
            description: "Visualize como o mecanismo de atenção funciona nos Transformers".to_string(),
            difficulty: "Intermediário".to_string(),
            category: "Arquitetura".to_string(),
            file_path: "attention.html".to_string(),
        });
        
        interactives.insert("tokenization".to_string(), InteractiveMetadata {
            id: "tokenization".to_string(),
            title: "Tokenização".to_string(),
            description: "Entenda como o texto é convertido em tokens para o modelo".to_string(),
            difficulty: "Iniciante".to_string(),
            category: "Pré-processamento".to_string(),
            file_path: "tokenization.html".to_string(),
        });
        
        interactives.insert("embeddings".to_string(), InteractiveMetadata {
            id: "embeddings".to_string(),
            title: "Embeddings".to_string(),
            description: "Explore como palavras são representadas como vetores".to_string(),
            difficulty: "Intermediário".to_string(),
            category: "Representação".to_string(),
            file_path: "embeddings.html".to_string(),
        });
        
        interactives.insert("transformer".to_string(), InteractiveMetadata {
            id: "transformer".to_string(),
            title: "Arquitetura Transformer".to_string(),
            description: "Visualize a arquitetura completa do Transformer".to_string(),
            difficulty: "Avançado".to_string(),
            category: "Arquitetura".to_string(),
            file_path: "transformer.html".to_string(),
        });
        
        interactives.insert("training".to_string(), InteractiveMetadata {
            id: "training".to_string(),
            title: "Processo de Treinamento".to_string(),
            description: "Acompanhe como o modelo aprende durante o treinamento".to_string(),
            difficulty: "Avançado".to_string(),
            category: "Treinamento".to_string(),
            file_path: "training.html".to_string(),
        });
        
        interactives.insert("inference".to_string(), InteractiveMetadata {
            id: "inference".to_string(),
            title: "Processo de Inferência".to_string(),
            description: "Veja como o modelo gera texto passo a passo".to_string(),
            difficulty: "Intermediário".to_string(),
            category: "Inferência".to_string(),
            file_path: "inference.html".to_string(),
        });
        
        Self {
            config,
            interactives,
        }
    }
    
    /// Inicia o servidor web
    pub async fn start(&self) -> Result<()> {
        let app = self.create_router().await?;
        
        let addr = SocketAddr::new(
            self.config.host.parse()?,
            self.config.port,
        );
        
        println!("🚀 Servidor de interativos iniciado em http://{}:{}", 
                 self.config.host, self.config.port);
        println!("📚 Acesse a página principal para ver todos os interativos disponíveis");
        
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        
        Ok(())
    }
    
    /// Cria o router do Axum com todas as rotas
    async fn create_router(&self) -> Result<Router> {
        let interactives = self.interactives.clone();
        let interativos_dir = self.config.interativos_dir.clone();
        
        let mut app = Router::new()
            // Página principal (índice)
            .route("/", get(move || index_handler(interactives.clone())))
            // Interativo específico
            .route("/interactive/:id", get(move |path| {
                interactive_handler(path, interativos_dir.clone())
            }))
            // Servir arquivos estáticos do diretório interativos
            .nest_service("/static", ServeDir::new(&self.config.interativos_dir))
            // Servir arquivos JavaScript específicos
            .nest_service("/js", ServeDir::new(self.config.interativos_dir.join("js")));
        
        // Integrar WebSocket e API REST se habilitado
        if self.config.enable_integration {
            let integration_state = IntegrationState::new();
            initialize_default_modules(&integration_state);
            
            let integration_router = create_integration_router(integration_state);
            app = app.merge(integration_router);
        }
        
        // Middleware
        app = app.layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
        );
        
        Ok(app)
    }
}

/// Handler para a página de índice
async fn index_handler(interactives: HashMap<String, InteractiveMetadata>) -> impl IntoResponse {
    let mut categories: HashMap<String, Vec<&InteractiveMetadata>> = HashMap::new();
    
    // Agrupa interativos por categoria
    for interactive in interactives.values() {
        categories
            .entry(interactive.category.clone())
            .or_insert_with(Vec::new)
            .push(interactive);
    }
    
    let mut html = format!(r#"
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mini-GPT-Rust - Interativos Educacionais</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="/static/js/integration.js"></script>
    <script src="/static/js/advanced-visualizations.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; }}
        .card-hover {{ transition: all 0.3s ease; }}
        .card-hover:hover {{ transform: translateY(-4px); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); }}
        .difficulty-badge {{
            @apply px-3 py-1 rounded-full text-xs font-medium;
        }}
        .difficulty-iniciante {{ @apply bg-green-100 text-green-800; }}
        .difficulty-intermediario {{ @apply bg-yellow-100 text-yellow-800; }}
        .difficulty-avancado {{ @apply bg-red-100 text-red-800; }}
    </style>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="text-center mb-12">
            <h1 class="text-5xl font-bold text-gray-800 mb-4">🤖 Mini-GPT-Rust</h1>
            <p class="text-xl text-gray-600 mb-2">Interativos Educacionais para Entender GPT</p>
            <p class="text-gray-500">Explore os conceitos fundamentais por trás dos modelos de linguagem</p>
        </header>
        
        <!-- Stats -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div class="bg-white rounded-xl p-6 text-center shadow-lg">
                <div class="text-3xl font-bold text-blue-600 mb-2">{}</div>
                <div class="text-gray-600">Interativos Disponíveis</div>
            </div>
            <div class="bg-white rounded-xl p-6 text-center shadow-lg">
                <div class="text-3xl font-bold text-green-600 mb-2">{}</div>
                <div class="text-gray-600">Categorias</div>
            </div>
            <div class="bg-white rounded-xl p-6 text-center shadow-lg">
                <div class="text-3xl font-bold text-purple-600 mb-2">100%</div>
                <div class="text-gray-600">Gratuito & Open Source</div>
            </div>
        </div>
"#, interactives.len(), categories.len());
    
    // Adiciona seções por categoria
    for (category, items) in categories {
        html.push_str(&format!(r#"
        <!-- Categoria: {} -->
        <section class="mb-12">
            <h2 class="text-3xl font-semibold text-gray-800 mb-6 border-b-2 border-blue-200 pb-2">{}</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
"#, category, category));
        
        for item in items {
            let difficulty_class = match item.difficulty.to_lowercase().as_str() {
                "iniciante" => "difficulty-iniciante",
                "intermediário" | "intermediario" => "difficulty-intermediario",
                "avançado" | "avancado" => "difficulty-avancado",
                _ => "difficulty-iniciante",
            };
            
            html.push_str(&format!(r#"
                <div class="bg-white rounded-xl p-6 shadow-lg card-hover">
                    <div class="flex justify-between items-start mb-4">
                        <h3 class="text-xl font-semibold text-gray-800">{}</h3>
                        <span class="difficulty-badge {}">{}</span>
                    </div>
                    <p class="text-gray-600 mb-4">{}</p>
                    <a href="/interactive/{}" class="inline-block bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors font-medium">
                        Explorar →
                    </a>
                </div>
"#, item.title, difficulty_class, item.difficulty, item.description, item.id));
        }
        
        html.push_str("            </div>\n        </section>\n");
    }
    
    html.push_str(r#"
        <!-- Visualizações Avançadas -->
        <section class="mt-16">
            <h2 class="text-3xl font-bold text-gray-800 mb-8 text-center">📊 Visualizações em Tempo Real</h2>
            
            <!-- Painel de Controle de Demonstrações -->
            <div class="bg-white rounded-xl p-6 shadow-lg mb-8">
                <h3 class="text-xl font-semibold text-gray-800 mb-4">🎮 Controle de Demonstrações</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <button onclick="runDemo('tokenizer')" class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded transition-colors">
                        🔤 Tokenizer
                    </button>
                    <button onclick="runDemo('attention')" class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded transition-colors">
                        🎯 Atenção
                    </button>
                    <button onclick="runDemo('embeddings')" class="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded transition-colors">
                        📊 Embeddings
                    </button>
                    <button onclick="runDemo('transformer')" class="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded transition-colors">
                        🤖 Transformer
                    </button>
                    <button onclick="runDemo('training')" class="bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded transition-colors">
                        🎓 Treinamento
                    </button>
                    <button onclick="runDemo('inference')" class="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded transition-colors">
                        🚀 Inferência
                    </button>
                    <button onclick="runDemo('chunking')" class="bg-pink-500 hover:bg-pink-600 text-white px-4 py-2 rounded transition-colors">
                        📝 Chunking
                    </button>
                    <button onclick="loadDemoHistory()" class="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded transition-colors">
                        📈 Histórico
                    </button>
                </div>
            </div>
            
            <!-- Métricas em Tempo Real -->
            <div class="bg-white rounded-xl p-6 shadow-lg mb-8">
                <h3 class="text-xl font-semibold text-gray-800 mb-4">⚡ Métricas em Tempo Real</h3>
                <div id="performance-metrics" class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div class="bg-blue-50 p-3 rounded-lg">⚡ Latência: <span id="latency-metric">--ms</span></div>
                    <div class="bg-green-50 p-3 rounded-lg">🧠 CPU: <span id="cpu-metric">--%</span></div>
                    <div class="bg-yellow-50 p-3 rounded-lg">💾 Memória: <span id="memory-metric">--MB</span></div>
                    <div class="bg-purple-50 p-3 rounded-lg">🚀 Throughput: <span id="throughput-metric">--/s</span></div>
                </div>
            </div>
            
            <!-- Visualizações Avançadas -->
            <div class="bg-white rounded-xl p-6 shadow-lg mb-8">
                <h3 class="text-xl font-semibold text-gray-800 mb-4">📊 Visualizações Avançadas</h3>
                
                <div class="flex flex-wrap gap-2 mb-6">
                    <button class="viz-tab bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors" onclick="showVisualization('attention')">🎯 Atenção</button>
                    <button class="viz-tab bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors" onclick="showVisualization('embeddings')">🌌 Embeddings</button>
                    <button class="viz-tab bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors" onclick="showVisualization('architecture')">🏗️ Arquitetura</button>
                    <button class="viz-tab bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors" onclick="showVisualization('flow')">🌊 Fluxo</button>
                    <button class="viz-tab bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors" onclick="showVisualization('performance')">📈 Performance</button>
                    <button class="viz-tab bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors" onclick="showVisualization('loss')">🏔️ Loss</button>
                </div>

                <div class="viz-content">
                    <div id="viz-attention" class="viz-panel">
                        <h4 class="text-lg font-semibold text-gray-800 mb-3">🎯 Mapa de Calor de Atenção</h4>
                        <div id="attention-heatmap" class="w-full h-64 border border-gray-200 rounded-lg bg-gray-50 flex items-center justify-center">
                            <span class="text-gray-500">Conecte-se para ver visualizações em tempo real</span>
                        </div>
                        <p class="text-sm text-gray-600 mt-2">Visualização das pontuações de atenção entre tokens. Cores mais escuras indicam maior atenção.</p>
                    </div>

                    <div id="viz-embeddings" class="viz-panel hidden">
                        <h4 class="text-lg font-semibold text-gray-800 mb-3">🌌 Espaço de Embeddings</h4>
                        <div id="embedding-space" class="w-full h-64 border border-gray-200 rounded-lg bg-gray-50 flex items-center justify-center">
                            <span class="text-gray-500">Projeção 2D do espaço de embeddings</span>
                        </div>
                        <p class="text-sm text-gray-600 mt-2">Projeção 2D do espaço de embeddings de alta dimensão. Palavras similares aparecem próximas.</p>
                    </div>

                    <div id="viz-architecture" class="viz-panel hidden">
                        <h4 class="text-lg font-semibold text-gray-800 mb-3">🏗️ Arquitetura do Transformer</h4>
                        <div id="transformer-architecture" class="w-full h-64 border border-gray-200 rounded-lg bg-gray-50 flex items-center justify-center">
                            <span class="text-gray-500">Diagrama da arquitetura do modelo</span>
                        </div>
                        <p class="text-sm text-gray-600 mt-2">Visualização das camadas do modelo Transformer. Clique nas camadas para destacá-las.</p>
                    </div>

                    <div id="viz-flow" class="viz-panel hidden">
                        <h4 class="text-lg font-semibold text-gray-800 mb-3">🌊 Fluxo de Tokens</h4>
                        <div id="token-flow" class="w-full h-64 border border-gray-200 rounded-lg bg-gray-50 flex items-center justify-center">
                            <span class="text-gray-500">Animação do processamento de tokens</span>
                        </div>
                        <p class="text-sm text-gray-600 mt-2">Animação do processamento sequencial de tokens através do modelo.</p>
                    </div>

                    <div id="viz-performance" class="viz-panel hidden">
                        <h4 class="text-lg font-semibold text-gray-800 mb-3">📈 Métricas de Performance</h4>
                        <canvas id="performance-metrics-chart" class="w-full h-64 border border-gray-200 rounded-lg"></canvas>
                        <p class="text-sm text-gray-600 mt-2">Gráfico em tempo real das métricas de performance do sistema.</p>
                    </div>

                    <div id="viz-loss" class="viz-panel hidden">
                        <h4 class="text-lg font-semibold text-gray-800 mb-3">🏔️ Paisagem de Loss</h4>
                        <div id="loss-landscape" class="w-full h-64 border border-gray-200 rounded-lg bg-gray-50 flex items-center justify-center">
                            <span class="text-gray-500">Evolução da função de loss</span>
                        </div>
                        <p class="text-sm text-gray-600 mt-2">Evolução da função de loss durante o treinamento.</p>
                    </div>
                </div>
            </div>
            
            <!-- Controles de Parâmetros Dinâmicos -->
            <div class="bg-white rounded-xl p-6 shadow-lg mb-8">
                <h3 class="text-xl font-semibold text-gray-800 mb-4">⚙️ Controles de Parâmetros</h3>
                
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <!-- Parâmetros do Modelo -->
                    <div class="parameter-group">
                        <h4 class="text-lg font-medium text-gray-700 mb-3">🧠 Modelo</h4>
                        
                        <div class="parameter-control mb-4">
                            <label class="block text-sm font-medium text-gray-600 mb-2">Learning Rate</label>
                            <div class="flex items-center space-x-3">
                                <input type="range" id="learning-rate" min="0.0001" max="0.1" step="0.0001" value="0.001" 
                                       class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                       onchange="updateParameter('learning_rate', this.value)">
                                <span id="learning-rate-value" class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">0.001</span>
                            </div>
                        </div>
                        
                        <div class="parameter-control mb-4">
                            <label class="block text-sm font-medium text-gray-600 mb-2">Batch Size</label>
                            <div class="flex items-center space-x-3">
                                <input type="range" id="batch-size" min="1" max="128" step="1" value="32" 
                                       class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                       onchange="updateParameter('batch_size', this.value)">
                                <span id="batch-size-value" class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">32</span>
                            </div>
                        </div>
                        
                        <div class="parameter-control mb-4">
                            <label class="block text-sm font-medium text-gray-600 mb-2">Dropout Rate</label>
                            <div class="flex items-center space-x-3">
                                <input type="range" id="dropout-rate" min="0" max="0.5" step="0.01" value="0.1" 
                                       class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                       onchange="updateParameter('dropout_rate', this.value)">
                                <span id="dropout-rate-value" class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">0.1</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Parâmetros de Atenção -->
                    <div class="parameter-group">
                        <h4 class="text-lg font-medium text-gray-700 mb-3">🎯 Atenção</h4>
                        
                        <div class="parameter-control mb-4">
                            <label class="block text-sm font-medium text-gray-600 mb-2">Num Heads</label>
                            <div class="flex items-center space-x-3">
                                <input type="range" id="num-heads" min="1" max="16" step="1" value="8" 
                                       class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                       onchange="updateParameter('num_heads', this.value)">
                                <span id="num-heads-value" class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">8</span>
                            </div>
                        </div>
                        
                        <div class="parameter-control mb-4">
                            <label class="block text-sm font-medium text-gray-600 mb-2">Head Dim</label>
                            <div class="flex items-center space-x-3">
                                <input type="range" id="head-dim" min="32" max="128" step="8" value="64" 
                                       class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                       onchange="updateParameter('head_dim', this.value)">
                                <span id="head-dim-value" class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">64</span>
                            </div>
                        </div>
                        
                        <div class="parameter-control mb-4">
                            <label class="block text-sm font-medium text-gray-600 mb-2">Temperature</label>
                            <div class="flex items-center space-x-3">
                                <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="1.0" 
                                       class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                       onchange="updateParameter('temperature', this.value)">
                                <span id="temperature-value" class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">1.0</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Parâmetros de Geração -->
                    <div class="parameter-group">
                        <h4 class="text-lg font-medium text-gray-700 mb-3">📝 Geração</h4>
                        
                        <div class="parameter-control mb-4">
                            <label class="block text-sm font-medium text-gray-600 mb-2">Max Length</label>
                            <div class="flex items-center space-x-3">
                                <input type="range" id="max-length" min="10" max="1000" step="10" value="100" 
                                       class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                       onchange="updateParameter('max_length', this.value)">
                                <span id="max-length-value" class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">100</span>
                            </div>
                        </div>
                        
                        <div class="parameter-control mb-4">
                            <label class="block text-sm font-medium text-gray-600 mb-2">Top-K</label>
                            <div class="flex items-center space-x-3">
                                <input type="range" id="top-k" min="1" max="100" step="1" value="50" 
                                       class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                       onchange="updateParameter('top_k', this.value)">
                                <span id="top-k-value" class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">50</span>
                            </div>
                        </div>
                        
                        <div class="parameter-control mb-4">
                            <label class="block text-sm font-medium text-gray-600 mb-2">Top-P</label>
                            <div class="flex items-center space-x-3">
                                <input type="range" id="top-p" min="0.1" max="1.0" step="0.05" value="0.9" 
                                       class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                       onchange="updateParameter('top_p', this.value)">
                                <span id="top-p-value" class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">0.9</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Botões de Controle -->
                <div class="flex flex-wrap gap-3 mt-6 pt-4 border-t border-gray-200">
                    <button onclick="resetParameters()" 
                            class="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors">
                        🔄 Reset
                    </button>
                    <button onclick="saveParameterPreset()" 
                            class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors">
                        💾 Salvar Preset
                    </button>
                    <button onclick="loadParameterPreset()" 
                            class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors">
                        📂 Carregar Preset
                    </button>
                    <button onclick="exportParameters()" 
                            class="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg transition-colors">
                        📤 Exportar
                    </button>
                </div>
                
                <!-- Status de Sincronização -->
                <div class="mt-4 p-3 bg-gray-50 rounded-lg">
                    <div class="flex items-center justify-between">
                        <span class="text-sm text-gray-600">Status da Sincronização:</span>
                        <span id="sync-status" class="text-sm font-medium text-green-600">🟢 Conectado</span>
                    </div>
                    <div class="text-xs text-gray-500 mt-1">
                        Última atualização: <span id="last-sync-time">--:--:--</span>
                    </div>
                </div>
            </div>
            
            <!-- Seção de Monitoramento de Performance -->
            <div class="bg-white rounded-xl p-6 shadow-lg mb-8">
                <h3 class="text-xl font-semibold text-gray-800 mb-6">📊 Monitoramento de Performance em Tempo Real</h3>
                
                <!-- Métricas Principais -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div class="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-lg">
                        <div class="text-sm text-blue-600 font-medium">CPU Usage</div>
                        <div id="cpu-usage" class="text-2xl font-bold text-blue-800">--</div>
                        <div class="text-xs text-blue-500">Processamento</div>
                    </div>
                    <div class="bg-gradient-to-r from-green-50 to-green-100 p-4 rounded-lg">
                        <div class="text-sm text-green-600 font-medium">Memory</div>
                        <div id="memory-usage" class="text-2xl font-bold text-green-800">--</div>
                        <div class="text-xs text-green-500">RAM utilizada</div>
                    </div>
                    <div class="bg-gradient-to-r from-purple-50 to-purple-100 p-4 rounded-lg">
                        <div class="text-sm text-purple-600 font-medium">Latência</div>
                        <div id="latency" class="text-2xl font-bold text-purple-800">--</div>
                        <div class="text-xs text-purple-500">Tempo resposta</div>
                    </div>
                    <div class="bg-gradient-to-r from-orange-50 to-orange-100 p-4 rounded-lg">
                        <div class="text-sm text-orange-600 font-medium">Throughput</div>
                        <div id="throughput" class="text-2xl font-bold text-orange-800">--</div>
                        <div class="text-xs text-orange-500">Tokens/seg</div>
                    </div>
                </div>
                
                <!-- Gráficos de Performance -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div>
                        <h4 class="text-lg font-semibold text-gray-700 mb-3">📈 Histórico de CPU/Memória</h4>
                        <canvas id="system-metrics-chart" class="w-full h-48 border border-gray-200 rounded-lg"></canvas>
                    </div>
                    <div>
                        <h4 class="text-lg font-semibold text-gray-700 mb-3">⚡ Latência e Throughput</h4>
                        <canvas id="performance-chart" class="w-full h-48 border border-gray-200 rounded-lg"></canvas>
                    </div>
                </div>
                
                <!-- Controles de Monitoramento -->
                <div class="flex items-center justify-between mt-6 pt-4 border-t border-gray-200">
                    <div class="flex items-center space-x-4">
                        <button id="monitoring-toggle" onclick="toggleMonitoring()" 
                                class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors">
                            ▶️ Iniciar Monitoramento
                        </button>
                        <button onclick="resetMetrics()" 
                                class="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors">
                            🔄 Reset Métricas
                        </button>
                        <button onclick="exportMetrics()" 
                                class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors">
                            📊 Exportar Dados
                        </button>
                    </div>
                    <div class="text-sm text-gray-600">
                        Status: <span id="monitoring-status" class="font-medium text-gray-500">Parado</span>
                    </div>
                </div>
                
                <!-- Alertas de Performance -->
                <div id="performance-alerts" class="mt-4 space-y-2 hidden">
                    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                        <div class="flex items-center">
                            <span class="text-yellow-600 mr-2">⚠️</span>
                            <span class="text-sm text-yellow-800">Alertas de performance aparecerão aqui</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div class="bg-white rounded-xl p-6 shadow-lg">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4">⚡ Performance Metrics</h3>
                    <canvas id="performance-chart" width="400" height="200"></canvas>
                </div>
                <div class="bg-white rounded-xl p-6 shadow-lg">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4">🧠 Attention Heatmap</h3>
                    <div id="attention-visualization" class="w-full h-48 border border-gray-200 rounded-lg"></div>
                </div>
            </div>
        </section>
        
        <!-- Footer -->
        <footer class="text-center mt-16 py-8 border-t border-gray-200">
            <p class="text-gray-600 mb-2">Desenvolvido com ❤️ usando Rust + Axum</p>
            <p class="text-sm text-gray-500">Projeto educacional open source para aprender sobre GPT e Transformers</p>
        </footer>
    </div>
    
    <!-- Scripts de Integração -->
    <script src="/js/integration.js"></script>
    <script>
        // Inicializa visualizações quando a página carrega
        document.addEventListener('DOMContentLoaded', function() {
            // Chart.js para métricas de performance
            const ctx = document.getElementById('performance-chart');
            if (ctx) {
                window.performanceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Latência (ms)',
                            data: [],
                            borderColor: 'rgb(59, 130, 246)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.4
                        }, {
                            label: 'Memória (MB)',
                            data: [],
                            borderColor: 'rgb(16, 185, 129)',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        },
                        plugins: {
                            legend: {
                                position: 'top'
                            }
                        }
                    }
                });
            }
            
            // Funções globais para demonstrações
            window.runDemo = function(module) {
                const button = document.querySelector(`button[onclick="runDemo('${module}')"]`);
                if (button) {
                    button.disabled = true;
                    button.textContent = 'Executando...';
                }

                fetch(`/api/v1/demo/${module}/run`, {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Demo executado:', data);
                    if (data.success) {
                        displayDemoResult(module, data.result);
                        updatePerformanceMetrics(data.result.metrics);
                    }
                })
                .catch(error => {
                    console.error('Erro ao executar demo:', error);
                    alert('Erro ao executar demonstração: ' + error.message);
                })
                .finally(() => {
                    if (button) {
                        button.disabled = false;
                        button.textContent = getButtonText(module);
                    }
                });
                
                // Atualizar visualizações com dados da demonstração
                if (data && data.success) {
                    updateVisualizationsWithDemoData({
                        module: module,
                        execution_time: data.result?.execution_time,
                        metrics: data.result?.metrics
                    });
                }
            };

            window.loadDemoHistory = function() {
                fetch('/api/v1/demo/history')
                .then(response => response.json())
                .then(data => {
                    console.log('Histórico carregado:', data);
                    if (data.success) {
                        displayDemoHistory(data.history);
                    }
                })
                .catch(error => {
                    console.error('Erro ao carregar histórico:', error);
                });
            };

            // Função para alternar entre visualizações
            window.showVisualization = function(vizType) {
                // Remover classe ativa de todas as abas
                document.querySelectorAll('.viz-tab').forEach(tab => {
                    tab.classList.remove('bg-blue-500', 'text-white');
                    tab.classList.add('bg-gray-200', 'text-gray-700');
                });
                
                // Ocultar todos os painéis
                document.querySelectorAll('.viz-panel').forEach(panel => {
                    panel.classList.add('hidden');
                });
                
                // Ativar aba selecionada
                event.target.classList.remove('bg-gray-200', 'text-gray-700');
                event.target.classList.add('bg-blue-500', 'text-white');
                
                // Mostrar painel selecionado
                const panel = document.getElementById(`viz-${vizType}`);
                if (panel) {
                    panel.classList.remove('hidden');
                }
                
                // Inicializar visualização se necessário
                if (window.advancedViz) {
                    switch(vizType) {
                        case 'attention':
                            console.log('🎯 Mostrando mapa de atenção');
                            break;
                        case 'embeddings':
                            console.log('🌌 Mostrando espaço de embeddings');
                            break;
                        case 'architecture':
                            console.log('🏗️ Mostrando arquitetura do transformer');
                            break;
                        case 'flow':
                            console.log('🌊 Mostrando fluxo de tokens');
                            break;
                        case 'performance':
                            console.log('📈 Mostrando métricas de performance');
                            break;
                        case 'loss':
                            console.log('🏔️ Mostrando paisagem de loss');
                            break;
                    }
                }
            };

            // Função para atualizar visualizações com dados de demonstração
            function updateVisualizationsWithDemoData(demoResult) {
                if (!window.advancedViz) return;
                
                // Simular dados baseados no tipo de demonstração
                const vizData = {
                    attention: demoResult.module === 'attention' ? generateMockAttentionData() : null,
                    embeddings: demoResult.module === 'embeddings' ? generateMockEmbeddingData() : null,
                    metrics: {
                        latency: demoResult.execution_time || 0,
                        cpu_usage: Math.random() * 100,
                        memory_usage: Math.random() * 1000
                    }
                };
                
                window.advancedViz.updateAll(vizData);
            }

            // Funções auxiliares para gerar dados mock
            function generateMockAttentionData() {
                const size = 7;
                const data = [];
                for (let i = 0; i < size; i++) {
                    for (let j = 0; j < size; j++) {
                        data.push({
                            from: i,
                            to: j,
                            value: Math.random() * 0.8 + 0.1
                        });
                    }
                }
                return data;
            }

            function generateMockEmbeddingData() {
                const embeddings = [];
                const words = ['gato', 'casa', 'livro', 'correr', 'grande'];
                
                words.forEach((word, i) => {
                    embeddings.push({
                        x: (Math.random() - 0.5) * 10,
                        y: (Math.random() - 0.5) * 10,
                        word: word,
                        cluster: Math.floor(i / 2)
                    });
                });
                
                return embeddings;
            }

            function getButtonText(module) {
                const buttonTexts = {
                    'tokenizer': '🔤 Tokenizer',
                    'attention': '🎯 Atenção',
                    'embeddings': '📊 Embeddings',
                    'transformer': '🤖 Transformer',
                    'training': '🎓 Treinamento',
                    'inference': '🚀 Inferência',
                    'chunking': '📝 Chunking'
                };
                return buttonTexts[module] || module;
            }

            function displayDemoResult(module, result) {
                const resultContainer = document.getElementById('demo-results') || createDemoResultsContainer();
                
                const resultDiv = document.createElement('div');
                resultDiv.className = 'demo-result bg-gray-50 p-4 rounded-lg mb-4 border-l-4 border-blue-500';
                resultDiv.innerHTML = `
                    <h4 class="text-lg font-semibold mb-2 text-gray-800">🎯 ${module.toUpperCase()}</h4>
                    <div class="result-metrics flex flex-wrap gap-2 mb-3 text-sm">
                        <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded">⏱️ ${result.execution_time}ms</span>
                        <span class="bg-green-100 text-green-800 px-2 py-1 rounded">🧠 ${result.metrics?.cpu_usage?.toFixed(1) || 'N/A'}%</span>
                        <span class="bg-yellow-100 text-yellow-800 px-2 py-1 rounded">💾 ${result.metrics?.memory_usage?.toFixed(1) || 'N/A'}MB</span>
                        <span class="bg-purple-100 text-purple-800 px-2 py-1 rounded">🚀 ${result.metrics?.throughput?.toFixed(1) || 'N/A'}/s</span>
                    </div>
                    <div class="result-output bg-gray-800 text-green-400 p-3 rounded text-sm overflow-auto max-h-32">
                        <pre>${result.output || 'Sem saída'}</pre>
                    </div>
                `;
                
                resultContainer.appendChild(resultDiv);
                
                // Remove resultados antigos (mantém apenas os 3 mais recentes)
                const results = resultContainer.querySelectorAll('.demo-result');
                if (results.length > 3) {
                    results[0].remove();
                }
            }

            function createDemoResultsContainer() {
                const container = document.createElement('div');
                container.id = 'demo-results';
                container.className = 'fixed bottom-4 right-4 max-w-md max-h-96 overflow-y-auto bg-white shadow-xl rounded-lg p-4 z-50 border';
                container.innerHTML = '<h3 class="text-lg font-semibold mb-3 text-gray-800">📊 Resultados</h3>';
                
                document.body.appendChild(container);
                return container;
            }

            function updatePerformanceMetrics(metrics) {
                if (metrics) {
                    document.getElementById('latency-metric').textContent = `${metrics.latency?.toFixed(1) || 'N/A'}ms`;
                    document.getElementById('cpu-metric').textContent = `${metrics.cpu_usage?.toFixed(1) || 'N/A'}%`;
                    document.getElementById('memory-metric').textContent = `${metrics.memory_usage?.toFixed(1) || 'N/A'}MB`;
                    document.getElementById('throughput-metric').textContent = `${metrics.throughput?.toFixed(1) || 'N/A'}/s`;
                    
                    // Atualiza gráficos
                    updateCharts(metrics);
                }
            }

            function updateCharts(metrics) {
                const now = new Date().toLocaleTimeString();
                
                // Atualiza gráfico de performance
                if (window.performanceChart && metrics.latency !== undefined) {
                    window.performanceChart.data.labels.push(now);
                    window.performanceChart.data.datasets[0].data.push(metrics.latency);
                    window.performanceChart.data.datasets[1].data.push(metrics.memory_usage || 0);
                    
                    if (window.performanceChart.data.labels.length > 20) {
                        window.performanceChart.data.labels.shift();
                        window.performanceChart.data.datasets[0].data.shift();
                        window.performanceChart.data.datasets[1].data.shift();
                    }
                    
                    window.performanceChart.update('none');
                }
            }

            function displayDemoHistory(history) {
                console.log('Histórico de demonstrações:', history);
                alert(`Histórico carregado: ${history.length} registros`);
                
                // Atualizar gráficos com dados históricos
                if (history.length > 0) {
                    const latestMetrics = history[history.length - 1].metrics;
                    if (latestMetrics) {
                        updatePerformanceMetrics(latestMetrics);
                    }
                }
            }
            
            // Sistema de Gerenciamento de Estado Global
            class StateManager {
                constructor() {
                    this.state = {
                        parameters: {
                            learning_rate: 0.001,
                            batch_size: 32,
                            dropout_rate: 0.1,
                            num_heads: 8,
                            head_dim: 64,
                            temperature: 1.0,
                            max_length: 100,
                            top_k: 50,
                            top_p: 0.9
                        },
                        demo: {
                            isRunning: false,
                            currentModule: null,
                            lastResults: null,
                            executionHistory: []
                        },
                        visualizations: {
                            activeTab: 'attention',
                            data: {
                                attention: null,
                                embeddings: null,
                                architecture: null,
                                flow: null,
                                performance: null,
                                loss: null
                            }
                        },
                        websocket: {
                            connected: false,
                            lastMessage: null,
                            connectionAttempts: 0
                        },
                        sync: {
                            lastUpdate: null,
                            status: 'idle', // idle, syncing, error
                            pendingUpdates: []
                        }
                    };
                    this.listeners = new Map();
                    this.middleware = [];
                }

                // Subscrever mudanças de estado
                subscribe(path, callback) {
                    if (!this.listeners.has(path)) {
                        this.listeners.set(path, new Set());
                    }
                    this.listeners.get(path).add(callback);
                    
                    // Retornar função de unsubscribe
                    return () => {
                        const pathListeners = this.listeners.get(path);
                        if (pathListeners) {
                            pathListeners.delete(callback);
                        }
                    };
                }

                // Atualizar estado
                setState(path, value) {
                    const pathParts = path.split('.');
                    let current = this.state;
                    
                    // Navegar até o penúltimo nível
                    for (let i = 0; i < pathParts.length - 1; i++) {
                        if (!current[pathParts[i]]) {
                            current[pathParts[i]] = {};
                        }
                        current = current[pathParts[i]];
                    }
                    
                    const lastKey = pathParts[pathParts.length - 1];
                    const oldValue = current[lastKey];
                    current[lastKey] = value;
                    
                    // Executar middleware
                    this.middleware.forEach(fn => fn(path, value, oldValue));
                    
                    // Notificar listeners
                    this.notifyListeners(path, value, oldValue);
                }

                // Obter estado
                getState(path) {
                    if (!path) return this.state;
                    
                    const pathParts = path.split('.');
                    let current = this.state;
                    
                    for (const part of pathParts) {
                        if (current[part] === undefined) {
                            return undefined;
                        }
                        current = current[part];
                    }
                    
                    return current;
                }

                // Notificar listeners
                notifyListeners(path, newValue, oldValue) {
                    // Notificar listeners exatos
                    const exactListeners = this.listeners.get(path);
                    if (exactListeners) {
                        exactListeners.forEach(callback => {
                            try {
                                callback(newValue, oldValue, path);
                            } catch (error) {
                                console.error('Error in state listener:', error);
                            }
                        });
                    }
                    
                    // Notificar listeners de caminhos pais
                    const pathParts = path.split('.');
                    for (let i = 1; i <= pathParts.length; i++) {
                        const parentPath = pathParts.slice(0, i).join('.');
                        const parentListeners = this.listeners.get(parentPath);
                        if (parentListeners) {
                            parentListeners.forEach(callback => {
                                try {
                                    callback(this.getState(parentPath), undefined, parentPath);
                                } catch (error) {
                                    console.error('Error in parent state listener:', error);
                                }
                            });
                        }
                    }
                }

                // Adicionar middleware
                addMiddleware(fn) {
                    this.middleware.push(fn);
                }

                // Ações específicas para parâmetros
                updateParameter(name, value) {
                    this.setState(`parameters.${name}`, parseFloat(value));
                }

                // Ações específicas para demo
                setDemoRunning(isRunning, module = null) {
                    this.setState('demo.isRunning', isRunning);
                    if (module) {
                        this.setState('demo.currentModule', module);
                    }
                }

                addDemoResult(result) {
                    const history = this.getState('demo.executionHistory') || [];
                    history.push({
                        timestamp: new Date().toISOString(),
                        module: this.getState('demo.currentModule'),
                        result: result
                    });
                    this.setState('demo.lastResults', result);
                    this.setState('demo.executionHistory', history.slice(-10)); // Manter apenas últimos 10
                }

                // Ações específicas para visualizações
                setVisualizationData(type, data) {
                    this.setState(`visualizations.data.${type}`, data);
                }

                setActiveVisualization(tab) {
                    this.setState('visualizations.activeTab', tab);
                }

                // Ações específicas para WebSocket
                setWebSocketStatus(connected) {
                    this.setState('websocket.connected', connected);
                    if (connected) {
                        this.setState('websocket.connectionAttempts', 0);
                    } else {
                        const attempts = this.getState('websocket.connectionAttempts') || 0;
                        this.setState('websocket.connectionAttempts', attempts + 1);
                    }
                }

                addWebSocketMessage(message) {
                    this.setState('websocket.lastMessage', {
                        timestamp: new Date().toISOString(),
                        data: message
                    });
                }

                // Ações específicas para sincronização
                setSyncStatus(status) {
                    this.setState('sync.status', status);
                    this.setState('sync.lastUpdate', new Date().toISOString());
                }

                addPendingUpdate(update) {
                    const pending = this.getState('sync.pendingUpdates') || [];
                    pending.push(update);
                    this.setState('sync.pendingUpdates', pending);
                }

                clearPendingUpdates() {
                    this.setState('sync.pendingUpdates', []);
                }
            }

            // Instância global do gerenciador de estado
            const stateManager = new StateManager();
            window.stateManager = stateManager;
            
            // Estado global dos parâmetros (compatibilidade)
             let currentParameters = stateManager.getState('parameters');
             
             let parameterPresets = {
                 'default': { ...currentParameters },
                 'fast_training': {
                     learning_rate: 0.01,
                     batch_size: 64,
                     dropout_rate: 0.05,
                     num_heads: 4,
                     head_dim: 32,
                     temperature: 0.8,
                     max_length: 50,
                     top_k: 20,
                     top_p: 0.8
                 },
                 'high_quality': {
                     learning_rate: 0.0001,
                     batch_size: 16,
                     dropout_rate: 0.2,
                     num_heads: 16,
                     head_dim: 128,
                     temperature: 1.2,
                     max_length: 200,
                     top_k: 100,
                     top_p: 0.95
                 }
             };
             
             // Middleware para logging de mudanças de estado
             stateManager.addMiddleware((path, newValue, oldValue) => {
                 console.log(`🔄 Estado alterado: ${path}`, { old: oldValue, new: newValue });
             });
             
             // Middleware para sincronização automática de parâmetros
             stateManager.addMiddleware(async (path, newValue, oldValue) => {
                 if (path.startsWith('parameters.')) {
                     const paramName = path.split('.')[1];
                     try {
                         stateManager.setSyncStatus('syncing');
                         
                         const response = await fetch('/api/v1/parameters/update', {
                             method: 'POST',
                             headers: {
                                 'Content-Type': 'application/json'
                             },
                             body: JSON.stringify({
                                 module: 'model',
                                 parameters: { [paramName]: newValue }
                             })
                         });
                         
                         if (response.ok) {
                             stateManager.setSyncStatus('idle');
                             console.log(`✅ Parâmetro ${paramName} sincronizado: ${newValue}`);
                         } else {
                             stateManager.setSyncStatus('error');
                             console.error(`❌ Erro ao sincronizar ${paramName}`);
                         }
                     } catch (error) {
                         stateManager.setSyncStatus('error');
                         console.error('❌ Erro de conexão:', error);
                     }
                 }
             });
             
             // Subscrever mudanças de parâmetros para atualizar UI
             stateManager.subscribe('parameters', (parameters) => {
                 Object.keys(parameters).forEach(paramName => {
                     const value = parameters[paramName];
                     const inputId = paramName.replace('_', '-');
                     const input = document.getElementById(inputId);
                     const valueDisplay = document.getElementById(`${inputId}-value`);
                     
                     if (input && input.value != value) {
                         input.value = value;
                     }
                     if (valueDisplay) {
                         valueDisplay.textContent = value;
                     }
                 });
             });
             
             // Subscrever mudanças de status de sincronização
             stateManager.subscribe('sync.status', (status) => {
                 const statusMap = {
                     'idle': { text: '🟢 Sincronizado', class: 'text-green-600' },
                     'syncing': { text: '🔄 Sincronizando...', class: 'text-blue-600' },
                     'error': { text: '🔴 Erro', class: 'text-red-600' }
                 };
                 
                 const statusInfo = statusMap[status] || statusMap['error'];
                 updateSyncStatus(statusInfo.text, statusInfo.class);
                 
                 if (status === 'idle') {
                     updateLastSyncTime();
                 }
             });
             
             // Função para atualizar parâmetro individual (usando StateManager)
             window.updateParameter = function(paramName, value) {
                 stateManager.updateParameter(paramName, value);
             };
             
             // Função para resetar parâmetros (usando StateManager)
             window.resetParameters = function() {
                 const defaultParams = parameterPresets['default'];
                 
                 Object.keys(defaultParams).forEach(param => {
                     stateManager.updateParameter(param, defaultParams[param]);
                 });
                 
                 console.log('🔄 Parâmetros resetados para valores padrão');
             };
             
             // Função para salvar preset (usando StateManager)
             window.saveParameterPreset = function() {
                 const presetName = prompt('Nome do preset:', 'meu_preset');
                 if (presetName && presetName.trim()) {
                     const currentParams = stateManager.getState('parameters');
                     parameterPresets[presetName.trim()] = { ...currentParams };
                     
                     // Salvar no localStorage
                     localStorage.setItem('mini-gpt-presets', JSON.stringify(parameterPresets));
                     
                     console.log(`💾 Preset '${presetName}' salvo com sucesso`);
                     alert(`Preset '${presetName}' salvo!`);
                 }
             };
             
             // Função para carregar preset (usando StateManager)
             window.loadParameterPreset = function() {
                 const presetNames = Object.keys(parameterPresets);
                 const presetName = prompt(`Presets disponíveis: ${presetNames.join(', ')}\n\nDigite o nome do preset:`);
                 
                 if (presetName && parameterPresets[presetName]) {
                     const preset = parameterPresets[presetName];
                     
                     Object.keys(preset).forEach(param => {
                         stateManager.updateParameter(param, preset[param]);
                     });
                     
                     console.log(`📂 Preset '${presetName}' carregado`);
                     alert(`Preset '${presetName}' carregado!`);
                 } else if (presetName) {
                     alert('Preset não encontrado!');
                 }
             };
             
             // Função para exportar parâmetros (usando StateManager)
             window.exportParameters = function() {
                 const currentParams = stateManager.getState('parameters');
                 const exportData = {
                     timestamp: new Date().toISOString(),
                     parameters: currentParams,
                     presets: parameterPresets,
                     state: stateManager.getState()
                 };
                 
                 const dataStr = JSON.stringify(exportData, null, 2);
                 const dataBlob = new Blob([dataStr], { type: 'application/json' });
                 
                 const link = document.createElement('a');
                 link.href = URL.createObjectURL(dataBlob);
                 link.download = `mini-gpt-parameters-${new Date().toISOString().split('T')[0]}.json`;
                 link.click();
                 
                 console.log('📤 Parâmetros e estado exportados');
             };
             
             // Funções auxiliares para status
             function updateSyncStatus(status, className) {
                 const statusElement = document.getElementById('sync-status');
                 if (statusElement) {
                     statusElement.textContent = status;
                     statusElement.className = `text-sm font-medium ${className}`;
                 }
             }
             
             function updateLastSyncTime() {
                 const timeElement = document.getElementById('last-sync-time');
                 if (timeElement) {
                     timeElement.textContent = new Date().toLocaleTimeString();
                 }
             }
             
             // Integração com WebSocket usando StateManager
             function initializeWebSocketIntegration() {
                 if (typeof connectWebSocket === 'function') {
                     // Subscrever mudanças de conexão WebSocket
                     stateManager.subscribe('websocket.connected', (connected) => {
                         console.log(`🔌 WebSocket ${connected ? 'conectado' : 'desconectado'}`);
                         
                         // Atualizar indicadores visuais
                         const indicators = document.querySelectorAll('.websocket-indicator');
                         indicators.forEach(indicator => {
                             indicator.className = `websocket-indicator ${
                                 connected ? 'text-green-500' : 'text-red-500'
                             }`;
                             indicator.textContent = connected ? '🟢' : '🔴';
                         });
                     });
                     
                     // Interceptar mensagens WebSocket
                     const originalOnMessage = window.onWebSocketMessage;
                     window.onWebSocketMessage = function(data) {
                         stateManager.addWebSocketMessage(data);
                         
                         // Processar mensagens específicas
                         if (data.type === 'parameter_update') {
                             stateManager.updateParameter(data.parameter, data.value);
                         } else if (data.type === 'demo_result') {
                             stateManager.addDemoResult(data.result);
                             stateManager.setDemoRunning(false);
                         } else if (data.type === 'visualization_data') {
                             stateManager.setVisualizationData(data.visualization_type, data.data);
                         }
                         
                         // Chamar handler original se existir
                         if (originalOnMessage) {
                             originalOnMessage(data);
                         }
                     };
                     
                     connectWebSocket();
                 }
             }
             
             // Integração com visualizações usando StateManager
             function initializeVisualizationIntegration() {
                 // Subscrever mudanças de dados de visualização
                 Object.keys(stateManager.getState('visualizations.data')).forEach(vizType => {
                     stateManager.subscribe(`visualizations.data.${vizType}`, (data) => {
                         if (data && typeof window[`update${vizType.charAt(0).toUpperCase() + vizType.slice(1)}Visualization`] === 'function') {
                             window[`update${vizType.charAt(0).toUpperCase() + vizType.slice(1)}Visualization`](data);
                         }
                     });
                 });
                 
                 // Subscrever mudanças de aba ativa
                 stateManager.subscribe('visualizations.activeTab', (activeTab) => {
                     if (typeof showVisualization === 'function') {
                         showVisualization(activeTab);
                     }
                 });
                 
                 // Interceptar função showVisualization
                 const originalShowVisualization = window.showVisualization;
                 window.showVisualization = function(tab) {
                     stateManager.setActiveVisualization(tab);
                     if (originalShowVisualization) {
                         originalShowVisualization(tab);
                     }
                 };
             }
             
             // Integração com sistema de demonstração
             function initializeDemoIntegration() {
                 // Subscrever mudanças de estado de demo
                 stateManager.subscribe('demo.isRunning', (isRunning) => {
                     const runButtons = document.querySelectorAll('.run-demo-btn');
                     runButtons.forEach(btn => {
                         btn.disabled = isRunning;
                         btn.textContent = isRunning ? '🔄 Executando...' : '▶️ Executar';
                     });
                 });
                 
                 // Subscrever resultados de demo
                 stateManager.subscribe('demo.lastResults', (results) => {
                     if (results && typeof updateVisualizationsWithDemoData === 'function') {
                         updateVisualizationsWithDemoData(results);
                     }
                 });
                 
                 // Interceptar execução de demos
                 const originalRunDemo = window.runDemo;
                 window.runDemo = function(module) {
                     stateManager.setDemoRunning(true, module);
                     if (originalRunDemo) {
                         return originalRunDemo(module);
                     }
                 };
             }
             
             // Sistema de Monitoramento de Performance
             let performanceMonitoring = {
                 isRunning: false,
                 interval: null,
                 charts: {},
                 data: {
                     cpu: [],
                     memory: [],
                     latency: [],
                     throughput: [],
                     timestamps: []
                 },
                 maxDataPoints: 50
             };
             
             // Inicializar gráficos de performance
             function initializePerformanceCharts() {
                 const systemCtx = document.getElementById('system-metrics-chart')?.getContext('2d');
                 const perfCtx = document.getElementById('performance-chart')?.getContext('2d');
                 
                 if (systemCtx) {
                     performanceMonitoring.charts.system = new Chart(systemCtx, {
                         type: 'line',
                         data: {
                             labels: [],
                             datasets: [{
                                 label: 'CPU (%)',
                                 data: [],
                                 borderColor: 'rgb(59, 130, 246)',
                                 backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                 tension: 0.4
                             }, {
                                 label: 'Memory (%)',
                                 data: [],
                                 borderColor: 'rgb(34, 197, 94)',
                                 backgroundColor: 'rgba(34, 197, 94, 0.1)',
                                 tension: 0.4
                             }]
                         },
                         options: {
                             responsive: true,
                             maintainAspectRatio: false,
                             scales: {
                                 y: {
                                     beginAtZero: true,
                                     max: 100
                                 }
                             },
                             plugins: {
                                 legend: {
                                     position: 'top'
                                 }
                             }
                         }
                     });
                 }
                 
                 if (perfCtx) {
                     performanceMonitoring.charts.performance = new Chart(perfCtx, {
                         type: 'line',
                         data: {
                             labels: [],
                             datasets: [{
                                 label: 'Latência (ms)',
                                 data: [],
                                 borderColor: 'rgb(147, 51, 234)',
                                 backgroundColor: 'rgba(147, 51, 234, 0.1)',
                                 yAxisID: 'y',
                                 tension: 0.4
                             }, {
                                 label: 'Throughput (tokens/s)',
                                 data: [],
                                 borderColor: 'rgb(249, 115, 22)',
                                 backgroundColor: 'rgba(249, 115, 22, 0.1)',
                                 yAxisID: 'y1',
                                 tension: 0.4
                             }]
                         },
                         options: {
                             responsive: true,
                             maintainAspectRatio: false,
                             scales: {
                                 y: {
                                     type: 'linear',
                                     display: true,
                                     position: 'left',
                                     title: {
                                         display: true,
                                         text: 'Latência (ms)'
                                     }
                                 },
                                 y1: {
                                     type: 'linear',
                                     display: true,
                                     position: 'right',
                                     title: {
                                         display: true,
                                         text: 'Throughput (tokens/s)'
                                     },
                                     grid: {
                                         drawOnChartArea: false
                                     }
                                 }
                             },
                             plugins: {
                                 legend: {
                                     position: 'top'
                                 }
                             }
                         }
                     });
                 }
             }
             
             // Alternar monitoramento
             function toggleMonitoring() {
                 const button = document.getElementById('monitoring-toggle');
                 const status = document.getElementById('monitoring-status');
                 
                 if (performanceMonitoring.isRunning) {
                     // Parar monitoramento
                     clearInterval(performanceMonitoring.interval);
                     performanceMonitoring.isRunning = false;
                     button.innerHTML = '▶️ Iniciar Monitoramento';
                     button.className = 'bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors';
                     status.textContent = 'Parado';
                     status.className = 'font-medium text-gray-500';
                     
                     stateManager.setState('performance', {
                         ...stateManager.getState('performance'),
                         monitoring: false
                     });
                 } else {
                     // Iniciar monitoramento
                     performanceMonitoring.isRunning = true;
                     button.innerHTML = '⏸️ Parar Monitoramento';
                     button.className = 'bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors';
                     status.textContent = 'Ativo';
                     status.className = 'font-medium text-green-600';
                     
                     // Inicializar gráficos se necessário
                     if (!performanceMonitoring.charts.system) {
                         initializePerformanceCharts();
                     }
                     
                     // Iniciar coleta de dados
                     performanceMonitoring.interval = setInterval(collectPerformanceData, 1000);
                     
                     stateManager.setState('performance', {
                         ...stateManager.getState('performance'),
                         monitoring: true
                     });
                 }
             }
             
             // Coletar dados de performance
             function collectPerformanceData() {
                 // Simular dados de performance (em produção, viria da API)
                 const now = new Date();
                 const timestamp = now.toLocaleTimeString();
                 
                 const cpu = Math.random() * 80 + 10; // 10-90%
                 const memory = Math.random() * 60 + 20; // 20-80%
                 const latency = Math.random() * 100 + 50; // 50-150ms
                 const throughput = Math.random() * 500 + 100; // 100-600 tokens/s
                 
                 // Atualizar displays
                 document.getElementById('cpu-usage').textContent = `${cpu.toFixed(1)}%`;
                 document.getElementById('memory-usage').textContent = `${memory.toFixed(1)}%`;
                 document.getElementById('latency').textContent = `${latency.toFixed(0)}ms`;
                 document.getElementById('throughput').textContent = `${throughput.toFixed(0)}`;
                 
                 // Adicionar aos dados
                 performanceMonitoring.data.cpu.push(cpu);
                 performanceMonitoring.data.memory.push(memory);
                 performanceMonitoring.data.latency.push(latency);
                 performanceMonitoring.data.throughput.push(throughput);
                 performanceMonitoring.data.timestamps.push(timestamp);
                 
                 // Limitar número de pontos
                 if (performanceMonitoring.data.cpu.length > performanceMonitoring.maxDataPoints) {
                     performanceMonitoring.data.cpu.shift();
                     performanceMonitoring.data.memory.shift();
                     performanceMonitoring.data.latency.shift();
                     performanceMonitoring.data.throughput.shift();
                     performanceMonitoring.data.timestamps.shift();
                 }
                 
                 // Atualizar gráficos
                 updatePerformanceCharts();
                 
                 // Verificar alertas
                 checkPerformanceAlerts(cpu, memory, latency, throughput);
                 
                 // Atualizar estado
                 stateManager.setState('performance', {
                     ...stateManager.getState('performance'),
                     lastUpdate: now,
                     metrics: { cpu, memory, latency, throughput }
                 });
             }
             
             // Atualizar gráficos
             function updatePerformanceCharts() {
                 if (performanceMonitoring.charts.system) {
                     const chart = performanceMonitoring.charts.system;
                     chart.data.labels = performanceMonitoring.data.timestamps;
                     chart.data.datasets[0].data = performanceMonitoring.data.cpu;
                     chart.data.datasets[1].data = performanceMonitoring.data.memory;
                     chart.update('none');
                 }
                 
                 if (performanceMonitoring.charts.performance) {
                     const chart = performanceMonitoring.charts.performance;
                     chart.data.labels = performanceMonitoring.data.timestamps;
                     chart.data.datasets[0].data = performanceMonitoring.data.latency;
                     chart.data.datasets[1].data = performanceMonitoring.data.throughput;
                     chart.update('none');
                 }
             }
             
             // Verificar alertas de performance
             function checkPerformanceAlerts(cpu, memory, latency, throughput) {
                 const alertsContainer = document.getElementById('performance-alerts');
                 let hasAlerts = false;
                 let alertMessages = [];
                 
                 if (cpu > 85) {
                     alertMessages.push('🔥 CPU usage alto: ' + cpu.toFixed(1) + '%');
                     hasAlerts = true;
                 }
                 
                 if (memory > 90) {
                     alertMessages.push('💾 Memória crítica: ' + memory.toFixed(1) + '%');
                     hasAlerts = true;
                 }
                 
                 if (latency > 200) {
                     alertMessages.push('🐌 Latência alta: ' + latency.toFixed(0) + 'ms');
                     hasAlerts = true;
                 }
                 
                 if (throughput < 50) {
                     alertMessages.push('📉 Throughput baixo: ' + throughput.toFixed(0) + ' tokens/s');
                     hasAlerts = true;
                 }
                 
                 if (hasAlerts) {
                     alertsContainer.innerHTML = alertMessages.map(msg => 
                         `<div class="bg-red-50 border border-red-200 rounded-lg p-3">
                             <div class="flex items-center">
                                 <span class="text-red-600 mr-2">⚠️</span>
                                 <span class="text-sm text-red-800">${msg}</span>
                             </div>
                         </div>`
                     ).join('');
                     alertsContainer.classList.remove('hidden');
                 } else {
                     alertsContainer.classList.add('hidden');
                 }
             }
             
             // Reset métricas
             function resetMetrics() {
                 performanceMonitoring.data = {
                     cpu: [],
                     memory: [],
                     latency: [],
                     throughput: [],
                     timestamps: []
                 };
                 
                 document.getElementById('cpu-usage').textContent = '--';
                 document.getElementById('memory-usage').textContent = '--';
                 document.getElementById('latency').textContent = '--';
                 document.getElementById('throughput').textContent = '--';
                 
                 updatePerformanceCharts();
                 
                 const alertsContainer = document.getElementById('performance-alerts');
                 alertsContainer.classList.add('hidden');
                 
                 console.log('🔄 Métricas de performance resetadas');
             }
             
             // Exportar métricas
             function exportMetrics() {
                 const data = {
                     timestamp: new Date().toISOString(),
                     metrics: performanceMonitoring.data,
                     summary: {
                         avgCpu: performanceMonitoring.data.cpu.reduce((a, b) => a + b, 0) / performanceMonitoring.data.cpu.length || 0,
                         avgMemory: performanceMonitoring.data.memory.reduce((a, b) => a + b, 0) / performanceMonitoring.data.memory.length || 0,
                         avgLatency: performanceMonitoring.data.latency.reduce((a, b) => a + b, 0) / performanceMonitoring.data.latency.length || 0,
                         avgThroughput: performanceMonitoring.data.throughput.reduce((a, b) => a + b, 0) / performanceMonitoring.data.throughput.length || 0
                     }
                 };
                 
                 const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                 const url = URL.createObjectURL(blob);
                 const a = document.createElement('a');
                 a.href = url;
                 a.download = `performance-metrics-${new Date().toISOString().slice(0, 19)}.json`;
                 document.body.appendChild(a);
                 a.click();
                 document.body.removeChild(a);
                 URL.revokeObjectURL(url);
                 
                 console.log('📊 Métricas exportadas');
             }
             
             // Carregar presets salvos do localStorage
             function loadSavedPresets() {
                 const saved = localStorage.getItem('mini-gpt-presets');
                 if (saved) {
                     try {
                         const savedPresets = JSON.parse(saved);
                         parameterPresets = { ...parameterPresets, ...savedPresets };
                         console.log('📂 Presets carregados do localStorage');
                     } catch (error) {
                         console.warn('⚠️ Erro ao carregar presets salvos:', error);
                     }
                 }
             }
             
             // Inicializar sistema completo quando a página carregar
             window.addEventListener('load', () => {
                 console.log('🚀 Inicializando sistema StateManager...');
                 
                 // Inicializar StateManager com estado inicial
                 const initialState = {
                     parameters: stateManager.getState('parameters'),
                     demos: { running: false, results: null },
                     visualizations: { activeTab: 'attention', data: {} },
                     websocket: { connected: false, lastMessage: null },
                     sync: { status: 'idle', lastSync: null }
                 };
                 
                 // Carregar presets salvos
                 loadSavedPresets();
                 
                 // Inicializar integrações
                 initializeWebSocketIntegration();
                 initializeVisualizationIntegration();
                 initializeDemoIntegration();
                 
                 // Inicializar valores dos parâmetros na UI
                 const currentParams = stateManager.getState('parameters');
                 Object.keys(currentParams).forEach(param => {
                     const inputId = param.replace('_', '-');
                     const valueId = `${inputId}-value`;
                     const valueElement = document.getElementById(valueId);
                     
                     if (valueElement) {
                         valueElement.textContent = currentParams[param];
                     }
                 });
                 
                 // Verificar inicialização das visualizações
                 setTimeout(() => {
                     if (window.advancedViz) {
                         console.log('✅ Visualizações avançadas inicializadas');
                         stateManager.setState('visualizations', {
                             ...stateManager.getState('visualizations'),
                             initialized: true
                         });
                     } else {
                         console.warn('⚠️ Visualizações avançadas não carregadas');
                     }
                     
                     console.log('🎉 Sistema StateManager inicializado com sucesso!');
                      console.log('📊 Estado atual:', stateManager.getState());
                      
                      console.log('⚙️ Sistema de controle de parâmetros inicializado');
                      updateSyncStatus('🟢 Conectado', 'text-green-600');
                 }, 1000);
             });
            
            // D3.js para visualização de atenção
            const attentionDiv = d3.select('#attention-visualization');
            if (!attentionDiv.empty()) {
                // Placeholder para heatmap de atenção
                attentionDiv
                    .append('svg')
                    .attr('width', '100%')
                    .attr('height', '100%')
                    .append('text')
                    .attr('x', '50%')
                    .attr('y', '50%')
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'middle')
                    .style('fill', '#6B7280')
                    .text('🔗 Conecte-se para ver visualizações em tempo real');
            }
        });
    </script>
</body>
</html>
"#);
    
    Html(html)
}

/// Handler para servir interativos específicos
async fn interactive_handler(
    Path(id): Path<String>,
    interativos_dir: PathBuf,
) -> Result<Response, StatusCode> {
    // Mapeia IDs para arquivos
    let file_name = match id.as_str() {
        "chunking" => "sample.html",
        "attention" => "attention.html",
        "tokenization" => "tokenization.html",
        "embeddings" => "embeddings.html",
        "transformer" => "transformer.html",
        "training" => "training.html",
        "inference" => "inference.html",
        _ => return Err(StatusCode::NOT_FOUND),
    };
    
    let file_path = interativos_dir.join(file_name);
    
    match fs::read_to_string(&file_path).await {
        Ok(content) => Ok(Html(content).into_response()),
        Err(_) => {
            // Se o arquivo não existe, retorna uma página de "em construção"
            let under_construction = format!(r#"
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Em Construção - {}</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-yellow-50 to-orange-100 min-h-screen flex items-center justify-center">
    <div class="text-center">
        <div class="text-6xl mb-4">🚧</div>
        <h1 class="text-4xl font-bold text-gray-800 mb-4">Em Construção</h1>
        <p class="text-xl text-gray-600 mb-8">O interativo "{}" está sendo desenvolvido!</p>
        <a href="/" class="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors">
            ← Voltar ao Índice
        </a>
    </div>
</body>
</html>
"#, id, id);
            Ok(Html(under_construction).into_response())
        }
    }
}

/// Inicia o servidor web com a configuração padrão
pub async fn start_web_server(config: Option<WebServerConfig>) -> Result<()> {
    let config = config.unwrap_or_default();
    let server = WebServer::new(config);
    server.start().await
}