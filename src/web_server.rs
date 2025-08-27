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
}

impl Default for WebServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            interativos_dir: PathBuf::from("interativos"),
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
        
        let app = Router::new()
            // Página principal (índice)
            .route("/", get(move || index_handler(interactives.clone())))
            // Interativo específico
            .route("/interactive/:id", get(move |path| {
                interactive_handler(path, interativos_dir.clone())
            }))
            // Servir arquivos estáticos do diretório interativos
            .nest_service("/static", ServeDir::new(&self.config.interativos_dir))
            // Middleware
            .layer(
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
        <!-- Footer -->
        <footer class="text-center mt-16 py-8 border-t border-gray-200">
            <p class="text-gray-600 mb-2">Desenvolvido com ❤️ usando Rust + Axum</p>
            <p class="text-sm text-gray-500">Projeto educacional open source para aprender sobre GPT e Transformers</p>
        </footer>
    </div>
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