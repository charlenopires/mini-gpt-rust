//! # Mini-GPT: Um Large Language Model (LLM) Educacional em Rust
//! 
//! ## 🧠 O que é um Large Language Model?
//! 
//! Imagine que estamos construindo um "cérebro artificial" que aprende a escrever em português.
//! Como uma criança aprendendo a falar, nosso modelo vai observar padrões no texto e tentar
//! reproduzi-los. Mas como exatamente isso funciona?
//! 
//! ### 📚 Processo de Aprendizado (similar ao cérebro humano):
//! 1. **Tokenização**: Quebra o texto em pedaços menores (tokens) - como sílabas para uma criança
//! 2. **Embeddings**: Converte palavras em números que o computador entende - cada palavra vira um vetor
//! 3. **Atenção**: O modelo aprende quais palavras são importantes para o contexto - como focar na conversa
//! 4. **Transformers**: Arquitetura que processa sequências de texto de forma paralela e eficiente
//! 5. **Treinamento**: Ajusta milhões de parâmetros para prever a próxima palavra corretamente
//! 
//! ### 🔬 Conceitos Fundamentais:
//! - **Tokens**: Unidades básicas de texto (palavras, subpalavras ou caracteres)
//! - **Embeddings**: Representações vetoriais densas que capturam significado semântico
//! - **Atenção**: Mecanismo que permite ao modelo focar em partes relevantes da entrada
//! - **Backpropagation**: Algoritmo que ajusta pesos da rede neural baseado nos erros
//! - **Gradient Descent**: Método de otimização que minimiza a função de perda iterativamente

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod tokenizer;
mod attention;
mod transformer;
mod model;
mod training;
mod educational_logger;
mod kernels;
mod chunking;
mod benchmarks;
mod device;
mod api;
mod web_server;

use model::{MiniGPT, CheckpointMetadata};
use training::Trainer;
use educational_logger::EducationalLogger;
use kernels::{FusionBenchmark, FusionConfig};
use chunking::{ChunkProcessor, ChunkingConfig, ChunkingStrategy};
use benchmarks::{BenchmarkRunner, BenchmarkConfig};

/// 🖥️ **INTERFACE DE LINHA DE COMANDO (CLI)**
/// 
/// Define a estrutura principal da aplicação usando o crate `clap`
/// para parsing automático de argumentos e geração de help.
/// 
/// ## 🎯 Funcionalidades Disponíveis:
/// 1. **Train**: Treina o modelo do zero com dados fornecidos
/// 2. **Generate**: Gera texto a partir de um prompt
/// 3. **Chat**: Modo interativo de conversação
/// 
/// ## 📖 Exemplo de Uso:
/// ```bash
/// # Treinar o modelo
/// cargo run -- train --data corpus.txt --epochs 50
/// 
/// # Gerar texto
/// cargo run -- generate --prompt "Era uma vez" --max-tokens 100
/// 
/// # Modo chat interativo
/// cargo run -- chat
/// ```
#[derive(Parser)]
#[command(name = "mini-gpt")]
#[command(about = "Mini-GPT: LLM Educacional em Rust 🦀🧠", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,  // 🎯 Comando principal a ser executado
}

/// 🎮 **COMANDOS DISPONÍVEIS**
/// 
/// Enumera todas as operações possíveis que o usuário pode realizar
/// com o Mini-GPT, cada uma com seus próprios parâmetros específicos.
#[derive(Subcommand)]
enum Commands {
    /// 🎓 **TREINAMENTO: Ensinar o modelo a escrever**
    /// 
    /// Treina o modelo do zero usando um corpus de texto em português.
    /// O processo envolve:
    /// 1. Carregar e tokenizar o dataset
    /// 2. Dividir em batches para processamento eficiente
    /// 3. Executar forward/backward passes
    /// 4. Otimizar parâmetros usando Adam
    /// 5. Salvar checkpoints periodicamente
    Train {
        /// 📁 Caminho para o arquivo de dados de treinamento
        /// Deve conter texto em português, preferencialmente limpo e bem formatado
        #[arg(short, long, default_value = "data/corpus_pt_br.txt")]
        data: PathBuf,
        
        /// 🔄 Número de épocas de treinamento
        /// Uma época = uma passada completa pelo dataset
        /// Mais épocas = melhor aprendizado, mas risco de overfitting
        #[arg(short, long, default_value = "100")]
        epochs: usize,
    },
    
    /// 🎨 **GERAÇÃO: Criar texto criativo**
    /// 
    /// Gera texto a partir de um prompt inicial usando o modelo treinado.
    /// O processo utiliza:
    /// 1. Tokenização do prompt de entrada
    /// 2. Geração autoregressiva (uma palavra por vez)
    /// 3. Sampling com temperatura para controlar criatividade
    /// 4. Decodificação de volta para texto legível
    Generate {
        /// 💭 Prompt inicial para geração de texto
        /// Exemplo: "Era uma vez", "O futuro da tecnologia", etc.
        #[arg(short, long)]
        prompt: String,
        
        /// 🎯 Número máximo de tokens a gerar
        /// Controla o comprimento do texto gerado
        /// 1 token ≈ 0.75 palavras em português
        #[arg(short, long, default_value = "100")]
        max_tokens: usize,
        
        /// 📚 Ativa logs educacionais detalhados
        #[arg(long, help = "Ativa logs educacionais detalhados")]
        educational: bool,
        
        /// 🔍 Mostra informações de tensores
        #[arg(long, help = "Mostra informações de tensores")]
        show_tensors: bool,
    },
    
    /// 💬 **CHAT: Conversação interativa**
    /// 
    /// Modo interativo onde você pode conversar com o modelo
    /// em tempo real, simulando um chatbot inteligente.
    /// 
    /// ## 🎮 Como usar:
    /// - Digite suas mensagens e pressione Enter
    /// - Digite 'quit' ou 'exit' para sair
    /// - O modelo mantém contexto da conversa
    Chat {
        /// 📚 Ativa logs educacionais detalhados
        #[arg(long, help = "Ativa logs educacionais detalhados")]
        educational: bool,
        
        /// 🔍 Mostra informações de tensores
        #[arg(long, help = "Mostra informações de tensores")]
        show_tensors: bool,
    },
    
    /// 📂 **LOAD: Carregar modelo de checkpoint**
    /// 
    /// Carrega um modelo previamente treinado de um arquivo SafeTensors
    /// e permite gerar texto ou iniciar chat com o modelo carregado.
    /// 
    /// ## 🎯 **Modos de Carregamento:**
    /// 1. **Direto**: Especifica caminho exato do checkpoint
    /// 2. **Interativo**: Lista checkpoints disponíveis para seleção
    /// 3. **Automático**: Carrega o melhor checkpoint (menor loss)
    /// 4. **Por Nome**: Busca checkpoint por nome/padrão
    /// 
    /// ## 📊 **Filtros Disponíveis:**
    /// - Por data de criação (mais recente/antigo)
    /// - Por performance (menor/maior loss)
    /// - Por step de treinamento
    /// - Por descrição/tags
    Load {
        /// 📁 Caminho para o arquivo de checkpoint (.safetensors)
        /// Se não especificado, entra em modo interativo
        #[arg(short, long)]
        checkpoint: Option<PathBuf>,
        
        /// 📂 Diretório para buscar checkpoints (modo interativo)
        #[arg(short, long, default_value = "models")]
        dir: PathBuf,
        
        /// 🎯 Carrega automaticamente o melhor checkpoint (menor loss)
        #[arg(long, help = "Carrega automaticamente o checkpoint com menor loss")]
        best: bool,
        
        /// 📅 Carrega o checkpoint mais recente
        #[arg(long, help = "Carrega o checkpoint mais recente por timestamp")]
        latest: bool,
        
        /// 🔍 Busca checkpoint por nome/padrão
        #[arg(long, help = "Busca checkpoint que contenha este padrão no nome")]
        name_pattern: Option<String>,
        
        /// 📊 Filtra por loss máximo
        #[arg(long, help = "Carrega apenas checkpoints com loss menor que este valor")]
        max_loss: Option<f32>,
        
        /// 🔢 Filtra por step mínimo de treinamento
        #[arg(long, help = "Carrega apenas checkpoints com step maior que este valor")]
        min_step: Option<usize>,
        
        /// 💭 Prompt para geração (opcional)
        #[arg(short, long)]
        prompt: Option<String>,
        
        /// 🎯 Número máximo de tokens a gerar
        #[arg(short, long, default_value = "100")]
        max_tokens: usize,
        
        /// 💬 Inicia modo chat após carregar
        #[arg(long, help = "Inicia modo chat interativo")]
        chat: bool,
        
        /// 📚 Ativa logs educacionais detalhados
        #[arg(long, help = "Ativa logs educacionais detalhados")]
        educational: bool,
        
        /// 🔍 Mostra informações detalhadas do checkpoint antes de carregar
        #[arg(long, help = "Exibe metadados detalhados do checkpoint")]
        info: bool,
    },
    
    /// 📋 **LIST: Listar checkpoints disponíveis**
    /// 
    /// Lista todos os checkpoints disponíveis em um diretório
    /// com informações sobre timestamp, loss e configuração.
    List {
        /// 📁 Diretório para buscar checkpoints
        #[arg(short, long, default_value = "models")]
        dir: PathBuf,
    },
    
    /// ⚡ **BENCHMARK: Testar performance de kernel fusion**
    /// 
    /// Executa benchmarks para medir ganhos de performance
    /// das otimizações de kernel fusion em diferentes cenários.
    Benchmark {
        /// 🔢 Tamanho do batch para teste
        #[arg(long, default_value = "4")]
        batch_size: usize,
        
        /// 📏 Comprimento da sequência
        #[arg(long, default_value = "128")]
        seq_len: usize,
        
        /// 🧮 Dimensão do modelo
        #[arg(long, default_value = "512")]
        d_model: usize,
        
        /// 🔄 Número de iterações para benchmark
        #[arg(long, default_value = "100")]
        iterations: usize,
        
        /// 🎯 Tipo de benchmark (attention, feedforward, all)
        #[arg(long, default_value = "all")]
        benchmark_type: String,
    },
    
    /// 📄 **CHUNKING: Demonstra sistema de chunking de texto**
    /// 
    /// Testa diferentes estratégias de chunking em texto:
    /// - Chunking fixo: divide em pedaços de tamanho fixo
    /// - Chunking semântico: preserva significado e estrutura
    /// - Chunking adaptativo: ajusta tamanho baseado no conteúdo
    /// - Chunking com sobreposição: mantém contexto entre chunks
    Chunk {
        /// 📁 Caminho para o arquivo de texto a ser processado
        #[arg(short, long, default_value = "data/sample_text.txt")]
        input: PathBuf,
        
        /// 🎯 Estratégia de chunking (fixed, semantic, adaptive, overlap)
        #[arg(short, long, default_value = "semantic")]
        strategy: String,
        
        /// 📏 Tamanho máximo do chunk em tokens
        #[arg(long, default_value = "512")]
        max_size: usize,
        
        /// 📐 Tamanho mínimo do chunk em tokens
        #[arg(long, default_value = "64")]
        min_size: usize,
        
        /// 🔄 Razão de sobreposição (0.0 a 1.0)
        #[arg(long, default_value = "0.1")]
        overlap: f32,
        
        /// 📊 Exibe estatísticas detalhadas
        #[arg(long, help = "Mostra análise detalhada dos chunks")]
        analyze: bool,
        
        /// 🎯 Preserva sentenças completas
        #[arg(long, help = "Evita quebrar sentenças no meio")]
        preserve_sentences: bool,
        
        /// 📝 Preserva parágrafos completos
        #[arg(long, help = "Evita quebrar parágrafos no meio")]
        preserve_paragraphs: bool,
        
        /// 💾 Salva chunks em arquivo
        #[arg(short, long, help = "Arquivo para salvar os chunks processados")]
        output: Option<PathBuf>,
    },
    
    /// 📊 **CHUNK BENCHMARK: Testa performance de chunking**
    /// 
    /// Executa benchmarks abrangentes das estratégias de chunking
    /// para avaliar performance, qualidade e uso de memória.
    ChunkBench {
        /// 📁 Arquivo de texto para benchmark
        #[arg(short, long)]
        input: PathBuf,
        
        /// 📏 Tamanhos de texto para testar (separados por vírgula)
        #[arg(long, default_value = "1000,5000,10000,50000")]
        sizes: String,
        
        /// 🔄 Número de iterações por teste
        #[arg(long, default_value = "10")]
        iterations: usize,
        
        /// 🎯 Estratégias para testar (separadas por vírgula)
        #[arg(long, default_value = "fixed,semantic,adaptive,overlapping")]
        strategies: String,
        
        /// 📁 Arquivo de saída para relatório
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// 🧪 Executar testes de stress
        #[arg(long)]
        stress: bool,
    },

    /// 🎓 **DEMO: Demonstrações educacionais dos módulos**
    /// 
    /// Executa demonstrações interativas dos diferentes módulos do Mini-GPT
    /// para fins educacionais e de aprendizado. Cada demo mostra conceitos
    /// fundamentais e implementações práticas.
    Demo {
        /// 🎯 Módulo para demonstrar (attention, tokenizer, model, transformer, benchmarks, kernels, educational_logger, all)
        #[arg(short, long, default_value = "all")]
        module: String,
        
        /// 📚 Ativa logs educacionais detalhados
        #[arg(long, help = "Ativa logs educacionais detalhados")]
        educational: bool,
        
        /// 🔍 Mostra informações de tensores
        #[arg(long, help = "Mostra informações de tensores")]
        show_tensors: bool,
        
        /// 🗺️ Exibe mapas de atenção (quando aplicável)
        #[arg(long, help = "Exibe mapas de atenção visuais")]
        show_attention: bool,
        
        /// ⚡ Executa benchmarks de performance
        #[arg(long, help = "Inclui benchmarks de performance")]
        benchmark: bool,
        
        /// 🎮 Modo interativo com pausas para explicações
        #[arg(long, help = "Modo interativo com pausas educacionais")]
        interactive: bool,
    },
    
    /// 🌐 **WEB: Servidor web para interativos educacionais**
    /// 
    /// Inicia um servidor web local que hospeda interativos educacionais
    /// acessíveis através do navegador. Inclui visualizações interativas
    /// de todos os conceitos fundamentais do GPT e Transformers.
    Web {
        /// 🌍 Endereço IP para bind do servidor
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// 🔌 Porta do servidor web
        #[arg(short, long, default_value = "3000")]
        port: u16,

        /// 📁 Diretório dos arquivos interativos
        #[arg(long, default_value = "interativos")]
        dir: PathBuf,

        /// 📄 Caminho do corpus usado pelas demonstrações
        #[arg(long, default_value = "data/corpus_pt_br.txt")]
        corpus: PathBuf,

        /// 📂 Diretório de checkpoints do modelo
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,
    },
}

/// 🚀 **FUNÇÃO PRINCIPAL: PONTO DE ENTRADA DA APLICAÇÃO**
/// 
/// Esta é a função principal que coordena toda a execução do Mini-GPT.
/// Funciona como um "maestro" que dirige a orquestra de funcionalidades.
/// 
/// ## 🎯 Responsabilidades:
/// 1. **Parsing CLI**: Interpreta argumentos da linha de comando
/// 2. **Inicialização**: Configura dispositivo de computação (CPU/GPU)
/// 3. **Roteamento**: Direciona para a função apropriada baseada no comando
/// 4. **Tratamento de Erros**: Propaga erros usando o tipo `Result`
/// 
/// ## 🖥️ **Seleção de Dispositivo:**
/// 
/// ### 💻 **CPU (Padrão):**
/// - **Vantagens**: Compatibilidade universal, debugging mais fácil
/// - **Desvantagens**: Mais lento para operações matriciais grandes
/// - **Uso**: Ideal para desenvolvimento e modelos pequenos
/// 
/// ### 🚀 **GPU (Metal/CUDA):**
/// - **Vantagens**: Paralelização massiva, muito mais rápido
/// - **Desvantagens**: Requer hardware específico, mais complexo
/// - **Uso**: Essencial para modelos grandes e treinamento intensivo
/// 
/// ## 📊 **Performance Esperada:**
/// ```text
/// CPU (M3):     ~1000 tokens/segundo (geração)
/// GPU (Metal):  ~5000 tokens/segundo (geração)
/// GPU (CUDA):   ~10000 tokens/segundo (geração)
/// ```
fn main() -> Result<()> {
    // 🎮 **PARSING DOS ARGUMENTOS CLI**
    // Usa o crate `clap` para interpretar automaticamente
    // os argumentos passados na linha de comando
    let cli = Cli::parse();
    
    // 🎯 **ROTEAMENTO BASEADO NO COMANDO**
    // Pattern matching para executar a função apropriada
    // baseada no comando escolhido pelo usuário
    match cli.command {
        // 🌐 **MODO SERVIDOR WEB**
        // Inicia servidor web para interativos educacionais
        Commands::Web { host, port, dir, corpus, models_dir } => {
            let device = device::resolve_device();

            println!("🌐 Iniciando servidor web para interativos educacionais...");
            println!("📍 Host: {}", host);
            println!("🔌 Porta: {}", port);
            println!("📁 Diretório: {:?}", dir);

            let config = web_server::WebServerConfig {
                host,
                port,
                interativos_dir: dir,
                corpus_path: corpus,
                models_dir,
            };

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                web_server::start_web_server(config, device).await
            })?;
        }
        
        // 🎓 **MODO TREINAMENTO**
        // Treina o modelo do zero usando dados fornecidos
        Commands::Train { data, epochs } => {
            let device = device::resolve_device();
            println!("📚 Iniciando treinamento com dados de: {:?}", data);
            println!("🔄 Épocas configuradas: {}", epochs);
            train_model(data, epochs, &device)?
        }
        
        // 🎨 **MODO GERAÇÃO**
        // Gera texto criativo a partir de um prompt
        Commands::Generate { prompt, max_tokens, educational, show_tensors } => {
            let device = device::resolve_device();
            println!("✨ Gerando texto a partir de: '{}'", prompt);
            println!("🎯 Máximo de tokens: {}", max_tokens);
            generate_text(&prompt, max_tokens, &device, educational, show_tensors)?
        }
        
        // 💬 **MODO CHAT INTERATIVO**
        // Permite conversação em tempo real com o modelo
        Commands::Chat { educational, show_tensors } => {
            let device = device::resolve_device();

            println!("💬 Modo chat ativado! Digite 'quit' ou 'exit' para terminar.");
            println!("🤖 Aguardando suas mensagens...");
            interactive_chat(&device, educational, show_tensors)?
        }
        
        // 📂 **MODO CARREGAMENTO DE MODELO**
        // Carrega modelo de checkpoint e executa geração ou chat
        Commands::Load { 
            checkpoint, 
            dir, 
            best, 
            latest, 
            name_pattern, 
            max_loss, 
            min_step, 
            prompt, 
            max_tokens, 
            chat, 
            educational, 
            info 
        } => {
            let device = device::resolve_device();

            // 🎯 **SELEÇÃO INTELIGENTE DE CHECKPOINT**
            let selected_checkpoint = select_checkpoint(
                checkpoint,
                &dir,
                best,
                latest,
                name_pattern,
                max_loss,
                min_step,
                info
            )?;
            
            load_and_run_model(selected_checkpoint, prompt, max_tokens, chat, educational, &device)?
        }
        
        // 📋 **MODO LISTAGEM DE CHECKPOINTS**
        // Lista todos os checkpoints disponíveis
        Commands::List { dir } => {
            list_checkpoints(dir)?
        }
        
        // ⚡ **MODO BENCHMARK DE KERNEL FUSION**
        // Testa performance das otimizações
        Commands::Benchmark { batch_size, seq_len, d_model, iterations, benchmark_type } => {
            let device = device::resolve_device();
            run_kernel_fusion_benchmark(batch_size, seq_len, d_model, iterations, &benchmark_type, &device)?
        }
        
        // 📄 **MODO CHUNKING DE TEXTO**
        // Demonstra diferentes estratégias de chunking
        Commands::Chunk { 
            input, 
            strategy, 
            max_size, 
            min_size, 
            overlap, 
            analyze, 
            preserve_sentences, 
            preserve_paragraphs, 
            output 
        } => {
            run_chunking_demo(
                input, 
                &strategy, 
                max_size, 
                min_size, 
                overlap, 
                analyze, 
                preserve_sentences, 
                preserve_paragraphs, 
                output
            )?
        }

        // 📊 **MODO BENCHMARK DE CHUNKING**
        // Executa testes de performance para diferentes estratégias
        Commands::ChunkBench {
            input,
            sizes,
            iterations,
            strategies,
            output,
            stress,
        } => {
            run_chunking_benchmark(
                input,
                &sizes,
                iterations,
                &strategies,
                output,
                stress,
            )?
        }
        
        // 🎓 **MODO DEMONSTRAÇÃO EDUCACIONAL**
        // Executa demonstrações interativas dos módulos
        Commands::Demo {
            module,
            educational,
            show_tensors,
            show_attention,
            benchmark,
            interactive,
        } => {
            run_educational_demos(
                &module,
                educational,
                show_tensors,
                show_attention,
                benchmark,
                interactive,
            )?
        }
    }
    
    // ✅ **FINALIZAÇÃO BEM-SUCEDIDA**
    // Retorna Ok(()) indicando que tudo correu bem
    println!("✨ Execução concluída com sucesso!");
    Ok(())
}

/// 📄 **DEMONSTRAÇÃO DO SISTEMA DE CHUNKING**
/// 
/// Executa uma demonstração completa do sistema de chunking,
/// mostrando diferentes estratégias e suas características.
fn run_chunking_demo(
    input_path: PathBuf,
    strategy: &str,
    max_size: usize,
    min_size: usize,
    overlap: f32,
    analyze: bool,
    preserve_sentences: bool,
    preserve_paragraphs: bool,
    output_path: Option<PathBuf>,
) -> Result<()> {
    use std::fs;
    use tokenizer::BPETokenizer;
    
    println!("📄 === DEMONSTRAÇÃO DO SISTEMA DE CHUNKING ===");
    println!("📁 Arquivo de entrada: {:?}", input_path);
    println!("🎯 Estratégia: {}", strategy);
    println!("📏 Tamanho máximo: {} tokens", max_size);
    println!("📐 Tamanho mínimo: {} tokens", min_size);
    println!("🔄 Sobreposição: {:.1}%", overlap * 100.0);
    println!();
    
    // 📖 **CARREGAMENTO DO TEXTO**
    let text = fs::read_to_string(&input_path)
        .map_err(|e| anyhow::anyhow!("Erro ao ler arquivo: {}", e))?;
    
    println!("📊 Texto carregado: {} caracteres", text.len());
    
    // 🔧 **INICIALIZAÇÃO DO TOKENIZER**
    let mut tokenizer = BPETokenizer::new(50000)
        .map_err(|e| anyhow::anyhow!("Erro ao inicializar tokenizer: {}", e))?;
    
    // Treina o tokenizer com uma amostra do texto para demonstração
    let sample_text = if text.len() > 10000 { &text[..10000] } else { &text };
    tokenizer.train(sample_text)
        .map_err(|e| anyhow::anyhow!("Erro ao treinar tokenizer: {}", e))?;
    
    // 📝 **CONFIGURAÇÃO DO CHUNKING**
    let strategy_enum = match strategy.to_lowercase().as_str() {
        "fixed" => ChunkingStrategy::Fixed,
        "semantic" => ChunkingStrategy::Semantic,
        "adaptive" => ChunkingStrategy::Adaptive,
        "overlapping" => ChunkingStrategy::Overlapping,
        _ => {
            println!("⚠️  Estratégia desconhecida '{}', usando 'semantic'", strategy);
            ChunkingStrategy::Semantic
        }
    };
    
    let config = ChunkingConfig {
        max_chunk_size: max_size,
        min_chunk_size: min_size,
        overlap_ratio: overlap,
        strategy: strategy_enum,
        preserve_sentences,
        preserve_paragraphs,
    };
    
    let mut processor = ChunkProcessor::new(config);
    
    // 🚀 **PROCESSAMENTO DOS CHUNKS**
    println!("🔄 Processando chunks...");
    let chunks = processor.process_text(&text, &tokenizer)
        .map_err(|e| anyhow::anyhow!("Erro no chunking: {}", e))?;
    
    println!("✅ Processamento concluído!");
    println!("📊 Total de chunks gerados: {}", chunks.len());
    println!();
    
    // 📈 **ANÁLISE DETALHADA (se solicitada)**
    if analyze {
        println!("📈 === ANÁLISE DETALHADA DOS CHUNKS ===");
        
        let total_tokens: usize = chunks.iter().map(|c| c.tokens.len()).sum();
        let avg_tokens = if !chunks.is_empty() { total_tokens / chunks.len() } else { 0 };
        let min_tokens = chunks.iter().map(|c| c.tokens.len()).min().unwrap_or(0);
        let max_tokens = chunks.iter().map(|c| c.tokens.len()).max().unwrap_or(0);
        
        println!("📊 Estatísticas gerais:");
        println!("   • Total de tokens: {}", total_tokens);
        println!("   • Média de tokens por chunk: {}", avg_tokens);
        println!("   • Menor chunk: {} tokens", min_tokens);
        println!("   • Maior chunk: {} tokens", max_tokens);
        println!();
        
        // 🔍 **DETALHES DE CADA CHUNK**
        for (i, chunk) in chunks.iter().enumerate().take(5) {
            println!("📄 Chunk {} (Índice: {}):", i + 1, chunk.chunk_index);
            println!("   • Tokens: {}", chunk.tokens.len());
            println!("   • Caracteres: {}", chunk.text.len());
            println!("   • Densidade: {:.3}", chunk.metadata.information_density);
            println!("   • Sentenças: {}", chunk.metadata.sentence_count);
            println!("   • Parágrafos: {}", chunk.metadata.paragraph_count);
            
            // Mostra preview do conteúdo
            let preview = if chunk.text.len() > 100 {
                format!("{}...", &chunk.text[..100])
            } else {
                chunk.text.clone()
            };
            println!("   • Preview: {}", preview.replace('\n', " "));
            println!();
        }
        
        if chunks.len() > 5 {
            println!("   ... e mais {} chunks", chunks.len() - 5);
            println!();
        }
        
        // 📊 **ESTATÍSTICAS GERAIS**
        let stats = processor.calculate_statistics(&chunks);
        println!("📊 Estatísticas detalhadas:");
        println!("   • Tamanho médio: {:.1} tokens", stats.avg_chunk_size);
        println!("   • Densidade média: {:.3}", stats.avg_information_density);
        println!("   • Taxa de preservação: {:.1}%", stats.boundary_preservation_rate * 100.0);
        println!();
    }
    
    // 💾 **SALVAMENTO (se solicitado)**
    if let Some(output) = output_path {
        println!("💾 Salvando chunks em: {:?}", output);
        
        let mut output_content = String::new();
        output_content.push_str(&format!("# Chunks processados com estratégia: {}\n\n", strategy));
        output_content.push_str(&format!("Total de chunks: {}\n", chunks.len()));
        output_content.push_str(&format!("Configuração: max={}, min={}, overlap={:.1}%\n\n", 
                                       max_size, min_size, overlap * 100.0));
        
        for (i, chunk) in chunks.iter().enumerate() {
            output_content.push_str(&format!("## Chunk {} (Índice: {})\n", i + 1, chunk.chunk_index));
            output_content.push_str(&format!("- Tokens: {}\n", chunk.tokens.len()));
            output_content.push_str(&format!("- Densidade: {:.3}\n", chunk.metadata.information_density));
            output_content.push_str(&format!("- Sentenças: {}\n", chunk.metadata.sentence_count));
            output_content.push_str(&format!("- Parágrafos: {}\n\n", chunk.metadata.paragraph_count));
            output_content.push_str(&chunk.text);
            output_content.push_str("\n\n---\n\n");
        }
        
        fs::write(&output, output_content)
            .map_err(|e| anyhow::anyhow!("Erro ao salvar arquivo: {}", e))?;
        
        println!("✅ Chunks salvos com sucesso!");
    }
    
    println!("🎉 Demonstração de chunking concluída!");
    Ok(())
}

/// 📊 **BENCHMARK DO SISTEMA DE CHUNKING**
/// 
/// Executa testes de performance para diferentes estratégias de chunking,
/// medindo tempo, qualidade e uso de memória.
fn run_chunking_benchmark(
    input_path: PathBuf,
    sizes: &str,
    iterations: usize,
    strategies: &str,
    output_path: Option<PathBuf>,
    stress: bool,
) -> Result<()> {
    println!("🚀 Iniciando benchmarks de chunking...");
    
    // Carregamento do texto
    let text = std::fs::read_to_string(&input_path)?;
    println!("📄 Texto carregado: {} caracteres", text.len());
    
    // Parse das estratégias
    let strategy_list: Vec<ChunkingStrategy> = strategies
        .split(',')
        .map(|s| match s.trim() {
            "fixed" => ChunkingStrategy::Fixed,
            "semantic" => ChunkingStrategy::Semantic,
            "adaptive" => ChunkingStrategy::Adaptive,
            "overlapping" => ChunkingStrategy::Overlapping,
            _ => ChunkingStrategy::Fixed,
        })
        .collect();
    
    // Configuração do benchmark com valores padrão
    let mut chunking_configs = std::collections::HashMap::new();
    
    // Configurações padrão para cada estratégia
    for strategy in &strategy_list {
        let config = ChunkingConfig {
            max_chunk_size: 512,
            min_chunk_size: 64,
            overlap_ratio: 0.1,
            strategy: strategy.clone(),
            preserve_sentences: true,
            preserve_paragraphs: false,
        };
        chunking_configs.insert(strategy.clone(), config);
    }
    
    let config = BenchmarkConfig {
        text_sizes: vec![1000, 5000, 10000],
        iterations,
        strategies: strategy_list,
        chunking_configs,
        warmup_iterations: 3,
    };
    
    // Execução dos benchmarks
    let mut runner = BenchmarkRunner::new(config)?;
    let results = runner.run_all_benchmarks(&text)?;
    
    // Geração do relatório
    let report = runner.generate_report(&results);
    
    // Exibição dos resultados
    println!("\n📈 Resultados dos Benchmarks:");
    println!("{}", report);
    
    // Salvamento opcional
    if let Some(output) = output_path {
        std::fs::write(&output, &report)?;
        println!("💾 Relatório salvo em: {}", output.display());
    }
    
    // Execução de testes de stress se solicitado
    if stress {
        println!("\n🔥 Executando testes de stress...");
        benchmarks::run_stress_tests(&text)?;
    }
    
    Ok(())
}

/// 🔄 **CARREGAMENTO E EXECUÇÃO DE MODELO**
/// 
/// Carrega um modelo de checkpoint e executa geração ou chat
fn load_and_run_model(
    checkpoint_path: PathBuf,
    prompt: Option<String>,
    max_tokens: usize,
    chat_mode: bool,
    educational: bool,
    device: &candle_core::Device,
) -> Result<()> {
    println!("📂 Carregando modelo de: {:?}", checkpoint_path);

    // Carregar modelo do checkpoint
    let (model, metadata) = MiniGPT::load_from_checkpoint(&checkpoint_path, device)
        .map_err(|e| anyhow::anyhow!("Erro ao carregar checkpoint: {}", e))?;
    println!("✅ Modelo carregado com sucesso! (Step: {:?})", metadata.training_step);

    // 🔤 **CARREGAR O TOKENIZER EXATO DO TREINAMENTO**
    // Sem o vocabulário original, os IDs de token não correspondem às mesmas
    // palavras/subpalavras que o modelo aprendeu — a geração seria lixo.
    let tokenizer_path = format!("{}.tokenizer.json", checkpoint_path.display());
    let tokenizer = tokenizer::BPETokenizer::load_json(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!(
            "Não foi possível carregar o tokenizer em {} ({}). Checkpoints salvos antes desta versão não possuem tokenizer.json — retreine o modelo.",
            tokenizer_path, e
        ))?;

    if chat_mode {
        // Modo chat interativo
        println!("💬 Iniciando chat com modelo carregado...");
        interactive_chat_with_model(&model, &tokenizer, educational)
    } else if let Some(prompt_text) = prompt {
        // Geração de texto
        println!("🎨 Gerando texto a partir do prompt...");
        generate_text_with_model(&model, &tokenizer, &prompt_text, max_tokens, educational)
    } else {
        println!("⚠️  Especifique um prompt (-p) ou use modo chat (--chat)");
        Ok(())
    }
}

/// 🎓 **SISTEMA DE DEMONSTRAÇÕES EDUCACIONAIS**
/// 
/// Executa demonstrações interativas dos módulos do Mini-GPT para fins educacionais.
/// Cada demonstração mostra conceitos fundamentais, implementações práticas e
/// otimizações de performance específicas de cada componente.
fn run_educational_demos(
    module: &str,
    educational: bool,
    show_tensors: bool,
    show_attention: bool,
    benchmark: bool,
    interactive: bool,
) -> Result<()> {
    println!("\n🎓 **SISTEMA DE DEMONSTRAÇÕES EDUCACIONAIS - MINI-GPT**");
    println!("📚 Explorando conceitos fundamentais de LLMs em Rust\n");
    
    if interactive {
        println!("🔄 **MODO INTERATIVO ATIVADO** - Pressione Enter para continuar entre seções\n");
    }
    
    match module {
        "attention" => {
            println!("🧠 **DEMONSTRAÇÃO: MECANISMO DE ATENÇÃO**");
            run_command("cargo run --example attention_demo", interactive)?;
        }
        "tokenizer" => {
            println!("🔤 **DEMONSTRAÇÃO: SISTEMA DE TOKENIZAÇÃO**");
            run_command("cargo run --example tokenizer_demo", interactive)?;
        }
        "model" => {
            println!("🤖 **DEMONSTRAÇÃO: ARQUITETURA DO MODELO**");
            run_command("cargo run --example model_demo", interactive)?;
        }
        "transformer" => {
            println!("🔄 **DEMONSTRAÇÃO: BLOCOS TRANSFORMER**");
            run_command("cargo run --example transformer_demo", interactive)?;
        }
        "benchmarks" => {
            println!("⚡ **DEMONSTRAÇÃO: SISTEMA DE BENCHMARKS**");
            run_command("cargo run --example benchmarks_demo", interactive)?;
        }
        "kernels" => {
            println!("🚀 **DEMONSTRAÇÃO: OTIMIZAÇÕES DE KERNEL**");
            run_command("cargo run --example kernels_demo", interactive)?;
        }
        "educational_logger" => {
            println!("📊 **DEMONSTRAÇÃO: LOGGING EDUCACIONAL**");
            run_command("cargo run --example educational_logger_demo", interactive)?;
        }
        "all" => {
            println!("🌟 **DEMONSTRAÇÃO COMPLETA: TODOS OS MÓDULOS**\n");
            
            let modules = [
                ("attention", "🧠 Mecanismo de Atenção"),
                ("tokenizer", "🔤 Sistema de Tokenização"),
                ("model", "🤖 Arquitetura do Modelo"),
                ("transformer", "🔄 Blocos Transformer"),
                ("benchmarks", "⚡ Sistema de Benchmarks"),
                ("kernels", "🚀 Otimizações de Kernel"),
                ("educational_logger", "📊 Logging Educacional"),
            ];
            
            for (i, (module_name, description)) in modules.iter().enumerate() {
                println!("\n{} **{}** ({}/{})", description, module_name.to_uppercase(), i + 1, modules.len());
                println!("{}", "=".repeat(60));
                
                if interactive {
                    println!("\nPressione Enter para continuar...");
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input)?;
                }
                
                run_educational_demos(
                    module_name,
                    educational,
                    show_tensors,
                    show_attention,
                    benchmark,
                    false, // Não usar modo interativo recursivamente
                )?;
            }
            
            println!("\n✅ **DEMONSTRAÇÕES CONCLUÍDAS COM SUCESSO!**");
            println!("🎯 Todos os módulos foram demonstrados com conceitos educacionais.");
        }
        _ => {
            println!("❌ **ERRO**: Módulo '{}' não encontrado.", module);
            println!("📋 **Módulos disponíveis**: attention, tokenizer, model, transformer, benchmarks, kernels, educational_logger, all");
            return Ok(());
        }
    }
    
    if benchmark && module != "all" {
        println!("\n⚡ **EXECUTANDO BENCHMARKS DE PERFORMANCE**");
        println!("📊 Medindo performance do módulo {}...", module);
        // Aqui poderia adicionar benchmarks específicos por módulo
    }
    
    Ok(())
}

/// 🛠️ **EXECUTOR DE COMANDOS AUXILIAR**
/// 
/// Executa comandos do sistema com tratamento de erros e modo interativo opcional.
fn run_command(command: &str, interactive: bool) -> Result<()> {
    if interactive {
        println!("\n🔧 **Executando**: {}", command);
        println!("Pressione Enter para continuar...");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
    }
    
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()?;
    
    if output.status.success() {
        println!("{}", String::from_utf8_lossy(&output.stdout));
    } else {
        println!("❌ **ERRO**: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    Ok(())
}

/// 🎯 **SELEÇÃO INTELIGENTE DE CHECKPOINT**
/// 
/// Implementa lógica avançada para seleção de checkpoints baseada em critérios
/// específicos como performance, data, nome e filtros customizados.
/// 
/// ## 🧠 **Algoritmo de Seleção:**
/// 1. **Modo Direto**: Se caminho específico fornecido, usa diretamente
/// 2. **Modo Automático**: Aplica filtros e critérios de ordenação
/// 3. **Modo Interativo**: Apresenta lista filtrada para seleção manual
/// 
/// ## 📊 **Critérios de Priorização:**
/// - **Best**: Menor loss (melhor performance)
/// - **Latest**: Timestamp mais recente
/// - **Pattern**: Correspondência de nome/descrição
/// - **Filtros**: Loss máximo, step mínimo
fn select_checkpoint(
    direct_path: Option<PathBuf>,
    search_dir: &PathBuf,
    auto_best: bool,
    auto_latest: bool,
    name_pattern: Option<String>,
    max_loss_filter: Option<f32>,
    min_step_filter: Option<usize>,
    show_info: bool,
) -> Result<PathBuf> {
    // 🎯 **MODO DIRETO: Caminho específico fornecido**
    if let Some(path) = direct_path {
        if !path.exists() {
            return Err(anyhow::anyhow!("❌ Checkpoint não encontrado: {:?}", path));
        }
        
        if show_info {
            println!("📋 Carregando checkpoint específico: {:?}", path);
            // Carrega apenas para mostrar informações, sem usar o modelo
            if let Ok((_, metadata)) = MiniGPT::load_from_checkpoint(&path, &candle_core::Device::Cpu) {
                display_checkpoint_info(&path, &metadata);
            }
        }
        
        return Ok(path);
    }
    
    // 📂 **BUSCA E FILTRAGEM DE CHECKPOINTS**
    println!("🔍 Buscando checkpoints em: {:?}", search_dir);
    
    let mut checkpoints = MiniGPT::list_checkpoints(search_dir)
        .map_err(|e| anyhow::anyhow!("Erro ao listar checkpoints: {}", e))?;
    
    if checkpoints.is_empty() {
        return Err(anyhow::anyhow!("📭 Nenhum checkpoint encontrado em {:?}", search_dir));
    }
    
    println!("📊 Encontrados {} checkpoints", checkpoints.len());
    
    // 🔍 **APLICAÇÃO DE FILTROS**
    
    // Filtro por padrão de nome
    if let Some(pattern) = &name_pattern {
        checkpoints.retain(|(path, metadata)| {
            let filename = std::path::Path::new(path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            
            let description = metadata.description
                .as_ref()
                .map(|d| d.to_lowercase())
                .unwrap_or_default();
            
            let pattern_lower = pattern.to_lowercase();
            filename.contains(&pattern_lower) || description.contains(&pattern_lower)
        });
        
        println!("🔍 Após filtro por padrão '{}': {} checkpoints", pattern, checkpoints.len());
    }
    
    // Filtro por loss máximo
    if let Some(max_loss) = max_loss_filter {
        checkpoints.retain(|(_, metadata)| {
            metadata.loss.map_or(false, |loss| loss <= max_loss)
        });
        
        println!("📊 Após filtro por loss ≤ {}: {} checkpoints", max_loss, checkpoints.len());
    }
    
    // Filtro por step mínimo
    if let Some(min_step) = min_step_filter {
        checkpoints.retain(|(_, metadata)| {
            metadata.training_step.map_or(false, |step| step >= min_step)
        });
        
        println!("🔢 Após filtro por step ≥ {}: {} checkpoints", min_step, checkpoints.len());
    }
    
    if checkpoints.is_empty() {
        return Err(anyhow::anyhow!("❌ Nenhum checkpoint atende aos critérios especificados"));
    }
    
    // 🎯 **SELEÇÃO AUTOMÁTICA**
    
    if auto_best {
        // Seleciona checkpoint com menor loss
        checkpoints.sort_by(|a, b| {
            let loss_a = a.1.loss.unwrap_or(f32::INFINITY);
            let loss_b = b.1.loss.unwrap_or(f32::INFINITY);
            loss_a.partial_cmp(&loss_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let (best_path, best_metadata) = &checkpoints[0];
        println!("🏆 Selecionado melhor checkpoint (loss: {:?}): {}", 
                best_metadata.loss, 
                std::path::Path::new(best_path).file_name().unwrap().to_string_lossy());
        
        if show_info {
            display_checkpoint_info(&PathBuf::from(best_path), best_metadata);
        }
        
        return Ok(PathBuf::from(best_path));
    }
    
    if auto_latest {
        // Seleciona checkpoint mais recente
        checkpoints.sort_by(|a, b| b.1.timestamp.cmp(&a.1.timestamp));
        
        let (latest_path, latest_metadata) = &checkpoints[0];
        println!("📅 Selecionado checkpoint mais recente: {}", 
                std::path::Path::new(latest_path).file_name().unwrap().to_string_lossy());
        
        if show_info {
            display_checkpoint_info(&PathBuf::from(latest_path), latest_metadata);
        }
        
        return Ok(PathBuf::from(latest_path));
    }
    
    // 🎮 **MODO INTERATIVO: Seleção manual**
    println!("\n🎮 Modo de seleção interativa ativado!");
    println!("{}", "=".repeat(80));
    
    // Ordena por loss (melhor primeiro) para apresentação
    checkpoints.sort_by(|a, b| {
        let loss_a = a.1.loss.unwrap_or(f32::INFINITY);
        let loss_b = b.1.loss.unwrap_or(f32::INFINITY);
        loss_a.partial_cmp(&loss_b).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    for (i, (path, metadata)) in checkpoints.iter().enumerate() {
        let filename = std::path::Path::new(path).file_name().unwrap().to_string_lossy();
        println!("{}. 📁 {}", i + 1, filename);
        println!("   📊 Loss: {:?} | 📅 {}", metadata.loss, metadata.timestamp);
        
        if let Some(step) = metadata.training_step {
            println!("   🔢 Step: {}", step);
        }
        
        if let Some(desc) = &metadata.description {
            println!("   📝 {}", desc);
        }
        
        println!();
    }
    
    println!("Digite o número do checkpoint desejado (1-{}) ou 'q' para cancelar:", checkpoints.len());
    
    use std::io::{self, Write};
    loop {
        print!("🎯 Sua escolha: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.eq_ignore_ascii_case("q") || input.eq_ignore_ascii_case("quit") {
            return Err(anyhow::anyhow!("❌ Seleção cancelada pelo usuário"));
        }
        
        if let Ok(choice) = input.parse::<usize>() {
            if choice >= 1 && choice <= checkpoints.len() {
                let (selected_path, selected_metadata) = &checkpoints[choice - 1];
                println!("✅ Checkpoint selecionado: {}", 
                        std::path::Path::new(selected_path).file_name().unwrap().to_string_lossy());
                
                if show_info {
                    display_checkpoint_info(&PathBuf::from(selected_path), selected_metadata);
                }
                
                return Ok(PathBuf::from(selected_path));
            }
        }
        
        println!("❌ Opção inválida. Digite um número entre 1 e {} ou 'q' para cancelar.", checkpoints.len());
    }
}

/// 📋 **EXIBIÇÃO DE INFORMAÇÕES DETALHADAS DO CHECKPOINT**
/// 
/// Mostra metadados completos de um checkpoint específico
fn display_checkpoint_info(path: &PathBuf, metadata: &CheckpointMetadata) {
    println!("\n📋 Informações Detalhadas do Checkpoint");
    println!("{}", "=".repeat(50));
    println!("📁 Arquivo: {}", path.file_name().unwrap().to_string_lossy());
    println!("📂 Caminho: {:?}", path);
    println!("📅 Timestamp: {}", metadata.timestamp);
    println!("🔧 Versão: {}", metadata.version);
    
    if let Some(loss) = metadata.loss {
        println!("📊 Loss: {:.6}", loss);
    }
    
    if let Some(step) = metadata.training_step {
        println!("🔢 Training Step: {}", step);
    }
    
    if let Some(desc) = &metadata.description {
        println!("📝 Descrição: {}", desc);
    }
    
    // Informações do arquivo
    if let Ok(file_metadata) = std::fs::metadata(path) {
        let size_mb = file_metadata.len() as f64 / (1024.0 * 1024.0);
        println!("💾 Tamanho: {:.2} MB", size_mb);
    }
    
    println!("{}", "=".repeat(50));
}

/// 📋 **LISTAGEM DE CHECKPOINTS**
/// 
/// Lista todos os checkpoints disponíveis em um diretório
fn list_checkpoints(dir: PathBuf) -> Result<()> {
    println!("📋 Listando checkpoints em: {:?}", dir);
    
    let checkpoints = MiniGPT::list_checkpoints(&dir)
        .map_err(|e| anyhow::anyhow!("Erro ao listar checkpoints: {}", e))?;
    
    if checkpoints.is_empty() {
        println!("📭 Nenhum checkpoint encontrado no diretório.");
        return Ok(());
    }
    
    println!("\n📊 Checkpoints encontrados:");
    println!("{}", "-".repeat(80));
    
    for (i, (path, metadata)) in checkpoints.iter().enumerate() {
        println!("{}. 📁 {}", i + 1, std::path::Path::new(path).file_name().unwrap().to_string_lossy());
        println!("   📅 Timestamp: {}", metadata.timestamp);
        println!("   📊 Loss: {:?}", metadata.loss);
        println!("   🔧 Versão: {}", metadata.version);
        
        if let Some(description) = &metadata.description {
            println!("   📝 Descrição: {}", description);
        }
        
        println!();
    }
    
    Ok(())
}

/// ⚡ **BENCHMARK DE KERNEL FUSION**
/// 
/// Executa benchmarks para medir ganhos de performance
fn run_kernel_fusion_benchmark(
    batch_size: usize,
    seq_len: usize,
    d_model: usize,
    iterations: usize,
    benchmark_type: &str,
    device: &candle_core::Device,
) -> Result<()> {
    println!("⚡ Executando benchmark de kernel fusion...");
    println!("📊 Configuração:");
    println!("   🔢 Batch size: {}", batch_size);
    println!("   📏 Sequence length: {}", seq_len);
    println!("   🧮 Model dimension: {}", d_model);
    println!("   🔄 Iterations: {}", iterations);
    println!("   🎯 Type: {}", benchmark_type);
    println!();
    
    let fusion_config = FusionConfig {
        enable_attention_fusion: true,
        enable_feedforward_fusion: true,
        enable_memory_optimization: true,
        fusion_threshold: 512,
    };
    
    let benchmark = FusionBenchmark::new(fusion_config, device.clone());
    
    match benchmark_type {
        "attention" => {
            let results = benchmark.benchmark_attention(batch_size, seq_len, d_model, iterations)?;
            println!("🎯 Resultados do Benchmark de Atenção:");
            println!("   ⚡ Fusionado: {:.2}ms (média)", results.fused_time_ms);
            println!("   🐌 Não-fusionado: {:.2}ms (média)", results.unfused_time_ms);
            println!("   🚀 Speedup: {:.2}x", results.speedup);
            println!("   💾 Economia de memória: {:.1}%", results.memory_saved_percent);
        }
        "feedforward" => {
            let results = benchmark.benchmark_feedforward(batch_size, seq_len, d_model, iterations)?;
            println!("🎯 Resultados do Benchmark de Feed-Forward:");
            println!("   ⚡ Fusionado: {:.2}ms (média)", results.fused_time_ms);
            println!("   🐌 Não-fusionado: {:.2}ms (média)", results.unfused_time_ms);
            println!("   🚀 Speedup: {:.2}x", results.speedup);
            println!("   💾 Economia de memória: {:.1}%", results.memory_saved_percent);
        }
        "all" => {
            println!("🎯 Executando benchmark completo...");
            
            let attention_results = benchmark.benchmark_attention(batch_size, seq_len, d_model, iterations)?;
            println!("\n📊 Atenção Multi-Head:");
            println!("   ⚡ Fusionado: {:.2}ms", attention_results.fused_time_ms);
            println!("   🐌 Não-fusionado: {:.2}ms", attention_results.unfused_time_ms);
            println!("   🚀 Speedup: {:.2}x", attention_results.speedup);
            
            let ff_results = benchmark.benchmark_feedforward(batch_size, seq_len, d_model, iterations)?;
            println!("\n📊 Feed-Forward:");
            println!("   ⚡ Fusionado: {:.2}ms", ff_results.fused_time_ms);
            println!("   🐌 Não-fusionado: {:.2}ms", ff_results.unfused_time_ms);
            println!("   🚀 Speedup: {:.2}x", ff_results.speedup);
            
            let total_speedup = (attention_results.speedup + ff_results.speedup) / 2.0;
            println!("\n🏆 Speedup médio total: {:.2}x", total_speedup);
        }
        _ => {
            println!("❌ Tipo de benchmark inválido. Use: attention, feedforward, ou all");
        }
    }
    
    Ok(())
}

/// 🎨 **GERAÇÃO DE TEXTO COM MODELO CARREGADO**
fn generate_text_with_model(
    model: &MiniGPT,
    tokenizer: &tokenizer::BPETokenizer,
    prompt: &str,
    max_tokens: usize,
    educational: bool,
) -> Result<()> {
    println!("🎨 Gerando texto com modelo carregado...");
    println!("💭 Prompt: {}", prompt);
    println!("🎯 Max tokens: {}", max_tokens);

    if educational {
        let tokens = tokenizer.encode(prompt).map_err(|e| anyhow::anyhow!("{}", e))?;
        println!("📚 Prompt tokenizado em {} tokens: {:?}", tokens.len(), tokens);
    }

    let generated = model
        .generate(prompt, max_tokens, tokenizer, 0.8)
        .map_err(|e| anyhow::anyhow!("Erro ao gerar texto: {}", e))?;

    println!("\n📝 Texto gerado:\n{}", generated);

    Ok(())
}

/// 💬 **CHAT INTERATIVO COM MODELO CARREGADO**
fn interactive_chat_with_model(
    model: &MiniGPT,
    tokenizer: &tokenizer::BPETokenizer,
    educational: bool,
) -> Result<()> {
    use std::io::{self, Write};

    println!("💬 Chat interativo com modelo carregado...");
    println!("🔤 Vocabulário: {} tokens", tokenizer.vocab_size());
    println!("💡 Digite suas mensagens e pressione Enter. Digite 'quit' ou 'exit' para sair.");
    if educational {
        println!("📚 Modo educacional ativado — cada prompt será mostrado tokenizado.");
    }

    let temperature = 0.8;
    let max_tokens = 100;

    loop {
        print!("\n🧑 Você: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" || input == "sair" {
            println!("👋 Até logo!");
            break;
        }

        if educational {
            let tokens = tokenizer.encode(input).map_err(|e| anyhow::anyhow!("{}", e))?;
            println!("📚 Tokens: {:?}", tokens);
        }

        let response = model
            .generate(input, max_tokens, tokenizer, temperature)
            .map_err(|e| anyhow::anyhow!("Erro ao gerar resposta: {}", e))?;

        println!("🤖 Mini-GPT: {}", response);
    }

    Ok(())
}

/// 🎓 **FUNÇÃO DE TREINAMENTO DO MODELO**
/// 
/// Esta função implementa o processo completo de treinamento de um modelo de linguagem.
/// É como ensinar uma criança a escrever: mostramos exemplos e ela aprende os padrões.
/// 
/// ## 📋 Etapas do Treinamento:
/// 
/// ### 1. **Tokenização** 📝
/// - Converte texto bruto em tokens (unidades processáveis)
/// - Similar a como dividimos frases em palavras para entender
/// - Cada token recebe um ID numérico único
/// 
/// ### 2. **Criação do Modelo** 🧠
/// - Inicializa a arquitetura Transformer com pesos aleatórios
/// - Define o tamanho do vocabulário baseado nos tokens encontrados
/// - Cria camadas de atenção, embeddings e redes neurais
/// 
/// ### 3. **Processo de Treinamento** 🎯
/// - **Forward Pass**: Modelo faz predições sobre próxima palavra
/// - **Loss Calculation**: Compara predição com resposta correta
/// - **Backpropagation**: Calcula gradientes (direção para melhorar)
/// - **Optimization**: Ajusta pesos usando gradiente descendente
/// - Repete por várias épocas até convergir
fn train_model(data_path: PathBuf, epochs: usize, device: &candle_core::Device) -> Result<()> {
    // 📖 **ETAPA 1: CARREGAMENTO DOS DADOS**
    // Lê o arquivo de texto que será usado como corpus de treinamento
    // Este texto contém os padrões que o modelo vai aprender
    let text = std::fs::read_to_string(data_path)?;
    
    // 🔤 **ETAPA 2: TOKENIZAÇÃO**
    // O tokenizador converte texto em números que o modelo pode processar
    // BPE (Byte Pair Encoding) é eficiente para vocabulários grandes
    // Processo similar a criar um "dicionário" onde cada palavra/subpalavra tem um número
    //
    // ## 🎯 **Como funciona o BPE (Byte Pair Encoding):**
    // 1. Começa com caracteres individuais
    // 2. Encontra pares de bytes mais frequentes
    // 3. Substitui pares por novos tokens
    // 4. Repete até atingir tamanho de vocabulário desejado
    //
    // **Exemplo prático:**
    // - Texto: "Brasil Brasil brasileiro"
    // - Passo 1: ['B', 'r', 'a', 's', 'i', 'l', ' ', ...]
    // - Passo 2: Encontra 'Br' frequente → token 256
    // - Passo 3: Encontra 'as' frequente → token 257
    // - Resultado: [256, 257, 'i', 'l', ' ', 256, 257, 'i', 'l', ...]
    let mut tokenizer = tokenizer::BPETokenizer::new(1000)?; // 1000 tokens de vocabulário
    tokenizer.train(&text)?;  // Analisa o texto e cria vocabulário
    
    // 🔢 **CONVERSÃO TEXTO → NÚMEROS**
    // Transforma todo o corpus em sequência de IDs numéricos
    // Estes números são o que o modelo realmente "vê" durante o treinamento
    //
    // ## 📊 **Exemplo de Tokenização:**
    // - Texto: "O Brasil é um país"
    // - Tokens: [15, 42, 89, 156, 203]
    // - Cada número representa uma palavra/subpalavra no vocabulário
    let tokens = tokenizer.encode(&text)?;
    println!("📊 Total de tokens: {}", tokens.len());
    
    // 🧠 **ETAPA 3: CONFIGURAÇÃO E CRIAÇÃO DO MODELO**
    // Define a arquitetura do Transformer - como o "DNA" do modelo
    //
    // ## 📊 **Parâmetros Explicados:**
    // - `vocab_size`: Tamanho do vocabulário (quantas palavras/tokens únicos)
    // - `n_embd`: Dimensão dos embeddings (128 = modelo pequeno educacional)
    // - `n_head`: Cabeças de atenção (4 = permite focar em 4 aspectos diferentes)
    // - `n_layer`: Camadas Transformer (4 = profundidade moderada)
    // - `block_size`: Contexto máximo (64 tokens = ~48 palavras em português)
    // - `dropout`: Regularização para evitar overfitting
    let config = model::GPTConfig {
        vocab_size: tokenizer.vocab_size(),  // Quantas palavras diferentes o modelo conhece
        n_embd: 128,      // Dimensão dos embeddings (pequena para educacional)
        n_head: 4,        // 4 cabeças de atenção (permite focar em aspectos diferentes)
        n_layer: 4,       // 4 camadas transformer (profundidade do modelo)
        block_size: 64,   // Contexto de 64 tokens (quantas palavras o modelo "lembra")
        dropout: 0.1,     // Regularização para evitar overfitting
    };
    
    // 🏗️ **CONSTRUÇÃO DA ARQUITETURA NEURAL**
    // Cria todas as camadas, pesos e conexões do modelo
    // Inicialmente com valores aleatórios - o treinamento vai ajustá-los
    //
    // ## 🧠 **Componentes Criados:**
    // - Embeddings de tokens e posições
    // - Camadas Transformer com atenção multi-head
    // - Redes feed-forward
    // - Layer normalization
    // - Cabeça de linguagem para predição
    let model = MiniGPT::new(config, device).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("🧠 Modelo criado com {} parâmetros", model.num_parameters());
    
    // 🎯 **ETAPA 4: TREINAMENTO PROPRIAMENTE DITO**
    // Aqui acontece a "mágica": o modelo aprende padrões através de:
    // - Múltiplas passadas pelos dados (épocas)
    // - Ajuste gradual dos pesos neurais via backpropagation
    // - Minimização da função de perda (cross-entropy loss)
    //
    // ## 🔄 **Processo de Treinamento:**
    // 1. **Forward Pass**: Modelo faz predições sobre próxima palavra
    // 2. **Loss Calculation**: Compara predição com resposta correta
    // 3. **Backpropagation**: Calcula gradientes (direção para melhorar)
    // 4. **Optimization**: Ajusta pesos usando gradiente descendente
    // 5. Repete por várias épocas até convergir
    let mut trainer = Trainer::new(model, tokenizer, device.clone()).map_err(|e| anyhow::anyhow!("{}", e))?;
    trainer.train(&tokens, epochs).map_err(|e| anyhow::anyhow!("{}", e))?;
    
    // 💾 **ETAPA 5: PERSISTÊNCIA DO MODELO TREINADO**
    // Salva os pesos aprendidos para uso posterior
    // Formato SafeTensors é seguro e eficiente para modelos ML
    //
    // ## 💾 **Processo de Salvamento:**
    // 1. Serializa todos os tensores do modelo
    // 2. Salva metadados da arquitetura
    // 3. Cria checkpoint para recuperação
    // 4. Valida integridade dos dados salvos
    trainer.save("models/mini_gpt.safetensors").map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("💾 Modelo salvo com sucesso!");
    
    Ok(())
}

/// 🎨 **FUNÇÃO DE GERAÇÃO DE TEXTO**
/// 
/// Esta função demonstra como um modelo treinado pode gerar texto novo.
/// É como pedir para uma pessoa continuar uma frase - ela usa o que aprendeu
/// para criar algo coerente e contextualmente apropriado.
/// 
/// ## 🔄 Processo de Geração:
/// 
/// ### 1. **Tokenização do Prompt** 📝
/// - Converte o texto inicial em tokens que o modelo entende
/// - Mesmo processo usado no treinamento
/// 
/// ### 2. **Inferência Autoregressiva** 🔮
/// - Modelo prediz próximo token baseado no contexto
/// - Adiciona predição ao contexto e repete
/// - Processo iterativo até atingir limite de tokens
/// 
/// ### 3. **Sampling com Temperatura** 🌡️
/// - Controla criatividade vs. determinismo
/// - Temperatura baixa = mais conservador
/// - Temperatura alta = mais criativo/aleatório
/// 
/// ## 🎯 Parâmetros:
/// - `prompt`: Texto inicial para começar a geração
/// - `max_tokens`: Limite máximo de tokens a gerar
/// - `device`: Dispositivo de computação (CPU/GPU)
/// 
/// ## 📊 **Processo Detalhado:**
/// ```text
/// 1. Tokenizar prompt: "Era uma vez" → [15, 42, 89]
/// 2. Forward pass: modelo prediz distribuição do próximo token
/// 3. Sampling: escolhe token baseado na distribuição + temperatura
/// 4. Adicionar ao contexto: [15, 42, 89, 156]
/// 5. Repetir até max_tokens ou token de fim
/// ```
fn generate_text(prompt: &str, max_tokens: usize, device: &candle_core::Device, educational: bool, show_tensors: bool) -> Result<()> {
    use std::time::Instant;
    
    let start_time = Instant::now();
    
    // 🎓 **INICIALIZAÇÃO DO LOGGER EDUCACIONAL**
    let verbosity_level = if educational { if show_tensors { 3 } else { 2 } } else { 0 };
    let logger = EducationalLogger::new(verbosity_level);
    
    // 🔧 **ETAPA 1: INICIALIZAÇÃO DO TOKENIZADOR**
    // 
    // ⚠️ **NOTA IMPORTANTE**: Em produção, carregaríamos o tokenizador
    // exato usado durante o treinamento para garantir consistência.
    // Aqui criamos um novo apenas para demonstração educacional.
    let mut tokenizer = tokenizer::BPETokenizer::new(1000)?;
    
    // 📚 **ETAPA 2: TREINAMENTO RÁPIDO DO TOKENIZADOR**
    // 
    // Para demonstração, treina com um corpus pequeno em português.
    // Em produção, usaríamos o mesmo vocabulário do treinamento.
    // 
    // ## 🎯 **Por que Consistência é Crucial:**
    // - Tokens diferentes = embeddings diferentes
    // - Modelo não reconhece tokens "novos"
    // - Pode gerar texto incoerente ou falhar
    // 
    // 📚 **CORPUS DE DEMONSTRAÇÃO EM PORTUGUÊS BRASILEIRO**
    // 
    // Este corpus pequeno serve apenas para demonstração educacional.
    // Em produção, usaríamos:
    // - Datasets gigantes (GB ou TB de texto)
    // - Texto limpo e pré-processado
    // - Múltiplos domínios (notícias, literatura, web, etc.)
    // - Balanceamento de tópicos e estilos
    let sample_text = "O Brasil é um país tropical. A inteligência artificial está revolucionando o mundo. \
                      A programação em Rust é segura e eficiente. O aprendizado de máquina utiliza dados para fazer previsões.";
    
    // 🎯 **TREINAMENTO DO TOKENIZADOR BPE**
    // 
    // ## 🔤 **Como funciona o BPE (Byte Pair Encoding):**
    // 1. Começa com caracteres individuais
    // 2. Encontra pares de bytes mais frequentes
    // 3. Substitui pares por novos tokens
    // 4. Repete até atingir tamanho de vocabulário desejado
    // 
    // **Exemplo prático:**
    // - Texto: "Brasil Brasil brasileiro"
    // - Passo 1: ['B', 'r', 'a', 's', 'i', 'l', ' ', ...]
    // - Passo 2: Encontra 'Br' frequente → token 256
    // - Passo 3: Encontra 'as' frequente → token 257
    // - Resultado: [256, 257, 'i', 'l', ' ', 256, 257, 'i', 'l', ...]
    tokenizer.train(sample_text)?;
    
    // 🎓 **LOG EDUCACIONAL: TOKENIZAÇÃO**
    // 
    // Mostra como o texto foi dividido em tokens para ajudar
    // a entender o processo de tokenização. Útil para:
    // - Debugar problemas de tokenização
    // - Entender como o modelo "vê" o texto
    // - Otimizar prompts para melhor performance
    if educational {
        let tokens = tokenizer.encode(prompt)?;
        logger.log_tokenization(prompt, &tokens, &tokenizer)?;
    }
    
    // 🏗️ **ETAPA 3: RECRIAÇÃO DA ARQUITETURA DO MODELO**
    // 
    // ⚠️ **CRÍTICO**: A arquitetura deve ser IDÊNTICA à usada no treinamento!
    // Qualquer diferença (n_embd, n_head, n_layer) causará erro de carregamento.
    // 
    // ## 📊 **Parâmetros Explicados:**
    // - `vocab_size`: Tamanho do vocabulário (quantas palavras/tokens únicos)
    // - `n_embd`: Dimensão dos embeddings (128 = modelo pequeno educacional)
    // - `n_head`: Cabeças de atenção (4 = permite focar em 4 aspectos diferentes)
    // - `n_layer`: Camadas Transformer (4 = profundidade moderada)
    // - `block_size`: Contexto máximo (64 tokens = ~48 palavras em português)
    // - `dropout`: 0.0 durante inferência (sem regularização)
    let config = model::GPTConfig {
        vocab_size: tokenizer.vocab_size(),  // Baseado no vocabulário treinado
        n_embd: 128,      // Embeddings de 128 dimensões
        n_head: 4,        // 4 cabeças de atenção multi-head
        n_layer: 4,       // 4 camadas Transformer empilhadas
        block_size: 64,   // Contexto de 64 tokens
        dropout: 0.0,     // Sem dropout durante inferência
    };
    
    // 🧠 **ETAPA 4: CARREGAMENTO DO MODELO**
    // 
    // Em produção, carregaríamos os pesos salvos do treinamento:
    // ```rust
    // let model = MiniGPT::load_from_checkpoint("model.safetensors", device)?;
    // ```
    // 
    // Aqui criamos um modelo "virgem" apenas para demonstração.
    // 
    // 🏗️ **CRIAÇÃO DA ARQUITETURA DO MODELO**
    // 
    // Inicializa todas as camadas neurais:
    // - Embeddings de tokens e posições
    // - Camadas Transformer com atenção multi-head
    // - Redes feed-forward
    // - Layer normalization
    // - Cabeça de linguagem para predição
    let model = MiniGPT::new(config, device).map_err(|e| anyhow::anyhow!("{}", e))?;
    
    // 🎯 **ETAPA 5: CONFIGURAÇÃO DA GERAÇÃO**
    if !educational {
        println!("🎯 Prompt: '{}'", prompt);
        println!("🔄 Gerando {} tokens...", max_tokens);
        println!("🌡️ Temperatura: 0.8 (criatividade moderada)");
    }
    
    // 🎲 **ETAPA 6: GERAÇÃO AUTOREGRESSIVA COM SAMPLING**
    // 
    // ## 🌡️ **Controle de Temperatura:**
    // ```text
    // Temperatura 0.1: Muito conservador, repetitivo
    // Temperatura 0.8: Equilíbrio ideal (usado aqui)
    // Temperatura 1.5: Muito criativo, pode ser incoerente
    // ```
    // 
    // ## 🔄 **Processo Autoregressivo:**
    // 1. Tokeniza prompt inicial
    // 2. Forward pass → distribuição de probabilidades
    // 3. Aplica temperatura para controlar aleatoriedade
    // 4. Faz sampling da distribuição
    // 5. Adiciona token escolhido ao contexto
    // 6. Repete até atingir max_tokens ou token especial
    match model.generate(prompt, max_tokens, &tokenizer, 0.8) {
        Ok(generated_text) => {
            let processing_time = start_time.elapsed().as_secs_f32();
            
            // 🎓 **LOG EDUCACIONAL: RESUMO FINAL**
            if educational {
                let full_text = format!("{}{}", prompt, generated_text);
                let token_count = tokenizer.encode(&full_text)?.len();
                logger.log_process_summary(prompt, &full_text, token_count, processing_time)?;
            } else {
                println!("\n✨ **RESULTADO DA GERAÇÃO:**");
                println!("📝 Texto completo:");
                println!("{}{}", prompt, generated_text);
                println!("\n📊 Estatísticas:");
                println!("   • Tokens gerados: ~{}", generated_text.split_whitespace().count());
                println!("   • Caracteres: {}", generated_text.len());
                println!("   • Tempo de processamento: {:.2}ms", processing_time * 1000.0);
            }
        }
        Err(e) => {
            println!("❌ Erro na geração: {}", e);
            println!("💡 Dica: Treine o modelo primeiro com 'mini-gpt train'");
        }
    }
    
    Ok(())
}

/// 💬 **FUNÇÃO DE CHAT INTERATIVO**
/// 
/// Esta função cria uma interface de conversação em tempo real com o modelo.
/// É como ter uma conversa com o modelo treinado, onde você pode:
/// - Fazer perguntas e receber respostas
/// - Ajustar parâmetros de geração dinamicamente
/// - Experimentar com diferentes configurações
/// 
/// ## 🎛️ Parâmetros Ajustáveis:
/// 
/// ### 🌡️ **Temperatura**
/// - Controla a "criatividade" do modelo
/// - 0.1 = Muito conservador, respostas previsíveis
/// - 1.0 = Equilibrado entre criatividade e coerência
/// - 2.0 = Muito criativo, pode ser incoerente
/// 
/// ### 🔢 **Max Tokens**
/// - Define o comprimento máximo da resposta
/// - Mais tokens = respostas mais longas
/// - Menos tokens = respostas mais concisas
/// 💬 **CHAT INTERATIVO: CONVERSAÇÃO EM TEMPO REAL**
/// 
/// Implementa um sistema de chat onde o usuário pode conversar
/// com o modelo de linguagem em tempo real, mantendo contexto
/// e permitindo ajustes dinâmicos de parâmetros.
/// 
/// ## 🎯 **Funcionalidades Principais:**
/// 
/// ### 1. **Conversação Contínua** 🔄
/// - Mantém histórico da conversa
/// - Contexto preservado entre mensagens
/// - Respostas baseadas no histórico completo
/// 
/// ### 2. **Comandos Especiais** 🎛️
/// - `/temp <valor>`: Ajusta criatividade (0.1-2.0)
/// - `/tokens <num>`: Define tamanho da resposta (10-200)
/// - `/help`: Mostra ajuda dos comandos
/// - `quit`/`exit`: Sai do chat
/// 
/// ### 3. **Interface Amigável** 🎨
/// - Prompts coloridos e informativos
/// - Feedback em tempo real
/// - Tratamento de erros gracioso
/// 
/// ## 🧠 **Arquitetura do Sistema:**
/// ```text
/// Input do Usuário → Tokenização → Contexto + Nova Mensagem
///                                        ↓
/// Resposta Formatada ← Decodificação ← Geração Autoregressiva
/// ```
/// 
/// ## 🎯 Parâmetros:
/// - `device`: Dispositivo de computação (CPU/GPU)
/// 
/// ## 📊 **Configurações Otimizadas para Chat:**
/// - **Temperatura padrão**: 0.8 (equilíbrio criatividade/coerência)
/// - **Max tokens padrão**: 50 (respostas concisas)
/// - **Block size**: 64 (contexto suficiente para conversação)
fn interactive_chat(device: &candle_core::Device, educational: bool, show_tensors: bool) -> Result<()> {
    use std::io::{self, Write};
    use std::time::Instant;
    
    // 🎓 **INICIALIZAÇÃO DO LOGGER EDUCACIONAL**
    let verbosity_level = if educational { if show_tensors { 3 } else { 2 } } else { 0 };
    let logger = EducationalLogger::new(verbosity_level);
    
    // 🔧 **ETAPA 1: INICIALIZAÇÃO DO TOKENIZADOR**
    // Prepara o sistema de tokenização para conversação interativa
    let mut tokenizer = tokenizer::BPETokenizer::new(1000)?;
    
    // 📖 **ETAPA 2: CARREGAMENTO DO CORPUS DE TREINAMENTO**
    // 
    // Tenta carregar arquivo de corpus do disco, caso contrário
    // usa um corpus de exemplo em português para demonstração.
    // 
    // ## 🎯 **Estratégia de Fallback:**
    // 1. Tenta ler "corpus_pt_br.txt" do diretório atual
    // 2. Se falhar, usa texto de exemplo embutido
    // 3. Garante que o sistema sempre funcione
    let sample_text = std::fs::read_to_string("corpus_pt_br.txt")
        .unwrap_or_else(|_| {
            println!("⚠️  Arquivo corpus_pt_br.txt não encontrado, usando corpus de exemplo.");
            "O Brasil é um país tropical localizado na América do Sul. \
             A inteligência artificial está transformando o mundo. \
             Rust é uma linguagem de programação segura e eficiente. \
             O aprendizado de máquina utiliza dados para fazer previsões inteligentes. \
             A conversação é uma forma natural de comunicação humana. \
             Os chatbots modernos podem manter diálogos coerentes e úteis.".to_string()
        });
    
    // 🔤 **ETAPA 3: TREINAMENTO DO TOKENIZADOR**
    // Constrói o vocabulário baseado no corpus disponível
    println!("🔤 Treinando tokenizador com {} caracteres...", sample_text.len());
    tokenizer.train(&sample_text)?;
    println!("✅ Vocabulário criado com {} tokens", tokenizer.vocab_size());
    
    // 🏗️ **ETAPA 4: CONFIGURAÇÃO DO MODELO PARA CHAT**
    // 
    // Configuração otimizada para conversação interativa:
    // - Modelo pequeno para respostas rápidas
    // - Block size adequado para manter contexto
    // - Sem dropout para inferência determinística
    let config = model::GPTConfig {
        vocab_size: tokenizer.vocab_size(),  // 📊 Baseado no vocabulário treinado
        n_embd: 128,                         // 🧮 Embeddings compactos para velocidade
        n_head: 4,                           // 🎯 Atenção suficiente para coerência
        n_layer: 4,                          // 🏗️ Profundidade balanceada
        block_size: 64,                      // 📏 Contexto adequado para chat
        dropout: 0.0,                        // ⚠️ Sem dropout para inferência!
    };
    
    // 🧠 **ETAPA 5: INICIALIZAÇÃO DO MODELO**
    // Cria o modelo com a configuração otimizada para chat
    println!("🧠 Inicializando modelo Mini-GPT...");
    let model = MiniGPT::new(config, device).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("✅ Modelo carregado com {} parâmetros", model.num_parameters());
    
    // 🎨 **ETAPA 6: APRESENTAÇÃO DA INTERFACE**
    // Mostra informações do sistema e comandos disponíveis
    println!("\n🤖 ===== MINI-GPT CHAT INTERATIVO =====");
    if educational {
        println!("🎓 **MODO EDUCACIONAL ATIVADO**");
        println!("   Logs detalhados serão exibidos para as primeiras interações.");
        println!("   Use comandos especiais para explorar o funcionamento interno.");
    }
    println!("💡 Digite suas mensagens e pressione Enter");
    println!("🚪 Digite 'quit' ou 'exit' para sair");
    println!("\n🎛️  **COMANDOS ESPECIAIS:**");
    if educational {
        println!("   /tokens-demo <texto> : Demonstra tokenização de um texto");
        println!("   /explain             : Explica o processo de geração atual");
    }
    println!("   /temp <0.1-2.0>  : Ajusta criatividade (atual: 0.8)");
    println!("   /tokens <10-200> : Define tamanho da resposta (atual: 50)");
    println!("   /help            : Mostra esta ajuda");
    println!("   /stats           : Mostra estatísticas do modelo");
    println!("\n🎯 **DICAS DE USO:**");
    println!("   • Temperatura baixa (0.1-0.5): Respostas mais conservadoras");
    println!("   • Temperatura alta (1.0-2.0): Respostas mais criativas");
    println!("   • Menos tokens: Respostas mais concisas");
    println!("   • Mais tokens: Respostas mais elaboradas\n");
    
    // 📊 **ETAPA 7: CONFIGURAÇÃO DOS PARÂMETROS DE GERAÇÃO**
    // Valores padrão equilibrados para uma boa experiência de chat
    let mut temperature = 0.8;  // 🌡️ Criatividade moderada
    let mut max_tokens = 50;    // 📏 Respostas de tamanho médio
    let mut conversation_history = String::new();  // 📚 Histórico da conversa
    let mut interaction_count = 0;  // 📊 Contador de interações para logs educacionais
    
    // 🔄 **ETAPA 8: LOOP PRINCIPAL DE CONVERSAÇÃO**
    // Loop infinito que processa mensagens do usuário
    loop {
        // 📝 **CAPTURA DE INPUT DO USUÁRIO**
        print!("\n🧑 Você: ");
        io::stdout().flush()?;  // Força exibição do prompt
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        // ⏭️ **PULA ENTRADAS VAZIAS**
        if input.is_empty() {
            continue;
        }
        
        // 🚪 **CONDIÇÕES DE SAÍDA**
        if input == "sair" || input == "exit" {
            println!("👋 Até logo!");
            break;
        }
        
        // 🌡️ **COMANDO: AJUSTE DE TEMPERATURA**
        // Permite modificar criatividade do modelo em tempo real
        if input.starts_with("/temp ") {
            if let Ok(new_temp) = input[6..].parse::<f32>() {
                if (0.1..=2.0).contains(&new_temp) {
                    temperature = new_temp;
                    println!("🌡️  Temperatura ajustada para: {:.1}", temperature);
                } else {
                    println!("❌ Temperatura deve estar entre 0.1 e 2.0");
                }
            }
            continue;
        }
        
        // 🔢 **COMANDO: AJUSTE DE TOKENS MÁXIMOS**
        // Controla comprimento das respostas geradas
        if input.starts_with("/tokens ") {
            if let Ok(new_tokens) = input[8..].parse::<usize>() {
                if (10..=200).contains(&new_tokens) {
                    max_tokens = new_tokens;
                    println!("🔢 Max tokens ajustado para: {}", max_tokens);
                } else {
                    println!("❌ Tokens deve estar entre 10 e 200");
                }
            }
            continue;
        }
        
        // 🎓 **COMANDOS EDUCACIONAIS**
        if educational && input.starts_with("/tokens-demo ") {
            let demo_text = &input[13..];
            println!("\n🔍 **DEMONSTRAÇÃO DE TOKENIZAÇÃO:**");
            let demo_tokens = tokenizer.encode(demo_text)?;
            logger.log_tokenization(demo_text, &demo_tokens, &tokenizer)?;
            continue;
        }
        
        if educational && input == "/explain" {
            println!("\n🎓 **EXPLICAÇÃO DO PROCESSO DE GERAÇÃO:**");
            println!("1. 📝 **Tokenização**: Converte texto em números (IDs de tokens)");
            println!("2. 🔢 **Embeddings**: Transforma IDs em vetores densos de significado");
            println!("3. 🧠 **Transformer**: Processa sequência com atenção e feed-forward");
            println!("4. 🎯 **Predição**: Calcula probabilidades para próximo token");
            println!("5. 🎲 **Amostragem**: Seleciona token baseado em temperatura");
            println!("6. 🔄 **Repetição**: Processo continua até atingir limite ou EOS\n");
            continue;
        }
        
        // ❓ **COMANDO: AJUDA**
        // Mostra comandos disponíveis e configurações atuais
        if input == "/help" {
            println!("\n🎛️  **COMANDOS DISPONÍVEIS:**");
            if educational {
                println!("   /tokens-demo <texto> : Demonstra tokenização de um texto");
                println!("   /explain             : Explica o processo de geração atual");
            }
            println!("   /temp <0.1-2.0>  : Ajusta criatividade (atual: {:.1})", temperature);
            println!("   /tokens <10-200> : Define tamanho da resposta (atual: {})", max_tokens);
            println!("   /stats           : Mostra estatísticas do modelo");
            println!("   /help            : Mostra esta ajuda");
            println!("   quit/exit        : Encerra o chat");
            println!("\n🎯 **CONFIGURAÇÕES ATUAIS:**");
            println!("   🌡️  Temperatura: {:.1} (criatividade)", temperature);
            println!("   📏 Max Tokens: {} (tamanho da resposta)", max_tokens);
            println!("   📚 Histórico: {} caracteres", conversation_history.len());
            continue;
        }
        
        // 📊 **COMANDO: ESTATÍSTICAS**
         // Mostra informações detalhadas sobre o modelo e conversa
         if input == "/stats" {
             println!("\n📊 **ESTATÍSTICAS DO MODELO:**");
             println!("   🧠 Parâmetros: {} (aprox. {:.1}K)", 
                      model.num_parameters(), 
                      model.num_parameters() as f32 / 1000.0);
             println!("   🔤 Vocabulário: {} tokens", tokenizer.vocab_size());
             println!("   🏗️  Arquitetura: 4 camadas, 4 cabeças");
             println!("   📐 Embeddings: 128 dimensões");
             println!("   📏 Contexto: 64 tokens");
             println!("\n💬 **ESTATÍSTICAS DA CONVERSA:**");
             println!("   📚 Histórico: {} caracteres", conversation_history.len());
             println!("   🌡️  Temperatura: {:.1}", temperature);
             println!("   📏 Max Tokens: {}", max_tokens);
             if educational {
                 println!("   🎓 Modo educacional: Ativo");
                 println!("   📊 Interações com logs: {}", interaction_count);
             }
             continue;
         }
        
        // 🎯 **ETAPA 9: GERAÇÃO DE RESPOSTA**
        // 
        // Processa a mensagem do usuário e gera uma resposta contextual.
        // 
        // ## 🔄 **Fluxo de Geração:**
        // 1. **Preparação do Contexto**: Combina histórico + nova mensagem
        // 2. **Tokenização**: Converte texto em tokens numéricos
        // 3. **Forward Pass**: Processa através das camadas do modelo
        // 4. **Sampling**: Aplica temperatura para controlar criatividade
        // 5. **Decodificação**: Converte tokens de volta para texto
        // 6. **Atualização**: Adiciona ao histórico para próximas interações
        
        interaction_count += 1;
        let start_time = Instant::now();
        
        // 📝 **PREPARAÇÃO DO PROMPT CONTEXTUAL**
        // Combina histórico da conversa com a nova mensagem do usuário
        let contextual_prompt = if conversation_history.is_empty() {
            input.to_string()  // 🆕 Primeira mensagem
        } else {
            format!("{} {}", conversation_history, input)  // 📚 Com contexto
        };
        
        // 🎓 **LOGGING EDUCACIONAL** (apenas para as primeiras 3 interações)
        if educational && interaction_count <= 3 {
            println!("\n🎓 ===== ANÁLISE EDUCACIONAL (Interação {}) =====", interaction_count);
            let tokens = tokenizer.encode(&contextual_prompt)?;
            logger.log_tokenization(&contextual_prompt, &tokens, &tokenizer)?;
            
            println!("\n🧠 **PROCESSAMENTO TRANSFORMER:**");
            println!("   • Sequência de entrada: {} tokens", tokens.len());
            println!("   • Processando através de {} camadas...", 4); // 4 camadas conforme config
        }
        
        // 🤖 **INDICADOR DE PROCESSAMENTO**
        print!("🤖 Mini-GPT: ");
        io::stdout().flush()?;  // Força exibição imediata
        
        // 🔮 **PROCESSO DE INFERÊNCIA NEURAL**
        // 
        // Aplica o modelo treinado para gerar uma resposta coerente
        // baseada no contexto da conversa e configurações atuais.
        // 
        // ## ⚙️ **Parâmetros de Geração:**
        // - **Input**: Prompt contextual (histórico + nova mensagem)
        // - **Max Tokens**: Limite de tokens para a resposta
        // - **Tokenizer**: Sistema de codificação/decodificação
        // - **Temperature**: Controle de criatividade/aleatoriedade
        match model.generate(&contextual_prompt, max_tokens, &tokenizer, temperature) {
            Ok(response) => {
                // ✅ **SUCESSO: EXIBE E ATUALIZA HISTÓRICO**
                println!("{}", response);
                
                // 🎓 **LOGGING DE PREDIÇÃO** (apenas para as primeiras 3 interações)
                if educational && interaction_count <= 3 {
                    let generated_tokens = tokenizer.encode(&response)?;
                    
                    let duration = start_time.elapsed();
                    println!("\n⏱️ **ESTATÍSTICAS DE GERAÇÃO:**");
                    println!("   • Tempo total: {:.2}ms", duration.as_millis());
                    println!("   • Tokens gerados: {}", generated_tokens.len());
                    println!("   • Velocidade: {:.1} tokens/s", generated_tokens.len() as f64 / duration.as_secs_f64());
                    println!("   • Temperatura usada: {:.1}", temperature);
                    println!("\n{}", "=".repeat(60));
                }
                
                // 📚 **ATUALIZAÇÃO DO HISTÓRICO DA CONVERSA**
                // Mantém contexto para próximas interações
                conversation_history.push_str(&format!(" {} {}", input, response));
                
                // 🧹 **LIMPEZA DE HISTÓRICO (PREVENÇÃO DE OVERFLOW)**
                // Mantém apenas os últimos 500 caracteres para evitar
                // que o contexto cresça indefinidamente
                if conversation_history.len() > 500 {
                    let start = conversation_history.len() - 400;
                    conversation_history = conversation_history[start..].to_string();
                }
            }
            Err(e) => {
                // ❌ **ERRO: TRATAMENTO GRACIOSO**
                println!("❌ Erro na geração: {}", e);
                println!("💡 **Sugestões:**");
                println!("   • Tente um prompt mais simples");
                println!("   • Reduza o número de tokens (/tokens <num>)");
                println!("   • Ajuste a temperatura (/temp <valor>)");
                println!("   • Verifique se o modelo foi treinado adequadamente");
            }
        }
        
        // 🎨 **SEPARADOR VISUAL**
        // Adiciona espaço entre interações para melhor legibilidade
        println!();
    }
    
    // 🏁 **FINALIZAÇÃO GRACOSA**
    // Retorna sucesso após saída do loop principal
    Ok(())
}