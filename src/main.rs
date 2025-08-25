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

use model::MiniGPT;
use training::Trainer;
use educational_logger::EducationalLogger;

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
        // 🎓 **MODO TREINAMENTO**
        // Treina o modelo do zero usando dados fornecidos
        Commands::Train { data, epochs } => {
            // 🚀 **DISPOSITIVO PARA TREINAMENTO: METAL GPU ARM APPLE**
        // Prioriza Metal GPU para máxima performance no ARM Apple
            let device = match candle_core::Device::new_metal(0) {
                Ok(metal_device) => {
                    println!("🚀 Usando dispositivo: Metal GPU");
                    println!("⚡ Aceleração de hardware ativada para treinamento!");
                    metal_device
                }
                Err(e) => {
                    println!("⚠️  Metal GPU não disponível ({}), usando CPU", e);
                    candle_core::Device::Cpu
                }
            };
            println!("📚 Iniciando treinamento com dados de: {:?}", data);
            println!("🔄 Épocas configuradas: {}", epochs);
            train_model(data, epochs, &device)?
        }
        
        // 🎨 **MODO GERAÇÃO**
        // Gera texto criativo a partir de um prompt
        Commands::Generate { prompt, max_tokens, educational, show_tensors } => {
            // 🚀 **DISPOSITIVO PARA GERAÇÃO: METAL GPU ARM APPLE**
            // Prioriza Metal GPU para geração ultra-rápida
            let device = match candle_core::Device::new_metal(0) {
                Ok(metal_device) => {
                    println!("🚀 Usando dispositivo: Metal GPU");
                    println!("⚡ Geração acelerada por hardware ativada!");
                    metal_device
                }
                Err(e) => {
                    println!("⚠️  Metal GPU não disponível ({}), usando CPU", e);
                    candle_core::Device::Cpu
                }
            };
            println!("🚀 Usando dispositivo: {:?}", device);
            println!("✨ Gerando texto a partir de: '{}'", prompt);
            println!("🎯 Máximo de tokens: {}", max_tokens);
            generate_text(&prompt, max_tokens, &device, educational, show_tensors)?
        }
        
        // 💬 **MODO CHAT INTERATIVO**
        // Permite conversação em tempo real com o modelo
        Commands::Chat { educational, show_tensors } => {
            // 🚀 **DISPOSITIVO PARA CHAT: GPU (com fallback para CPU)**
            // Prioriza GPU para melhor performance em chat interativo
            let device = match candle_core::Device::new_metal(0) {
                Ok(metal_device) => {
                    println!("🚀 Usando dispositivo: Metal GPU (aceleração de hardware)");
                    metal_device
                }
                Err(_) => {
                    // Fallback para CUDA se Metal não estiver disponível
                    match candle_core::Device::new_cuda(0) {
                        Ok(cuda_device) => {
                            println!("🚀 Usando dispositivo: CUDA GPU (aceleração de hardware)");
                            cuda_device
                        }
                        Err(_) => {
                            println!("⚠️  GPU não disponível, usando CPU");
                            candle_core::Device::Cpu
                        }
                    }
                }
            };
            
            println!("💬 Modo chat ativado! Digite 'quit' ou 'exit' para terminar.");
            println!("🤖 Aguardando suas mensagens...");
            interactive_chat(&device, educational, show_tensors)?
        }
    }
    
    // ✅ **FINALIZAÇÃO BEM-SUCEDIDA**
    // Retorna Ok(()) indicando que tudo correu bem
    println!("✨ Execução concluída com sucesso!");
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
    let mut trainer = Trainer::new(model, tokenizer, device.clone());
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
    let logger = EducationalLogger::new()
        .with_verbosity(educational)
        .with_tensor_info(show_tensors)
        .with_attention_maps(false);
    
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
    let logger = EducationalLogger::new()
        .with_verbosity(educational)
        .with_tensor_info(show_tensors)
        .with_attention_maps(false);
    
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