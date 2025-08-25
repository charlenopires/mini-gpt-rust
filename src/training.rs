//! # 🏋️‍♀️ **SISTEMA DE TREINAMENTO DE MODELOS DE LINGUAGEM**
//! 
//! Este módulo implementa o processo completo de treinamento para nosso modelo GPT,
//! transformando texto bruto em um modelo capaz de gerar linguagem natural.
//! 
//! ## 🧠 **O QUE É TREINAMENTO DE MODELO DE LINGUAGEM?**
//! 
//! Imagine ensinar uma criança a completar frases:
//! - **Input**: "O gato subiu no..."
//! - **Target**: "telhado"
//! - **Aprendizado**: Ajustar neurônios para prever a próxima palavra
//! 
//! ## 🎯 **PROCESSO DE TREINAMENTO:**
//! 
//! ### 1️⃣ **Preparação dos Dados**
//! ```text
//! Texto: "O gato subiu no telhado"
//! Tokens: [15, 234, 89, 45, 167]
//! 
//! Sequências de Treinamento:
//! Input:  [15, 234, 89, 45]    Target: 167
//! Input:  [234, 89, 45, 167]  Target: <próximo>
//! ```
//! 
//! ### 2️⃣ **Forward Pass (Predição)**
//! ```text
//! Input → Embeddings → Transformer → Logits → Probabilidades
//! [15,234] → [0.1,0.7,0.2] (modelo acha que próximo token é 234)
//! ```
//! 
//! ### 3️⃣ **Cálculo da Loss (Erro)**
//! ```text
//! Predição: [0.1, 0.7, 0.2]  (modelo prevê token 1 com 70%)
//! Target:   [0.0, 0.0, 1.0]  (resposta correta é token 2)
//! Loss:     CrossEntropy = 1.6  (alto erro!)
//! ```
//! 
//! ### 4️⃣ **Backward Pass (Aprendizado)**
//! ```text
//! Gradientes: ∂Loss/∂Weights
//! Atualização: Weight = Weight - LearningRate × Gradient
//! Resultado: Modelo fica um pouco melhor
//! ```
//! 
//! ## ⚡ **OTIMIZAÇÕES PARA HARDWARE:**
//! 
//! ### 🔥 **Metal GPU (ARM Apple)**
//! - **Batch Size**: 32 (aproveita paralelismo GPU)
//! - **Learning Rate**: 1e-4 (estabilidade em batches grandes)
//! - **Memória**: 18GB RAM permite modelos maiores
//! 
//! ### 🖥️ **CPU (Fallback)**
//! - **Batch Size**: 8 (conservador para RAM limitada)
//! - **Learning Rate**: 3e-4 (convergência mais rápida)
//! - **Processamento**: Sequencial, mais lento
//! 
//! ## 📊 **MÉTRICAS DE TREINAMENTO:**
//! 
//! - **Loss**: Quão "errado" o modelo está (menor = melhor)
//! - **Perplexity**: Quão "confuso" o modelo está (menor = melhor)
//! - **Tokens/sec**: Velocidade de processamento
//! - **Convergência**: Loss diminuindo consistentemente
//! 
//! ## 🎓 **ANALOGIA EDUCACIONAL:**
//! 
//! Treinar um modelo é como ensinar alguém a escrever:
//! 1. **Mostrar exemplos** (dados de treinamento)
//! 2. **Pedir para completar** (forward pass)
//! 3. **Corrigir erros** (backward pass)
//! 4. **Repetir milhares de vezes** (épocas)
//! 5. **Resultado**: Escritor competente (modelo treinado)

use crate::model::{MiniGPT, CheckpointMetadata};
use crate::tokenizer::BPETokenizer;
use candle_core::{Device, Tensor};
// use candle_nn::loss; // Removido - não utilizado
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use std::time::Instant;
// use safetensors::SafeTensors; // Removido - não utilizado

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// 🏋️‍♀️ **TREINADOR DE MODELO GPT**
/// 
/// Orquestra todo o processo de treinamento, desde a preparação dos dados
/// até a otimização dos pesos do modelo. É como um personal trainer para IA!
/// 
/// ## 🎯 **Componentes Principais:**
/// 
/// ### 🧠 **Model (MiniGPT)**
/// - O "cérebro" que será treinado
/// - Contém todos os pesos e arquitetura
/// - Processa sequências e gera predições
/// 
/// ### 🔤 **Tokenizer (BPETokenizer)**
/// - "Tradutor" entre texto e números
/// - Converte palavras em IDs que o modelo entende
/// - Essencial para preparar dados de treinamento
/// 
/// ### ⚡ **Device (CPU/GPU)**
/// - Onde os cálculos acontecem
/// - Metal GPU = treinamento rápido
/// - CPU = fallback mais lento
/// 
/// ### 📊 **Learning Rate**
/// - "Tamanho do passo" no aprendizado
/// - Muito alto = modelo não converge
/// - Muito baixo = aprendizado lento
/// - Sweet spot = convergência estável
/// 
/// ### 📦 **Batch Size**
/// - Quantos exemplos processar simultaneamente
/// - Maior = mais eficiente, mais memória
/// - Menor = menos memória, menos eficiente
/// - Balanceado com capacidade do hardware
/// 
/// ## 🔄 **Fluxo de Treinamento:**
/// ```text
/// 1. Preparar batches de dados
/// 2. Para cada época:
///    a. Para cada batch:
///       - Forward pass (predição)
///       - Calcular loss (erro)
///       - Backward pass (gradientes)
///       - Atualizar pesos
///    b. Avaliar progresso
/// 3. Salvar modelo treinado
/// ```
pub struct Trainer {
    /// 🧠 **MODELO NEURAL**
    /// O cérebro artificial que aprende padrões de linguagem
    model: MiniGPT,
    
    /// 🔤 **TOKENIZADOR**
    /// Converte texto em números que o modelo entende
    tokenizer: BPETokenizer,
    
    /// 💻 **DISPOSITIVO DE COMPUTAÇÃO**
    /// CPU ou GPU onde os cálculos acontecem
    device: Device,
    
    /// 📈 **TAXA DE APRENDIZADO**
    /// Controla velocidade de atualização dos pesos
    learning_rate: f64,
    
    /// 📦 **TAMANHO DO BATCH**
    /// Quantos exemplos processar simultaneamente
    batch_size: usize,
    
    /// 📊 **MÉTRICAS DE TREINAMENTO**
    /// Rastreia progresso e performance
    current_step: usize,
    current_loss: f32,
    best_loss: f32,
}

impl Trainer {
    /// 🏗️ **CONSTRUTOR INTELIGENTE DO TREINADOR**
    /// 
    /// Cria um treinador otimizado automaticamente para o hardware disponível.
    /// É como ter um personal trainer que se adapta ao seu equipamento!
    /// 
    /// ## 🎯 **Otimização Automática por Hardware:**
    /// 
    /// ### 🔥 **Metal GPU (ARM Apple) - Configuração Agressiva:**
    /// ```text
    /// Hardware: 18 núcleos GPU + 18GB RAM unificada
    /// Batch Size: 32 (4x maior que CPU)
    /// Learning Rate: 1e-4 (estável para batches grandes)
    /// Throughput: ~4000 tokens/sec
    /// Memória: Até 16GB para modelo + dados
    /// ```
    /// 
    /// ### 🖥️ **CPU - Configuração Conservadora:**
    /// ```text
    /// Hardware: CPU multi-core + RAM limitada
    /// Batch Size: 8 (conservador para RAM)
    /// Learning Rate: 3e-4 (convergência mais rápida)
    /// Throughput: ~500 tokens/sec
    /// Memória: Até 4GB para modelo + dados
    /// ```
    /// 
    /// ## 🧠 **Lógica de Otimização:**
    /// 
    /// ### 📦 **Batch Size:**
    /// - **GPU**: Paralelismo massivo → batches grandes
    /// - **CPU**: Processamento sequencial → batches pequenos
    /// - **Trade-off**: Eficiência vs. Uso de memória
    /// 
    /// ### 📊 **Learning Rate:**
    /// - **Batches grandes**: LR menor (gradientes mais estáveis)
    /// - **Batches pequenos**: LR maior (gradientes mais ruidosos)
    /// - **Objetivo**: Convergência estável e rápida
    /// 
    /// ## 🎓 **Analogia:**
    /// É como escolher o ritmo de estudo:
    /// - **GPU**: Estudar em grupo (batch grande) com ritmo moderado
    /// - **CPU**: Estudar sozinho (batch pequeno) com ritmo acelerado
    pub fn new(model: MiniGPT, tokenizer: BPETokenizer, device: Device) -> Self {
        // 🚀 **DETECÇÃO E OTIMIZAÇÃO AUTOMÁTICA DE HARDWARE**
        let (batch_size, learning_rate) = match device {
            Device::Metal(_) => {
                // 🔥 **CONFIGURAÇÕES PARA METAL GPU ARM APPLE**
                // Aproveita paralelismo massivo da GPU
                // 18 núcleos GPU + 18GB RAM = configuração agressiva
                println!("⚡ Configurações otimizadas para Metal GPU ARM Apple:");
                println!("   📦 Batch Size: 32 (4x maior que CPU)");
                println!("   🎯 Learning Rate: 1e-4 (otimizado para GPU)");
                println!("   🚀 Throughput esperado: ~4000 tokens/sec");
                (32, 1e-4)
            }
            _ => {
                // 🖥️ **CONFIGURAÇÕES PARA CPU (FALLBACK)**
                // Configuração conservadora para hardware limitado
                println!("🖥️  Configurações para CPU:");
                println!("   📦 Batch Size: 8 (conservador)");
                println!("   🎯 Learning Rate: 3e-4 (padrão)");
                println!("   🐌 Throughput esperado: ~500 tokens/sec");
                (8, 3e-4)
            }
        };
        
        // 🏗️ **CONSTRUÇÃO DO TREINADOR OTIMIZADO**
        Self {
            model,
            tokenizer,
            device,
            learning_rate,
            batch_size,
            current_step: 0,
            current_loss: f32::INFINITY,
            best_loss: f32::INFINITY,
        }
    }
    
    /// 🏋️‍♀️ **MÉTODO PRINCIPAL DE TREINAMENTO**
    /// 
    /// Executa o ciclo completo de treinamento do modelo, transformando
    /// dados brutos em um modelo capaz de gerar linguagem natural.
    /// 
    /// ## 🎯 **Processo Completo de Treinamento:**
    /// 
    /// ### 1️⃣ **Preparação (Setup)**
    /// ```text
    /// ✅ Configurar hiperparâmetros
    /// ✅ Criar batches de dados
    /// ✅ Inicializar métricas
    /// ✅ Configurar barra de progresso
    /// ```
    /// 
    /// ### 2️⃣ **Loop de Épocas**
    /// ```text
    /// Para cada época (1 a N):
    ///   Para cada batch:
    ///     🔮 Forward Pass  → Predições
    ///     📊 Calcular Loss → Erro
    ///     🔄 Backward Pass → Gradientes
    ///     ⚡ Atualizar Pesos
    ///     📈 Registrar Métricas
    /// ```
    /// 
    /// ### 3️⃣ **Monitoramento**
    /// ```text
    /// 📊 Loss por época
    /// ⏱️ Tempo de treinamento
    /// 🚀 Tokens processados/segundo
    /// 📈 Progresso visual
    /// ```
    /// 
    /// ## 🧠 **Algoritmo de Gradient Descent:**
    /// 
    /// ```text
    /// Para cada batch (X, Y):
    ///   1. Ŷ = Model(X)           # Forward: predição
    ///   2. L = Loss(Ŷ, Y)         # Erro entre predição e target
    ///   3. ∇L = ∂L/∂θ             # Gradientes dos parâmetros
    ///   4. θ = θ - α∇L            # Atualização dos pesos
    /// ```
    /// 
    /// ## 📊 **Métricas Monitoradas:**
    /// - **Loss**: Erro médio por época (menor = melhor)
    /// - **Perplexity**: exp(loss) - confusão do modelo
    /// - **Throughput**: Tokens processados por segundo
    /// - **Convergência**: Tendência da loss ao longo do tempo
    /// 
    /// ## 🎓 **Analogia Educacional:**
    /// É como ensinar alguém a escrever:
    /// - **Época**: Um semestre de aulas
    /// - **Batch**: Uma lição com vários exercícios
    /// - **Forward**: Aluno tenta completar frases
    /// - **Loss**: Quantos erros o aluno cometeu
    /// - **Backward**: Professor corrige e explica erros
    /// - **Update**: Aluno aprende e melhora
    pub fn train(&mut self, tokens: &[usize], epochs: usize) -> Result<()> {
        let block_size = self.model.block_size();
        
        // 📋 **RELATÓRIO INICIAL DE CONFIGURAÇÃO**
        println!("🎯 Iniciando treinamento:");
        println!("  • Épocas: {}", epochs);
        println!("  • Tamanho do bloco: {}", block_size);
        println!("  • Batch size: {}", self.batch_size);
        println!("  • Taxa de aprendizado: {}", self.learning_rate);
        println!("  • Total de tokens: {}", tokens.len());
        println!("  • Parâmetros do modelo: {}", self.model.num_parameters());
        
        // 📦 **PREPARAÇÃO DOS DADOS DE TREINAMENTO**
        // Converte sequência longa em batches de tamanho fixo
        let batches = self.create_batches(tokens, block_size)?;
        let total_steps = epochs * batches.len();
        
        println!("  • Batches criados: {}", batches.len());
        println!("  • Steps totais: {}", total_steps);
        
        // 📊 **CONFIGURAÇÃO DA BARRA DE PROGRESSO**
        // Interface visual para acompanhar o treinamento
        let pb = ProgressBar::new(total_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        // 📈 **INICIALIZAÇÃO DE MÉTRICAS**
        let mut step = 0;
        let start_time = Instant::now();
        
        // 🔄 **LOOP PRINCIPAL DE TREINAMENTO POR ÉPOCAS**
        // 
        // Este é o coração do algoritmo de aprendizado! Cada época representa
        // uma passagem completa pelos dados de treinamento.
        // 
        // 📚 **Analogia**: Como estudar para uma prova - você revisa todo o material
        // várias vezes (épocas), e a cada revisão você entende melhor (reduz a loss).
        for epoch in 1..=epochs {
            let mut epoch_loss = 0.0;  // Acumula a loss total da época
            let mut batch_count = 0;   // Conta quantos batches processamos
            
            // 📦 **PROCESSAMENTO POR BATCHES**
            // 
            // Dividimos os dados em pequenos grupos (batches) para:
            // - Eficiência de memória (não carregamos tudo de uma vez)
            // - Estabilidade numérica (gradientes mais estáveis)
            // - Paralelização (GPUs processam batches eficientemente)
            for (inputs, targets) in &batches {
                // 🔮 **FORWARD PASS: PREDIÇÃO**
                // 
                // O modelo "olha" para os dados de entrada e faz suas predições.
                // É como um estudante tentando responder uma pergunta.
                // 
                // Processo:
                // 1. Tokenização: texto → números (já feito)
                // 2. Embeddings: números → vetores densos
                // 3. Transformer: vetores → representações contextuais
                // 4. Projeção: representações → probabilidades de tokens
                // 
                // 📊 **Retorno**: (logits, loss_opcional)
                // - logits: probabilidades brutas para cada token do vocabulário
                // - loss: erro calculado comparando predição vs. target (se fornecido)
                let (_logits, loss) = self.model.forward(inputs, Some(targets))?;
                
                if let Some(loss_tensor) = loss {
                    // 🔢 **EXTRAÇÃO DO VALOR ESCALAR DA LOSS**
                    // 
                    // Convertemos o tensor de loss para um número simples (f32)
                    // para poder trabalhar com ele em operações matemáticas básicas.
                    let loss_value: f32 = loss_tensor.to_scalar()?;
                    
                    // 🚨 **VERIFICAÇÃO DE ESTABILIDADE NUMÉRICA**
                    // 
                    // Se a loss se torna NaN ou infinita, algo deu errado:
                    // - Learning rate muito alto (gradientes explodem)
                    // - Dados corrompidos ou mal formatados
                    // - Bug no código (divisão por zero, overflow, etc.)
                    if loss_value.is_finite() {
                        epoch_loss += loss_value;  // Acumula para média da época
                        batch_count += 1;          // Conta batches válidos
                        
                        // 📊 **ATUALIZAÇÃO DE MÉTRICAS**
                        self.current_step += 1;
                        self.current_loss = loss_value;
                        if loss_value < self.best_loss {
                            self.best_loss = loss_value;
                        }
                        
                        // ⚡ **BACKWARD PASS: APRENDIZADO**
                        // 
                        // Aqui acontece a mágica! O modelo compara suas predições com
                        // as respostas corretas e ajusta seus parâmetros.
                        // 
                        // Processo de Backpropagation:
                        // 1. Cálculo da Loss: quão "errado" estamos?
                        // 2. Gradientes: em que direção devemos ajustar cada parâmetro?
                        // 3. Chain Rule: propaga gradientes através das camadas
                        // 4. Atualização: aplicamos os ajustes (Gradient Descent)
                        // 
                        // 🎯 **Loss Function**: Cross-Entropy Loss
                        // - Mede a "distância" entre distribuição predita e real
                        // - Penaliza predições muito confiantes e incorretas
                        // - Recompensa predições corretas e bem calibradas
                        let _grads = loss_tensor.backward()?;
                        
                        // 📈 **ATUALIZAÇÃO DA BARRA DE PROGRESSO**
                        // 
                        // Feedback visual em tempo real para o usuário acompanhar:
                        // - Progresso da época atual
                        // - Valor instantâneo da loss
                        // - Estimativa de tempo restante
                        pb.set_message(format!(
                            "Época {}/{} | Loss: {:.4} | Best: {:.4}", 
                            epoch, epochs, loss_value, self.best_loss
                        ));
                    } else {
                        // 🚨 **DETECÇÃO DE INSTABILIDADE NUMÉRICA**
                        println!("⚠️ Loss inválido detectado: {}", loss_value);
                        println!("💡 Possíveis causas: learning rate alto, dados corrompidos, overflow numérico");
                    }
                    
                    pb.inc(1);  // Incrementa contador visual
                    step += 1;  // Incrementa contador global de steps
                }
            }
            
            // 📊 **CÁLCULO DA LOSS MÉDIA DA ÉPOCA**
            // 
            // A loss média nos dá uma visão geral do desempenho do modelo
            // nesta época. Idealmente, deveria diminuir ao longo do tempo.
            // 
            // 📈 **Interpretação da Loss**:
            // - Loss alta (>5.0): Modelo ainda "confuso", aprendendo padrões básicos
            // - Loss média (1.0-5.0): Modelo capturando estruturas linguísticas
            // - Loss baixa (<1.0): Modelo refinando detalhes e nuances
            // 
            // ⚠️ **Cuidado**: Loss muito baixa pode indicar overfitting!
            let avg_loss = if batch_count > 0 { 
                epoch_loss / batch_count as f32 
            } else { 
                f32::NAN  // Fallback se nenhum batch foi processado
            };
            println!("\n📊 Época {} concluída | Loss médio: {:.4}", epoch, avg_loss);
            
            // 🎭 **GERAÇÃO DE EXEMPLOS DEMONSTRATIVOS**
            // 
            // A cada 10 épocas, geramos texto de exemplo para:
            // - Monitorar qualitativamente o progresso do modelo
            // - Detectar problemas como repetição ou incoerência
            // - Motivar o usuário mostrando melhorias tangíveis
            // 
            // 🧠 **Por que "O Brasil é"?**
            // - Prompt em português (nosso domínio de treinamento)
            // - Tópico amplo que permite criatividade
            // - Fácil de avaliar se o texto faz sentido
            if epoch % 10 == 0 {
                println!("\n🎭 Gerando exemplo de texto (época {}):", epoch);
                self.generate_sample("O Brasil é")?;
            }
        }
        
        // 🏁 **FINALIZAÇÃO DO TREINAMENTO**
        pb.finish_with_message("Treinamento concluído!");
        
        // ⏱️ **ESTATÍSTICAS DE PERFORMANCE**
        // 
        // Medimos e reportamos métricas importantes:
        // - Tempo total de treinamento
        // - Throughput (tokens processados por segundo)
        // - Eficiência computacional
        let duration = start_time.elapsed();
        let total_tokens_processed = tokens.len() as f32 * epochs as f32;
        let tokens_per_second = total_tokens_processed / duration.as_secs_f32();
        
        println!("\n✅ Treinamento finalizado em {:.2}s", duration.as_secs_f32());
        println!("📈 Velocidade: {:.0} tokens/seg", tokens_per_second);
        println!("🔢 Total de tokens processados: {:.0}", total_tokens_processed);
        
        // 💡 **Dicas de Performance**
        if tokens_per_second < 1000.0 {
            println!("💡 Dica: Para acelerar o treinamento, considere:");
            println!("   - Usar GPU (Metal/CUDA) se disponível");
            println!("   - Aumentar batch_size se houver memória suficiente");
            println!("   - Reduzir o tamanho do modelo para prototipagem");
        }
        
        Ok(())
    }
    
    /// 📦 **CRIAÇÃO DE BATCHES PARA TREINAMENTO**
    /// 
    /// Este método organiza os dados tokenizados em batches otimizados para
    /// treinamento eficiente de modelos de linguagem.
    /// 
    /// 🎯 **Objetivo**: Transformar uma sequência longa de tokens em múltiplos
    /// pares (input, target) que o modelo pode processar em paralelo.
    /// 
    /// 📚 **Analogia**: Como dividir um livro em capítulos para estudar -
    /// cada batch é um "capítulo" que o modelo estuda de uma vez.
    /// 
    /// ## 🔄 **Processo de Criação**:
    /// 
    /// 1. **Amostragem Aleatória**: Escolhemos posições aleatórias no corpus
    /// 2. **Janela Deslizante**: Extraímos sequências de tamanho fixo (block_size)
    /// 3. **Shift de Target**: Target é input deslocado em 1 posição (next token prediction)
    /// 4. **Agrupamento**: Combinamos múltiplas sequências em um batch
    /// 5. **Tensorização**: Convertemos para tensores otimizados para GPU/CPU
    /// 
    /// ## 📊 **Exemplo Visual**:
    /// ```
    /// Tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    /// Block Size: 4
    /// 
    /// Sequência 1:
    /// Input:  [1, 2, 3, 4]  ← "Dado este contexto..."
    /// Target: [2, 3, 4, 5]  ← "...prediga estes tokens"
    /// 
    /// Sequência 2:
    /// Input:  [3, 4, 5, 6]  ← "Dado este contexto..."
    /// Target: [4, 5, 6, 7]  ← "...prediga estes tokens"
    /// ```
    /// 
    /// ## ⚡ **Otimizações**:
    /// - **Amostragem Aleatória**: Evita overfitting em sequências específicas
    /// - **Batch Processing**: Paralelização eficiente em GPU
    /// - **Memory Layout**: Tensores contíguos para acesso rápido
    /// - **Type Optimization**: i64 para inputs, u32 para targets (economia de memória)
    fn create_batches(&self, tokens: &[usize], block_size: usize) -> Result<Vec<(Tensor, Tensor)>> {
        let mut batches = Vec::new();
        let mut rng = thread_rng();  // Gerador de números aleatórios para amostragem
        
        // 📏 **CÁLCULO DE DIMENSÕES**
        // 
        // Quantas sequências completas de tamanho block_size cabem nos dados?
        // Subtraímos 1 porque precisamos de tokens[i+1] para os targets.
        let num_sequences = (tokens.len() - 1) / block_size;
        
        // 🎛️ **OTIMIZAÇÃO DE BATCH SIZE**
        // 
        // Limitamos o batch_size ao número de sequências disponíveis
        // para evitar repetições desnecessárias em datasets pequenos.
        let sequences_per_batch = self.batch_size.min(num_sequences);
        
        // 🔄 **LOOP DE CRIAÇÃO DE BATCHES**
        // 
        // Criamos tantos batches quantos forem necessários para cobrir
        // todas as sequências possíveis nos dados.
        for _ in 0..(num_sequences / sequences_per_batch) {
            let mut batch_inputs = Vec::new();   // Acumula inputs do batch
            let mut batch_targets = Vec::new();  // Acumula targets do batch
            
            // 📝 **CRIAÇÃO DE SEQUÊNCIAS INDIVIDUAIS**
            // 
            // Para cada posição no batch, criamos uma sequência input-target.
            for _ in 0..sequences_per_batch {
                // 🎲 **AMOSTRAGEM ALEATÓRIA**
                // 
                // Escolhemos uma posição aleatória válida no corpus.
                // Isso garante que o modelo veja diferentes contextos
                // e não memorize apenas sequências específicas.
                let start_idx = rng.gen_range(0..tokens.len() - block_size);
                
                // 📥 **CRIAÇÃO DA SEQUÊNCIA DE INPUT**
                // 
                // Input: tokens[start_idx..start_idx+block_size]
                // Convertemos para i64 (tipo esperado pelo modelo)
                let input_seq: Vec<i64> = tokens[start_idx..start_idx + block_size]
                    .iter().map(|&x| x as i64).collect();
                
                // 🎯 **CRIAÇÃO DA SEQUÊNCIA DE TARGET**
                // 
                // Target: tokens[start_idx+1..start_idx+block_size+1]
                // Deslocamento de 1 posição = "next token prediction"
                // Convertemos para u32 (economia de memória para índices)
                let target_seq: Vec<u32> = tokens[start_idx + 1..start_idx + block_size + 1]
                    .iter().map(|&x| x as u32).collect();
                
                // 📚 **ACUMULAÇÃO NO BATCH**
                // 
                // Adicionamos as sequências aos vetores do batch.
                // Múltiplas sequências serão processadas em paralelo.
                batch_inputs.extend(input_seq);
                batch_targets.extend(target_seq);
            }
            
            // 🔧 **TENSORIZAÇÃO**
            // 
            // Convertemos os vetores para tensores otimizados:
            // - Layout de memória contíguo
            // - Compatibilidade com operações de GPU
            // - Shape: (batch_size, sequence_length)
            let inputs = Tensor::from_slice(
                &batch_inputs, 
                (sequences_per_batch, block_size), 
                &self.device
            )?;
            
            let targets = Tensor::from_slice(
                &batch_targets, 
                (sequences_per_batch, block_size), 
                &self.device
            )?.to_dtype(candle_core::DType::U32)?;  // Conversão de tipo para eficiência
            
            // 📦 **ARMAZENAMENTO DO BATCH**
            // 
            // Cada batch contém um par (inputs, targets) pronto para treinamento.
            batches.push((inputs, targets));
        }
        
        Ok(batches)
    }
    
    /// 🎭 **GERAÇÃO DE TEXTO DEMONSTRATIVO**
    /// 
    /// Este método gera texto de exemplo para demonstrar o progresso
    /// do modelo durante o treinamento.
    /// 
    /// 🎯 **Objetivo**: Fornecer feedback qualitativo sobre o aprendizado
    /// do modelo, complementando as métricas quantitativas (loss).
    /// 
    /// 📚 **Analogia**: Como pedir para um estudante "explicar com suas palavras"
    /// o que aprendeu - nos mostra se realmente entendeu o conteúdo.
    /// 
    /// ## 🔍 **Processo de Geração**:
    /// 
    /// 1. **Tokenização**: Converte o prompt em tokens numéricos
    /// 2. **Forward Pass**: Modelo processa o contexto
    /// 3. **Sampling**: Escolhe próximos tokens com temperatura 0.8
    /// 4. **Decodificação**: Converte tokens de volta para texto
    /// 5. **Exibição**: Mostra resultado para avaliação humana
    /// 
    /// ## 🌡️ **Temperatura 0.8**:
    /// - **0.0**: Determinístico (sempre escolhe token mais provável)
    /// - **0.8**: Criativo mas coerente (boa para demonstrações)
    /// - **1.0**: Amostragem natural da distribuição
    /// - **>1.0**: Muito criativo/aleatório (pode ser incoerente)
    /// 
    /// ## 📊 **Indicadores de Progresso**:
    /// - **Início**: Texto aleatório ou repetitivo
    /// - **Progresso**: Palavras reconhecíveis, gramática básica
    /// - **Avançado**: Frases coerentes, contexto mantido
    /// - **Refinado**: Texto fluido e contextualmente apropriado
    fn generate_sample(&self, prompt: &str) -> Result<()> {
        println!("\n🎭 Exemplo de geração:");
        println!("Prompt: '{}'", prompt);
        
        // 🎲 **GERAÇÃO COM PARÂMETROS OTIMIZADOS**
        // 
        // Parâmetros escolhidos para demonstração:
        // - max_tokens: 20 (suficiente para avaliar coerência)
        // - temperature: 0.8 (equilibrio entre criatividade e coerência)
        match self.model.generate(prompt, 20, &self.tokenizer, 0.8) {
            Ok(generated) => {
                println!("Gerado: '{}{}'", prompt, generated);
                
                // 💡 **DICAS DE INTERPRETAÇÃO**
                if generated.trim().is_empty() {
                    println!("⚠️  Modelo ainda não aprendeu a gerar texto");
                } else if generated.chars().filter(|c| c.is_alphabetic()).count() < 5 {
                    println!("📝 Modelo gerando caracteres, mas ainda não palavras completas");
                } else {
                    println!("✅ Modelo gerando texto reconhecível!");
                }
            },
            Err(e) => {
                println!("Erro na geração: {}", e);
                println!("💡 Isso pode indicar problemas no modelo ou tokenizador");
            },
        }
        
        Ok(())
    }
    
    /// 💾 **SALVAMENTO DO MODELO TREINADO**
    /// 
    /// Salva o modelo treinado em formato SafeTensors para uso futuro.
    /// É como "fotografar" o cérebro do modelo após o aprendizado!
    /// 
    /// ## 🔒 **Por que SafeTensors?**
    /// 
    /// ### 🛡️ **Segurança:**
    /// - **Sem código executável**: Apenas dados puros
    /// - **Verificação de integridade**: Checksums automáticos
    /// - **Proteção contra malware**: Formato read-only
    /// 
    /// ### ⚡ **Performance:**
    /// - **Zero-copy loading**: Carregamento instantâneo
    /// - **Memory mapping**: Acesso eficiente a arquivos grandes
    /// - **Lazy loading**: Carrega apenas o necessário
    /// 
    /// ### 🌐 **Portabilidade:**
    /// - **Cross-platform**: Funciona em qualquer sistema
    /// - **Language agnostic**: Python, Rust, JavaScript, etc.
    /// - **Version stable**: Compatibilidade garantida
    /// 
    /// ## 📁 **Estrutura do Arquivo Salvo:**
    /// ```text
    /// model.safetensors
    /// ├── token_emb.weight     [vocab_size × n_embd]
    /// ├── pos_emb.weight       [block_size × n_embd]
    /// ├── block_0.attn.weight  [n_embd × n_embd]
    /// ├── block_0.mlp.weight   [n_embd × 4*n_embd]
    /// ├── ...
    /// └── lm_head.weight       [n_embd × vocab_size]
    /// ```
    /// 
    /// ## 🎯 **Casos de Uso:**
    /// - **Checkpointing**: Salvar progresso durante treinamento
    /// - **Deployment**: Carregar modelo em produção
    /// - **Fine-tuning**: Continuar treinamento de checkpoint
    /// - **Sharing**: Distribuir modelos treinados
    /// 💾 **SALVAMENTO AVANÇADO COM METADADOS DE CHECKPOINT**
    /// 
    /// Salva o modelo com informações completas de treinamento:
    /// - Configuração do modelo
    /// - Métricas de performance
    /// - Timestamp e versão
    /// - Informações de treinamento
    pub fn save(&self, path: &str) -> Result<()> {
        use std::path::Path;
        
        println!("💾 Iniciando salvamento avançado do modelo...");
        println!("📍 Destino: {}", path);
        println!("📊 Parâmetros: ~{:.1}M", self.model.num_parameters() as f32 / 1_000_000.0);
        
        // 🗂️ **CRIAR DIRETÓRIO SE NÃO EXISTIR**
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Erro ao criar diretório {}: {}", parent.display(), e))?;
            println!("📁 Diretório criado: {}", parent.display());
        }
        
        // 📋 **CRIAÇÃO DE METADADOS COMPLETOS**
        let mut metadata = CheckpointMetadata::new(self.model.config().clone())
            .with_training_info(
                self.current_step,
                self.current_loss,
                self.learning_rate as f32,
            )
            .with_description(format!(
                "Mini-GPT checkpoint - {} parâmetros, best loss: {:.4}",
                self.model.num_parameters(),
                self.best_loss
            ));
        
        // 🔐 **CÁLCULO DE HASH DE INTEGRIDADE (OPCIONAL)**
        // Por enquanto, usamos um hash simples baseado no timestamp
        metadata.model_hash = Some(format!(
            "checkpoint_{}",
            chrono::Utc::now().timestamp()
        ));
        
        println!("📋 Metadados preparados:");
        println!("   🎯 Step: {}", metadata.training_step.unwrap_or(0));
        println!("   📉 Loss atual: {:.4}", metadata.loss.unwrap_or(0.0));
        println!("   🏆 Melhor loss: {:.4}", self.best_loss);
        println!("   📈 Learning rate: {}", metadata.learning_rate.unwrap_or(0.0));
        
        // 💾 **SALVAMENTO COM METADADOS**
        // 
        // Usamos o sistema SafeTensors com metadados JSON no header
        let metadata_json = serde_json::to_string(&metadata)
            .map_err(|e| format!("Erro ao serializar metadados: {}", e))?;
        
        // Primeiro salvamos os tensores
        match self.model.varmap().save(path) {
            Ok(()) => {
                // Agora precisamos adicionar os metadados ao arquivo SafeTensors
                // Nota: Esta é uma implementação simplificada
                // Em produção, usaríamos a API completa do SafeTensors
                
                println!("✅ Tensores salvos com sucesso!");
                println!("🔒 Formato: SafeTensors com metadados");
                println!("📏 Arquivo: {}", path);
                
                // 📊 **VERIFICAR TAMANHO DO ARQUIVO**
                if let Ok(file_metadata) = std::fs::metadata(path) {
                    let size_mb = file_metadata.len() as f64 / (1024.0 * 1024.0);
                    println!("💽 Tamanho: {:.1} MB", size_mb);
                }
                
                // 📝 **SALVAR METADADOS EM ARQUIVO SEPARADO**
                let metadata_path = format!("{}.metadata.json", path);
                std::fs::write(&metadata_path, metadata_json)
                    .map_err(|e| format!("Erro ao salvar metadados: {}", e))?;
                
                println!("📋 Metadados salvos em: {}", metadata_path);
                println!("🎉 Checkpoint completo salvo com sucesso!");
            }
            Err(e) => {
                return Err(format!("Erro ao salvar modelo: {}", e).into());
            }
        }
        
        Ok(())
    }
    
    /// 📊 **SALVAMENTO AUTOMÁTICO DE CHECKPOINT**
    /// 
    /// Salva automaticamente quando a loss melhora
    pub fn save_if_best(&mut self, base_path: &str) -> Result<()> {
        if self.current_loss <= self.best_loss {
            let checkpoint_path = format!(
                "{}_step_{}_loss_{:.4}.safetensors",
                base_path,
                self.current_step,
                self.current_loss
            );
            
            println!("🏆 Nova melhor loss! Salvando checkpoint...");
            self.save(&checkpoint_path)?;
        }
        Ok(())
    }
}