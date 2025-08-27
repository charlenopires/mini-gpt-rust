//! # 🧠 Demonstração do Modelo Mini-GPT
//!
//! Este exemplo demonstra como construir, configurar e usar um modelo GPT completo,
//! desde a inicialização até a geração de texto.
//!
//! ## 🎯 O que você vai aprender:
//! - Como configurar um modelo GPT com diferentes tamanhos
//! - Processo de forward pass e cálculo de loss
//! - Geração de texto com diferentes estratégias
//! - Análise de parâmetros e complexidade computacional
//! - Salvamento e carregamento de checkpoints
//! - Otimizações de memória e performance

use std::collections::HashMap;
use std::time::Instant;
use anyhow::Result;

// Estruturas simplificadas para demonstração
// Em um projeto real, estas viriam de mini_gpt_rust::model

#[derive(Debug, Clone)]
struct GPTConfig {
    pub vocab_size: usize,   // 📚 Tamanho do vocabulário
    pub n_embd: usize,       // 🧮 Dimensão dos embeddings
    pub n_head: usize,       // 👁️ Número de cabeças de atenção
    pub n_layer: usize,      // 🏗️ Número de blocos transformer
    pub block_size: usize,   // 📏 Tamanho máximo do contexto
    pub dropout: f32,        // 🎲 Taxa de dropout
}

impl GPTConfig {
    /// 🏠 Configuração para modelo pequeno (para testes)
    fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            n_embd: 128,
            n_head: 4,
            n_layer: 4,
            block_size: 64,
            dropout: 0.1,
        }
    }
    
    /// 🏢 Configuração para modelo pequeno (experimentos)
    fn small() -> Self {
        Self {
            vocab_size: 5000,
            n_embd: 256,
            n_head: 8,
            n_layer: 6,
            block_size: 128,
            dropout: 0.1,
        }
    }
    
    /// 🏭 Configuração para modelo médio (produção leve)
    fn medium() -> Self {
        Self {
            vocab_size: 10000,
            n_embd: 512,
            n_head: 8,
            n_layer: 12,
            block_size: 256,
            dropout: 0.1,
        }
    }
    
    /// 🚀 Configuração para modelo grande (alta performance)
    fn large() -> Self {
        Self {
            vocab_size: 50000,
            n_embd: 768,
            n_head: 12,
            n_layer: 24,
            block_size: 512,
            dropout: 0.1,
        }
    }
    
    /// 📊 Calcula número total de parâmetros
    fn num_parameters(&self) -> usize {
        // Token embeddings: vocab_size × n_embd
        let token_emb = self.vocab_size * self.n_embd;
        
        // Position embeddings: block_size × n_embd
        let pos_emb = self.block_size * self.n_embd;
        
        // Transformer blocks
        let mut transformer_params = 0;
        for _ in 0..self.n_layer {
            // Multi-head attention
            let qkv_proj = 3 * self.n_embd * self.n_embd; // Q, K, V projections
            let attn_out = self.n_embd * self.n_embd;     // Output projection
            let attn_total = qkv_proj + attn_out;
            
            // Feed-forward network
            let ff_hidden = 4 * self.n_embd; // Typical expansion factor
            let ff_up = self.n_embd * ff_hidden;
            let ff_down = ff_hidden * self.n_embd;
            let ff_total = ff_up + ff_down;
            
            // Layer norms (2 per block)
            let ln_params = 2 * self.n_embd * 2; // weight + bias
            
            transformer_params += attn_total + ff_total + ln_params;
        }
        
        // Final layer norm
        let final_ln = self.n_embd * 2;
        
        // Language model head
        let lm_head = self.n_embd * self.vocab_size;
        
        token_emb + pos_emb + transformer_params + final_ln + lm_head
    }
    
    /// 💾 Estima uso de memória em MB
    fn memory_usage_mb(&self) -> f32 {
        let params = self.num_parameters();
        let bytes_per_param = 4; // float32
        let total_bytes = params * bytes_per_param;
        
        // Adiciona overhead para ativações (estimativa)
        let activation_overhead = 1.5;
        
        (total_bytes as f32 * activation_overhead) / (1024.0 * 1024.0)
    }
    
    /// ⚡ Estima FLOPs por token
    fn flops_per_token(&self) -> usize {
        let mut total_flops = 0;
        
        // Embedding lookup (negligível)
        
        // Transformer blocks
        for _ in 0..self.n_layer {
            // Attention: Q@K^T
            let qk_flops = self.block_size * self.n_embd * self.block_size;
            
            // Attention: softmax@V
            let av_flops = self.block_size * self.block_size * self.n_embd;
            
            // Feed-forward
            let ff_flops = 2 * self.n_embd * (4 * self.n_embd); // up + down projections
            
            total_flops += qk_flops + av_flops + ff_flops;
        }
        
        // Final projection
        total_flops += self.n_embd * self.vocab_size;
        
        total_flops
    }
}

/// 🎭 Estrutura principal do modelo (simplificada para demonstração)
#[derive(Debug)]
struct MiniGPT {
    config: GPTConfig,
    parameters: HashMap<String, Vec<f32>>, // Simulação de parâmetros
    training_step: usize,
    device: String,
}

impl MiniGPT {
    /// 🏗️ Cria um novo modelo com configuração especificada
    fn new(config: GPTConfig, device: &str) -> Result<Self> {
        println!("🏗️ Inicializando modelo Mini-GPT...");
        println!("📊 Configuração:");
        println!("  - Vocabulário: {} tokens", config.vocab_size);
        println!("  - Embedding: {} dimensões", config.n_embd);
        println!("  - Atenção: {} cabeças", config.n_head);
        println!("  - Camadas: {} blocos", config.n_layer);
        println!("  - Contexto: {} tokens", config.block_size);
        println!("  - Dropout: {:.1}%", config.dropout * 100.0);
        
        let num_params = config.num_parameters();
        let memory_mb = config.memory_usage_mb();
        
        println!("\n💾 Recursos computacionais:");
        println!("  - Parâmetros: {:.2}M", num_params as f32 / 1_000_000.0);
        println!("  - Memória: {:.1} MB", memory_mb);
        println!("  - FLOPs/token: {:.2}M", config.flops_per_token() as f32 / 1_000_000.0);
        
        // Simula inicialização de parâmetros
        let mut parameters = HashMap::new();
        
        // Token embeddings
        parameters.insert(
            "token_embedding".to_string(),
            vec![0.0; config.vocab_size * config.n_embd]
        );
        
        // Position embeddings
        parameters.insert(
            "position_embedding".to_string(),
            vec![0.0; config.block_size * config.n_embd]
        );
        
        // Transformer blocks (simplificado)
        for i in 0..config.n_layer {
            parameters.insert(
                format!("block_{}_attention", i),
                vec![0.0; config.n_embd * config.n_embd * 4] // Q, K, V, O
            );
            parameters.insert(
                format!("block_{}_ffn", i),
                vec![0.0; config.n_embd * config.n_embd * 8] // up + down
            );
        }
        
        // Language model head
        parameters.insert(
            "lm_head".to_string(),
            vec![0.0; config.n_embd * config.vocab_size]
        );
        
        println!("✅ Modelo inicializado com sucesso!");
        
        Ok(Self {
            config,
            parameters,
            training_step: 0,
            device: device.to_string(),
        })
    }
    
    /// 🔄 Simula forward pass do modelo
    fn forward(&self, input_ids: &[usize], targets: Option<&[usize]>) -> Result<(Vec<f32>, Option<f32>)> {
        let batch_size = 1;
        let seq_len = input_ids.len();
        
        println!("🔄 Executando forward pass...");
        println!("  - Input shape: [{}]", seq_len);
        println!("  - Batch size: {}", batch_size);
        
        // Simula processamento através das camadas
        let start_time = Instant::now();
        
        // 1. Token + Position Embeddings
        println!("  📚 Aplicando embeddings...");
        let mut hidden_states = vec![vec![0.0; self.config.n_embd]; seq_len];
        
        // 2. Transformer Blocks
        for layer in 0..self.config.n_layer {
            println!("  🧠 Processando bloco {}...", layer + 1);
            
            // Simula atenção multi-cabeça
            self.simulate_attention(&mut hidden_states, layer)?;
            
            // Simula feed-forward
            self.simulate_feedforward(&mut hidden_states, layer)?;
        }
        
        // 3. Final Layer Norm + LM Head
        println!("  🎪 Aplicando projeção final...");
        let logits = self.simulate_lm_head(&hidden_states)?;
        
        let forward_time = start_time.elapsed();
        println!("  ⏱️ Tempo de forward: {:?}", forward_time);
        
        // Calcula loss se targets fornecidos
        let loss = if let Some(targets) = targets {
            Some(self.compute_loss(&logits, targets)?)
        } else {
            None
        };
        
        Ok((logits, loss))
    }
    
    /// 👁️ Simula camada de atenção multi-cabeça
    fn simulate_attention(&self, hidden_states: &mut Vec<Vec<f32>>, layer: usize) -> Result<()> {
        let seq_len = hidden_states.len();
        let head_dim = self.config.n_embd / self.config.n_head;
        
        println!("    👁️ Atenção: {} cabeças, {} dim/cabeça", self.config.n_head, head_dim);
        
        // Simula cálculo de atenção para cada cabeça
        for head in 0..self.config.n_head {
            // Q @ K^T / sqrt(d_k)
            let attention_scores = self.compute_attention_scores(seq_len, head_dim)?;
            
            // Softmax
            let attention_weights = self.softmax(&attention_scores)?;
            
            // Attention @ V
            self.apply_attention_weights(&attention_weights, hidden_states, head)?;
        }
        
        Ok(())
    }
    
    /// 🧮 Simula cálculo de scores de atenção
    fn compute_attention_scores(&self, seq_len: usize, head_dim: usize) -> Result<Vec<Vec<f32>>> {
        let mut scores = vec![vec![0.0; seq_len]; seq_len];
        
        // Simula Q @ K^T
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Simula produto escalar normalizado
                let score = if j <= i { // Causal mask
                    (i as f32 - j as f32).exp() / (head_dim as f32).sqrt()
                } else {
                    f32::NEG_INFINITY
                };
                scores[i][j] = score;
            }
        }
        
        Ok(scores)
    }
    
    /// 🎯 Aplica softmax aos scores de atenção
    fn softmax(&self, scores: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut weights = vec![vec![0.0; scores[0].len()]; scores.len()];
        
        for i in 0..scores.len() {
            let mut sum = 0.0;
            let mut max_score = f32::NEG_INFINITY;
            
            // Encontra máximo para estabilidade numérica
            for &score in &scores[i] {
                if score > max_score {
                    max_score = score;
                }
            }
            
            // Calcula exponenciais
            for j in 0..scores[i].len() {
                if scores[i][j] != f32::NEG_INFINITY {
                    weights[i][j] = (scores[i][j] - max_score).exp();
                    sum += weights[i][j];
                }
            }
            
            // Normaliza
            for j in 0..weights[i].len() {
                weights[i][j] /= sum;
            }
        }
        
        Ok(weights)
    }
    
    /// 🔗 Aplica pesos de atenção aos valores
    fn apply_attention_weights(
        &self,
        weights: &[Vec<f32>],
        hidden_states: &mut Vec<Vec<f32>>,
        head: usize
    ) -> Result<()> {
        // Simula aplicação dos pesos de atenção
        // Em implementação real, isso modificaria hidden_states
        let _head_offset = head * (self.config.n_embd / self.config.n_head);
        
        // Placeholder para demonstração
        for i in 0..hidden_states.len() {
            for j in 0..hidden_states[i].len() {
                hidden_states[i][j] += weights[i].iter().sum::<f32>() * 0.01;
            }
        }
        
        Ok(())
    }
    
    /// ⚡ Simula rede feed-forward
    fn simulate_feedforward(&self, hidden_states: &mut Vec<Vec<f32>>, layer: usize) -> Result<()> {
        let ff_dim = self.config.n_embd * 4; // Expansão típica
        
        println!("    ⚡ Feed-forward: {} → {} → {}", 
                self.config.n_embd, ff_dim, self.config.n_embd);
        
        // Simula transformação linear + ativação + projeção
        for i in 0..hidden_states.len() {
            for j in 0..hidden_states[i].len() {
                // Simula: Linear -> GELU -> Linear
                let x = hidden_states[i][j];
                let expanded = x * 2.0; // Simula expansão
                let activated = self.gelu(expanded); // GELU activation
                hidden_states[i][j] = activated * 0.5; // Simula projeção de volta
            }
        }
        
        Ok(())
    }
    
    /// 🌊 Função de ativação GELU
    fn gelu(&self, x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }
    
    /// 🎪 Simula projeção final para vocabulário
    fn simulate_lm_head(&self, hidden_states: &[Vec<f32>]) -> Result<Vec<f32>> {
        let seq_len = hidden_states.len();
        let mut logits = vec![0.0; self.config.vocab_size];
        
        // Simula projeção linear: hidden_states @ W_lm_head
        for i in 0..self.config.vocab_size {
            let mut sum = 0.0;
            for j in 0..self.config.n_embd {
                // Usa último token para predição
                sum += hidden_states[seq_len - 1][j] * (i as f32 * 0.01);
            }
            logits[i] = sum;
        }
        
        Ok(logits)
    }
    
    /// 📉 Calcula cross-entropy loss
    fn compute_loss(&self, logits: &[f32], targets: &[usize]) -> Result<f32> {
        if targets.is_empty() {
            return Ok(0.0);
        }
        
        let target = targets[targets.len() - 1]; // Último token como target
        
        // Softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut exp_sum = 0.0;
        
        for &logit in logits {
            exp_sum += (logit - max_logit).exp();
        }
        
        let log_prob = logits[target] - max_logit - exp_sum.ln();
        let loss = -log_prob;
        
        println!("  📉 Loss: {:.4}", loss);
        
        Ok(loss)
    }
    
    /// 🎲 Gera texto usando sampling
    fn generate(
        &self,
        prompt_ids: &[usize],
        max_tokens: usize,
        temperature: f32
    ) -> Result<Vec<usize>> {
        println!("\n🎲 Gerando {} tokens com temperatura {:.2}...", max_tokens, temperature);
        
        let mut generated = prompt_ids.to_vec();
        
        for step in 0..max_tokens {
            // Limita contexto ao block_size
            let context_start = if generated.len() > self.config.block_size {
                generated.len() - self.config.block_size
            } else {
                0
            };
            
            let context = &generated[context_start..];
            
            // Forward pass
            let (logits, _) = self.forward(context, None)?;
            
            // Aplica temperatura
            let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
            
            // Sample próximo token
            let next_token = self.sample_from_logits(&scaled_logits)?;
            generated.push(next_token);
            
            if step < 10 || step % 10 == 0 {
                println!("  Token {}: {} (logit: {:.3})", 
                        step + 1, next_token, logits[next_token]);
            }
        }
        
        println!("✅ Geração concluída!");
        Ok(generated)
    }
    
    /// 🎯 Faz sampling dos logits
    fn sample_from_logits(&self, logits: &[f32]) -> Result<usize> {
        // Softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut probs = Vec::new();
        let mut sum = 0.0;
        
        for &logit in logits {
            let prob = (logit - max_logit).exp();
            probs.push(prob);
            sum += prob;
        }
        
        // Normaliza
        for prob in &mut probs {
            *prob /= sum;
        }
        
        // Sampling multinomial (simplificado)
        let mut cumsum = 0.0;
        let random_val = 0.5; // Simulação de random
        
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if random_val <= cumsum {
                return Ok(i);
            }
        }
        
        Ok(0) // Fallback
    }
    
    /// 💾 Simula salvamento de checkpoint
    fn save_checkpoint(&self, path: &str, description: Option<String>) -> Result<()> {
        println!("💾 Salvando checkpoint em '{}'...", path);
        
        let metadata = CheckpointMetadata {
            config: self.config.clone(),
            training_step: self.training_step,
            timestamp: chrono::Utc::now().to_rfc3339(),
            description,
            model_size_mb: self.config.memory_usage_mb(),
            parameter_count: self.config.num_parameters(),
        };
        
        println!("📊 Metadados do checkpoint:");
        println!("  - Passo de treinamento: {}", metadata.training_step);
        println!("  - Parâmetros: {:.2}M", metadata.parameter_count as f32 / 1_000_000.0);
        println!("  - Tamanho: {:.1} MB", metadata.model_size_mb);
        
        if let Some(desc) = &metadata.description {
            println!("  - Descrição: {}", desc);
        }
        
        // Simula salvamento dos parâmetros
        let total_params: usize = self.parameters.values().map(|v| v.len()).sum();
        println!("  - Salvando {} parâmetros...", total_params);
        
        println!("✅ Checkpoint salvo com sucesso!");
        Ok(())
    }
    
    /// 📊 Retorna estatísticas do modelo
    fn get_stats(&self) -> ModelStats {
        ModelStats {
            config: self.config.clone(),
            parameter_count: self.config.num_parameters(),
            memory_usage_mb: self.config.memory_usage_mb(),
            flops_per_token: self.config.flops_per_token(),
            training_step: self.training_step,
            device: self.device.clone(),
        }
    }
}

/// 📊 Estrutura para metadados de checkpoint
#[derive(Debug, Clone)]
struct CheckpointMetadata {
    config: GPTConfig,
    training_step: usize,
    timestamp: String,
    description: Option<String>,
    model_size_mb: f32,
    parameter_count: usize,
}

/// 📈 Estrutura para estatísticas do modelo
#[derive(Debug)]
struct ModelStats {
    config: GPTConfig,
    parameter_count: usize,
    memory_usage_mb: f32,
    flops_per_token: usize,
    training_step: usize,
    device: String,
}

/// 🎭 Estrutura para demonstrar conceitos do modelo
struct ModelDemo;

impl ModelDemo {
    /// 🏗️ Demonstra criação de modelos com diferentes tamanhos
    fn demo_model_sizes() -> Result<()> {
        println!("\n🏗️ === DEMONSTRAÇÃO DE TAMANHOS DE MODELO ===");
        
        let configs = vec![
            ("Tiny", GPTConfig::tiny()),
            ("Small", GPTConfig::small()),
            ("Medium", GPTConfig::medium()),
            ("Large", GPTConfig::large()),
        ];
        
        println!("\n📊 Comparação de configurações:");
        println!("Tamanho | Params | Memory | Layers | Heads | Context | FLOPs/token");
        println!("--------|--------|--------|--------|-------|---------|------------");
        
        for (name, config) in configs {
            let params_m = config.num_parameters() as f32 / 1_000_000.0;
            let memory_mb = config.memory_usage_mb();
            let flops_m = config.flops_per_token() as f32 / 1_000_000.0;
            
            println!("{:7} | {:6.1}M | {:6.1}MB | {:6} | {:5} | {:7} | {:8.1}M",
                    name, params_m, memory_mb, config.n_layer, 
                    config.n_head, config.block_size, flops_m);
        }
        
        println!("\n💡 Observações:");
        println!("  - Modelos maiores = mais parâmetros = mais capacidade");
        println!("  - Mais camadas = representações mais abstratas");
        println!("  - Mais cabeças = diferentes tipos de atenção");
        println!("  - Contexto maior = memória de longo prazo");
        
        Ok(())
    }
    
    /// 🔄 Demonstra forward pass detalhado
    fn demo_forward_pass() -> Result<()> {
        println!("\n🔄 === DEMONSTRAÇÃO DE FORWARD PASS ===");
        
        let config = GPTConfig::small();
        let model = MiniGPT::new(config, "cpu")?;
        
        // Simula sequência de entrada
        let input_ids = vec![1, 15, 42, 128, 7]; // IDs de tokens
        let targets = vec![15, 42, 128, 7, 99];  // Targets para loss
        
        println!("\n📝 Entrada:");
        println!("  Input IDs: {:?}", input_ids);
        println!("  Targets: {:?}", targets);
        
        // Executa forward pass
        let (logits, loss) = model.forward(&input_ids, Some(&targets))?;
        
        println!("\n📊 Saída:");
        println!("  Logits shape: [{}]", logits.len());
        println!("  Loss: {:?}", loss);
        
        // Analisa distribuição dos logits
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_logit = logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let mean_logit = logits.iter().sum::<f32>() / logits.len() as f32;
        
        println!("\n📈 Estatísticas dos logits:");
        println!("  Máximo: {:.4}", max_logit);
        println!("  Mínimo: {:.4}", min_logit);
        println!("  Média: {:.4}", mean_logit);
        
        // Mostra top-5 predições
        let mut indexed_logits: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("\n🏆 Top-5 predições:");
        for (rank, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
            println!("  {}: Token {} (logit: {:.4})", rank + 1, token_id, logit);
        }
        
        Ok(())
    }
    
    /// 🎲 Demonstra geração de texto
    fn demo_text_generation() -> Result<()> {
        println!("\n🎲 === DEMONSTRAÇÃO DE GERAÇÃO DE TEXTO ===");
        
        let config = GPTConfig::tiny(); // Modelo pequeno para demo rápida
        let model = MiniGPT::new(config, "cpu")?;
        
        let prompt_ids = vec![1, 15, 42]; // Prompt inicial
        
        println!("\n🎯 Testando diferentes temperaturas:");
        
        let temperatures = vec![0.1, 0.7, 1.0, 1.5];
        
        for temp in temperatures {
            println!("\n🌡️ Temperatura: {:.1}", temp);
            
            let generated = model.generate(&prompt_ids, 10, temp)?;
            
            println!("  Prompt: {:?}", &generated[..prompt_ids.len()]);
            println!("  Gerado: {:?}", &generated[prompt_ids.len()..]);
            
            // Analisa diversidade
            let unique_tokens: std::collections::HashSet<_> = generated[prompt_ids.len()..].iter().collect();
            let diversity = unique_tokens.len() as f32 / (generated.len() - prompt_ids.len()) as f32;
            
            println!("  Diversidade: {:.2} ({} tokens únicos)", diversity, unique_tokens.len());
        }
        
        println!("\n💡 Efeitos da temperatura:");
        println!("  - Baixa (0.1): Mais determinística, menos criativa");
        println!("  - Média (0.7): Balanceada, boa para texto geral");
        println!("  - Alta (1.0+): Mais criativa, potencialmente incoerente");
        
        Ok(())
    }
    
    /// 💾 Demonstra sistema de checkpoints
    fn demo_checkpoint_system() -> Result<()> {
        println!("\n💾 === DEMONSTRAÇÃO DE SISTEMA DE CHECKPOINTS ===");
        
        let config = GPTConfig::small();
        let mut model = MiniGPT::new(config, "cpu")?;
        
        // Simula progresso de treinamento
        let training_steps = vec![100, 500, 1000, 2000];
        
        for &step in &training_steps {
            model.training_step = step;
            
            let description = match step {
                100 => Some("Checkpoint inicial - modelo começando a convergir".to_string()),
                500 => Some("Checkpoint intermediário - loss estabilizando".to_string()),
                1000 => Some("Checkpoint avançado - boa performance".to_string()),
                2000 => Some("Checkpoint final - modelo treinado".to_string()),
                _ => None,
            };
            
            let checkpoint_path = format!("model_step_{}.safetensors", step);
            model.save_checkpoint(&checkpoint_path, description)?;
            
            println!();
        }
        
        println!("💡 Boas práticas para checkpoints:");
        println!("  - Salve regularmente durante o treinamento");
        println!("  - Inclua metadados (loss, learning rate, etc.)");
        println!("  - Use formato SafeTensors para segurança");
        println!("  - Mantenha múltiplas versões para rollback");
        
        Ok(())
    }
    
    /// 📊 Demonstra análise de performance
    fn demo_performance_analysis() -> Result<()> {
        println!("\n📊 === ANÁLISE DE PERFORMANCE ===");
        
        let configs = vec![
            ("Tiny", GPTConfig::tiny()),
            ("Small", GPTConfig::small()),
            ("Medium", GPTConfig::medium()),
        ];
        
        println!("\n⚡ Benchmark de forward pass:");
        println!("Modelo | Seq Len | Tempo (ms) | Throughput (tokens/s) | Memory (MB)");
        println!("-------|---------|------------|----------------------|------------");
        
        for (name, config) in configs {
            let model = MiniGPT::new(config, "cpu")?;
            
            let sequence_lengths = vec![16, 64, 128];
            
            for seq_len in sequence_lengths {
                let input_ids: Vec<usize> = (0..seq_len).collect();
                
                let start_time = Instant::now();
                let _ = model.forward(&input_ids, None)?;
                let duration = start_time.elapsed();
                
                let duration_ms = duration.as_millis() as f32;
                let throughput = seq_len as f32 / (duration_ms / 1000.0);
                let memory_mb = config.memory_usage_mb();
                
                println!("{:6} | {:7} | {:10.1} | {:20.1} | {:10.1}",
                        name, seq_len, duration_ms, throughput, memory_mb);
            }
        }
        
        println!("\n🔍 Fatores que afetam performance:");
        println!("  - Tamanho do modelo (parâmetros)");
        println!("  - Comprimento da sequência (quadrático na atenção)");
        println!("  - Tamanho do batch (paralelização)");
        println!("  - Dispositivo (CPU vs GPU vs TPU)");
        println!("  - Precisão (FP32 vs FP16 vs INT8)");
        
        Ok(())
    }
}

/// 🎯 Exercícios práticos para aprofundar o entendimento
struct ModelExercises;

impl ModelExercises {
    /// 📝 Exercício 1: Otimização de arquitetura
    fn exercise_architecture_optimization() {
        println!("\n📝 === EXERCÍCIO 1: OTIMIZAÇÃO DE ARQUITETURA ===");
        println!("\n🎯 Objetivo: Encontrar configuração ótima para seu caso de uso");
        
        println!("\n🔍 Parâmetros para experimentar:");
        
        println!("\n1. 📏 Tamanho do Modelo:");
        println!("   - n_embd: 128, 256, 512, 768, 1024");
        println!("   - n_layer: 4, 6, 12, 24, 48");
        println!("   - n_head: 4, 8, 12, 16");
        
        println!("\n2. 📝 Contexto e Vocabulário:");
        println!("   - block_size: 128, 256, 512, 1024, 2048");
        println!("   - vocab_size: 1K, 5K, 10K, 50K, 100K");
        
        println!("\n3. 🎛️ Regularização:");
        println!("   - dropout: 0.0, 0.1, 0.2, 0.3");
        println!("   - weight_decay: 0.01, 0.1, 0.3");
        
        println!("\n💡 Experimentos sugeridos:");
        println!("  1. Teste diferentes ratios n_embd/n_head");
        println!("  2. Compare modelos largos vs profundos");
        println!("  3. Analise trade-off memória vs performance");
        println!("  4. Meça impacto do tamanho do contexto");
    }
    
    /// 🔬 Exercício 2: Análise de atenção
    fn exercise_attention_analysis() {
        println!("\n🔬 === EXERCÍCIO 2: ANÁLISE DE ATENÇÃO ===");
        println!("\n🎯 Objetivo: Entender como o modelo 'presta atenção'");
        
        println!("\n🔍 Técnicas de análise:");
        
        println!("\n1. 🎨 Visualização de Mapas de Atenção:");
        println!("   - Extraia pesos de atenção de cada cabeça");
        println!("   - Crie heatmaps token-to-token");
        println!("   - Identifique padrões (local, global, sintático)");
        
        println!("\n2. 📊 Análise Estatística:");
        println!("   - Entropia dos pesos de atenção");
        println!("   - Distância média de atenção");
        println!("   - Especialização por cabeça");
        
        println!("\n3. 🧪 Experimentos de Ablação:");
        println!("   - Remova cabeças específicas");
        println!("   - Teste com diferentes números de cabeças");
        println!("   - Compare atenção local vs global");
        
        println!("\n💡 Implementação sugerida:");
        println!("  1. Adicione hooks para capturar atenção");
        println!("  2. Implemente visualizações interativas");
        println!("  3. Analise em diferentes tipos de texto");
        println!("  4. Compare com modelos de diferentes tamanhos");
    }
    
    /// 🚀 Exercício 3: Otimização de inferência
    fn exercise_inference_optimization() {
        println!("\n🚀 === EXERCÍCIO 3: OTIMIZAÇÃO DE INFERÊNCIA ===");
        println!("\n🎯 Objetivo: Acelerar geração de texto em produção");
        
        println!("\n⚡ Técnicas de otimização:");
        
        println!("\n1. 🧠 KV-Cache:");
        println!("   - Cache keys e values computados");
        println!("   - Evita recomputação em geração autoregressiva");
        println!("   - Reduz complexidade de O(n²) para O(n)");
        
        println!("\n2. 🔢 Quantização:");
        println!("   - FP32 → FP16 (2x speedup)");
        println!("   - FP16 → INT8 (4x speedup)");
        println!("   - Quantização dinâmica vs estática");
        
        println!("\n3. 📦 Batching Inteligente:");
        println!("   - Agrupe sequências de tamanhos similares");
        println!("   - Use padding mínimo");
        println!("   - Implemente continuous batching");
        
        println!("\n4. 🔧 Otimizações de Kernel:");
        println!("   - Fused attention kernels");
        println!("   - Flash Attention");
        println!("   - Operações in-place");
        
        println!("\n💡 Métricas para medir:");
        println!("  - Latência (tempo por token)");
        println!("  - Throughput (tokens por segundo)");
        println!("  - Uso de memória");
        println!("  - Qualidade da saída (perplexity)");
    }
}

/// 🚀 Função principal que executa todas as demonstrações
fn main() -> Result<()> {
    println!("🧠 === DEMONSTRAÇÃO DO MODELO MINI-GPT ===");
    println!("Explorando como construir e usar um Large Language Model");
    
    // Demonstrações básicas
    ModelDemo::demo_model_sizes()?;
    ModelDemo::demo_forward_pass()?;
    ModelDemo::demo_text_generation()?;
    ModelDemo::demo_checkpoint_system()?;
    ModelDemo::demo_performance_analysis()?;
    
    // Exercícios educacionais
    println!("\n\n🎓 === EXERCÍCIOS PRÁTICOS ===");
    ModelExercises::exercise_architecture_optimization();
    ModelExercises::exercise_attention_analysis();
    ModelExercises::exercise_inference_optimization();
    
    println!("\n\n✅ === DEMONSTRAÇÃO CONCLUÍDA ===");
    println!("🎯 Próximos passos:");
    println!("  1. Experimente com diferentes configurações");
    println!("  2. Implemente os exercícios sugeridos");
    println!("  3. Teste com dados reais");
    println!("  4. Explore técnicas avançadas de otimização");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_parameter_calculation() {
        let config = GPTConfig::tiny();
        let params = config.num_parameters();
        assert!(params > 0);
        assert!(params < 10_000_000); // Modelo tiny deve ser pequeno
    }
    
    #[test]
    fn test_model_creation() {
        let config = GPTConfig::tiny();
        let model = MiniGPT::new(config, "cpu").unwrap();
        assert_eq!(model.device, "cpu");
        assert_eq!(model.training_step, 0);
    }
    
    #[test]
    fn test_forward_pass() {
        let config = GPTConfig::tiny();
        let model = MiniGPT::new(config, "cpu").unwrap();
        
        let input_ids = vec![1, 2, 3];
        let (logits, loss) = model.forward(&input_ids, None).unwrap();
        
        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(loss.is_none());
    }
    
    #[test]
    fn test_generation() {
        let config = GPTConfig::tiny();
        let model = MiniGPT::new(config, "cpu").unwrap();
        
        let prompt = vec![1, 2];
        let generated = model.generate(&prompt, 5, 1.0).unwrap();
        
        assert_eq!(generated.len(), prompt.len() + 5);
        assert_eq!(&generated[..prompt.len()], &prompt[..]);
    }
    
    #[test]
    fn test_gelu_activation() {
        let config = GPTConfig::tiny();
        let model = MiniGPT::new(config, "cpu").unwrap();
        
        // Testa propriedades da GELU
        assert_eq!(model.gelu(0.0), 0.0);
        assert!(model.gelu(1.0) > 0.0);
        assert!(model.gelu(-1.0) < 0.0);
    }
}