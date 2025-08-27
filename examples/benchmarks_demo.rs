//! # 🚀 Demonstração do Sistema de Benchmarks
//!
//! Este exemplo demonstra como usar o sistema de benchmarks para avaliar
//! a performance e qualidade das diferentes estratégias de chunking.
//!
//! ## 🎯 O que você aprenderá:
//! - Como configurar e executar benchmarks
//! - Análise de métricas de performance e qualidade
//! - Comparação entre diferentes estratégias
//! - Otimização baseada em dados
//! - Testes de stress e escalabilidade

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::fmt;

// Estruturas simplificadas para demonstração
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChunkingStrategy {
    Fixed,
    Semantic,
    Adaptive,
    Sliding,
}

impl fmt::Display for ChunkingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkingStrategy::Fixed => write!(f, "Fixed"),
            ChunkingStrategy::Semantic => write!(f, "Semantic"),
            ChunkingStrategy::Adaptive => write!(f, "Adaptive"),
            ChunkingStrategy::Sliding => write!(f, "Sliding"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub max_chunk_size: usize,
    pub min_chunk_size: usize,
    pub overlap_ratio: f64,
    pub strategy: ChunkingStrategy,
    pub preserve_sentences: bool,
    pub preserve_paragraphs: bool,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub strategy_name: String,
    
    // Métricas temporais
    pub processing_time: Duration,
    pub chars_per_second: f64,
    pub tokens_per_second: f64,
    
    // Métricas de qualidade
    pub total_chunks: usize,
    pub avg_chunk_size: f64,
    pub chunk_size_stddev: f64,
    pub avg_information_density: f64,
    pub boundary_preservation_rate: f64,
    
    // Métricas de memória
    pub peak_memory_mb: f64,
    pub memory_per_chunk_kb: f64,
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub text_sizes: Vec<usize>,
    pub iterations: usize,
    pub strategies: Vec<ChunkingStrategy>,
    pub chunking_configs: HashMap<ChunkingStrategy, ChunkingConfig>,
    pub warmup_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        let mut chunking_configs = HashMap::new();
        
        chunking_configs.insert(
            ChunkingStrategy::Fixed,
            ChunkingConfig {
                max_chunk_size: 512,
                min_chunk_size: 64,
                overlap_ratio: 0.0,
                strategy: ChunkingStrategy::Fixed,
                preserve_sentences: false,
                preserve_paragraphs: false,
            },
        );
        
        chunking_configs.insert(
            ChunkingStrategy::Semantic,
            ChunkingConfig {
                max_chunk_size: 512,
                min_chunk_size: 128,
                overlap_ratio: 0.1,
                strategy: ChunkingStrategy::Semantic,
                preserve_sentences: true,
                preserve_paragraphs: true,
            },
        );
        
        chunking_configs.insert(
            ChunkingStrategy::Adaptive,
            ChunkingConfig {
                max_chunk_size: 768,
                min_chunk_size: 256,
                overlap_ratio: 0.15,
                strategy: ChunkingStrategy::Adaptive,
                preserve_sentences: true,
                preserve_paragraphs: false,
            },
        );
        
        chunking_configs.insert(
            ChunkingStrategy::Sliding,
            ChunkingConfig {
                max_chunk_size: 400,
                min_chunk_size: 200,
                overlap_ratio: 0.25,
                strategy: ChunkingStrategy::Sliding,
                preserve_sentences: false,
                preserve_paragraphs: false,
            },
        );
        
        Self {
            text_sizes: vec![1000, 5000, 10000, 50000],
            iterations: 5,
            strategies: vec![
                ChunkingStrategy::Fixed,
                ChunkingStrategy::Semantic,
                ChunkingStrategy::Adaptive,
                ChunkingStrategy::Sliding,
            ],
            chunking_configs,
            warmup_iterations: 2,
        }
    }
}

pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }
    
    pub fn run_all_benchmarks(&self, test_text: &str) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();
        
        println!("🚀 Iniciando benchmarks com {} estratégias...", self.config.strategies.len());
        
        for strategy in &self.config.strategies {
            if let Some(config) = self.config.chunking_configs.get(strategy) {
                println!("\n📊 Testando estratégia: {}", strategy);
                
                match self.benchmark_strategy(test_text, strategy, config) {
                    Ok(result) => {
                        println!("   ✅ Concluído em {:?}", result.processing_time);
                        results.push(result);
                    }
                    Err(e) => {
                        println!("   ❌ Erro: {:?}", e);
                    }
                }
            }
        }
        
        results
    }
    
    fn benchmark_strategy(
        &self,
        text: &str,
        strategy: &ChunkingStrategy,
        config: &ChunkingConfig,
    ) -> Result<BenchmarkResult, String> {
        // Aquecimento
        for _ in 0..self.config.warmup_iterations {
            let _ = self.simulate_chunking(text, strategy, config);
        }
        
        let mut total_time = Duration::new(0, 0);
        let mut total_chunks = 0;
        let mut chunk_sizes = Vec::new();
        
        // Executa benchmarks
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let chunks = self.simulate_chunking(text, strategy, config)?;
            let duration = start.elapsed();
            
            total_time += duration;
            total_chunks += chunks.len();
            
            for chunk in &chunks {
                chunk_sizes.push(chunk.len());
            }
        }
        
        // Calcula métricas
        let avg_time = total_time / self.config.iterations as u32;
        let chars_per_second = (text.len() as f64) / avg_time.as_secs_f64();
        let tokens_per_second = (text.split_whitespace().count() as f64) / avg_time.as_secs_f64();
        
        let avg_chunks = total_chunks as f64 / self.config.iterations as f64;
        let avg_chunk_size = chunk_sizes.iter().sum::<usize>() as f64 / chunk_sizes.len() as f64;
        
        // Calcula desvio padrão
        let variance = chunk_sizes.iter()
            .map(|&size| (size as f64 - avg_chunk_size).powi(2))
            .sum::<f64>() / chunk_sizes.len() as f64;
        let chunk_size_stddev = variance.sqrt();
        
        // Simula outras métricas
        let avg_information_density = self.calculate_information_density(strategy);
        let boundary_preservation_rate = self.calculate_boundary_preservation(strategy);
        let peak_memory_mb = self.estimate_memory_usage(text.len(), avg_chunks as usize);
        let memory_per_chunk_kb = (peak_memory_mb * 1024.0) / avg_chunks;
        
        Ok(BenchmarkResult {
            strategy_name: strategy.to_string(),
            processing_time: avg_time,
            chars_per_second,
            tokens_per_second,
            total_chunks: avg_chunks as usize,
            avg_chunk_size,
            chunk_size_stddev,
            avg_information_density,
            boundary_preservation_rate,
            peak_memory_mb,
            memory_per_chunk_kb,
        })
    }
    
    fn simulate_chunking(
        &self,
        text: &str,
        strategy: &ChunkingStrategy,
        config: &ChunkingConfig,
    ) -> Result<Vec<String>, String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        
        match strategy {
            ChunkingStrategy::Fixed => {
                let chunk_size = config.max_chunk_size;
                for chunk_words in words.chunks(chunk_size) {
                    chunks.push(chunk_words.join(" "));
                }
            }
            ChunkingStrategy::Semantic => {
                // Simula chunking semântico baseado em sentenças
                let sentences: Vec<&str> = text.split('.').collect();
                let mut current_chunk = String::new();
                
                for sentence in sentences {
                    if current_chunk.len() + sentence.len() > config.max_chunk_size {
                        if !current_chunk.is_empty() {
                            chunks.push(current_chunk.trim().to_string());
                            current_chunk = String::new();
                        }
                    }
                    current_chunk.push_str(sentence);
                    current_chunk.push('.');
                }
                
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                }
            }
            ChunkingStrategy::Adaptive => {
                // Simula chunking adaptativo
                let mut i = 0;
                while i < words.len() {
                    let mut chunk_size = config.min_chunk_size;
                    
                    // Adapta o tamanho baseado no conteúdo
                    if i + chunk_size < words.len() {
                        let next_words = &words[i..i + chunk_size + 50.min(words.len() - i - chunk_size)];
                        if next_words.iter().any(|w| w.ends_with('.')) {
                            chunk_size += 50;
                        }
                    }
                    
                    chunk_size = chunk_size.min(config.max_chunk_size).min(words.len() - i);
                    let chunk_words = &words[i..i + chunk_size];
                    chunks.push(chunk_words.join(" "));
                    i += chunk_size;
                }
            }
            ChunkingStrategy::Sliding => {
                // Simula sliding window
                let window_size = config.max_chunk_size;
                let step_size = (window_size as f64 * (1.0 - config.overlap_ratio)) as usize;
                
                let mut i = 0;
                while i < words.len() {
                    let end = (i + window_size).min(words.len());
                    let chunk_words = &words[i..end];
                    chunks.push(chunk_words.join(" "));
                    
                    i += step_size;
                    if i >= words.len() {
                        break;
                    }
                }
            }
        }
        
        Ok(chunks)
    }
    
    fn calculate_information_density(&self, strategy: &ChunkingStrategy) -> f64 {
        match strategy {
            ChunkingStrategy::Fixed => 0.75,
            ChunkingStrategy::Semantic => 0.90,
            ChunkingStrategy::Adaptive => 0.85,
            ChunkingStrategy::Sliding => 0.80,
        }
    }
    
    fn calculate_boundary_preservation(&self, strategy: &ChunkingStrategy) -> f64 {
        match strategy {
            ChunkingStrategy::Fixed => 0.60,
            ChunkingStrategy::Semantic => 0.95,
            ChunkingStrategy::Adaptive => 0.85,
            ChunkingStrategy::Sliding => 0.70,
        }
    }
    
    fn estimate_memory_usage(&self, text_len: usize, num_chunks: usize) -> f64 {
        // Estimativa simples de uso de memória
        let base_memory = text_len as f64 / 1024.0 / 1024.0; // MB
        let chunk_overhead = (num_chunks * 64) as f64 / 1024.0 / 1024.0; // MB
        base_memory + chunk_overhead
    }
    
    pub fn generate_report(&self, results: &[BenchmarkResult]) -> String {
        let mut report = String::new();
        
        report.push_str("\n📊 === RELATÓRIO DE BENCHMARKS ===\n\n");
        
        // Tabela de performance
        report.push_str("⚡ **PERFORMANCE TEMPORAL:**\n");
        report.push_str(&format!(
            "{:<12} {:<12} {:<15} {:<15}\n",
            "Estratégia", "Tempo (ms)", "Chars/s", "Tokens/s"
        ));
        report.push_str(&"-".repeat(60));
        report.push('\n');
        
        for result in results {
            report.push_str(&format!(
                "{:<12} {:<12} {:<15.0} {:<15.0}\n",
                result.strategy_name,
                result.processing_time.as_millis(),
                result.chars_per_second,
                result.tokens_per_second
            ));
        }
        
        // Tabela de qualidade
        report.push_str("\n🎯 **QUALIDADE DOS CHUNKS:**\n");
        report.push_str(&format!(
            "{:<12} {:<8} {:<12} {:<12} {:<12}\n",
            "Estratégia", "Chunks", "Tam.Médio", "Desvio", "Densidade"
        ));
        report.push_str(&"-".repeat(70));
        report.push('\n');
        
        for result in results {
            report.push_str(&format!(
                "{:<12} {:<8} {:<12.1} {:<12.1} {:<12.2}\n",
                result.strategy_name,
                result.total_chunks,
                result.avg_chunk_size,
                result.chunk_size_stddev,
                result.avg_information_density
            ));
        }
        
        // Tabela de memória
        report.push_str("\n💾 **USO DE MEMÓRIA:**\n");
        report.push_str(&format!(
            "{:<12} {:<15} {:<15} {:<15}\n",
            "Estratégia", "Pico (MB)", "Por Chunk (KB)", "Preservação"
        ));
        report.push_str(&"-".repeat(65));
        report.push('\n');
        
        for result in results {
            report.push_str(&format!(
                "{:<12} {:<15.2} {:<15.2} {:<15.1}%\n",
                result.strategy_name,
                result.peak_memory_mb,
                result.memory_per_chunk_kb,
                result.boundary_preservation_rate * 100.0
            ));
        }
        
        // Análise e recomendações
        report.push_str("\n🏆 **ANÁLISE E RECOMENDAÇÕES:**\n");
        
        if let Some(fastest) = results.iter().min_by(|a, b| a.processing_time.cmp(&b.processing_time)) {
            report.push_str(&format!("   • ⚡ Mais rápido: {} ({:?})\n", fastest.strategy_name, fastest.processing_time));
        }
        
        if let Some(best_quality) = results.iter().max_by(|a, b| a.avg_information_density.partial_cmp(&b.avg_information_density).unwrap()) {
            report.push_str(&format!("   • 🎯 Melhor qualidade: {} ({:.2})\n", best_quality.strategy_name, best_quality.avg_information_density));
        }
        
        if let Some(most_efficient) = results.iter().min_by(|a, b| a.peak_memory_mb.partial_cmp(&b.peak_memory_mb).unwrap()) {
            report.push_str(&format!("   • 💾 Mais eficiente: {} ({:.2} MB)\n", most_efficient.strategy_name, most_efficient.peak_memory_mb));
        }
        
        report
    }
}

// === DEMONSTRAÇÕES ===

fn demo_basic_benchmarks() {
    println!("\n🚀 === DEMONSTRAÇÃO: BENCHMARKS BÁSICOS ===");
    
    let test_text = generate_test_text(5000);
    let config = BenchmarkConfig::default();
    let mut runner = BenchmarkRunner::new(config);
    
    println!("📝 Texto de teste: {} caracteres", test_text.len());
    println!("🔧 Configuração: {} estratégias, {} iterações", 
             runner.config.strategies.len(), runner.config.iterations);
    
    let results = runner.run_all_benchmarks(&test_text);
    let report = runner.generate_report(&results);
    
    println!("{}", report);
}

fn demo_scalability_analysis() {
    println!("\n📈 === DEMONSTRAÇÃO: ANÁLISE DE ESCALABILIDADE ===");
    
    let text_sizes = vec![1000, 5000, 10000, 25000];
    let strategy = ChunkingStrategy::Semantic;
    
    println!("🔍 Analisando escalabilidade da estratégia: {}", strategy);
    
    for &size in &text_sizes {
        let test_text = generate_test_text(size);
        
        let mut config = BenchmarkConfig::default();
        config.strategies = vec![strategy.clone()];
        config.iterations = 3;
        
        let mut runner = BenchmarkRunner::new(config);
        let results = runner.run_all_benchmarks(&test_text);
        
        if let Some(result) = results.first() {
            println!("\n📊 Tamanho: {} chars", size);
            println!("   • Tempo: {:?}", result.processing_time);
            println!("   • Throughput: {:.0} chars/s", result.chars_per_second);
            println!("   • Chunks: {}", result.total_chunks);
            println!("   • Memória: {:.2} MB", result.peak_memory_mb);
        }
    }
}

fn demo_strategy_comparison() {
    println!("\n⚖️ === DEMONSTRAÇÃO: COMPARAÇÃO DE ESTRATÉGIAS ===");
    
    let test_text = generate_test_text(10000);
    
    println!("🎯 Comparando estratégias em diferentes cenários:");
    
    // Cenário 1: Performance
    println!("\n🏃 **Cenário 1: Foco em Performance**");
    let mut perf_config = BenchmarkConfig::default();
    perf_config.iterations = 10;
    
    let mut runner = BenchmarkRunner::new(perf_config);
    let results = runner.run_all_benchmarks(&test_text);
    
    let fastest = results.iter().min_by(|a, b| a.processing_time.cmp(&b.processing_time)).unwrap();
    println!("   🏆 Vencedor: {} ({:?})", fastest.strategy_name, fastest.processing_time);
    
    // Cenário 2: Qualidade
    println!("\n🎯 **Cenário 2: Foco em Qualidade**");
    let best_quality = results.iter().max_by(|a, b| {
        a.avg_information_density.partial_cmp(&b.avg_information_density).unwrap()
    }).unwrap();
    println!("   🏆 Vencedor: {} (densidade: {:.2})", 
             best_quality.strategy_name, best_quality.avg_information_density);
    
    // Cenário 3: Eficiência de memória
    println!("\n💾 **Cenário 3: Foco em Memória**");
    let most_efficient = results.iter().min_by(|a, b| {
        a.peak_memory_mb.partial_cmp(&b.peak_memory_mb).unwrap()
    }).unwrap();
    println!("   🏆 Vencedor: {} ({:.2} MB)", 
             most_efficient.strategy_name, most_efficient.peak_memory_mb);
}

fn demo_custom_benchmarks() {
    println!("\n🔧 === DEMONSTRAÇÃO: BENCHMARKS CUSTOMIZADOS ===");
    
    // Configuração customizada para documentos técnicos
    let mut custom_config = BenchmarkConfig {
        text_sizes: vec![2000, 8000],
        iterations: 5,
        strategies: vec![ChunkingStrategy::Semantic, ChunkingStrategy::Adaptive],
        chunking_configs: HashMap::new(),
        warmup_iterations: 1,
    };
    
    // Configuração otimizada para documentos técnicos
    custom_config.chunking_configs.insert(
        ChunkingStrategy::Semantic,
        ChunkingConfig {
            max_chunk_size: 1024,
            min_chunk_size: 256,
            overlap_ratio: 0.05,
            strategy: ChunkingStrategy::Semantic,
            preserve_sentences: true,
            preserve_paragraphs: true,
        },
    );
    
    custom_config.chunking_configs.insert(
        ChunkingStrategy::Adaptive,
        ChunkingConfig {
            max_chunk_size: 1200,
            min_chunk_size: 400,
            overlap_ratio: 0.1,
            strategy: ChunkingStrategy::Adaptive,
            preserve_sentences: true,
            preserve_paragraphs: false,
        },
    );
    
    let technical_text = generate_technical_text(8000);
    let mut runner = BenchmarkRunner::new(custom_config);
    
    println!("📚 Testando com documento técnico ({} chars)", technical_text.len());
    
    let results = runner.run_all_benchmarks(&technical_text);
    let report = runner.generate_report(&results);
    
    println!("{}", report);
    
    println!("\n💡 **Insights para documentos técnicos:**");
    println!("   • Preserve sentenças para manter contexto técnico");
    println!("   • Use chunks maiores para conceitos complexos");
    println!("   • Overlap mínimo para evitar redundância");
}

fn demo_stress_testing() {
    println!("\n🔥 === DEMONSTRAÇÃO: TESTES DE STRESS ===");
    
    let stress_sizes = vec![50000, 100000, 200000];
    
    println!("💪 Executando testes de stress com documentos grandes:");
    
    for &size in &stress_sizes {
        println!("\n🧪 Testando com {} caracteres...", size);
        
        let large_text = generate_test_text(size);
        
        // Testa apenas estratégias mais eficientes para textos grandes
        let mut stress_config = BenchmarkConfig {
            text_sizes: vec![size],
            iterations: 2,
            strategies: vec![ChunkingStrategy::Fixed, ChunkingStrategy::Sliding],
            chunking_configs: HashMap::new(),
            warmup_iterations: 1,
        };
        
        // Configurações otimizadas para textos grandes
        stress_config.chunking_configs.insert(
            ChunkingStrategy::Fixed,
            ChunkingConfig {
                max_chunk_size: 2048,
                min_chunk_size: 512,
                overlap_ratio: 0.0,
                strategy: ChunkingStrategy::Fixed,
                preserve_sentences: false,
                preserve_paragraphs: false,
            },
        );
        
        stress_config.chunking_configs.insert(
            ChunkingStrategy::Sliding,
            ChunkingConfig {
                max_chunk_size: 1536,
                min_chunk_size: 768,
                overlap_ratio: 0.1,
                strategy: ChunkingStrategy::Sliding,
                preserve_sentences: false,
                preserve_paragraphs: false,
            },
        );
        
        let mut runner = BenchmarkRunner::new(stress_config);
        let start = Instant::now();
        let results = runner.run_all_benchmarks(&large_text);
        let total_time = start.elapsed();
        
        println!("   ⏱️ Tempo total: {:?}", total_time);
        
        for result in &results {
            println!("   📊 {}: {:.0} chars/s, {:.2} MB", 
                     result.strategy_name, 
                     result.chars_per_second,
                     result.peak_memory_mb);
        }
        
        // Verifica se alguma estratégia falhou
        if results.is_empty() {
            println!("   ❌ Todas as estratégias falharam!");
        } else {
            println!("   ✅ {} estratégias completaram com sucesso", results.len());
        }
    }
}

fn demo_optimization_insights() {
    println!("\n🎯 === DEMONSTRAÇÃO: INSIGHTS DE OTIMIZAÇÃO ===");
    
    let test_text = generate_test_text(15000);
    let config = BenchmarkConfig::default();
    let mut runner = BenchmarkRunner::new(config);
    
    let results = runner.run_all_benchmarks(&test_text);
    
    println!("🔍 Analisando resultados para insights de otimização:");
    
    // Análise de trade-offs
    println!("\n⚖️ **Trade-offs identificados:**");
    
    for result in &results {
        let speed_score = 1.0 / result.processing_time.as_secs_f64();
        let quality_score = result.avg_information_density;
        let memory_score = 1.0 / result.peak_memory_mb;
        
        let overall_score = (speed_score * 0.3 + quality_score * 0.5 + memory_score * 0.2) * 100.0;
        
        println!("   📊 {}: Score geral = {:.1}", result.strategy_name, overall_score);
        println!("      • Velocidade: {:.2}, Qualidade: {:.2}, Memória: {:.2}", 
                 speed_score * 100.0, quality_score * 100.0, memory_score * 100.0);
    }
    
    // Recomendações baseadas em cenários
    println!("\n💡 **Recomendações por cenário:**");
    
    println!("\n🚀 **Para aplicações em tempo real:**");
    let fastest = results.iter().min_by(|a, b| a.processing_time.cmp(&b.processing_time)).unwrap();
    println!("   • Use: {}", fastest.strategy_name);
    println!("   • Razão: Menor latência ({:?})", fastest.processing_time);
    
    println!("\n🎯 **Para análise de documentos:**");
    let best_quality = results.iter().max_by(|a, b| {
        a.avg_information_density.partial_cmp(&b.avg_information_density).unwrap()
    }).unwrap();
    println!("   • Use: {}", best_quality.strategy_name);
    println!("   • Razão: Melhor preservação de contexto ({:.2})", best_quality.avg_information_density);
    
    println!("\n💾 **Para sistemas com pouca memória:**");
    let most_efficient = results.iter().min_by(|a, b| {
        a.peak_memory_mb.partial_cmp(&b.peak_memory_mb).unwrap()
    }).unwrap();
    println!("   • Use: {}", most_efficient.strategy_name);
    println!("   • Razão: Menor uso de memória ({:.2} MB)", most_efficient.peak_memory_mb);
}

// === EXERCÍCIOS PRÁTICOS ===

fn practical_exercises() {
    println!("\n🎓 === EXERCÍCIOS PRÁTICOS ===");
    
    println!("\n📝 **Exercício 1: Benchmark Personalizado**");
    println!("   Crie um benchmark para avaliar estratégias com:");
    println!("   - Textos de diferentes idiomas");
    println!("   - Documentos com formatação específica");
    println!("   - Métricas customizadas de qualidade");
    
    println!("\n📝 **Exercício 2: Otimização Automática**");
    println!("   Implemente um sistema que:");
    println!("   - Analisa características do texto");
    println!("   - Seleciona automaticamente a melhor estratégia");
    println!("   - Ajusta parâmetros dinamicamente");
    
    println!("\n📝 **Exercício 3: Análise de Regressão**");
    println!("   Desenvolva testes que detectem:");
    println!("   - Degradação de performance");
    println!("   - Mudanças na qualidade dos chunks");
    println!("   - Vazamentos de memória");
    
    println!("\n📝 **Exercício 4: Benchmark Distribuído**");
    println!("   Implemente benchmarks que:");
    println!("   - Executem em paralelo");
    println!("   - Testem diferentes arquiteturas");
    println!("   - Comparem com baselines externos");
    
    println!("\n📝 **Exercício 5: Visualização de Resultados**");
    println!("   Crie dashboards que mostrem:");
    println!("   - Gráficos de performance ao longo do tempo");
    println!("   - Heatmaps de uso de memória");
    println!("   - Comparações interativas entre estratégias");
}

// === FUNÇÕES AUXILIARES ===

fn generate_test_text(size: usize) -> String {
    let base_text = "Este é um texto de exemplo para demonstrar o sistema de chunking. \
                     Ele contém várias sentenças e parágrafos para simular documentos reais. \
                     O objetivo é avaliar como diferentes estratégias de chunking processam \
                     este conteúdo e mantêm a coerência semântica. ";
    
    let mut result = String::new();
    while result.len() < size {
        result.push_str(base_text);
        if result.len() % 1000 == 0 {
            result.push_str("\n\nNovo parágrafo para adicionar estrutura ao documento. ");
        }
    }
    
    result.truncate(size);
    result
}

fn generate_technical_text(size: usize) -> String {
    let technical_content = "A arquitetura de microserviços representa um paradigma de desenvolvimento \
                            onde aplicações são decompostas em serviços pequenos e independentes. \
                            Cada microserviço é responsável por uma funcionalidade específica e \
                            comunica-se através de APIs bem definidas. Esta abordagem oferece \
                            benefícios como escalabilidade independente, tecnologias heterogêneas \
                            e deployment isolado. No entanto, introduz complexidades relacionadas \
                            à comunicação entre serviços, consistência de dados e monitoramento \
                            distribuído. ";
    
    let mut result = String::new();
    let mut section = 1;
    
    while result.len() < size {
        result.push_str(&format!("\n\n## Seção {}: Conceitos Fundamentais\n\n", section));
        result.push_str(technical_content);
        
        result.push_str("\n\n### Implementação Prática\n\n");
        result.push_str("Para implementar esta arquitetura, considere os seguintes aspectos: \
                        1) Definição de bounded contexts, 2) Estratégias de comunicação, \
                        3) Padrões de dados, 4) Observabilidade e monitoramento. ");
        
        section += 1;
    }
    
    result.truncate(size);
    result
}

fn main() {
    println!("🚀 === DEMONSTRAÇÃO DO SISTEMA DE BENCHMARKS ===");
    println!("\nEste exemplo mostra como usar benchmarks para avaliar e otimizar");
    println!("estratégias de chunking em diferentes cenários e requisitos.");
    
    demo_basic_benchmarks();
    demo_scalability_analysis();
    demo_strategy_comparison();
    demo_custom_benchmarks();
    demo_stress_testing();
    demo_optimization_insights();
    practical_exercises();
    
    println!("\n🎉 === DEMONSTRAÇÃO CONCLUÍDA ===");
    println!("\n🚀 **Próximos Passos:**");
    println!("   • Execute benchmarks com seus próprios dados");
    println!("   • Customize métricas para suas necessidades");
    println!("   • Implemente otimizações baseadas nos resultados");
    println!("   • Configure monitoramento contínuo de performance");
    
    println!("\n📚 **Recursos Adicionais:**");
    println!("   • Implementação completa em src/benchmarks.rs");
    println!("   • Exemplos de chunking em examples/chunking_examples.rs");
    println!("   • Documentação detalhada no README.md");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_runner_creation() {
        let config = BenchmarkConfig::default();
        let runner = BenchmarkRunner::new(config);
        assert_eq!(runner.config.strategies.len(), 4);
    }
    
    #[test]
    fn test_chunking_simulation() {
        let config = BenchmarkConfig::default();
        let runner = BenchmarkRunner::new(config);
        
        let test_text = "Este é um teste. Com várias sentenças. Para verificar o chunking.";
        let chunking_config = &runner.config.chunking_configs[&ChunkingStrategy::Fixed];
        
        let chunks = runner.simulate_chunking(test_text, &ChunkingStrategy::Fixed, chunking_config).unwrap();
        assert!(!chunks.is_empty());
    }
    
    #[test]
    fn test_benchmark_metrics() {
        let result = BenchmarkResult {
            strategy_name: "Test".to_string(),
            processing_time: Duration::from_millis(100),
            chars_per_second: 1000.0,
            tokens_per_second: 200.0,
            total_chunks: 10,
            avg_chunk_size: 50.0,
            chunk_size_stddev: 5.0,
            avg_information_density: 0.8,
            boundary_preservation_rate: 0.9,
            peak_memory_mb: 2.5,
            memory_per_chunk_kb: 0.25,
        };
        
        assert_eq!(result.strategy_name, "Test");
        assert_eq!(result.total_chunks, 10);
        assert!((result.avg_information_density - 0.8).abs() < f64::EPSILON);
    }
}