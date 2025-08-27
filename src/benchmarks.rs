//! 🚀 **BENCHMARKS E TESTES DE PERFORMANCE**
//!
//! Este módulo implementa benchmarks abrangentes para avaliar o desempenho
//! das diferentes estratégias de chunking em cenários realistas.
//!
//! ## 📊 **Métricas Avaliadas:**
//!
//! ### ⏱️ **Performance Temporal:**
//! - **Latência**: Tempo para processar um único documento
//! - **Throughput**: Documentos processados por segundo
//! - **Escalabilidade**: Performance com diferentes tamanhos de texto
//!
//! ### 🎯 **Qualidade dos Chunks:**
//! - **Densidade de Informação**: Concentração de conteúdo relevante
//! - **Preservação de Contexto**: Manutenção de limites semânticos
//! - **Distribuição de Tamanhos**: Uniformidade dos chunks gerados
//!
//! ### 💾 **Eficiência de Memória:**
//! - **Pico de Uso**: Máximo de memória durante processamento
//! - **Fragmentação**: Eficiência na alocação de memória
//! - **Overhead**: Memória adicional por estratégia

use crate::chunking::*;
use crate::tokenizer::BPETokenizer;
use anyhow::Result;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// 📈 **RESULTADO DE BENCHMARK**
///
/// Estrutura que encapsula todas as métricas coletadas
/// durante a execução de um benchmark específico.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Nome da estratégia testada
    pub strategy_name: String,
    
    /// ⏱️ **MÉTRICAS TEMPORAIS**
    /// Tempo total de processamento
    pub processing_time: Duration,
    /// Throughput em caracteres por segundo
    pub chars_per_second: f64,
    /// Throughput em tokens por segundo
    pub tokens_per_second: f64,
    
    /// 📊 **MÉTRICAS DE QUALIDADE**
    /// Número total de chunks gerados
    pub total_chunks: usize,
    /// Tamanho médio dos chunks em tokens
    pub avg_chunk_size: f64,
    /// Desvio padrão do tamanho dos chunks
    pub chunk_size_stddev: f64,
    /// Densidade média de informação
    pub avg_information_density: f64,
    /// Taxa de preservação de limites
    pub boundary_preservation_rate: f64,
    
    /// 💾 **MÉTRICAS DE MEMÓRIA**
    /// Pico de uso de memória (estimado)
    pub peak_memory_mb: f64,
    /// Overhead de memória por chunk
    pub memory_per_chunk_kb: f64,
}

/// 🎯 **CONFIGURAÇÃO DE BENCHMARK**
///
/// Define os parâmetros para execução de benchmarks,
/// permitindo testes customizados e reproduzíveis.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Tamanhos de texto para testar escalabilidade
    pub text_sizes: Vec<usize>,
    /// Número de iterações por teste
    pub iterations: usize,
    /// Estratégias a serem testadas
    pub strategies: Vec<ChunkingStrategy>,
    /// Configurações de chunking para cada estratégia
    pub chunking_configs: HashMap<ChunkingStrategy, ChunkingConfig>,
    /// Aquecimento antes dos benchmarks
    pub warmup_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        let mut chunking_configs = HashMap::new();
        
        // Configuração para estratégia Fixed
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
        
        // Configuração para estratégia Semantic
        chunking_configs.insert(
            ChunkingStrategy::Semantic,
            ChunkingConfig {
                max_chunk_size: 512,
                min_chunk_size: 64,
                overlap_ratio: 0.1,
                strategy: ChunkingStrategy::Semantic,
                preserve_sentences: true,
                preserve_paragraphs: true,
            },
        );
        
        // Configuração para estratégia Adaptive
        chunking_configs.insert(
            ChunkingStrategy::Adaptive,
            ChunkingConfig {
                max_chunk_size: 768,
                min_chunk_size: 128,
                overlap_ratio: 0.15,
                strategy: ChunkingStrategy::Adaptive,
                preserve_sentences: true,
                preserve_paragraphs: true,
            },
        );
        
        // Configuração para estratégia Overlapping
        chunking_configs.insert(
            ChunkingStrategy::Overlapping,
            ChunkingConfig {
                max_chunk_size: 512,
                min_chunk_size: 64,
                overlap_ratio: 0.25,
                strategy: ChunkingStrategy::Overlapping,
                preserve_sentences: true,
                preserve_paragraphs: false,
            },
        );
        
        Self {
            text_sizes: vec![1000, 5000, 10000, 50000, 100000],
            iterations: 10,
            strategies: vec![
                ChunkingStrategy::Fixed,
                ChunkingStrategy::Semantic,
                ChunkingStrategy::Adaptive,
                ChunkingStrategy::Overlapping,
            ],
            chunking_configs,
            warmup_iterations: 3,
        }
    }
}

/// 🏃‍♂️ **EXECUTOR DE BENCHMARKS**
///
/// Classe principal responsável por executar benchmarks
/// e coletar métricas de performance detalhadas.
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    tokenizer: BPETokenizer,
}

impl BenchmarkRunner {
    /// 🔧 **CONSTRUTOR**
    ///
    /// Inicializa o executor com configuração personalizada.
    pub fn new(config: BenchmarkConfig) -> Result<Self> {
        let mut tokenizer = BPETokenizer::new(50000)?;
        
        // Treina o tokenizer com texto de exemplo
        let sample_text = "Este é um texto de exemplo para treinar o tokenizer. \
                          Ele contém várias palavras e frases que ajudam a criar \
                          um vocabulário básico para os benchmarks de chunking.";
        tokenizer.train(sample_text)?;
        
        Ok(Self {
            config,
            tokenizer,
        })
    }
    
    /// 🚀 **EXECUÇÃO COMPLETA DE BENCHMARKS**
    ///
    /// Executa todos os benchmarks configurados e retorna
    /// resultados detalhados para análise.
    pub fn run_all_benchmarks(&mut self, test_text: &str) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        println!("🚀 === INICIANDO BENCHMARKS DE CHUNKING ===");
        println!("📊 Configuração:");
        println!("   • Tamanhos de texto: {:?}", self.config.text_sizes);
        println!("   • Iterações por teste: {}", self.config.iterations);
        println!("   • Estratégias: {:?}", self.config.strategies);
        println!("   • Aquecimento: {} iterações", self.config.warmup_iterations);
        println!();
        
        let text_sizes = self.config.text_sizes.clone();
        let strategies = self.config.strategies.clone();
        let chunking_configs = self.config.chunking_configs.clone();
        
        for text_size in text_sizes {
            let text_sample = if test_text.len() >= text_size {
                // Encontra um boundary de caractere válido
                let mut end = text_size;
                while end > 0 && !test_text.is_char_boundary(end) {
                    end -= 1;
                }
                test_text[..end].to_string()
            } else {
                // Repete o texto se necessário
                let repetitions = (text_size / test_text.len()) + 1;
                let repeated_text = test_text.repeat(repetitions);
                let mut end = text_size.min(repeated_text.len());
                while end > 0 && !repeated_text.is_char_boundary(end) {
                    end -= 1;
                }
                repeated_text[..end].to_string()
            };
            
            println!("📏 Testando com {} caracteres...", text_size);
            
            for strategy in &strategies {
                let config = chunking_configs.get(strategy)
                    .ok_or_else(|| anyhow::anyhow!("Configuração não encontrada para estratégia {:?}", strategy))?;
                
                let result = self.benchmark_strategy(&text_sample, strategy, config)?;
                
                println!("   ✅ {:?}: {:.2}ms, {:.0} chars/s, {} chunks", 
                        strategy, 
                        result.processing_time.as_secs_f64() * 1000.0,
                        result.chars_per_second,
                        result.total_chunks);
                
                results.push(result);
            }
            println!();
        }
        
        println!("🎉 Benchmarks concluídos!");
        Ok(results)
    }
    
    /// 🎯 **BENCHMARK DE ESTRATÉGIA ESPECÍFICA**
    ///
    /// Executa benchmark detalhado para uma estratégia específica.
    fn benchmark_strategy(
        &mut self,
        text: &str,
        strategy: &ChunkingStrategy,
        config: &ChunkingConfig,
    ) -> Result<BenchmarkResult> {
        let mut processor = ChunkProcessor::new(config.clone());
        
        // 🔥 **AQUECIMENTO**
        for _ in 0..self.config.warmup_iterations {
            let _ = processor.process_text(text, &self.tokenizer)?;
        }
        
        // 📊 **MEDIÇÃO PRINCIPAL**
        let mut total_time = Duration::new(0, 0);
        let mut all_chunks = Vec::new();
        
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let chunks = processor.process_text(text, &self.tokenizer)?;
            let elapsed = start.elapsed();
            
            total_time += elapsed;
            if all_chunks.is_empty() {
                all_chunks = chunks; // Salva chunks da primeira iteração para análise
            }
        }
        
        let avg_time = total_time / self.config.iterations as u32;
        
        // 📈 **CÁLCULO DE MÉTRICAS**
        let chars_per_second = text.len() as f64 / avg_time.as_secs_f64();
        
        let total_tokens: usize = all_chunks.iter().map(|c| c.tokens.len()).sum();
        let tokens_per_second = total_tokens as f64 / avg_time.as_secs_f64();
        
        let chunk_sizes: Vec<f64> = all_chunks.iter().map(|c| c.tokens.len() as f64).collect();
        let avg_chunk_size = chunk_sizes.iter().sum::<f64>() / chunk_sizes.len() as f64;
        
        let variance = chunk_sizes.iter()
            .map(|&size| (size - avg_chunk_size).powi(2))
            .sum::<f64>() / chunk_sizes.len() as f64;
        let chunk_size_stddev = variance.sqrt();
        
        let stats = processor.calculate_statistics(&all_chunks);
        
        // 💾 **ESTIMATIVA DE MEMÓRIA**
        let estimated_memory_per_chunk = std::mem::size_of::<TextChunk>() as f64 / 1024.0; // KB
        let peak_memory_mb = (all_chunks.len() as f64 * estimated_memory_per_chunk) / 1024.0; // MB
        
        Ok(BenchmarkResult {
            strategy_name: format!("{:?}", strategy),
            processing_time: avg_time,
            chars_per_second,
            tokens_per_second,
            total_chunks: all_chunks.len(),
            avg_chunk_size,
            chunk_size_stddev,
            avg_information_density: stats.avg_information_density as f64,
            boundary_preservation_rate: stats.boundary_preservation_rate as f64,
            peak_memory_mb,
            memory_per_chunk_kb: estimated_memory_per_chunk,
        })
    }
    
    /// 📊 **RELATÓRIO DETALHADO**
    ///
    /// Gera relatório abrangente dos resultados de benchmark.
    pub fn generate_report(&self, results: &[BenchmarkResult]) -> String {
        let mut report = String::new();
        
        report.push_str("# 📊 RELATÓRIO DE BENCHMARKS DE CHUNKING\n\n");
        
        // 🏆 **RANKING POR PERFORMANCE**
        report.push_str("## 🏆 Ranking por Performance (chars/s)\n\n");
        let mut sorted_by_speed = results.to_vec();
        sorted_by_speed.sort_by(|a, b| b.chars_per_second.partial_cmp(&a.chars_per_second).unwrap());
        
        for (i, result) in sorted_by_speed.iter().enumerate() {
            report.push_str(&format!(
                "{}. **{}**: {:.0} chars/s ({:.2}ms)\n",
                i + 1,
                result.strategy_name,
                result.chars_per_second,
                result.processing_time.as_secs_f64() * 1000.0
            ));
        }
        
        // 🎯 **RANKING POR QUALIDADE**
        report.push_str("\n## 🎯 Ranking por Qualidade (densidade de informação)\n\n");
        let mut sorted_by_quality = results.to_vec();
        sorted_by_quality.sort_by(|a, b| b.avg_information_density.partial_cmp(&a.avg_information_density).unwrap());
        
        for (i, result) in sorted_by_quality.iter().enumerate() {
            report.push_str(&format!(
                "{}. **{}**: {:.3} densidade ({:.1}% preservação)\n",
                i + 1,
                result.strategy_name,
                result.avg_information_density,
                result.boundary_preservation_rate * 100.0
            ));
        }
        
        // 📈 **DETALHES COMPLETOS**
        report.push_str("\n## 📈 Detalhes Completos\n\n");
        
        for result in results {
            report.push_str(&format!("### {}\n\n", result.strategy_name));
            report.push_str(&format!("- **Performance**: {:.0} chars/s, {:.0} tokens/s\n", 
                           result.chars_per_second, result.tokens_per_second));
            report.push_str(&format!("- **Tempo**: {:.2}ms\n", 
                           result.processing_time.as_secs_f64() * 1000.0));
            report.push_str(&format!("- **Chunks**: {} total, {:.1} ± {:.1} tokens\n", 
                           result.total_chunks, result.avg_chunk_size, result.chunk_size_stddev));
            report.push_str(&format!("- **Qualidade**: {:.3} densidade, {:.1}% preservação\n", 
                           result.avg_information_density, result.boundary_preservation_rate * 100.0));
            report.push_str(&format!("- **Memória**: {:.2}MB pico, {:.2}KB/chunk\n\n", 
                           result.peak_memory_mb, result.memory_per_chunk_kb));
        }
        
        report
    }
}

/// 🧪 **TESTES DE STRESS**
///
/// Executa testes de stress para avaliar comportamento
/// em condições extremas de uso.
pub fn run_stress_tests(text: &str) -> Result<()> {
    println!("🧪 === TESTES DE STRESS ===");
    
    // Teste com texto muito grande
    let large_text = text.repeat(100);
    println!("📏 Testando com texto de {} caracteres...", large_text.len());
    
    let config = ChunkingConfig {
        max_chunk_size: 1024,
        min_chunk_size: 128,
        overlap_ratio: 0.1,
        strategy: ChunkingStrategy::Adaptive,
        preserve_sentences: true,
        preserve_paragraphs: true,
    };
    
    let mut processor = ChunkProcessor::new(config);
    let mut tokenizer = BPETokenizer::new(50000)?;
    tokenizer.train(&large_text[..10000.min(large_text.len())])?;
    
    let start = Instant::now();
    let chunks = processor.process_text(&large_text, &tokenizer)?;
    let elapsed = start.elapsed();
    
    println!("✅ Processamento concluído:");
    println!("   • Tempo: {:.2}s", elapsed.as_secs_f64());
    println!("   • Chunks: {}", chunks.len());
    println!("   • Performance: {:.0} chars/s", large_text.len() as f64 / elapsed.as_secs_f64());
    
    Ok(())
}