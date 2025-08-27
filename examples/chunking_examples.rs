//! # Exemplos Práticos do Sistema de Chunking
//!
//! Este arquivo demonstra como usar o sistema de chunking do Mini-GPT
//! em diferentes cenários práticos, desde casos básicos até implementações
//! avançadas para produção.

use std::time::Instant;
use candle_core::Device;

// Importações locais - ajuste conforme a estrutura do seu projeto
// Para compilar este exemplo, você precisa ter os módulos disponíveis
// ou usar: cargo run --example chunking_examples

// Simulação das estruturas para o exemplo compilar
// Em um projeto real, essas viriam dos módulos correspondentes

#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub max_chunk_size: usize,
    pub min_chunk_size: usize,
    pub overlap_ratio: f32,
    pub strategy: ChunkingStrategy,
    pub preserve_sentences: bool,
    pub preserve_paragraphs: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkingStrategy {
    Fixed,
    Semantic,
    Adaptive,
    Overlapping,
}

#[derive(Debug, Clone)]
pub struct TextChunk {
    pub tokens: Vec<u32>,
    pub text: String,
    pub start_position: usize,
    pub end_position: usize,
    pub chunk_index: usize,
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub information_density: f32,
    pub ends_at_boundary: bool,
}

pub struct ChunkProcessor {
    config: ChunkingConfig,
}

pub struct ChunkingStatistics {
    pub total_chunks: usize,
    pub total_tokens: usize,
    pub avg_chunk_size: f32,
    pub min_chunk_size: usize,
    pub max_chunk_size: usize,
    pub avg_information_density: f32,
    pub boundary_preservation_rate: f32,
}

pub struct ChunkQualityReport {
    pub avg_coherence_score: f32,
    pub total_coverage_gaps: usize,
    pub coherence_scores: Vec<f32>,
    pub coverage_gaps: Vec<usize>,
}

pub struct OptimizationSuggestion {
    pub category: OptimizationCategory,
    pub description: String,
    pub impact: OptimizationImpact,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationCategory {
    SizeConsistency,
    InformationDensity,
    BoundaryPreservation,
    Coverage,
    Performance,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationImpact {
    Low,
    Medium,
    High,
}

pub struct ChunkingAnalyzer;
pub struct Tokenizer;

// Implementações básicas para o exemplo funcionar
impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 512,
            min_chunk_size: 64,
            overlap_ratio: 0.1,
            strategy: ChunkingStrategy::Semantic,
            preserve_sentences: true,
            preserve_paragraphs: true,
        }
    }
}

impl ChunkProcessor {
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }
    
    pub fn process_text(&mut self, text: &str, _tokenizer: &Tokenizer) -> Result<Vec<TextChunk>, Box<dyn std::error::Error>> {
        // Implementação simplificada para demonstração
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        
        for (i, chunk_words) in words.chunks(self.config.max_chunk_size / 4).enumerate() {
            let chunk_text = chunk_words.join(" ");
            let tokens: Vec<u32> = chunk_words.iter().enumerate().map(|(j, _)| (i * 100 + j) as u32).collect();
            
            chunks.push(TextChunk {
                tokens,
                text: chunk_text,
                start_position: i * self.config.max_chunk_size,
                end_position: (i + 1) * self.config.max_chunk_size,
                chunk_index: i,
                metadata: ChunkMetadata {
                    sentence_count: chunk_words.len() / 10,
                    paragraph_count: 1,
                    information_density: 0.7,
                    ends_at_boundary: true,
                },
            });
        }
        
        Ok(chunks)
    }
    
    pub fn calculate_statistics(&self, chunks: &[TextChunk]) -> ChunkingStatistics {
        let total_chunks = chunks.len();
        let total_tokens: usize = chunks.iter().map(|c| c.tokens.len()).sum();
        let avg_chunk_size = if total_chunks > 0 { total_tokens as f32 / total_chunks as f32 } else { 0.0 };
        
        ChunkingStatistics {
            total_chunks,
            total_tokens,
            avg_chunk_size,
            min_chunk_size: chunks.iter().map(|c| c.tokens.len()).min().unwrap_or(0),
            max_chunk_size: chunks.iter().map(|c| c.tokens.len()).max().unwrap_or(0),
            avg_information_density: 0.7,
            boundary_preservation_rate: 0.9,
        }
    }
    
    pub fn chunks_to_tensor(&self, chunks: &[TextChunk], device: &Device) -> Result<candle_core::Tensor, Box<dyn std::error::Error>> {
        let max_len = self.config.max_chunk_size;
        let mut tensor_data = Vec::new();
        
        for chunk in chunks {
            let mut padded_tokens = chunk.tokens.clone();
            if padded_tokens.len() < max_len {
                padded_tokens.resize(max_len, 0);
            } else if padded_tokens.len() > max_len {
                padded_tokens.truncate(max_len);
            }
            tensor_data.extend(padded_tokens);
        }
        
        let shape = (chunks.len(), max_len);
        Ok(candle_core::Tensor::from_vec(tensor_data, shape, device)?)
    }
}

impl Tokenizer {
    pub fn new(_vocab_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self)
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        // Tokenização simplificada para demonstração
        Ok(text.split_whitespace().enumerate().map(|(i, _)| i as u32).collect())
    }
    
    pub fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
        // Decodificação simplificada
        Ok(format!("decoded_text_from_{}_tokens", tokens.len()))
    }
}

impl ChunkingAnalyzer {
    pub fn analyze_chunk_quality(chunks: &[TextChunk]) -> ChunkQualityReport {
        ChunkQualityReport {
            avg_coherence_score: 0.8,
            total_coverage_gaps: 0,
            coherence_scores: chunks.iter().map(|_| 0.8).collect(),
            coverage_gaps: Vec::new(),
        }
    }
    
    pub fn suggest_optimizations(_stats: &ChunkingStatistics, _quality: &ChunkQualityReport) -> Vec<OptimizationSuggestion> {
        vec![
            OptimizationSuggestion {
                category: OptimizationCategory::Performance,
                description: "Configuração otimizada detectada".to_string(),
                impact: OptimizationImpact::Low,
            }
        ]
    }
}

/// Exemplo 1: Chunking Básico com Estratégia Fixa
/// 
/// Este exemplo demonstra o uso mais simples do sistema de chunking,
/// dividindo um texto em blocos de tamanho fixo.
pub fn exemplo_chunking_fixo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Exemplo 1: Chunking Fixo ===");
    
    // Texto de exemplo - um artigo sobre inteligência artificial
    let texto = "
        A inteligência artificial representa uma das maiores revoluções tecnológicas da história humana.
        Desde os primeiros algoritmos de busca até os modernos modelos de linguagem, a IA tem transformado
        a forma como interagimos com a tecnologia. Os modelos de linguagem, em particular, demonstram
        capacidades impressionantes de compreensão e geração de texto. Eles são treinados em vastos
        conjuntos de dados textuais, aprendendo padrões linguísticos complexos. O processo de treinamento
        envolve técnicas sofisticadas como atenção multi-cabeça e redes neurais transformers. Essas
        arquiteturas permitem que os modelos capturem dependências de longo alcance no texto, resultando
        em gerações mais coerentes e contextualmente apropriadas. A aplicação prática desses modelos
        abrange desde assistentes virtuais até sistemas de tradução automática.
    ";
    
    // Configuração para chunking fixo
    let config = ChunkingConfig {
        max_chunk_size: 128,  // Chunks de até 128 tokens
        min_chunk_size: 32,   // Mínimo de 32 tokens
        overlap_ratio: 0.0,   // Sem sobreposição
        strategy: ChunkingStrategy::Fixed,
        preserve_sentences: false,
        preserve_paragraphs: false,
    };
    
    // Inicializa o processador e tokenizador
    let mut processor = ChunkProcessor::new(config.clone());
    let tokenizer = Tokenizer::new(1000)?; // Vocabulário de 1000 tokens
    
    // Processa o texto em chunks
    let inicio = Instant::now();
    let chunks = processor.process_text(texto, &tokenizer)?;
    let tempo_processamento = inicio.elapsed();
    
    // Exibe resultados
    println!("Texto original: {} caracteres", texto.len());
    println!("Número de chunks gerados: {}", chunks.len());
    println!("Tempo de processamento: {:?}", tempo_processamento);
    println!();
    
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: {} tokens", i + 1, chunk.tokens.len());
        println!("Texto: {}...", &chunk.text[..std::cmp::min(100, chunk.text.len())]);
        println!("Densidade de informação: {:.2}", chunk.metadata.information_density);
        println!();
    }
    
    // Calcula estatísticas
    let stats = processor.calculate_statistics(&chunks);
    println!("Estatísticas:");
    println!("- Tamanho médio dos chunks: {:.1} tokens", stats.avg_chunk_size);
    println!("- Densidade média de informação: {:.2}", stats.avg_information_density);
    
    Ok(())
}

/// Exemplo 2: Chunking Semântico para Preservar Contexto
/// 
/// Este exemplo mostra como usar chunking semântico para manter
/// a integridade do conteúdo, respeitando limites de sentenças.
pub fn exemplo_chunking_semantico() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Exemplo 2: Chunking Semântico ===");
    
    let texto = "
        O aprendizado de máquina é um subcampo da inteligência artificial. Ele se concentra no
        desenvolvimento de algoritmos que podem aprender e fazer previsões a partir de dados.
        Existem três tipos principais de aprendizado de máquina. O aprendizado supervisionado
        usa dados rotulados para treinar modelos. O aprendizado não supervisionado encontra
        padrões em dados não rotulados. O aprendizado por reforço aprende através de tentativa
        e erro, recebendo recompensas ou penalidades. Cada abordagem tem suas próprias aplicações
        e vantagens. A escolha da técnica depende do problema específico e dos dados disponíveis.
        Redes neurais são uma ferramenta poderosa em todas essas categorias. Elas podem modelar
        relações complexas entre variáveis. As redes neurais profundas, ou deep learning,
        revolucionaram muitas áreas da IA.
    ";
    
    let config = ChunkingConfig {
        max_chunk_size: 100,
        min_chunk_size: 30,
        overlap_ratio: 0.0,
        strategy: ChunkingStrategy::Semantic,
        preserve_sentences: true,
        preserve_paragraphs: true,
    };
    
    let mut processor = ChunkProcessor::new(config.clone());
    let tokenizer = Tokenizer::new(1000)?;
    
    let chunks = processor.process_text(texto, &tokenizer)?;
    
    println!("Chunks semânticos gerados: {}", chunks.len());
    println!();
    
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: {} tokens", i + 1, chunk.tokens.len());
        println!("Sentenças: {}", chunk.metadata.sentence_count);
        println!("Termina em limite semântico: {}", chunk.metadata.ends_at_boundary);
        println!("Texto: {}", chunk.text.trim());
        println!("{}", "-".repeat(50));
    }
    
    // Analisa qualidade dos chunks
    let quality_report = ChunkingAnalyzer::analyze_chunk_quality(&chunks);
    println!("Análise de Qualidade:");
    println!("- Score médio de coerência: {:.2}", quality_report.avg_coherence_score);
    println!("- Gaps de cobertura: {}", quality_report.total_coverage_gaps);
    
    Ok(())
}

/// Exemplo 3: Chunking Adaptativo para Otimização Automática
/// 
/// Este exemplo demonstra como o chunking adaptativo ajusta automaticamente
/// o tamanho dos chunks baseado na densidade de informação do conteúdo.
pub fn exemplo_chunking_adaptativo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Exemplo 3: Chunking Adaptativo ===");
    
    // Texto com diferentes densidades de informação
    let texto_denso = "
        Transformers utilizam mecanismos de atenção multi-cabeça para processar sequências.
        A atenção permite que o modelo foque em diferentes partes da entrada simultaneamente.
        Cada cabeça de atenção aprende diferentes tipos de relações entre tokens.
        A arquitetura encoder-decoder facilita tarefas de tradução e geração de texto.
    ";
    
    let texto_simples = "
        Este é um texto muito simples. Ele tem palavras básicas. As sentenças são curtas.
        Não há conceitos complexos aqui. É fácil de entender. Qualquer pessoa pode ler.
        As ideias são diretas. Não há jargão técnico. É um exemplo de texto simples.
    ";
    
    let config = ChunkingConfig {
        max_chunk_size: 80,
        min_chunk_size: 20,
        overlap_ratio: 0.0,
        strategy: ChunkingStrategy::Adaptive,
        preserve_sentences: true,
        preserve_paragraphs: false,
    };
    
    let mut processor = ChunkProcessor::new(config.clone());
    let tokenizer = Tokenizer::new(1000)?;
    
    println!("Processando texto denso (técnico):");
    let chunks_densos = processor.process_text(texto_denso, &tokenizer)?;
    for (i, chunk) in chunks_densos.iter().enumerate() {
        println!("Chunk {}: {} tokens, densidade: {:.2}", 
                i + 1, chunk.tokens.len(), chunk.metadata.information_density);
    }
    
    println!();
    println!("Processando texto simples:");
    let chunks_simples = processor.process_text(texto_simples, &tokenizer)?;
    for (i, chunk) in chunks_simples.iter().enumerate() {
        println!("Chunk {}: {} tokens, densidade: {:.2}", 
                i + 1, chunk.tokens.len(), chunk.metadata.information_density);
    }
    
    // Compara estatísticas
    let stats_denso = processor.calculate_statistics(&chunks_densos);
    let stats_simples = processor.calculate_statistics(&chunks_simples);
    
    println!();
    println!("Comparação de Estatísticas:");
    println!("Texto denso - Tamanho médio: {:.1}, Densidade: {:.2}", 
            stats_denso.avg_chunk_size, stats_denso.avg_information_density);
    println!("Texto simples - Tamanho médio: {:.1}, Densidade: {:.2}", 
            stats_simples.avg_chunk_size, stats_simples.avg_information_density);
    
    Ok(())
}

/// Exemplo 4: Chunking com Sobreposição para Manter Contexto
/// 
/// Este exemplo mostra como usar sobreposição entre chunks para preservar
/// contexto em tarefas que requerem continuidade entre segmentos.
pub fn exemplo_chunking_sobreposicao() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Exemplo 4: Chunking com Sobreposição ===");
    
    let texto = "
        A arquitetura Transformer revolucionou o processamento de linguagem natural.
        Ela introduziu o conceito de atenção como mecanismo principal de processamento.
        Diferentemente das RNNs, os Transformers processam sequências em paralelo.
        Isso resulta em treinamento mais eficiente e melhor captura de dependências.
        O mecanismo de atenção calcula pesos para cada posição na sequência.
        Esses pesos determinam a importância relativa de cada token.
        A atenção multi-cabeça permite capturar diferentes tipos de relações.
        Cada cabeça foca em aspectos específicos da entrada.
    ";
    
    let config = ChunkingConfig {
        max_chunk_size: 60,
        min_chunk_size: 20,
        overlap_ratio: 0.3, // 30% de sobreposição
        strategy: ChunkingStrategy::Overlapping,
        preserve_sentences: false,
        preserve_paragraphs: false,
    };
    
    let mut processor = ChunkProcessor::new(config.clone());
    let tokenizer = Tokenizer::new(1000)?;
    
    let chunks = processor.process_text(texto, &tokenizer)?;
    
    println!("Chunks com sobreposição gerados: {}", chunks.len());
    println!();
    
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: tokens {}-{} ({} tokens)", 
                i + 1, chunk.start_position, chunk.end_position, chunk.tokens.len());
        
        // Mostra sobreposição com chunk anterior
        if i > 0 {
            let prev_chunk = &chunks[i - 1];
            let overlap_start = chunk.start_position;
            let overlap_end = prev_chunk.end_position;
            if overlap_start < overlap_end {
                let overlap_size = overlap_end - overlap_start;
                println!("  Sobreposição com chunk anterior: {} tokens", overlap_size);
            }
        }
        
        println!("  Texto: {}...", &chunk.text[..std::cmp::min(80, chunk.text.len())]);
        println!();
    }
    
    Ok(())
}

/// Exemplo 5: Conversão de Chunks para Tensores
/// 
/// Este exemplo demonstra como converter chunks processados em tensores
/// prontos para uso em treinamento ou inferência.
pub fn exemplo_chunks_para_tensores() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Exemplo 5: Conversão para Tensores ===");
    
    let texto = "
        Os modelos de linguagem são treinados para prever o próximo token em uma sequência.
        Durante o treinamento, eles aprendem representações ricas de linguagem.
        Essas representações capturam sintaxe, semântica e conhecimento factual.
        O processo de treinamento usa grandes quantidades de texto da internet.
    ";
    
    let config = ChunkingConfig {
        max_chunk_size: 64,
        min_chunk_size: 16,
        overlap_ratio: 0.0,
        strategy: ChunkingStrategy::Semantic,
        preserve_sentences: true,
        preserve_paragraphs: false,
    };
    
    let mut processor = ChunkProcessor::new(config.clone());
    let tokenizer = Tokenizer::new(1000)?;
    let device = Device::Cpu;
    
    // Processa texto em chunks
    let chunks = processor.process_text(texto, &tokenizer)?;
    
    // Converte chunks para tensor
    let tensor = processor.chunks_to_tensor(&chunks, &device)?;
    
    println!("Chunks processados: {}", chunks.len());
    println!("Shape do tensor: {:?}", tensor.shape());
    println!("Dispositivo: {:?}", tensor.device());
    
    // Mostra informações detalhadas de cada chunk
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: {} tokens (padded para {})", 
                i + 1, chunk.tokens.len(), config.max_chunk_size);
        println!("  Primeiros tokens: {:?}", 
                &chunk.tokens[..std::cmp::min(10, chunk.tokens.len())]);
    }
    
    Ok(())
}

/// Exemplo 6: Análise de Performance e Otimização
/// 
/// Este exemplo compara diferentes estratégias de chunking em termos
/// de performance e qualidade dos resultados.
pub fn exemplo_analise_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Exemplo 6: Análise de Performance ===");
    
    let texto = std::fs::read_to_string("data/corpus_pt_br.txt")
        .unwrap_or_else(|_| {
            // Texto de fallback se o arquivo não existir
            "A inteligência artificial está transformando o mundo. ".repeat(1000)
        });
    
    let tokenizer = Tokenizer::new(1000)?;
    let estrategias = vec![
        ("Fixo", ChunkingStrategy::Fixed),
        ("Semântico", ChunkingStrategy::Semantic),
        ("Adaptativo", ChunkingStrategy::Adaptive),
        ("Sobreposição", ChunkingStrategy::Overlapping),
    ];
    
    println!("Comparando estratégias com texto de {} caracteres", texto.len());
    println!();
    
    for (nome, estrategia) in estrategias {
        let config = ChunkingConfig {
            max_chunk_size: 256,
            min_chunk_size: 64,
            overlap_ratio: 0.2,
            strategy: estrategia,
            preserve_sentences: true,
            preserve_paragraphs: true,
        };
        
        let mut processor = ChunkProcessor::new(config.clone());
        
        // Mede tempo de processamento
        let inicio = Instant::now();
        let chunks = processor.process_text(&texto, &tokenizer)?;
        let tempo = inicio.elapsed();
        
        // Calcula estatísticas
        let stats = processor.calculate_statistics(&chunks);
        let quality = ChunkingAnalyzer::analyze_chunk_quality(&chunks);
        
        println!("Estratégia: {}", nome);
        println!("  Tempo: {:?}", tempo);
        println!("  Chunks: {}", stats.total_chunks);
        println!("  Tamanho médio: {:.1} tokens", stats.avg_chunk_size);
        println!("  Densidade média: {:.2}", stats.avg_information_density);
        println!("  Preservação de limites: {:.1}%", stats.boundary_preservation_rate * 100.0);
        println!("  Score de coerência: {:.2}", quality.avg_coherence_score);
        println!();
        
        // Gera sugestões de otimização
        let sugestoes = ChunkingAnalyzer::suggest_optimizations(&stats, &quality);
        if !sugestoes.is_empty() {
            println!("  Sugestões de otimização:");
            for sugestao in sugestoes {
                println!("    - {:?}: {}", sugestao.impact, sugestao.description);
            }
            println!();
        }
    }
    
    Ok(())
}

/// Exemplo 7: Chunking para Diferentes Tipos de Conteúdo
/// 
/// Este exemplo mostra como adaptar a configuração de chunking
/// para diferentes tipos de conteúdo (código, prosa, dados estruturados).
pub fn exemplo_chunking_por_tipo_conteudo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Exemplo 7: Chunking por Tipo de Conteúdo ===");
    
    let tokenizer = Tokenizer::new(1000)?;
    
    // Configuração para código fonte
    let config_codigo = ChunkingConfig {
        max_chunk_size: 200,
        min_chunk_size: 50,
        overlap_ratio: 0.1,
        strategy: ChunkingStrategy::Semantic,
        preserve_sentences: false, // Código não tem sentenças tradicionais
        preserve_paragraphs: true, // Preserva blocos de código
    };
    
    // Configuração para prosa/literatura
    let config_prosa = ChunkingConfig {
        max_chunk_size: 300,
        min_chunk_size: 100,
        overlap_ratio: 0.15,
        strategy: ChunkingStrategy::Adaptive,
        preserve_sentences: true,
        preserve_paragraphs: true,
    };
    
    // Configuração para dados estruturados
    let config_dados = ChunkingConfig {
        max_chunk_size: 128,
        min_chunk_size: 32,
        overlap_ratio: 0.0,
        strategy: ChunkingStrategy::Fixed,
        preserve_sentences: false,
        preserve_paragraphs: false,
    };
    
    let exemplo_codigo = "
        fn calcular_atencao(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
            let scores = q.matmul(k.transpose(-2, -1)?)?;
            let scaled_scores = scores / (k.dim(-1)? as f64).sqrt();
            let attention_weights = softmax(&scaled_scores, -1)?;
            attention_weights.matmul(v)
        }
    ";
    
    let exemplo_prosa = "
        Era uma vez, em um reino distante, uma princesa que possuía o dom de compreender
        a linguagem dos animais. Ela passava seus dias conversando com os pássaros do jardim,
        aprendendo sobre os segredos da floresta e os mistérios do mundo natural.
    ";
    
    let exemplo_dados = "user_id:123,name:João,age:30,city:São Paulo;user_id:124,name:Maria,age:25,city:Rio de Janeiro";
    
    let exemplos = vec![
        ("Código", exemplo_codigo, config_codigo),
        ("Prosa", exemplo_prosa, config_prosa),
        ("Dados", exemplo_dados, config_dados),
    ];
    
    for (tipo, texto, config) in exemplos {
        println!("Processando: {}", tipo);
        let mut processor = ChunkProcessor::new(config.clone());
        let chunks = processor.process_text(texto, &tokenizer)?;
        let stats = processor.calculate_statistics(&chunks);
        
        println!("  Chunks gerados: {}", chunks.len());
        println!("  Tamanho médio: {:.1} tokens", stats.avg_chunk_size);
        println!("  Densidade de informação: {:.2}", stats.avg_information_density);
        println!();
    }
    
    Ok(())
}

/// Função principal que executa todos os exemplos
pub fn executar_todos_exemplos() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Executando Exemplos do Sistema de Chunking\n");
    
    exemplo_chunking_fixo()?;
    println!("\n{}", "=".repeat(60));
    
    exemplo_chunking_semantico()?;
    println!("\n{}", "=".repeat(60));
    
    exemplo_chunking_adaptativo()?;
    println!("\n{}", "=".repeat(60));
    
    exemplo_chunking_sobreposicao()?;
    println!("\n{}", "=".repeat(60));
    
    exemplo_chunks_para_tensores()?;
    println!("\n{}", "=".repeat(60));
    
    exemplo_analise_performance()?;
    println!("\n{}", "=".repeat(60));
    
    exemplo_chunking_por_tipo_conteudo()?;
    
    println!("\n✅ Todos os exemplos executados com sucesso!");
    Ok(())
}

/// Função principal do exemplo
fn main() -> Result<(), Box<dyn std::error::Error>> {
    executar_todos_exemplos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exemplo_chunking_fixo() {
        assert!(exemplo_chunking_fixo().is_ok());
    }
    
    #[test]
    fn test_exemplo_chunking_semantico() {
        assert!(exemplo_chunking_semantico().is_ok());
    }
    
    #[test]
    fn test_exemplo_chunking_adaptativo() {
        assert!(exemplo_chunking_adaptativo().is_ok());
    }
}