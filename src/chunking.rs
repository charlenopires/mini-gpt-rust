//! # Sistema de Chunking para Mini-GPT
//!
//! Este módulo implementa diferentes estratégias de chunking (divisão de dados em partes menores)
//! para otimizar o processamento de textos longos em modelos de linguagem.
//!
//! ## O que é Chunking?
//!
//! Chunking é o processo de dividir textos longos em segmentos menores e gerenciáveis.
//! Isso é essencial para:
//! - Respeitar limites de contexto do modelo (block_size)
//! - Otimizar uso de memória durante treinamento e inferência
//! - Melhorar a qualidade do aprendizado em sequências longas
//! - Permitir processamento paralelo de grandes volumes de texto
//!
//! ## Estratégias Implementadas
//!
//! 1. **Chunking Fixo**: Divide o texto em blocos de tamanho fixo
//! 2. **Chunking Semântico**: Respeita limites de sentenças e parágrafos
//! 3. **Chunking Adaptativo**: Ajusta o tamanho baseado no conteúdo
//! 4. **Chunking com Sobreposição**: Mantém contexto entre chunks adjacentes

use candle_core::{Result, Tensor};
use std::collections::VecDeque;
use anyhow::anyhow;

/// Configuração para estratégias de chunking
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Tamanho máximo de cada chunk em tokens
    pub max_chunk_size: usize,
    /// Tamanho mínimo de cada chunk em tokens
    pub min_chunk_size: usize,
    /// Sobreposição entre chunks adjacentes (0.0 a 1.0)
    pub overlap_ratio: f32,
    /// Estratégia de chunking a ser utilizada
    pub strategy: ChunkingStrategy,
    /// Se deve preservar limites de sentenças
    pub preserve_sentences: bool,
    /// Se deve preservar limites de parágrafos
    pub preserve_paragraphs: bool,
}

/// Diferentes estratégias de chunking disponíveis
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChunkingStrategy {
    /// Divisão em blocos de tamanho fixo
    /// Mais simples e rápida, mas pode quebrar contexto
    Fixed,
    /// Divisão respeitando limites semânticos
    /// Preserva integridade do conteúdo, mas chunks podem variar em tamanho
    Semantic,
    /// Divisão adaptativa baseada no conteúdo
    /// Balanceia tamanho e semântica dinamicamente
    Adaptive,
    /// Divisão com sobreposição para manter contexto
    /// Útil para tarefas que requerem continuidade entre chunks
    Overlapping,
}

/// Representa um chunk de texto processado
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// Tokens do chunk
    pub tokens: Vec<u32>,
    /// Texto original do chunk
    pub text: String,
    /// Posição inicial no texto original
    pub start_position: usize,
    /// Posição final no texto original
    pub end_position: usize,
    /// Índice do chunk na sequência
    pub chunk_index: usize,
    /// Metadados adicionais
    pub metadata: ChunkMetadata,
}

/// Metadados associados a cada chunk
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    /// Número de sentenças no chunk
    pub sentence_count: usize,
    /// Número de parágrafos no chunk
    pub paragraph_count: usize,
    /// Densidade de informação (tokens únicos / total de tokens)
    pub information_density: f32,
    /// Se o chunk termina em uma fronteira semântica natural
    pub ends_at_boundary: bool,
}

/// Processador principal de chunking
pub struct ChunkProcessor {
    config: ChunkingConfig,
    /// Cache de chunks processados para otimização
    chunk_cache: std::collections::HashMap<String, Vec<TextChunk>>,
}

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
    /// Cria um novo processador de chunking
    ///
    /// # Argumentos
    /// * `config` - Configuração para o processamento de chunks
    ///
    /// # Exemplo
    /// ```rust
    /// let config = ChunkingConfig::default();
    /// let processor = ChunkProcessor::new(config);
    /// ```
    pub fn new(config: ChunkingConfig) -> Self {
        Self {
            config,
            chunk_cache: std::collections::HashMap::new(),
        }
    }

    /// Processa um texto em chunks usando a estratégia configurada
    ///
    /// # Argumentos
    /// * `text` - Texto a ser dividido em chunks
    /// * `tokenizer` - Referência ao tokenizador para conversão texto->tokens
    ///
    /// # Retorna
    /// Vector de chunks processados
    ///
    /// # Exemplo
    /// ```rust
    /// let chunks = processor.process_text(&text, &tokenizer)?;
    /// for chunk in chunks {
    ///     println!("Chunk {}: {} tokens", chunk.chunk_index, chunk.tokens.len());
    /// }
    /// ```
    pub fn process_text(
        &mut self,
        text: &str,
        tokenizer: &crate::tokenizer::BPETokenizer,
    ) -> Result<Vec<TextChunk>> {
        // Verifica cache primeiro para otimização
        let cache_key = format!("{:?}:{}", self.config, text.len());
        if let Some(cached_chunks) = self.chunk_cache.get(&cache_key) {
            return Ok(cached_chunks.clone());
        }

        let chunks = match self.config.strategy {
            ChunkingStrategy::Fixed => self.process_fixed_chunking(text, tokenizer)?,
            ChunkingStrategy::Semantic => self.process_semantic_chunking(text, tokenizer)?,
            ChunkingStrategy::Adaptive => self.process_adaptive_chunking(text, tokenizer)?,
            ChunkingStrategy::Overlapping => self.process_overlapping_chunking(text, tokenizer)?,
        };

        // Armazena no cache para futuras consultas
        self.chunk_cache.insert(cache_key, chunks.clone());
        Ok(chunks)
    }

    /// Implementa chunking de tamanho fixo
    ///
    /// Esta estratégia divide o texto em blocos de tamanho fixo, ignorando
    /// limites semânticos. É a mais rápida, mas pode quebrar contexto.
    fn process_fixed_chunking(
        &self,
        text: &str,
        tokenizer: &crate::tokenizer::BPETokenizer,
    ) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        let tokens = tokenizer.encode(text).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let chunk_size = self.config.max_chunk_size;

        for (i, chunk_tokens) in tokens.chunks(chunk_size).enumerate() {
            let start_pos = i * chunk_size;
            let end_pos = std::cmp::min(start_pos + chunk_size, tokens.len());
            
            // Decodifica os tokens de volta para texto
            let chunk_text = tokenizer.decode(&chunk_tokens.to_vec()).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let chunk_tokens_u32: Vec<u32> = chunk_tokens.iter().map(|&x| x as u32).collect();
            
            let metadata = ChunkMetadata {
                sentence_count: self.count_sentences(&chunk_text),
                paragraph_count: self.count_paragraphs(&chunk_text),
                information_density: self.calculate_information_density(&chunk_tokens_u32),
                ends_at_boundary: false, // Chunking fixo não respeita limites
            };

            chunks.push(TextChunk {
                tokens: chunk_tokens_u32,
                text: chunk_text,
                start_position: start_pos,
                end_position: end_pos,
                chunk_index: i,
                metadata,
            });
        }

        Ok(chunks)
    }

    /// Implementa chunking semântico
    ///
    /// Esta estratégia respeita limites de sentenças e parágrafos,
    /// criando chunks mais coerentes semanticamente.
    fn process_semantic_chunking(
        &self,
        text: &str,
        tokenizer: &crate::tokenizer::BPETokenizer,
    ) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut chunk_index = 0;
        let mut start_position = 0;

        // Divide o texto em sentenças
        let sentences = self.split_into_sentences(text);
        
        for sentence in sentences {
            let potential_chunk = if current_chunk.is_empty() {
                sentence.clone()
            } else {
                format!("{} {}", current_chunk, sentence)
            };

            let potential_tokens = tokenizer.encode(&potential_chunk).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

            // Se adicionar esta sentença exceder o limite, finaliza o chunk atual
            if potential_tokens.len() > self.config.max_chunk_size && !current_chunk.is_empty() {
                let chunk_tokens = tokenizer.encode(&current_chunk).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let chunk_tokens_u32: Vec<u32> = chunk_tokens.iter().map(|&x| x as u32).collect();
                let metadata = ChunkMetadata {
                    sentence_count: self.count_sentences(&current_chunk),
                    paragraph_count: self.count_paragraphs(&current_chunk),
                    information_density: self.calculate_information_density(&chunk_tokens_u32),
                    ends_at_boundary: true, // Termina em limite de sentença
                };

                chunks.push(TextChunk {
                    tokens: chunk_tokens_u32,
                    text: current_chunk.clone(),
                    start_position,
                    end_position: start_position + current_chunk.len(),
                    chunk_index,
                    metadata,
                });

                // Inicia novo chunk
                current_chunk = sentence;
                start_position += current_chunk.len();
                chunk_index += 1;
            } else {
                current_chunk = potential_chunk;
            }
        }

        // Adiciona o último chunk se não estiver vazio
        if !current_chunk.is_empty() {
            let chunk_tokens = tokenizer.encode(&current_chunk).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let chunk_tokens_u32: Vec<u32> = chunk_tokens.iter().map(|&x| x as u32).collect();
            let metadata = ChunkMetadata {
                sentence_count: self.count_sentences(&current_chunk),
                paragraph_count: self.count_paragraphs(&current_chunk),
                information_density: self.calculate_information_density(&chunk_tokens_u32),
                ends_at_boundary: true,
            };

            chunks.push(TextChunk {
                tokens: chunk_tokens_u32,
                text: current_chunk,
                start_position,
                end_position: text.len(),
                chunk_index,
                metadata,
            });
        }

        Ok(chunks)
    }

    /// Implementa chunking adaptativo
    ///
    /// Esta estratégia ajusta o tamanho dos chunks baseado na densidade
    /// de informação e complexidade do conteúdo.
    fn process_adaptive_chunking(
        &self,
        text: &str,
        tokenizer: &crate::tokenizer::BPETokenizer,
    ) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        let sentences = self.split_into_sentences(text);
        let mut current_chunk = String::new();
        let mut chunk_index = 0;
        let mut start_position = 0;

        for sentence in sentences {
            let potential_chunk = if current_chunk.is_empty() {
                sentence.clone()
            } else {
                format!("{} {}", current_chunk, sentence)
            };

            let potential_tokens = tokenizer.encode(&potential_chunk).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let potential_tokens_u32: Vec<u32> = potential_tokens.iter().map(|&x| x as u32).collect();
            let density = self.calculate_information_density(&potential_tokens_u32);
            
            // Ajusta o tamanho máximo baseado na densidade de informação
            let adaptive_max_size = if density > 0.8 {
                // Alta densidade: chunks menores para melhor processamento
                (self.config.max_chunk_size as f32 * 0.7) as usize
            } else if density < 0.3 {
                // Baixa densidade: chunks maiores são aceitáveis
                (self.config.max_chunk_size as f32 * 1.2) as usize
            } else {
                self.config.max_chunk_size
            };

            if potential_tokens.len() > adaptive_max_size && !current_chunk.is_empty() {
                let chunk_tokens = tokenizer.encode(&current_chunk).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let chunk_tokens_u32: Vec<u32> = chunk_tokens.iter().map(|&x| x as u32).collect();
                let metadata = ChunkMetadata {
                    sentence_count: self.count_sentences(&current_chunk),
                    paragraph_count: self.count_paragraphs(&current_chunk),
                    information_density: self.calculate_information_density(&chunk_tokens_u32),
                    ends_at_boundary: true,
                };

                chunks.push(TextChunk {
                    tokens: chunk_tokens_u32,
                    text: current_chunk.clone(),
                    start_position,
                    end_position: start_position + current_chunk.len(),
                    chunk_index,
                    metadata,
                });

                current_chunk = sentence;
                start_position += current_chunk.len();
                chunk_index += 1;
            } else {
                current_chunk = potential_chunk;
            }
        }

        // Adiciona o último chunk
        if !current_chunk.is_empty() {
            let chunk_tokens = tokenizer.encode(&current_chunk).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let chunk_tokens_u32: Vec<u32> = chunk_tokens.iter().map(|&x| x as u32).collect();
            let metadata = ChunkMetadata {
                sentence_count: self.count_sentences(&current_chunk),
                paragraph_count: self.count_paragraphs(&current_chunk),
                information_density: self.calculate_information_density(&chunk_tokens_u32),
                ends_at_boundary: true,
            };

            chunks.push(TextChunk {
                tokens: chunk_tokens_u32,
                text: current_chunk,
                start_position,
                end_position: text.len(),
                chunk_index,
                metadata,
            });
        }

        Ok(chunks)
    }

    /// Implementa chunking com sobreposição
    ///
    /// Esta estratégia mantém uma sobreposição entre chunks adjacentes
    /// para preservar contexto e continuidade.
    fn process_overlapping_chunking(
        &self,
        text: &str,
        tokenizer: &crate::tokenizer::BPETokenizer,
    ) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        let tokens = tokenizer.encode(text).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let chunk_size = self.config.max_chunk_size;
        let overlap_size = (chunk_size as f32 * self.config.overlap_ratio) as usize;
        let step_size = chunk_size - overlap_size;

        let mut start_idx = 0;
        let mut chunk_index = 0;

        while start_idx < tokens.len() {
            let end_idx = std::cmp::min(start_idx + chunk_size, tokens.len());
            let chunk_tokens = tokens[start_idx..end_idx].to_vec();
            
            // Decodifica os tokens para texto
            let chunk_text = tokenizer.decode(&chunk_tokens).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let chunk_tokens_u32: Vec<u32> = chunk_tokens.iter().map(|&x| x as u32).collect();
            
            let metadata = ChunkMetadata {
                sentence_count: self.count_sentences(&chunk_text),
                paragraph_count: self.count_paragraphs(&chunk_text),
                information_density: self.calculate_information_density(&chunk_tokens_u32),
                ends_at_boundary: false, // Sobreposição pode quebrar limites
            };

            chunks.push(TextChunk {
                tokens: chunk_tokens_u32,
                text: chunk_text,
                start_position: start_idx,
                end_position: end_idx,
                chunk_index,
                metadata,
            });

            start_idx += step_size;
            chunk_index += 1;

            // Evita chunks muito pequenos no final
            if tokens.len() - start_idx < self.config.min_chunk_size {
                break;
            }
        }

        Ok(chunks)
    }

    /// Converte chunks em tensores para treinamento
    ///
    /// # Argumentos
    /// * `chunks` - Vector de chunks processados
    /// * `device` - Dispositivo onde criar os tensores
    ///
    /// # Retorna
    /// Tensor com shape [num_chunks, max_chunk_size] contendo os tokens
    pub fn chunks_to_tensor(
        &self,
        chunks: &[TextChunk],
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let max_len = self.config.max_chunk_size;
        let mut tensor_data = Vec::new();

        for chunk in chunks {
            let mut padded_tokens = chunk.tokens.clone();
            
            // Padding ou truncamento para tamanho fixo
            if padded_tokens.len() < max_len {
                padded_tokens.resize(max_len, 0); // Padding com zeros
            } else if padded_tokens.len() > max_len {
                padded_tokens.truncate(max_len);
            }

            tensor_data.extend(padded_tokens);
        }

        let shape = (chunks.len(), max_len);
        Tensor::from_vec(tensor_data, shape, device)
    }

    /// Calcula estatísticas dos chunks processados
    pub fn calculate_statistics(&self, chunks: &[TextChunk]) -> ChunkingStatistics {
        let total_chunks = chunks.len();
        let total_tokens: usize = chunks.iter().map(|c| c.tokens.len()).sum();
        let avg_chunk_size = if total_chunks > 0 {
            total_tokens as f32 / total_chunks as f32
        } else {
            0.0
        };

        let min_chunk_size = chunks.iter().map(|c| c.tokens.len()).min().unwrap_or(0);
        let max_chunk_size = chunks.iter().map(|c| c.tokens.len()).max().unwrap_or(0);
        
        let avg_information_density = if total_chunks > 0 {
            chunks.iter().map(|c| c.metadata.information_density).sum::<f32>() / total_chunks as f32
        } else {
            0.0
        };

        let boundary_preservation_rate = if total_chunks > 0 {
            let boundary_chunks = chunks.iter().filter(|c| c.metadata.ends_at_boundary).count();
            boundary_chunks as f32 / total_chunks as f32
        } else {
            0.0
        };

        ChunkingStatistics {
            total_chunks,
            total_tokens,
            avg_chunk_size,
            min_chunk_size,
            max_chunk_size,
            avg_information_density,
            boundary_preservation_rate,
        }
    }

    // Métodos auxiliares para análise de texto

    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        // Implementação simples de divisão por sentenças
        // Em produção, considere usar bibliotecas especializadas como spacy ou nltk
        text.split(&['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn count_sentences(&self, text: &str) -> usize {
        text.matches(&['.', '!', '?']).count()
    }

    fn count_paragraphs(&self, text: &str) -> usize {
        text.split("\n\n").filter(|p| !p.trim().is_empty()).count()
    }

    fn calculate_information_density(&self, tokens: &[u32]) -> f32 {
        if tokens.is_empty() {
            return 0.0;
        }

        let unique_tokens: std::collections::HashSet<_> = tokens.iter().collect();
        unique_tokens.len() as f32 / tokens.len() as f32
    }
}

/// Estatísticas de processamento de chunks
#[derive(Debug, Clone)]
pub struct ChunkingStatistics {
    pub total_chunks: usize,
    pub total_tokens: usize,
    pub avg_chunk_size: f32,
    pub min_chunk_size: usize,
    pub max_chunk_size: usize,
    pub avg_information_density: f32,
    pub boundary_preservation_rate: f32,
}

/// Utilitários para análise e otimização de chunking
pub struct ChunkingAnalyzer;

impl ChunkingAnalyzer {
    /// Analisa a qualidade dos chunks gerados
    pub fn analyze_chunk_quality(chunks: &[TextChunk]) -> ChunkQualityReport {
        let mut coherence_scores = Vec::new();
        let mut coverage_gaps = Vec::new();
        
        for (i, chunk) in chunks.iter().enumerate() {
            // Calcula score de coerência baseado em densidade de informação
            let coherence = chunk.metadata.information_density;
            coherence_scores.push(coherence);
            
            // Verifica gaps de cobertura entre chunks adjacentes
            if i > 0 {
                let prev_chunk = &chunks[i - 1];
                let gap = chunk.start_position.saturating_sub(prev_chunk.end_position);
                if gap > 0 {
                    coverage_gaps.push(gap);
                }
            }
        }
        
        let avg_coherence = if !coherence_scores.is_empty() {
            coherence_scores.iter().sum::<f32>() / coherence_scores.len() as f32
        } else {
            0.0
        };
        
        let total_coverage_gaps: usize = coverage_gaps.iter().sum();
        
        ChunkQualityReport {
            avg_coherence_score: avg_coherence,
            total_coverage_gaps,
            coherence_scores,
            coverage_gaps,
        }
    }
    
    /// Sugere otimizações para a configuração de chunking
    pub fn suggest_optimizations(
        stats: &ChunkingStatistics,
        quality: &ChunkQualityReport,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        
        // Analisa variação no tamanho dos chunks
        let size_variation = stats.max_chunk_size as f32 / stats.min_chunk_size as f32;
        if size_variation > 3.0 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::SizeConsistency,
                description: "Alta variação no tamanho dos chunks. Considere usar chunking adaptativo ou ajustar parâmetros.".to_string(),
                impact: OptimizationImpact::Medium,
            });
        }
        
        // Analisa densidade de informação
        if stats.avg_information_density < 0.3 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::InformationDensity,
                description: "Baixa densidade de informação. Considere aumentar o tamanho dos chunks ou usar pré-processamento.".to_string(),
                impact: OptimizationImpact::High,
            });
        }
        
        // Analisa preservação de limites
        if stats.boundary_preservation_rate < 0.7 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::BoundaryPreservation,
                description: "Baixa taxa de preservação de limites semânticos. Use chunking semântico ou adaptativo.".to_string(),
                impact: OptimizationImpact::High,
            });
        }
        
        // Analisa gaps de cobertura
        if quality.total_coverage_gaps > 0 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::Coverage,
                description: "Gaps de cobertura detectados. Considere usar chunking com sobreposição.".to_string(),
                impact: OptimizationImpact::Medium,
            });
        }
        
        suggestions
    }
}

/// Relatório de qualidade dos chunks
#[derive(Debug, Clone)]
pub struct ChunkQualityReport {
    pub avg_coherence_score: f32,
    pub total_coverage_gaps: usize,
    pub coherence_scores: Vec<f32>,
    pub coverage_gaps: Vec<usize>,
}

/// Sugestão de otimização
#[derive(Debug, Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking_config_default() {
        let config = ChunkingConfig::default();
        assert_eq!(config.max_chunk_size, 512);
        assert_eq!(config.strategy, ChunkingStrategy::Semantic);
    }

    #[test]
    fn test_information_density_calculation() {
        let processor = ChunkProcessor::new(ChunkingConfig::default());
        let tokens = vec![1, 2, 3, 2, 1]; // 3 tokens únicos de 5 total
        let density = processor.calculate_information_density(&tokens);
        assert_eq!(density, 0.6);
    }

    #[test]
    fn test_sentence_counting() {
        let processor = ChunkProcessor::new(ChunkingConfig::default());
        let text = "Esta é a primeira sentença. Esta é a segunda! E esta é a terceira?";
        let count = processor.count_sentences(text);
        assert_eq!(count, 3);
    }
}