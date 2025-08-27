//! # Exemplo Didático: Embeddings e Representações Vetoriais
//!
//! Este exemplo demonstra como funcionam os embeddings em modelos de linguagem:
//! - Criação de embeddings de tokens
//! - Embeddings posicionais
//! - Operações com vetores de embedding
//! - Similaridade semântica
//! - Visualização de relações
//!
//! ## Como executar:
//! ```bash
//! cargo run --example embeddings_explained
//! ```

use std::collections::HashMap;
use std::f32::consts::PI;

/// Representa um vetor de embedding
#[derive(Debug, Clone)]
struct Embedding {
    vector: Vec<f32>,
    dimension: usize,
}

impl Embedding {
    /// Cria um novo embedding com valores aleatórios
    fn new(dimension: usize) -> Self {
        let vector: Vec<f32> = (0..dimension)
            .map(|i| ((i as f32 * 0.1).sin() * 0.5))
            .collect();
        
        Self { vector, dimension }
    }
    
    /// Cria um embedding a partir de valores específicos
    fn from_values(values: Vec<f32>) -> Self {
        let dimension = values.len();
        Self {
            vector: values,
            dimension,
        }
    }
    
    /// Calcula a similaridade cosseno entre dois embeddings
    fn cosine_similarity(&self, other: &Embedding) -> f32 {
        assert_eq!(self.dimension, other.dimension);
        
        let dot_product: f32 = self.vector.iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    /// Adiciona dois embeddings (usado em conexões residuais)
    fn add(&self, other: &Embedding) -> Embedding {
        assert_eq!(self.dimension, other.dimension);
        
        let vector: Vec<f32> = self.vector.iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        Embedding::from_values(vector)
    }
    
    /// Multiplica o embedding por um escalar
    fn scale(&self, factor: f32) -> Embedding {
        let vector: Vec<f32> = self.vector.iter()
            .map(|x| x * factor)
            .collect();
        
        Embedding::from_values(vector)
    }
    
    /// Normaliza o embedding para ter norma unitária
    fn normalize(&self) -> Embedding {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm == 0.0 {
            return self.clone();
        }
        
        let vector: Vec<f32> = self.vector.iter()
            .map(|x| x / norm)
            .collect();
        
        Embedding::from_values(vector)
    }
    
    /// Calcula a distância euclidiana
    fn euclidean_distance(&self, other: &Embedding) -> f32 {
        assert_eq!(self.dimension, other.dimension);
        
        self.vector.iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// ## 1. Camada de Embedding de Tokens
/// 
/// Converte IDs de tokens em vetores densos de dimensão fixa
#[derive(Debug)]
struct TokenEmbedding {
    embeddings: HashMap<usize, Embedding>,
    vocab_size: usize,
    embedding_dim: usize,
}

impl TokenEmbedding {
    fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut embeddings = HashMap::new();
        
        // Inicializar embeddings para cada token no vocabulário
        for token_id in 0..vocab_size {
            // Usar uma semente baseada no ID para embeddings consistentes
            let embedding = Self::create_deterministic_embedding(token_id, embedding_dim);
            embeddings.insert(token_id, embedding);
        }
        
        Self {
            embeddings,
            vocab_size,
            embedding_dim,
        }
    }
    
    /// Cria um embedding determinístico baseado no ID do token
    fn create_deterministic_embedding(token_id: usize, dim: usize) -> Embedding {
        let mut vector = Vec::with_capacity(dim);
        
        for i in 0..dim {
            // Usar funções trigonométricas para criar padrões únicos
            let value = ((token_id as f32 * 0.1 + i as f32 * 0.05).sin() * 0.5) +
                       ((token_id as f32 * 0.07 + i as f32 * 0.03).cos() * 0.3);
            vector.push(value);
        }
        
        Embedding::from_values(vector)
    }
    
    /// Obtém o embedding para um token específico
    fn get_embedding(&self, token_id: usize) -> Option<&Embedding> {
        self.embeddings.get(&token_id)
    }
    
    /// Processa uma sequência de tokens
    fn forward(&self, token_ids: &[usize]) -> Vec<Embedding> {
        println!("\n🔤 Convertendo tokens em embeddings...");
        println!("Sequência de entrada: {:?}", token_ids);
        
        let mut embeddings = Vec::new();
        
        for &token_id in token_ids {
            if let Some(embedding) = self.get_embedding(token_id) {
                embeddings.push(embedding.clone());
                println!("   Token {} → Embedding[{}] (primeiros 4 valores: {:?})", 
                    token_id, 
                    self.embedding_dim,
                    &embedding.vector[0..4.min(embedding.vector.len())]
                );
            } else {
                println!("   ⚠️  Token {} não encontrado no vocabulário!", token_id);
            }
        }
        
        println!("✅ {} embeddings gerados", embeddings.len());
        embeddings
    }
}

/// ## 2. Embeddings Posicionais
/// 
/// Adicionam informação sobre a posição dos tokens na sequência
#[derive(Debug)]
struct PositionalEmbedding {
    max_length: usize,
    embedding_dim: usize,
    embeddings: Vec<Embedding>,
}

impl PositionalEmbedding {
    fn new(max_length: usize, embedding_dim: usize) -> Self {
        let mut embeddings = Vec::with_capacity(max_length);
        
        // Gerar embeddings posicionais usando encoding sinusoidal
        for pos in 0..max_length {
            embeddings.push(Self::create_sinusoidal_embedding(pos, embedding_dim));
        }
        
        Self {
            max_length,
            embedding_dim,
            embeddings,
        }
    }
    
    /// Cria embedding posicional usando funções sinusoidais
    /// 
    /// Fórmula do paper "Attention Is All You Need":
    /// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    fn create_sinusoidal_embedding(position: usize, dim: usize) -> Embedding {
        let mut vector = Vec::with_capacity(dim);
        
        for i in 0..dim {
            let angle = position as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / dim as f32);
            
            let value = if i % 2 == 0 {
                angle.sin()  // Posições pares: seno
            } else {
                angle.cos()  // Posições ímpares: cosseno
            };
            
            vector.push(value);
        }
        
        Embedding::from_values(vector)
    }
    
    /// Obtém embeddings posicionais para uma sequência
    fn forward(&self, sequence_length: usize) -> Vec<Embedding> {
        println!("\n📍 Gerando embeddings posicionais...");
        println!("Comprimento da sequência: {}", sequence_length);
        
        let mut pos_embeddings = Vec::new();
        
        for pos in 0..sequence_length {
            if pos < self.max_length {
                pos_embeddings.push(self.embeddings[pos].clone());
                println!("   Posição {} → Embedding posicional (primeiros 4: {:?})", 
                    pos, 
                    &self.embeddings[pos].vector[0..4.min(self.embedding_dim)]
                );
            } else {
                println!("   ⚠️  Posição {} excede comprimento máximo {}", pos, self.max_length);
                break;
            }
        }
        
        println!("✅ {} embeddings posicionais gerados", pos_embeddings.len());
        pos_embeddings
    }
}

/// ## 3. Combinação de Embeddings
/// 
/// Combina embeddings de tokens e posicionais
#[derive(Debug)]
struct EmbeddingLayer {
    token_embedding: TokenEmbedding,
    positional_embedding: PositionalEmbedding,
}

impl EmbeddingLayer {
    fn new(vocab_size: usize, embedding_dim: usize, max_length: usize) -> Self {
        Self {
            token_embedding: TokenEmbedding::new(vocab_size, embedding_dim),
            positional_embedding: PositionalEmbedding::new(max_length, embedding_dim),
        }
    }
    
    /// Processa uma sequência completa
    fn forward(&self, token_ids: &[usize]) -> Vec<Embedding> {
        println!("\n🏗️  === CAMADA DE EMBEDDING COMPLETA ===");
        
        // 1. Obter embeddings de tokens
        let token_embeddings = self.token_embedding.forward(token_ids);
        
        // 2. Obter embeddings posicionais
        let pos_embeddings = self.positional_embedding.forward(token_ids.len());
        
        // 3. Combinar embeddings (soma elemento a elemento)
        println!("\n➕ Combinando embeddings de tokens e posicionais...");
        let mut combined_embeddings = Vec::new();
        
        for (i, (token_emb, pos_emb)) in token_embeddings.iter().zip(pos_embeddings.iter()).enumerate() {
            let combined = token_emb.add(pos_emb);
            combined_embeddings.push(combined);
            
            println!("   Posição {}: Token + Posicional → Embedding final", i);
        }
        
        println!("✅ Embeddings finais prontos para o Transformer!");
        combined_embeddings
    }
}

/// ## 4. Análise de Similaridade Semântica
struct SemanticAnalyzer {
    embedding_layer: EmbeddingLayer,
    vocabulary: HashMap<String, usize>,
}

impl SemanticAnalyzer {
    fn new() -> Self {
        // Criar vocabulário de exemplo
        let mut vocabulary = HashMap::new();
        let words = vec![
            "gato", "cachorro", "animal", "pet",
            "carro", "bicicleta", "transporte", "veículo",
            "casa", "apartamento", "moradia", "lar",
            "livro", "revista", "jornal", "leitura",
            "feliz", "alegre", "contente", "satisfeito",
        ];
        
        for (i, word) in words.iter().enumerate() {
            vocabulary.insert(word.to_string(), i);
        }
        
        let embedding_layer = EmbeddingLayer::new(words.len(), 64, 100);
        
        Self {
            embedding_layer,
            vocabulary,
        }
    }
    
    fn get_word_embedding(&self, word: &str) -> Option<Embedding> {
        if let Some(&token_id) = self.vocabulary.get(word) {
            self.embedding_layer.token_embedding.get_embedding(token_id).cloned()
        } else {
            None
        }
    }
    
    fn find_similar_words(&self, target_word: &str, top_k: usize) -> Vec<(String, f32)> {
        if let Some(target_embedding) = self.get_word_embedding(target_word) {
            let mut similarities = Vec::new();
            
            for (word, &token_id) in &self.vocabulary {
                if word != target_word {
                    if let Some(word_embedding) = self.embedding_layer.token_embedding.get_embedding(token_id) {
                        let similarity = target_embedding.cosine_similarity(word_embedding);
                        similarities.push((word.clone(), similarity));
                    }
                }
            }
            
            // Ordenar por similaridade (maior primeiro)
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            similarities.into_iter().take(top_k).collect()
        } else {
            Vec::new()
        }
    }
    
    fn demonstrate_semantic_relationships(&self) {
        println!("\n\n🔍 === ANÁLISE DE SIMILARIDADE SEMÂNTICA ===");
        
        let test_words = vec!["gato", "carro", "casa", "feliz"];
        
        for word in test_words {
            println!("\n🎯 Palavras similares a '{}':", word);
            let similar = self.find_similar_words(word, 3);
            
            for (i, (similar_word, similarity)) in similar.iter().enumerate() {
                println!("   {}. {} (similaridade: {:.3})", 
                    i + 1, similar_word, similarity);
            }
        }
    }
}

/// ## 5. Demonstração de Operações Vetoriais
fn demonstrate_vector_operations() {
    println!("\n\n🧮 === OPERAÇÕES COM EMBEDDINGS ===");
    
    // Criar embeddings de exemplo
    let emb1 = Embedding::from_values(vec![1.0, 2.0, 3.0, 4.0]);
    let emb2 = Embedding::from_values(vec![2.0, 1.0, 4.0, 3.0]);
    let emb3 = Embedding::from_values(vec![1.0, 2.0, 3.0, 4.0]); // Idêntico ao emb1
    
    println!("\n📊 Embeddings de teste:");
    println!("   Embedding 1: {:?}", emb1.vector);
    println!("   Embedding 2: {:?}", emb2.vector);
    println!("   Embedding 3: {:?}", emb3.vector);
    
    // Similaridade cosseno
    println!("\n📐 Similaridades cosseno:");
    println!("   emb1 ↔ emb2: {:.3}", emb1.cosine_similarity(&emb2));
    println!("   emb1 ↔ emb3: {:.3}", emb1.cosine_similarity(&emb3));
    println!("   emb2 ↔ emb3: {:.3}", emb2.cosine_similarity(&emb3));
    
    // Distância euclidiana
    println!("\n📏 Distâncias euclidianas:");
    println!("   emb1 ↔ emb2: {:.3}", emb1.euclidean_distance(&emb2));
    println!("   emb1 ↔ emb3: {:.3}", emb1.euclidean_distance(&emb3));
    println!("   emb2 ↔ emb3: {:.3}", emb2.euclidean_distance(&emb3));
    
    // Operações aritméticas
    println!("\n➕ Operações aritméticas:");
    let sum = emb1.add(&emb2);
    println!("   emb1 + emb2: {:?}", sum.vector);
    
    let scaled = emb1.scale(2.0);
    println!("   emb1 × 2: {:?}", scaled.vector);
    
    let normalized = emb1.normalize();
    println!("   emb1 normalizado: {:?}", 
        normalized.vector.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>());
}

/// Função principal
fn main() {
    println!("🚀 === GUIA COMPLETO DE EMBEDDINGS ===");
    println!("\nEmbeddings são representações vetoriais densas que capturam");
    println!("significado semântico e relações entre tokens.\n");
    
    // === DEMONSTRAÇÃO BÁSICA ===
    println!("\n📚 === DEMONSTRAÇÃO BÁSICA ===");
    
    let vocab_size = 1000;
    let embedding_dim = 128;
    let max_length = 50;
    
    println!("\n⚙️  Configurações:");
    println!("   - Tamanho do vocabulário: {}", vocab_size);
    println!("   - Dimensão dos embeddings: {}", embedding_dim);
    println!("   - Comprimento máximo: {}", max_length);
    
    // Criar camada de embedding
    let embedding_layer = EmbeddingLayer::new(vocab_size, embedding_dim, max_length);
    
    // Sequência de exemplo
    let token_sequence = vec![10, 25, 7, 42, 3];
    println!("\n🔢 Sequência de tokens: {:?}", token_sequence);
    
    // Processar através da camada de embedding
    let embeddings = embedding_layer.forward(&token_sequence);
    
    println!("\n📊 Resultado:");
    println!("   - {} embeddings gerados", embeddings.len());
    println!("   - Cada embedding tem {} dimensões", embedding_dim);
    
    // === OPERAÇÕES VETORIAIS ===
    demonstrate_vector_operations();
    
    // === ANÁLISE SEMÂNTICA ===
    let analyzer = SemanticAnalyzer::new();
    analyzer.demonstrate_semantic_relationships();
    
    // === CONCEITOS FUNDAMENTAIS ===
    println!("\n\n📚 === CONCEITOS FUNDAMENTAIS ===");
    
    println!("\n🎯 O QUE SÃO EMBEDDINGS?");
    println!("   • Representações vetoriais densas de tokens");
    println!("   • Capturam significado semântico e sintático");
    println!("   • Permitem operações matemáticas com palavras");
    println!("   • Dimensão típica: 128, 256, 512, 768, 1024");
    
    println!("\n🔤 EMBEDDINGS DE TOKENS:");
    println!("   • Cada token do vocabulário → vetor único");
    println!("   • Aprendidos durante o treinamento");
    println!("   • Palavras similares → vetores similares");
    println!("   • Matriz de embedding: [vocab_size × embedding_dim]");
    
    println!("\n📍 EMBEDDINGS POSICIONAIS:");
    println!("   • Adicionam informação sobre posição na sequência");
    println!("   • Essenciais para modelos sem recorrência");
    println!("   • Encoding sinusoidal: padrão no Transformer original");
    println!("   • Embeddings aprendidos: alternativa moderna");
    
    println!("\n🧮 OPERAÇÕES IMPORTANTES:");
    println!("   • Similaridade cosseno: mede direção dos vetores");
    println!("   • Distância euclidiana: mede magnitude da diferença");
    println!("   • Soma: combina informações (token + posição)");
    println!("   • Normalização: estabiliza magnitudes");
    
    println!("\n🎨 PROPRIEDADES INTERESSANTES:");
    println!("   • Analogias: rei - homem + mulher ≈ rainha");
    println!("   • Clustering: palavras similares se agrupam");
    println!("   • Interpolação: pontos intermediários têm significado");
    println!("   • Transferência: embeddings pré-treinados são úteis");
    
    println!("\n⚡ OTIMIZAÇÕES:");
    println!("   • Embedding sharing: compartilhar entre entrada/saída");
    println!("   • Quantização: reduzir precisão para economizar memória");
    println!("   • Pruning: remover dimensões menos importantes");
    println!("   • Factorization: decompor matriz grande em menores");
    
    println!("\n\n🎓 === EXERCÍCIOS SUGERIDOS ===");
    println!("1. Implemente embeddings aprendidos vs. sinusoidais");
    println!("2. Experimente com diferentes dimensões");
    println!("3. Visualize embeddings em 2D usando t-SNE ou PCA");
    println!("4. Implemente analogias vetoriais");
    println!("5. Compare embeddings de diferentes modelos");
    println!("6. Implemente embedding de subpalavras");
    
    println!("\n✨ Embeddings explicados com sucesso! ✨");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_operations() {
        let emb1 = Embedding::from_values(vec![1.0, 0.0, 0.0]);
        let emb2 = Embedding::from_values(vec![0.0, 1.0, 0.0]);
        
        // Vetores ortogonais devem ter similaridade cosseno 0
        assert!((emb1.cosine_similarity(&emb2) - 0.0).abs() < 1e-6);
        
        // Distância euclidiana entre vetores unitários ortogonais
        assert!((emb1.euclidean_distance(&emb2) - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_token_embedding() {
        let token_emb = TokenEmbedding::new(100, 64);
        
        assert_eq!(token_emb.vocab_size, 100);
        assert_eq!(token_emb.embedding_dim, 64);
        
        let embedding = token_emb.get_embedding(0).unwrap();
        assert_eq!(embedding.dimension, 64);
    }

    #[test]
    fn test_positional_embedding() {
        let pos_emb = PositionalEmbedding::new(50, 64);
        
        let embeddings = pos_emb.forward(5);
        assert_eq!(embeddings.len(), 5);
        
        for emb in embeddings {
            assert_eq!(emb.dimension, 64);
        }
    }

    #[test]
    fn test_embedding_layer() {
        let layer = EmbeddingLayer::new(100, 64, 50);
        let token_ids = vec![1, 5, 10, 15];
        
        let embeddings = layer.forward(&token_ids);
        assert_eq!(embeddings.len(), 4);
        
        for emb in embeddings {
            assert_eq!(emb.dimension, 64);
        }
    }
}