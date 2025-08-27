//! # 🔤 Demonstração do Sistema de Tokenização BPE
//!
//! Este exemplo demonstra como o algoritmo Byte Pair Encoding (BPE) funciona na prática,
//! desde o treinamento até a tokenização e decodificação de texto.
//!
//! ## 🎯 O que você vai aprender:
//! - Como treinar um tokenizador BPE do zero
//! - Visualização do processo de merge de pares
//! - Comparação entre diferentes estratégias de tokenização
//! - Análise de eficiência e compressão
//! - Tratamento de texto multilíngue

use std::collections::HashMap;
use std::time::Instant;
use anyhow::Result;

// Estrutura simplificada do BPE Tokenizer para demonstração
// Em um projeto real, esta viria de mini_gpt_rust::tokenizer
#[derive(Debug, Clone)]
struct BPETokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    merges: Vec<(String, String)>,
    vocab_size: usize,
}

impl BPETokenizer {
    /// 🏗️ Cria um novo tokenizador BPE
    fn new(vocab_size: usize) -> Result<Self> {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        // Adiciona tokens especiais
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("<pad>".to_string(), 1);
        vocab.insert("<eos>".to_string(), 2);
        vocab.insert("<bos>".to_string(), 3);
        
        reverse_vocab.insert(0, "<unk>".to_string());
        reverse_vocab.insert(1, "<pad>".to_string());
        reverse_vocab.insert(2, "<eos>".to_string());
        reverse_vocab.insert(3, "<bos>".to_string());
        
        // Adiciona caracteres ASCII básicos
        let mut next_id = 4;
        for i in 32..127 {
            let ch = char::from(i as u8).to_string();
            vocab.insert(ch.clone(), next_id);
            reverse_vocab.insert(next_id, ch);
            next_id += 1;
        }
        
        Ok(Self {
            vocab,
            reverse_vocab,
            merges: Vec::new(),
            vocab_size,
        })
    }
    
    /// 🎓 Treina o tokenizador com texto fornecido
    fn train(&mut self, text: &str) -> Result<()> {
        println!("🎓 Iniciando treinamento do tokenizador BPE...");
        
        // Obtém frequências de palavras
        let mut word_freqs = self.get_word_frequencies(text);
        println!("📊 Encontradas {} palavras únicas", word_freqs.len());
        
        let target_merges = self.vocab_size - self.vocab.len();
        println!("🎯 Realizando {} merges para atingir vocabulário de {}", target_merges, self.vocab_size);
        
        for merge_step in 0..target_merges {
            // Encontra o par mais frequente
            let pair_freqs = self.get_pair_frequencies(&word_freqs);
            
            if pair_freqs.is_empty() {
                println!("⚠️  Não há mais pares para merge. Parando no passo {}", merge_step);
                break;
            }
            
            let best_pair = pair_freqs.iter()
                .max_by_key(|(_, &freq)| freq)
                .map(|(pair, _)| pair.clone())
                .unwrap();
            
            let freq = pair_freqs[&best_pair];
            
            if merge_step < 10 || merge_step % 100 == 0 {
                println!("🔄 Merge {}: ('{}', '{}') - frequência: {}", 
                        merge_step + 1, best_pair.0, best_pair.1, freq);
            }
            
            // Aplica o merge
            word_freqs = self.apply_merge(&word_freqs, &best_pair);
            
            // Adiciona ao vocabulário
            let new_token = format!("{}{}", best_pair.0, best_pair.1);
            let new_id = self.vocab.len();
            self.vocab.insert(new_token.clone(), new_id);
            self.reverse_vocab.insert(new_id, new_token);
            
            // Salva o merge
            self.merges.push(best_pair);
        }
        
        println!("✅ Treinamento concluído! Vocabulário final: {} tokens", self.vocab.len());
        Ok(())
    }
    
    /// 📊 Obtém frequências de palavras no texto
    fn get_word_frequencies(&self, text: &str) -> HashMap<Vec<String>, usize> {
        let mut word_freqs = HashMap::new();
        
        for word in text.split_whitespace() {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            *word_freqs.entry(chars).or_insert(0) += 1;
        }
        
        word_freqs
    }
    
    /// 🔍 Obtém frequências de pares adjacentes
    fn get_pair_frequencies(&self, word_freqs: &HashMap<Vec<String>, usize>) -> HashMap<(String, String), usize> {
        let mut pair_freqs = HashMap::new();
        
        for (word, &freq) in word_freqs {
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i].clone(), word[i + 1].clone());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }
        
        pair_freqs
    }
    
    /// 🔄 Aplica um merge específico
    fn apply_merge(&self, word_freqs: &HashMap<Vec<String>, usize>, pair: &(String, String)) -> HashMap<Vec<String>, usize> {
        let mut new_word_freqs = HashMap::new();
        
        for (word, &freq) in word_freqs {
            let new_word = self.apply_merge_to_word(word.clone(), pair);
            *new_word_freqs.entry(new_word).or_insert(0) += freq;
        }
        
        new_word_freqs
    }
    
    /// 🔧 Aplica merge a uma palavra específica
    fn apply_merge_to_word(&self, mut word: Vec<String>, merge: &(String, String)) -> Vec<String> {
        let mut i = 0;
        while i < word.len().saturating_sub(1) {
            if word[i] == merge.0 && word[i + 1] == merge.1 {
                let merged = format!("{}{}", merge.0, merge.1);
                word[i] = merged;
                word.remove(i + 1);
            } else {
                i += 1;
            }
        }
        word
    }
    
    /// 🔢 Codifica texto em IDs de tokens
    fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let mut tokens = Vec::new();
        
        for word in text.split_whitespace() {
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            
            // Aplica todos os merges na ordem
            for merge in &self.merges {
                chars = self.apply_merge_to_word(chars, merge);
            }
            
            // Converte para IDs
            for token in chars {
                if let Some(&id) = self.vocab.get(&token) {
                    tokens.push(id);
                } else {
                    tokens.push(0); // <unk>
                }
            }
        }
        
        Ok(tokens)
    }
    
    /// 🔤 Decodifica IDs de tokens em texto
    fn decode(&self, tokens: &[usize]) -> Result<String> {
        let mut result = String::new();
        
        for &token_id in tokens {
            if let Some(token) = self.reverse_vocab.get(&token_id) {
                result.push_str(token);
            } else {
                result.push_str("<unk>");
            }
        }
        
        Ok(result)
    }
    
    /// 📏 Retorna tamanho do vocabulário
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// 🎭 Estrutura para demonstrar conceitos de tokenização
struct TokenizerDemo {
    tokenizer: BPETokenizer,
}

impl TokenizerDemo {
    /// 🏗️ Cria uma nova instância de demonstração
    fn new(vocab_size: usize) -> Result<Self> {
        let tokenizer = BPETokenizer::new(vocab_size)?;
        Ok(Self { tokenizer })
    }
    
    /// 📚 Demonstra treinamento básico com texto simples
    fn demo_basic_training(&mut self) -> Result<()> {
        println!("\n📚 === DEMONSTRAÇÃO DE TREINAMENTO BÁSICO ===");
        
        let training_text = "
            O gato subiu no telhado.
            O gato desceu do telhado.
            O cachorro correu no jardim.
            O cachorro brincou no jardim.
            A criança brincou com o gato.
            A criança correu com o cachorro.
            O sol brilhou no jardim.
            A lua brilhou no telhado.
        ";
        
        println!("📝 Texto de treinamento:");
        println!("{}", training_text.trim());
        
        println!("\n🔢 Estatísticas do texto:");
        let words: Vec<&str> = training_text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let chars: Vec<char> = training_text.chars().filter(|c| !c.is_whitespace()).collect();
        let unique_chars: std::collections::HashSet<char> = chars.iter().cloned().collect();
        
        println!("  - Total de palavras: {}", words.len());
        println!("  - Palavras únicas: {}", unique_words.len());
        println!("  - Total de caracteres: {}", chars.len());
        println!("  - Caracteres únicos: {}", unique_chars.len());
        
        // Treina o tokenizador
        let start_time = Instant::now();
        self.tokenizer.train(training_text)?;
        let training_duration = start_time.elapsed();
        
        println!("\n⏱️  Tempo de treinamento: {:?}", training_duration);
        println!("📊 Vocabulário final: {} tokens", self.tokenizer.vocab_size());
        
        Ok(())
    }
    
    /// 🔍 Demonstra processo de tokenização passo a passo
    fn demo_tokenization_process(&self) -> Result<()> {
        println!("\n🔍 === PROCESSO DE TOKENIZAÇÃO ===");
        
        let test_sentences = vec![
            "O gato subiu",
            "programação em Rust",
            "inteligência artificial",
            "tokenização avançada",
        ];
        
        for sentence in test_sentences {
            println!("\n📝 Frase: '{}'", sentence);
            
            // Tokeniza
            let start_time = Instant::now();
            let tokens = self.tokenizer.encode(sentence)?;
            let encoding_duration = start_time.elapsed();
            
            // Decodifica
            let start_time = Instant::now();
            let decoded = self.tokenizer.decode(&tokens)?;
            let decoding_duration = start_time.elapsed();
            
            println!("🔢 Tokens: {:?}", tokens);
            println!("🔤 Decodificado: '{}'", decoded);
            println!("⏱️  Codificação: {:?}, Decodificação: {:?}", 
                    encoding_duration, decoding_duration);
            
            // Analisa compressão
            let original_chars = sentence.len();
            let token_count = tokens.len();
            let compression_ratio = original_chars as f32 / token_count as f32;
            
            println!("📊 Compressão: {} chars → {} tokens (ratio: {:.2}x)", 
                    original_chars, token_count, compression_ratio);
            
            // Mostra tokens individuais
            println!("🧩 Breakdown dos tokens:");
            for (i, &token_id) in tokens.iter().enumerate() {
                if let Some(token_str) = self.tokenizer.reverse_vocab.get(&token_id) {
                    println!("  {}: {} → '{}'", i, token_id, token_str);
                }
            }
        }
        
        Ok(())
    }
    
    /// 📊 Analisa eficiência do vocabulário
    fn analyze_vocabulary_efficiency(&self) -> Result<()> {
        println!("\n📊 === ANÁLISE DE EFICIÊNCIA DO VOCABULÁRIO ===");
        
        // Analisa distribuição de comprimentos de tokens
        let mut length_distribution: HashMap<usize, usize> = HashMap::new();
        
        for token in self.tokenizer.reverse_vocab.values() {
            let length = token.chars().count();
            *length_distribution.entry(length).or_insert(0) += 1;
        }
        
        println!("\n📏 Distribuição de comprimentos de tokens:");
        let mut lengths: Vec<_> = length_distribution.keys().cloned().collect();
        lengths.sort();
        
        for length in lengths {
            let count = length_distribution[&length];
            let percentage = count as f32 / self.tokenizer.vocab_size() as f32 * 100.0;
            let bar = "█".repeat((percentage / 2.0) as usize);
            println!("  {} chars: {:3} tokens ({:5.1}%) {}", 
                    length, count, percentage, bar);
        }
        
        // Analisa tokens mais comuns por categoria
        self.analyze_token_categories()?;
        
        Ok(())
    }
    
    /// 🏷️ Analisa categorias de tokens
    fn analyze_token_categories(&self) -> Result<()> {
        println!("\n🏷️ Categorias de tokens:");
        
        let mut categories = HashMap::new();
        
        for token in self.tokenizer.reverse_vocab.values() {
            let category = if token.starts_with('<') && token.ends_with('>') {
                "Especiais"
            } else if token.len() == 1 {
                if token.chars().next().unwrap().is_alphabetic() {
                    "Letras"
                } else if token.chars().next().unwrap().is_numeric() {
                    "Números"
                } else {
                    "Símbolos"
                }
            } else if token.chars().all(|c| c.is_alphabetic()) {
                "Palavras/Subpalavras"
            } else {
                "Mistos"
            };
            
            *categories.entry(category).or_insert(0) += 1;
        }
        
        for (category, count) in categories {
            let percentage = count as f32 / self.tokenizer.vocab_size() as f32 * 100.0;
            println!("  {}: {} tokens ({:.1}%)", category, count, percentage);
        }
        
        Ok(())
    }
    
    /// 🌍 Demonstra tokenização multilíngue
    fn demo_multilingual_tokenization(&self) -> Result<()> {
        println!("\n🌍 === TOKENIZAÇÃO MULTILÍNGUE ===");
        
        let multilingual_texts = vec![
            ("Português", "Olá, como você está?"),
            ("English", "Hello, how are you?"),
            ("Español", "Hola, ¿cómo estás?"),
            ("Français", "Salut, comment allez-vous?"),
        ];
        
        println!("\n🔍 Comparando tokenização entre idiomas:");
        println!("Idioma     | Texto                    | Tokens | Compressão");
        println!("-----------|--------------------------|--------|------------");
        
        for (language, text) in multilingual_texts {
            let tokens = self.tokenizer.encode(text)?;
            let compression = text.len() as f32 / tokens.len() as f32;
            
            println!("{:10} | {:24} | {:6} | {:8.2}x", 
                    language, text, tokens.len(), compression);
        }
        
        println!("\n💡 Observações:");
        println!("  - Idiomas com caracteres especiais podem ter compressão menor");
        println!("  - BPE se adapta aos padrões mais frequentes no treinamento");
        println!("  - Vocabulário maior melhora suporte multilíngue");
        
        Ok(())
    }
    
    /// ⚡ Benchmark de performance
    fn benchmark_performance(&self) -> Result<()> {
        println!("\n⚡ === BENCHMARK DE PERFORMANCE ===");
        
        let long_text = "Esta é uma frase muito longa que será repetida várias vezes para simular um texto extenso. ".repeat(10);
        let test_texts = vec![
            ("Curto", "Olá mundo"),
            ("Médio", "Esta é uma frase de tamanho médio para testar a performance do tokenizador"),
            ("Longo", long_text.as_str()),
        ];
        
        println!("\n⏱️  Resultados do benchmark:");
        println!("Tamanho | Chars | Tokens | Encode (μs) | Decode (μs) | Throughput");
        println!("--------|-------|--------|-------------|-------------|------------");
        
        for (size_name, text) in test_texts {
            let char_count = text.len();
            
            // Benchmark encoding
            let iterations = 1000;
            let start = Instant::now();
            
            for _ in 0..iterations {
                let _ = self.tokenizer.encode(text)?;
            }
            
            let encode_duration = start.elapsed();
            let avg_encode_micros = encode_duration.as_micros() / iterations;
            
            // Benchmark decoding
            let tokens = self.tokenizer.encode(text)?;
            let start = Instant::now();
            
            for _ in 0..iterations {
                let _ = self.tokenizer.decode(&tokens)?;
            }
            
            let decode_duration = start.elapsed();
            let avg_decode_micros = decode_duration.as_micros() / iterations;
            
            let throughput = char_count as f64 / (avg_encode_micros as f64 / 1_000_000.0);
            
            println!("{:7} | {:5} | {:6} | {:11} | {:11} | {:8.0} chars/s", 
                    size_name, char_count, tokens.len(), 
                    avg_encode_micros, avg_decode_micros, throughput);
        }
        
        Ok(())
    }
}

/// 🎯 Exercícios práticos para aprofundar o entendimento
struct TokenizerExercises;

impl TokenizerExercises {
    /// 📝 Exercício 1: Comparação de estratégias
    fn exercise_tokenization_strategies() {
        println!("\n📝 === EXERCÍCIO 1: ESTRATÉGIAS DE TOKENIZAÇÃO ===");
        println!("\n🎯 Objetivo: Comparar diferentes abordagens de tokenização");
        
        println!("\n🔍 Estratégias para implementar:");
        
        println!("\n1. 🔤 Tokenização por Caracteres:");
        println!("   - Cada caractere = 1 token");
        println!("   - Vantagem: Vocabulário pequeno");
        println!("   - Desvantagem: Sequências muito longas");
        
        println!("\n2. 📝 Tokenização por Palavras:");
        println!("   - Cada palavra = 1 token");
        println!("   - Vantagem: Preserva significado");
        println!("   - Desvantagem: Vocabulário gigante");
        
        println!("\n3. 🧩 BPE (Atual):");
        println!("   - Subpalavras baseadas em frequência");
        println!("   - Vantagem: Balanceado");
        println!("   - Desvantagem: Complexidade de treinamento");
        
        println!("\n4. 🎯 SentencePiece:");
        println!("   - Trata texto como sequência de bytes");
        println!("   - Vantagem: Independente de idioma");
        println!("   - Desvantagem: Pode quebrar caracteres");
        
        println!("\n💡 Experimento sugerido:");
        println!("  1. Implemente tokenizador por caracteres");
        println!("  2. Implemente tokenizador por palavras");
        println!("  3. Compare eficiência em diferentes tipos de texto");
        println!("  4. Analise trade-offs de cada abordagem");
    }
    
    /// 🔬 Exercício 2: Otimização de vocabulário
    fn exercise_vocabulary_optimization() {
        println!("\n🔬 === EXERCÍCIO 2: OTIMIZAÇÃO DE VOCABULÁRIO ===");
        println!("\n🎯 Objetivo: Otimizar tamanho e composição do vocabulário");
        
        println!("\n⚡ Técnicas de otimização:");
        
        println!("\n1. 📊 Análise de Frequência:");
        println!("   - Identifique tokens subutilizados");
        println!("   - Remova tokens com frequência < threshold");
        println!("   - Substitua por decomposição em subtokens");
        
        println!("\n2. 🎯 Vocabulário Adaptativo:");
        println!("   - Ajuste vocabulário para domínio específico");
        println!("   - Adicione termos técnicos relevantes");
        println!("   - Remova tokens irrelevantes");
        
        println!("\n3. 🔄 Re-treinamento Incremental:");
        println!("   - Atualize vocabulário com novos dados");
        println!("   - Mantenha compatibilidade com modelos existentes");
        println!("   - Use técnicas de transfer learning");
        
        println!("\n4. 📈 Métricas de Qualidade:");
        println!("   - Taxa de compressão");
        println!("   - Cobertura de vocabulário");
        println!("   - Eficiência de encoding/decoding");
        
        println!("\n💡 Implementação sugerida:");
        println!("  1. Analise distribuição de frequências");
        println!("  2. Implemente pruning de vocabulário");
        println!("  3. Teste em diferentes domínios");
        println!("  4. Meça impacto na qualidade do modelo");
    }
    
    /// 🌐 Exercício 3: Suporte multilíngue avançado
    fn exercise_multilingual_support() {
        println!("\n🌐 === EXERCÍCIO 3: SUPORTE MULTILÍNGUE AVANÇADO ===");
        println!("\n🎯 Objetivo: Melhorar tokenização para múltiplos idiomas");
        
        println!("\n🔍 Desafios multilíngues:");
        
        println!("\n1. 📝 Scripts Diferentes:");
        println!("   - Latino, Cirílico, Árabe, CJK");
        println!("   - Direções de escrita diferentes");
        println!("   - Sistemas de pontuação variados");
        
        println!("\n2. 🔤 Morfologia Complexa:");
        println!("   - Aglutinação (Turco, Finlandês)");
        println!("   - Flexão rica (Russo, Alemão)");
        println!("   - Composição (Alemão, Holandês)");
        
        println!("\n3. 🎯 Balanceamento de Idiomas:");
        println!("   - Evitar bias para idiomas dominantes");
        println!("   - Garantir cobertura adequada");
        println!("   - Otimizar para idiomas de baixo recurso");
        
        println!("\n💡 Soluções propostas:");
        println!("  1. Pré-processamento específico por script");
        println!("  2. Vocabulário balanceado por idioma");
        println!("  3. Normalização Unicode consistente");
        println!("  4. Teste em corpora multilíngues");
    }
}

/// 🚀 Função principal que executa todas as demonstrações
fn main() -> Result<()> {
    println!("🔤 === DEMONSTRAÇÃO DO SISTEMA DE TOKENIZAÇÃO BPE ===");
    println!("Explorando como texto se torna números que modelos entendem");
    
    let mut demo = TokenizerDemo::new(500)?; // Vocabulário de 500 tokens
    
    // Executa demonstrações básicas
    demo.demo_basic_training()?;
    demo.demo_tokenization_process()?;
    demo.analyze_vocabulary_efficiency()?;
    demo.demo_multilingual_tokenization()?;
    
    // Benchmark de performance
    demo.benchmark_performance()?;
    
    // Exercícios educacionais
    println!("\n\n🎓 === EXERCÍCIOS PRÁTICOS ===");
    TokenizerExercises::exercise_tokenization_strategies();
    TokenizerExercises::exercise_vocabulary_optimization();
    TokenizerExercises::exercise_multilingual_support();
    
    println!("\n\n✅ === DEMONSTRAÇÃO CONCLUÍDA ===");
    println!("🎯 Próximos passos:");
    println!("  1. Experimente com diferentes tamanhos de vocabulário");
    println!("  2. Teste com textos de domínios específicos");
    println!("  3. Implemente os exercícios sugeridos");
    println!("  4. Compare com outros algoritmos de tokenização");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = BPETokenizer::new(1000).unwrap();
        assert!(tokenizer.vocab_size() >= 100); // Pelo menos caracteres básicos
    }
    
    #[test]
    fn test_basic_encoding_decoding() {
        let mut tokenizer = BPETokenizer::new(200).unwrap();
        let text = "hello world";
        
        // Treina com texto simples
        tokenizer.train(text).unwrap();
        
        // Testa encoding/decoding
        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();
        
        assert!(!tokens.is_empty());
        assert_eq!(decoded.replace(" ", ""), text.replace(" ", ""));
    }
    
    #[test]
    fn test_special_tokens() {
        let tokenizer = BPETokenizer::new(100).unwrap();
        
        // Verifica se tokens especiais existem
        assert!(tokenizer.vocab.contains_key("<unk>"));
        assert!(tokenizer.vocab.contains_key("<pad>"));
        assert!(tokenizer.vocab.contains_key("<eos>"));
        assert!(tokenizer.vocab.contains_key("<bos>"));
    }
    
    #[test]
    fn test_merge_application() {
        let tokenizer = BPETokenizer::new(100).unwrap();
        let word = vec!["h".to_string(), "e".to_string(), "l".to_string(), "l".to_string(), "o".to_string()];
        let merge = ("l".to_string(), "l".to_string());
        
        let result = tokenizer.apply_merge_to_word(word, &merge);
        assert_eq!(result, vec!["h", "e", "ll", "o"]);
    }
}