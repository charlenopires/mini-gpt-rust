//! # Exemplo Didático: Processo de Tokenização
//!
//! Este exemplo demonstra como o texto é processado em modelos de linguagem:
//! - Divisão do texto em tokens
//! - Conversão de tokens para IDs numéricos
//! - Diferentes estratégias de tokenização
//! - Encoding e decoding de sequências
//!
//! ## Como executar:
//! ```bash
//! cargo run --example tokenization_process
//! ```

use std::collections::HashMap;
use std::collections::BTreeMap;

/// Representa um token individual
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Token {
    text: String,
    id: usize,
}

impl Token {
    fn new(text: String, id: usize) -> Self {
        Self { text, id }
    }
}

/// ## 1. Tokenizador Simples por Palavras
/// 
/// A forma mais básica de tokenização: dividir por espaços e pontuação
#[derive(Debug)]
struct WordTokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    next_id: usize,
}

impl WordTokenizer {
    fn new() -> Self {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            next_id: 0,
        };
        
        // Adicionar tokens especiais
        tokenizer.add_special_tokens();
        tokenizer
    }
    
    fn add_special_tokens(&mut self) {
        let special_tokens = vec![
            "<PAD>",   // Padding
            "<UNK>",   // Unknown
            "<BOS>",   // Beginning of sequence
            "<EOS>",   // End of sequence
        ];
        
        for token in special_tokens {
            self.add_token(token.to_string());
        }
    }
    
    fn add_token(&mut self, token: String) -> usize {
        if let Some(&id) = self.vocab.get(&token) {
            return id;
        }
        
        let id = self.next_id;
        self.vocab.insert(token.clone(), id);
        self.reverse_vocab.insert(id, token);
        self.next_id += 1;
        id
    }
    
    /// Treina o tokenizador com um corpus de texto
    fn train(&mut self, texts: &[&str]) {
        println!("🎓 Treinando tokenizador com {} textos...", texts.len());
        
        for text in texts {
            let words = self.simple_tokenize(text);
            for word in words {
                self.add_token(word);
            }
        }
        
        println!("✅ Vocabulário criado com {} tokens", self.vocab.len());
    }
    
    /// Tokenização simples: divide por espaços e pontuação
    fn simple_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
            } else if ch.is_ascii_punctuation() {
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
                tokens.push(ch.to_string());
            } else {
                current_word.push(ch.to_lowercase().next().unwrap_or(ch));
            }
        }
        
        if !current_word.is_empty() {
            tokens.push(current_word);
        }
        
        tokens
    }
    
    /// Converte texto em sequência de IDs
    fn encode(&self, text: &str) -> Vec<usize> {
        println!("\n🔤 Codificando texto: \"{}\"", text);
        
        let tokens = self.simple_tokenize(text);
        println!("📝 Tokens extraídos: {:?}", tokens);
        
        let mut ids = vec![self.vocab["<BOS>"]];
        
        for token in tokens {
            let id = self.vocab.get(&token)
                .copied()
                .unwrap_or(self.vocab["<UNK>"]);
            ids.push(id);
            
            if self.vocab.contains_key(&token) {
                println!("   '{}' → ID {}", token, id);
            } else {
                println!("   '{}' → ID {} (<UNK>)", token, id);
            }
        }
        
        ids.push(self.vocab["<EOS>"]);
        println!("🔢 Sequência final: {:?}", ids);
        
        ids
    }
    
    /// Converte sequência de IDs de volta para texto
    fn decode(&self, ids: &[usize]) -> String {
        println!("\n🔢 Decodificando IDs: {:?}", ids);
        
        let tokens: Vec<String> = ids.iter()
            .filter_map(|&id| self.reverse_vocab.get(&id))
            .filter(|token| !matches!(token.as_str(), "<BOS>" | "<EOS>" | "<PAD>"))
            .cloned()
            .collect();
        
        println!("📝 Tokens recuperados: {:?}", tokens);
        
        let text = tokens.join(" ");
        println!("📄 Texto final: \"{}\"", text);
        
        text
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// ## 2. Tokenizador BPE Simplificado
/// 
/// Byte Pair Encoding: une os pares de caracteres mais frequentes
#[derive(Debug)]
struct SimpleBPETokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    merges: Vec<(String, String)>,
    next_id: usize,
}

impl SimpleBPETokenizer {
    fn new() -> Self {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            merges: Vec::new(),
            next_id: 0,
        };
        
        tokenizer.add_special_tokens();
        tokenizer
    }
    
    fn add_special_tokens(&mut self) {
        let special_tokens = vec!["<PAD>", "<UNK>", "<BOS>", "<EOS>"];
        for token in special_tokens {
            self.add_token(token.to_string());
        }
    }
    
    fn add_token(&mut self, token: String) -> usize {
        if let Some(&id) = self.vocab.get(&token) {
            return id;
        }
        
        let id = self.next_id;
        self.vocab.insert(token.clone(), id);
        self.reverse_vocab.insert(id, token);
        self.next_id += 1;
        id
    }
    
    /// Treina o BPE com um corpus
    fn train(&mut self, texts: &[&str], num_merges: usize) {
        println!("\n🧠 Treinando BPE com {} merges...", num_merges);
        
        // 1. Inicializar vocabulário com caracteres individuais
        let mut word_freqs = HashMap::new();
        
        for text in texts {
            for word in text.split_whitespace() {
                let word = word.to_lowercase();
                *word_freqs.entry(word.clone()).or_insert(0) += 1;
                
                // Adicionar caracteres individuais ao vocabulário
                for ch in word.chars() {
                    self.add_token(ch.to_string());
                }
            }
        }
        
        println!("📚 Vocabulário inicial: {} caracteres", self.vocab.len() - 4);
        
        // 2. Executar merges BPE
        for merge_step in 0..num_merges {
            let best_pair = self.find_best_pair(&word_freqs);
            
            if let Some((left, right)) = best_pair {
                let merged = format!("{}{}", left, right);
                self.add_token(merged.clone());
                self.merges.push((left.clone(), right.clone()));
                
                println!("   Merge {}: '{}' + '{}' → '{}'", 
                    merge_step + 1, left, right, merged);
                
                // Atualizar frequências das palavras
                self.update_word_freqs(&mut word_freqs, &left, &right, &merged);
            } else {
                break;
            }
        }
        
        println!("✅ BPE treinado: {} tokens no vocabulário", self.vocab.len());
    }
    
    fn find_best_pair(&self, word_freqs: &HashMap<String, usize>) -> Option<(String, String)> {
        let mut pair_freqs = HashMap::new();
        
        for (word, freq) in word_freqs {
            let chars: Vec<char> = word.chars().collect();
            for i in 0..chars.len().saturating_sub(1) {
                let pair = (chars[i].to_string(), chars[i + 1].to_string());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }
        
        pair_freqs.into_iter()
            .max_by_key(|(_, freq)| *freq)
            .map(|(pair, _)| pair)
    }
    
    fn update_word_freqs(
        &self,
        word_freqs: &mut HashMap<String, usize>,
        left: &str,
        right: &str,
        merged: &str,
    ) {
        let pattern = format!("{}{}", left, right);
        let mut new_word_freqs = HashMap::new();
        
        for (word, freq) in word_freqs.iter() {
            let new_word = word.replace(&pattern, merged);
            new_word_freqs.insert(new_word, *freq);
        }
        
        *word_freqs = new_word_freqs;
    }
    
    /// Aplica BPE para tokenizar uma palavra
    fn apply_bpe(&self, word: &str) -> Vec<String> {
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        
        for (left, right) in &self.merges {
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == *left && tokens[i + 1] == *right {
                    let merged = format!("{}{}", left, right);
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        
        tokens
    }
    
    fn encode(&self, text: &str) -> Vec<usize> {
        println!("\n🔤 Codificando com BPE: \"{}\"", text);
        
        let mut ids = vec![self.vocab["<BOS>"]];
        
        for word in text.split_whitespace() {
            let word = word.to_lowercase();
            let tokens = self.apply_bpe(&word);
            
            println!("   '{}' → {:?}", word, tokens);
            
            for token in tokens {
                let id = self.vocab.get(&token)
                    .copied()
                    .unwrap_or(self.vocab["<UNK>"]);
                ids.push(id);
            }
        }
        
        ids.push(self.vocab["<EOS>"]);
        println!("🔢 Sequência BPE: {:?}", ids);
        
        ids
    }
}

/// ## 3. Demonstração Comparativa
fn demonstrate_tokenization() {
    println!("🚀 === DEMONSTRAÇÃO DE TOKENIZAÇÃO ===");
    
    // Corpus de treinamento
    let training_texts = vec![
        "o gato subiu no telhado",
        "o cachorro desceu da escada",
        "a menina brinca no jardim",
        "o menino estuda na escola",
        "os pássaros voam no céu",
        "as flores crescem no campo",
    ];
    
    let test_text = "o menino brinca com o cachorro";
    
    println!("\n📚 Corpus de treinamento:");
    for (i, text) in training_texts.iter().enumerate() {
        println!("   {}: \"{}\"", i + 1, text);
    }
    
    println!("\n🧪 Texto de teste: \"{}\"", test_text);
    
    // === TOKENIZAÇÃO POR PALAVRAS ===
    println!("\n\n🔤 === TOKENIZAÇÃO POR PALAVRAS ===");
    
    let mut word_tokenizer = WordTokenizer::new();
    word_tokenizer.train(&training_texts);
    
    println!("\n📊 Estatísticas do vocabulário:");
    println!("   - Tamanho: {} tokens", word_tokenizer.vocab_size());
    
    let word_ids = word_tokenizer.encode(test_text);
    let decoded_text = word_tokenizer.decode(&word_ids);
    
    println!("\n🔄 Verificação de consistência:");
    println!("   Original: \"{}\"", test_text);
    println!("   Decodificado: \"{}\"", decoded_text);
    println!("   ✅ Consistente: {}", test_text.to_lowercase() == decoded_text);
    
    // === TOKENIZAÇÃO BPE ===
    println!("\n\n🧬 === TOKENIZAÇÃO BPE ===");
    
    let mut bpe_tokenizer = SimpleBPETokenizer::new();
    bpe_tokenizer.train(&training_texts, 10);
    
    println!("\n📊 Estatísticas BPE:");
    println!("   - Tamanho do vocabulário: {} tokens", bpe_tokenizer.vocab.len());
    println!("   - Número de merges: {}", bpe_tokenizer.merges.len());
    
    let bpe_ids = bpe_tokenizer.encode(test_text);
    
    // === COMPARAÇÃO ===
    println!("\n\n⚖️  === COMPARAÇÃO DE MÉTODOS ===");
    
    println!("\n📏 Eficiência de compressão:");
    println!("   - Palavras: {} tokens → {} IDs", 
        test_text.split_whitespace().count(), word_ids.len());
    println!("   - BPE: {} tokens → {} IDs", 
        test_text.split_whitespace().count(), bpe_ids.len());
    
    println!("\n💾 Tamanho do vocabulário:");
    println!("   - Palavras: {} tokens", word_tokenizer.vocab_size());
    println!("   - BPE: {} tokens", bpe_tokenizer.vocab.len());
    
    // === ANÁLISE DE TOKENS DESCONHECIDOS ===
    println!("\n\n❓ === TESTE COM PALAVRAS DESCONHECIDAS ===");
    
    let unknown_text = "o elefante gigante dança";
    println!("Texto com palavras novas: \"{}\"", unknown_text);
    
    let _unknown_word_ids = word_tokenizer.encode(unknown_text);
    let _unknown_bpe_ids = bpe_tokenizer.encode(unknown_text);
    
    println!("\n🔍 Como cada método lida com palavras desconhecidas:");
    println!("   - Palavras: usa <UNK> para tokens não vistos");
    println!("   - BPE: decompõe em sub-palavras conhecidas");
}

/// Função principal
fn main() {
    println!("📖 === GUIA COMPLETO DE TOKENIZAÇÃO ===");
    println!("\nA tokenização é o primeiro passo no processamento de texto em LLMs.");
    println!("Vamos explorar diferentes abordagens e suas características.\n");
    
    demonstrate_tokenization();
    
    println!("\n\n📚 === CONCEITOS FUNDAMENTAIS ===");
    
    println!("\n🔤 TOKENIZAÇÃO POR PALAVRAS:");
    println!("   ✅ Vantagens:");
    println!("      • Simples de implementar e entender");
    println!("      • Preserva significado semântico das palavras");
    println!("      • Boa para idiomas com separação clara de palavras");
    println!("   ❌ Desvantagens:");
    println!("      • Vocabulário muito grande");
    println!("      • Muitos tokens <UNK> para palavras raras");
    println!("      • Não lida bem com morfologia complexa");
    
    println!("\n🧬 BYTE PAIR ENCODING (BPE):");
    println!("   ✅ Vantagens:");
    println!("      • Vocabulário de tamanho controlado");
    println!("      • Lida bem com palavras raras e novas");
    println!("      • Captura padrões morfológicos");
    println!("      • Usado em modelos modernos (GPT, BERT)");
    println!("   ❌ Desvantagens:");
    println!("      • Mais complexo de implementar");
    println!("      • Pode quebrar palavras de forma não intuitiva");
    
    println!("\n🔢 TOKENS ESPECIAIS:");
    println!("   • <PAD>: Preenchimento para sequências de tamanho fixo");
    println!("   • <UNK>: Tokens desconhecidos ou raros");
    println!("   • <BOS>: Início de sequência");
    println!("   • <EOS>: Fim de sequência");
    
    println!("\n🎯 ESCOLHA DO MÉTODO:");
    println!("   • Palavras: Bom para prototipagem e domínios específicos");
    println!("   • BPE: Padrão para modelos de produção");
    println!("   • SentencePiece: Extensão do BPE para mais idiomas");
    println!("   • WordPiece: Variação usada no BERT");
    
    println!("\n\n🎓 === EXERCÍCIOS SUGERIDOS ===");
    println!("1. Implemente um tokenizador por caracteres");
    println!("2. Adicione suporte a diferentes idiomas");
    println!("3. Experimente com diferentes números de merges BPE");
    println!("4. Implemente WordPiece tokenization");
    println!("5. Adicione métricas de qualidade da tokenização");
    
    println!("\n✨ Tokenização concluída com sucesso! ✨");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_tokenizer() {
        let mut tokenizer = WordTokenizer::new();
        tokenizer.train(&["hello world", "world peace"]);
        
        let ids = tokenizer.encode("hello world");
        let decoded = tokenizer.decode(&ids);
        
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_bpe_tokenizer() {
        let mut tokenizer = SimpleBPETokenizer::new();
        tokenizer.train(&["hello", "world", "help"], 2);
        
        assert!(tokenizer.vocab.len() > 4); // Mais que tokens especiais
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = WordTokenizer::new();
        
        assert!(tokenizer.vocab.contains_key("<PAD>"));
        assert!(tokenizer.vocab.contains_key("<UNK>"));
        assert!(tokenizer.vocab.contains_key("<BOS>"));
        assert!(tokenizer.vocab.contains_key("<EOS>"));
    }
}