# 🦀 Mini-GPT-Rust

> **Um Large Language Model (LLM) completo implementado em Rust com suporte nativo à GPU Metal ARM Apple**

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![Candle](https://img.shields.io/badge/candle-ML%20Framework-blue.svg)](https://github.com/huggingface/candle)
[![Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-green.svg)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 **Visão Geral**

Mini-GPT-Rust é uma implementação completa de um modelo GPT (Generative Pre-trained Transformer) em Rust, utilizando o framework [Candle](https://github.com/huggingface/candle) da Hugging Face. O projeto demonstra como construir, treinar e usar um LLM moderno com performance de sistemas e safety garantida pelo Rust.

### ✨ **Características Principais**

- 🧠 **Arquitetura Transformer Completa**: Self-attention, feed-forward networks, layer normalization
- 🚀 **GPU Metal ARM Apple Otimizada**: Configurações específicas para máxima performance
- 📝 **Tokenização BPE**: Byte Pair Encoding para processamento eficiente de texto
- 🏋️ **Treinamento Adaptativo**: Configurações dinâmicas baseadas no hardware disponível
- 💬 **Múltiplos Modos**: Treinamento, geração de texto e chat interativo
- 🎓 **Sistema Educacional**: Logging detalhado para entender o funcionamento interno dos LLMs
- 🔍 **Análise de Tokenização**: Visualização completa do processo de tokenização
- 📊 **Métricas de Performance**: Estatísticas em tempo real de geração e processamento
- 💾 **Persistência SafeTensors**: Salvamento e carregamento seguro de modelos
- 🛡️ **Memory Safety**: Garantias de segurança de memória do Rust
- ⚡ **Zero-Cost Abstractions**: Performance de baixo nível com ergonomia de alto nível

## 🏗️ **Arquitetura do Sistema**

```
┌─────────────────────────────────────────────────────────────┐
│                     Mini-GPT-Rust                          │
├─────────────────────────────────────────────────────────────┤
│  🎯 main.rs              │ Interface principal e modos      │
│  🧠 model.rs             │ Arquitetura GPT e forward pass  │
│  🏗️ transformer.rs       │ Blocos Transformer e Feed-Forward│
│  👁️ attention.rs         │ Multi-Head Self-Attention       │
│  🏋️ training.rs          │ Loop de treinamento e otimização │
│  📝 tokenizer.rs         │ Tokenização BPE                 │
│  🎓 educational_logger.rs │ Sistema de logging educacional   │
├─────────────────────────────────────────────────────────────┤
│                    Framework Candle                        │
├─────────────────────────────────────────────────────────────┤
│  🔥 Metal GPU ARM Apple │ 🖥️ CPU Fallback │ 🌐 WebGPU        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 **Instalação e Configuração**

### **Pré-requisitos**

- **Rust 1.70+**: [Instalar Rust](https://rustup.rs/)
- **macOS com Metal**: Para aceleração GPU (opcional)
- **18GB+ RAM**: Recomendado para treinamento com GPU

### **Clonagem e Build**

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/mini-gpt-rust.git
cd mini-gpt-rust

# Build em modo release (recomendado)
cargo build --release

# Ou build em modo debug para desenvolvimento
cargo build
```

### **Dependências Principais**

```toml
[dependencies]
# Framework Candle com suporte Metal GPU
candle-core = { version = "0.8", features = ["metal"] }
candle-nn = { version = "0.8", features = ["metal"] }
candle-transformers = { version = "0.8", features = ["metal"] }
candle-optimisers = "0.8"     # Otimizadores (Adam, SGD)

# Persistência e serialização
safetensors = "0.4"           # Persistência segura de modelos
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Tokenização e processamento
tokenizers = "0.15"           # Tokenização BPE
unicode-segmentation = "1.10" # Segmentação Unicode PT-BR

# Utilitários e CLI
indicatif = "0.17"            # Barras de progresso
rand = "0.8"                  # Geração de números aleatórios
anyhow = "1.0"                # Error handling
thiserror = "1.0"             # Error handling estruturado
clap = { version = "4.0", features = ["derive"] }  # CLI
ndarray = "0.15"              # Arrays multidimensionais

# Suporte Metal para macOS
[target.'cfg(target_os = "macos")'.dependencies]
candle-metal-kernels = "0.8"  # Kernels Metal ARM Apple
```

## 🎮 **Uso do Sistema**

### **1. 🏋️ Treinamento do Modelo**

```bash
# Treinamento básico (salva automaticamente em models/mini_gpt.safetensors)
cargo run --release -- train

# Treinamento com parâmetros customizados
cargo run --release -- train --epochs 10 --data data/corpus_pt_br.txt
```

**Saída esperada:**
```
⚡ Configurações otimizadas para Metal GPU ARM Apple:
   📦 Batch Size: 32 (4x maior que CPU)
   🎯 Learning Rate: 1e-4 (otimizado para GPU)
🚀 Inicializando modelo para Metal GPU
   ⚡ Precisão: F32 otimizada para Metal
   🧠 Memória: Configurado para 18GB ARM Apple
🏋️ Iniciando treinamento...
Época 1/5: Loss médio: 7.9030, Velocidade: 1996 tokens/seg
💾 Modelo salvo com sucesso: models/mini_gpt.safetensors (4.0 MB)
```

### **2. 📝 Geração de Texto**

```bash
# Geração simples
cargo run --release -- generate --prompt "O Brasil é" --max-tokens 50

# Geração com parâmetros avançados
cargo run --release -- generate --prompt "Era uma vez" --max-tokens 100 --temperature 0.8
```

### **3. 💬 Chat Interativo**

```bash
# Modo chat básico
cargo run --release -- chat

# Chat com modo educacional (recomendado para aprendizado)
cargo run --release -- chat --educational --show-tensors

# Chat com entrada via pipe
echo "Conte-me sobre inteligência artificial" | cargo run --release -- chat
```

#### **🎓 Comandos Especiais do Chat Educacional**

Quando o modo educacional está ativo, você pode usar:

```bash
/tokens-demo "texto"  # Demonstra tokenização passo a passo
/explain              # Explica o processo completo de geração
/temp 0.8            # Ajusta temperatura de geração
/tokens 100          # Define máximo de tokens
/stats               # Mostra estatísticas do modelo
/help                # Lista todos os comandos
```

**Exemplo de saída educacional:**
```
🎓 MODO EDUCACIONAL ATIVADO
📝 Tokenização: "Olá mundo" → [156, 89, 45]
🔢 Embeddings: 512 dimensões por token
🧠 Processamento: 4 camadas Transformer
⚡ Geração: 23 tokens em 0.45s (51.1 tok/s)
```

## 💾 **Sistema de Persistência SafeTensors**

O Mini-GPT-Rust implementa um sistema robusto de persistência usando o formato SafeTensors, garantindo segurança e portabilidade dos modelos treinados.

### **Características do Sistema**

- **🔒 Formato Seguro**: SafeTensors previne ataques de deserialização
- **📦 Portabilidade**: Modelos compatíveis entre diferentes plataformas
- **⚡ Performance**: Carregamento rápido com mapeamento de memória
- **🎯 Automático**: Salvamento automático após cada época de treinamento
- **📁 Organização**: Estrutura de diretórios automática (`models/`)

### **Uso do Sistema**

```bash
# Salvamento automático durante treinamento
cargo run --release -- train  # Salva em models/mini_gpt.safetensors

# Verificar modelo salvo
ls -lh models/
# -rw-r--r--  1 user  staff   4.0M  mini_gpt.safetensors
```

### **Estrutura do Arquivo SafeTensors**

```rust
// Tensores salvos automaticamente:
- token_embedding.weight     // Embeddings de tokens
- position_embedding.weight  // Embeddings posicionais  
- transformer.layers.*.ln1   // Layer normalization 1
- transformer.layers.*.ln2   // Layer normalization 2
- transformer.layers.*.attn  // Pesos de atenção
- transformer.layers.*.mlp   // Pesos do feed-forward
- lm_head.weight            // Cabeça de linguagem
```

## 🧠 **Detalhes Técnicos**

### **Arquitetura do Modelo**

| Componente | Descrição | Implementação |
|------------|-----------|---------------|
| **Embeddings** | Token + Position embeddings | `nn::Embedding` |
| **Transformer Blocks** | N camadas empilhadas | `TransformerBlock` |
| **Self-Attention** | Multi-head attention | `MultiHeadAttention` |
| **Feed-Forward** | MLP com expansão 4x | `FeedForward` |
| **Layer Norm** | Normalização por camada | `nn::LayerNorm` |
| **Language Head** | Projeção para vocabulário | `nn::Linear` |

### **Configurações de Hardware**

#### **🔥 GPU Metal ARM Apple (Otimizado)**
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Precisão**: F32 otimizada
- **Memória**: Configurado para 18GB
- **Performance**: ~2000 tokens/seg

#### **🖥️ CPU (Fallback)**
- **Batch Size**: 8
- **Learning Rate**: 3e-4
- **Precisão**: F32 padrão
- **Performance**: ~6400 tokens/seg

### **Tokenização BPE**

O sistema utiliza Byte Pair Encoding (BPE) para tokenização eficiente:

```rust
// Tokens especiais
<PAD>  = 0  // Padding
<UNK>  = 1  // Unknown
<BOS>  = 2  // Begin of Sentence
<EOS>  = 3  // End of Sentence
```

## 📊 **Benchmarks e Performance**

### **Resultados de Treinamento**

| Hardware | Batch Size | Velocidade | Loss Final | Tempo/Época |
|----------|------------|------------|------------|-------------|
| Metal ARM Apple | 32 | 1996 tok/s | 7.9030 | 7.21s |
| CPU (Fallback) | 8 | 6413 tok/s | 7.9119 | 2.25s |

### **Uso de Memória**

- **Modelo**: ~4MB (parâmetros salvos em SafeTensors)
- **Treinamento**: ~2GB (Metal GPU)
- **Inferência**: ~500MB (Metal GPU)
- **Modelo Salvo**: Formato SafeTensors portável e seguro

## 🔧 **Desenvolvimento e Contribuição**

### **Estrutura do Código**

```
src/
├── main.rs              # 🚀 CLI e modos de operação
├── model.rs             # 🧠 Arquitetura GPT principal
├── transformer.rs       # 🏗️ Blocos Transformer
├── attention.rs         # 👁️ Mecanismo de atenção
├── training.rs          # 🏋️ Loop de treinamento + persistência
├── tokenizer.rs         # 📝 Tokenização BPE
└── educational_logger.rs # 🎓 Sistema de logging educacional

data/
└── corpus_pt_br.txt     # 📚 Dataset em português brasileiro

models/
└── mini_gpt.safetensors # 💾 Modelo treinado (SafeTensors)

docs/
└── EDUCATIONAL_LOGGING.md # 📖 Documentação do sistema educacional
```

### **Padrões de Código**

- **Error Handling**: `Result<T>` em todas as operações
- **Memory Safety**: Ownership e borrowing do Rust
- **Performance**: Zero-cost abstractions
- **Documentation**: Comentários detalhados em português

### **Testes**

```bash
# Executar todos os testes
cargo test

# Testes com output detalhado
cargo test -- --nocapture

# Benchmark de performance
cargo bench
```

## 🎯 **Roadmap e Melhorias Futuras**

### **🔥 Próximas Features**

- [x] **Persistência SafeTensors**: Salvamento seguro de modelos ✅
- [ ] **Carregamento de Modelos**: Sistema completo de checkpoint
- [ ] **Quantização**: Suporte a INT8/FP16 para eficiência
- [ ] **Distributed Training**: Treinamento distribuído
- [ ] **Instruction Tuning**: Fine-tuning para seguir instruções
- [ ] **RLHF**: Reinforcement Learning from Human Feedback
- [ ] **Multi-Modal**: Suporte a imagens e áudio

### **🚀 Otimizações Planejadas**

- [ ] **Kernel Fusion**: Otimizações de baixo nível
- [x] **Memory Mapping**: Carregamento eficiente com SafeTensors ✅
- [ ] **Streaming**: Geração de texto em tempo real
- [ ] **Caching**: Cache inteligente de atenção
- [ ] **Model Compression**: Compressão de modelos salvos

### **📚 Melhorias de Qualidade**

- [ ] **Dataset Expansion**: Mais dados de treinamento
- [ ] **Hyperparameter Tuning**: Otimização automática
- [ ] **Architecture Improvements**: RoPE, SwiGLU, etc.
- [ ] **Evaluation Metrics**: Métricas de qualidade

## 🤝 **Contribuindo**

1. **Fork** o projeto
2. **Crie** uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. **Abra** um Pull Request

### **Guidelines de Contribuição**

- Siga os padrões de código Rust (use `cargo fmt`)
- Adicione testes para novas funcionalidades
- Documente APIs públicas
- Mantenha compatibilidade com Metal GPU

## 📄 **Licença**

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 **Agradecimentos**

- **[Hugging Face Candle](https://github.com/huggingface/candle)**: Framework ML em Rust
- **[Attention is All You Need](https://arxiv.org/abs/1706.03762)**: Paper original do Transformer
- **[OpenAI GPT](https://openai.com/research/language-unsupervised)**: Arquitetura GPT
- **Comunidade Rust**: Pelo ecossistema incrível

## 📞 **Contato e Suporte**

- **Issues**: [GitHub Issues](https://github.com/seu-usuario/mini-gpt-rust/issues)
- **Discussões**: [GitHub Discussions](https://github.com/seu-usuario/mini-gpt-rust/discussions)
- **Email**: seu-email@exemplo.com

---

<div align="center">

**🦀 Feito com ❤️ em Rust | ⚡ Acelerado por Metal GPU ARM Apple**

</div>