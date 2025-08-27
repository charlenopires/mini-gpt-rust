# Mini GPT em Rust 🦀🧠

Uma implementação educacional completa de um Large Language Model (LLM) em Rust, demonstrando desde os conceitos fundamentais até técnicas avançadas de otimização e treinamento. Agora com **Sistema de Demonstrações Educacionais Interativas** e **Servidor Web de Interativos**!

## 🎯 Objetivo

Este projeto foi criado para ensinar os componentes essenciais de um LLM moderno, implementados em Rust com foco em:

- **Performance**: Zero-cost abstractions e otimizações de baixo nível
- **Segurança**: Memory safety sem garbage collection
- **Concorrência**: Paralelismo seguro e eficiente
- **Educação**: Exemplos práticos e exercícios interativos
- **Demonstrações**: Sistema completo de demos educacionais integradas
- **Interface Web**: Servidor web com interativos educacionais visuais

## 🏗️ Arquitetura do Projeto

```
mini-gpt-rust/
├── src/
│   ├── main.rs               # CLI principal com comando demo
│   ├── attention.rs          # Mecanismo de atenção multi-head
│   ├── tokenizer.rs          # Sistema de tokenização BPE
│   ├── model.rs              # Arquitetura do Mini-GPT
│   ├── transformer.rs        # Blocos Transformer
│   ├── training.rs           # Sistema de treinamento
│   ├── educational_logger.rs # Logging educacional avançado
│   ├── kernels.rs            # Otimizações de baixo nível
│   ├── benchmarks.rs         # Sistema de benchmarks
│   └── chunking.rs           # Processamento de chunks
├── examples/
│   ├── attention_demo.rs     # 🧠 Demo do mecanismo de atenção
│   ├── tokenizer_demo.rs     # 🔤 Demo do sistema de tokenização
│   ├── model_demo.rs         # 🤖 Demo da arquitetura do modelo
│   ├── transformer_demo.rs   # 🔄 Demo dos blocos Transformer
│   ├── benchmarks_demo.rs    # ⚡ Demo do sistema de benchmarks
│   ├── kernels_demo.rs       # 🚀 Demo de otimizações de kernel
│   ├── educational_logger_demo.rs # 📊 Demo do logging educacional
│   └── educational/          # Exemplos educacionais avançados
│       ├── embeddings_explained.rs
│       ├── gradient_computation.rs
│       ├── inference_engine.rs
│       ├── memory_management.rs
│       ├── optimization_techniques.rs
│       ├── tokenization_process.rs
│       ├── training_system.rs
│       └── transformer_architecture.rs
├── models/                   # Modelos treinados
└── data/                     # Dados de treinamento
```

## 🚀 Componentes Implementados

### 🎓 **Sistema de Demonstrações Educacionais** (NOVO!)
- **Comando Demo Integrado**: `cargo run -- demo` para acesso fácil
- **7 Módulos Demonstrativos**: Cada um com exemplos práticos e interativos
- **Modo Interativo**: Pausas educacionais para melhor aprendizado
- **Benchmarks Integrados**: Medição de performance em tempo real
- **Visualizações**: Mapas de atenção, tensores e métricas

### 1. **🧠 Mecanismo de Atenção** (`attention_demo.rs`)
- Self-Attention e Multi-Head Attention
- Implementação matemática detalhada
- Visualização de mapas de atenção
- Benchmarks de performance
- **Exercícios**: Análise de padrões, comparação de heads, otimizações

### 2. **🔤 Sistema de Tokenização** (`tokenizer_demo.rs`)
- Byte-Pair Encoding (BPE) completo
- WordPiece e algoritmos alternativos
- Processamento de vocabulário
- Análise de eficiência de compressão
- **Exercícios**: Comparação de algoritmos, análise de vocabulário

### 3. **🤖 Arquitetura do Modelo** (`model_demo.rs`)
- Estrutura completa do Mini-GPT
- Camadas de embedding e projeção
- Forward pass detalhado
- Análise de parâmetros e memória
- **Exercícios**: Modificação de arquitetura, análise de complexidade

### 4. **🔄 Blocos Transformer** (`transformer_demo.rs`)
- Implementação completa de blocos
- Layer normalization e conexões residuais
- Feed-forward networks
- Análise de fluxo de dados
- **Exercícios**: Comparação de normalizações, análise de gradientes

### 5. **⚡ Sistema de Benchmarks** (`benchmarks_demo.rs`)
- Medição de performance de atenção
- Benchmarks de feed-forward
- Análise de throughput e latência
- Comparação de estratégias
- **Exercícios**: Otimização de performance, profiling detalhado

### 6. **🚀 Otimizações de Kernel** (`kernels_demo.rs`)
- Kernel fusion para atenção e feed-forward
- Gerenciamento inteligente de memória
- Técnicas de otimização SIMD
- Benchmarks de speedup
- **Exercícios**: Implementação de fusões, análise de cache

### 7. **📊 Logging Educacional** (`educational_logger_demo.rs`)
- Rastreamento de operações matemáticas
- Análise de fluxo de treinamento
- Visualização de métricas
- Debugging interativo
- **Exercícios**: Análise de convergência, debugging de modelos

### 📚 **Exemplos Educacionais Avançados** (`educational/`)
- **Embeddings**: Representações vetoriais e análise semântica
- **Gradientes**: Computação automática e backpropagation
- **Inferência**: Motor de inferência com continuous batching
- **Memória**: Gerenciamento otimizado de recursos
- **Otimização**: Quantização e técnicas avançadas
- **Treinamento**: Sistema completo com otimizadores
- **Transformer**: Arquitetura detalhada com visualizações

### 🌐 **Servidor Web de Interativos** (`src/web_server.rs`)
- **Framework Axum**: Servidor web assíncrono de alta performance
- **Interface Responsiva**: Design moderno com Tailwind CSS
- **Interativos Visuais**: 6 demonstrações educacionais interativas
  - **Tokenização**: Visualização do processo de tokenização de texto
  - **Atenção**: Demonstração do mecanismo de self-attention
  - **Embeddings**: Exploração de representações vetoriais
  - **Transformer**: Arquitetura completa com visualizações
  - **Treinamento**: Processo de treinamento com métricas
  - **Inferência**: Motor de inferência em tempo real
  - **Text Chunking**: Estratégias de divisão de texto
- **Servir Arquivos**: Sistema de roteamento para arquivos estáticos
- **CLI Integrado**: Comando `web` para iniciar o servidor

## 🛠️ Como Usar

### 🌐 **Servidor Web de Interativos (NOVO!)**
```bash
# Iniciar servidor web com interativos educacionais
cargo run -- web --host 127.0.0.1 --port 8080 --dir interativos

# Acesse no navegador
# http://127.0.0.1:8080

# Opções disponíveis:
# --host: Endereço IP do servidor (padrão: 127.0.0.1)
# --port: Porta do servidor (padrão: 8080)
# --dir: Diretório dos arquivos interativos (padrão: interativos)
```

### 📱 **Interativos Disponíveis**
- **Tokenização** (`/tokenization`): Visualize como texto é convertido em tokens
- **Atenção** (`/attention`): Explore o mecanismo de self-attention
- **Embeddings** (`/embeddings`): Entenda representações vetoriais
- **Transformer** (`/transformer`): Arquitetura completa interativa
- **Treinamento** (`/training`): Processo de treinamento com métricas
- **Inferência** (`/inference`): Motor de inferência em tempo real
- **Text Chunking** (`/chunking`): Estratégias de divisão de texto

### 🎯 **Sistema de Demonstrações**
```bash
# Execute todas as demonstrações
cargo run -- demo

# Execute demonstração específica
cargo run -- demo --module attention
cargo run -- demo --module tokenizer
cargo run -- demo --module model

# Modo interativo com pausas educacionais
cargo run -- demo --interactive

# Com visualizações detalhadas
cargo run -- demo --show-tensors --show-attention-maps

# Com benchmarks de performance
cargo run -- demo --benchmarks

# Combinando opções
cargo run -- demo --module transformer --interactive --benchmarks
```

### 📋 **Opções do Comando Demo**
- `--module <MODULE>`: Executa demonstração específica (attention, tokenizer, model, transformer, benchmarks, kernels, educational_logger)
- `--educational-logs`: Ativa logs educacionais detalhados
- `--show-tensors`: Exibe visualizações de tensores
- `--show-attention-maps`: Mostra mapas de atenção
- `--interactive`: Modo interativo com pausas
- `--benchmarks`: Inclui medições de performance

### Pré-requisitos

```bash
# Instalar Rust (se não tiver)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clonar o repositório
git clone <repository-url>
cd mini-gpt-rust
```

### Executar Exemplos

```bash
# Exemplo básico da arquitetura Transformer
cargo run --example transformer_architecture

# Sistema de tokenização com exercícios
cargo run --example tokenization_process

# Embeddings e análise semântica
cargo run --example embeddings_explained

# Sistema de treinamento completo
cargo run --example training_system

# Computação de gradientes
cargo run --example gradient_computation

# Técnicas de otimização
cargo run --example optimization_techniques

# Motor de inferência
cargo run --example inference_engine

# Gerenciamento de memória
cargo run --example memory_management
```

### 🧪 **Testes e Validação**

```bash
# Execute todos os testes
cargo test

# Testes específicos por módulo
cargo test transformer
cargo test tokenizer
cargo test attention
cargo test model

# Testes com output detalhado
cargo test -- --nocapture

# Validação das demonstrações
cargo run -- demo --module attention --benchmarks
cargo run -- demo --module tokenizer --show-tensors

# Teste completo do sistema
cargo run -- demo --interactive --benchmarks
```

### 📊 **Verificação de Performance**
```bash
# Benchmarks integrados
cargo run -- demo --benchmarks

# Benchmarks específicos
cargo run -- benchmark --help

# Análise de memória
cargo run -- demo --module kernels --show-tensors
```

## 📚 Conceitos Educacionais

### 1. **Transformer Architecture**
- **Self-Attention**: Como o modelo "presta atenção" a diferentes partes da entrada
- **Multi-Head Attention**: Múltiplas "perspectivas" de atenção em paralelo
- **Positional Encoding**: Como o modelo entende a ordem das palavras
- **Layer Normalization**: Estabilização do treinamento

### 2. **Tokenização**
- **Subword Tokenization**: Balanceando vocabulário e expressividade
- **BPE (Byte-Pair Encoding)**: Algoritmo para criar subwords eficientes
- **Tokens Especiais**: Marcadores para início, fim e padding

### 3. **Embeddings**
- **Representação Vetorial**: Como palavras se tornam números
- **Espaço Semântico**: Palavras similares ficam próximas no espaço
- **Analogias Vetoriais**: "rei" - "homem" + "mulher" ≈ "rainha"

### 4. **Treinamento**
- **Gradiente Descendente**: Como o modelo "aprende" ajustando pesos
- **Backpropagation**: Propagação do erro para trás na rede
- **Otimizadores**: Algoritmos para atualização eficiente de pesos

### 5. **Otimização**
- **Quantização**: Reduzindo precisão para economizar memória
- **KV Caching**: Reutilizando cálculos de atenção
- **Batching**: Processando múltiplas sequências simultaneamente

## 🔬 Exercícios Práticos

Cada exemplo inclui exercícios interativos que demonstram:

1. **Análise de Performance**: Benchmarks e profiling
2. **Visualização**: Mapas de atenção e embeddings
3. **Comparações**: Trade-offs entre diferentes abordagens
4. **Implementação**: Extensões e melhorias
5. **Debugging**: Análise de problemas comuns

## 🎓 Progressão de Aprendizado

### Iniciante
1. `transformer_architecture.rs` - Entenda a arquitetura básica
2. `tokenization_process.rs` - Como texto vira números
3. `embeddings_explained.rs` - Representações vetoriais

### Intermediário
4. `training_system.rs` - Como modelos aprendem
5. `gradient_computation.rs` - Matemática por trás do treinamento
6. `optimization_techniques.rs` - Tornando tudo mais eficiente

### Avançado
7. `inference_engine.rs` - Sistemas de produção
8. `memory_management.rs` - Otimizações de baixo nível

## 🚀 Próximos Passos

### ✅ **Recentemente Implementado**
- ✅ Sistema completo de demonstrações educacionais
- ✅ 7 módulos demonstrativos interativos
- ✅ Comando `demo` integrado ao CLI
- ✅ Benchmarks e visualizações em tempo real
- ✅ Modo interativo para aprendizado
- ✅ **Servidor Web com Axum**: Interface web completa
- ✅ **7 Interativos Educacionais**: Tokenização, Atenção, Embeddings, Transformer, Treinamento, Inferência e Text Chunking
- ✅ **Design Responsivo**: Interface moderna com Tailwind CSS
- ✅ **CLI Web**: Comando `web` para servidor local

### 🎯 **Em Desenvolvimento**

1. **🔗 Integração Web-Demo**
   - Conectar interativos web com sistema de demos
   - Parâmetros dinâmicos em tempo real
   - Visualizações avançadas no navegador
   - Sincronização entre CLI e interface web

2. **⚡ GPU Computing & Aceleração**
   - Integração com CUDA/ROCm
   - Kernels customizados para operações específicas
   - Memory management otimizado para GPU
   - Benchmarks GPU vs CPU

3. **🤖 Modelos Pré-treinados**
   - Sistema de download e cache de modelos
   - Compatibilidade com formatos populares (GGML, SafeTensors)
   - Fine-tuning de modelos existentes
   - Demonstrações com modelos reais

4. **🔬 Otimizações Avançadas**
   - Flash Attention implementation
   - Gradient checkpointing
   - Mixed precision training
   - Quantização dinâmica

5. **📊 Ferramentas de Análise Avançada**
   - Profiler integrado com visualizações
   - Análise de bottlenecks automática
   - Comparação de estratégias de otimização
   - Métricas educacionais detalhadas

## 🤝 Contribuindo

Contribuições são bem-vindas! Áreas de interesse:

- Novos exercícios educacionais
- Otimizações de performance
- Documentação e tutoriais
- Testes e benchmarks
- Exemplos de uso real

## 📖 Recursos Adicionais

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Paper original do Transformer
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visualização excelente
- [Rust Book](https://doc.rust-lang.org/book/) - Aprendendo Rust
- [Candle](https://github.com/huggingface/candle) - Framework de ML em Rust

## 📄 Licença

Este projeto é licenciado sob a MIT License - veja o arquivo LICENSE para detalhes.

---

**Feito com ❤️ e 🦀 por um Rust Systems Architect**

*"Fearless concurrency meets fearless learning"*