# 🦀 Mini GPT Rust - Sistema Educacional Completo

> **Um Large Language Model (LLM) educacional implementado em Rust, focado em demonstrar conceitos fundamentais de IA de forma interativa e didática.**

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Educational](https://img.shields.io/badge/purpose-educational-green.svg)](README.md)
[![Lines of Code](https://img.shields.io/badge/lines_of_code-51k+-brightgreen.svg)](README.md)
[![Web Demos](https://img.shields.io/badge/web_demos-8-blue.svg)](README.md)
[![Rust Files](https://img.shields.io/badge/rust_files-35-orange.svg)](README.md)

## 🎯 **Visão Geral**

O **Mini GPT Rust** é uma implementação educacional completa de um Large Language Model, projetado especificamente para ensinar os conceitos fundamentais por trás dos modelos de linguagem modernos. Este projeto combina teoria sólida com implementação prática, oferecendo uma experiência de aprendizado única e interativa.

### 📊 **Estatísticas do Projeto**

- **51.114+ linhas de código Rust** - Implementação robusta e completa
- **35 módulos Rust** - Arquitetura modular e bem organizada
- **8 demonstrações web interativas** - Interface visual para conceitos complexos
- **15+ exemplos educacionais** - Casos de uso práticos e didáticos
- **Sistema web completo** - Servidor Axum com WebSocket e API REST

### 🚀 **Características Principais**

Este projeto foi criado para ensinar os componentes essenciais de um LLM moderno, implementados em Rust com foco em:

- **Performance e Segurança de Memória** 🔒 - Rust garante zero-cost abstractions
- **Clareza Educacional** 📚 - Código documentado e exemplos práticos
- **Implementação Prática** ⚡ - Funcionalidades reais de um LLM
- **Demonstrações Interativas** 🎮 - Interface web para visualização
- **Sistema Web Completo** 🌐 - Servidor Axum com WebSocket e API REST
- **Benchmarks Avançados** 📊 - Métricas de performance detalhadas

### 🏗️ **Arquitetura do Projeto**

```
mini-gpt-rust/
├── src/                    # Código fonte principal (35 arquivos Rust)
│   ├── main.rs            # CLI principal com comandos educacionais
│   ├── tokenizer.rs       # Implementação BPE completa
│   ├── attention.rs       # Mecanismo de self-attention
│   ├── transformer.rs     # Blocos Transformer
│   ├── model.rs          # Arquitetura completa do modelo
│   ├── training.rs       # Sistema de treinamento
│   ├── chunking.rs       # Estratégias de chunking
│   ├── benchmarks.rs     # Sistema de benchmarks
│   ├── web_server.rs     # Servidor web Axum
│   ├── web_demo_integration.rs # WebSocket e API REST
│   └── ...               # Outros módulos especializados
├── examples/               # Exemplos educacionais (15+ demos)
│   ├── educational/       # Módulos educacionais avançados
│   └── ...               # Demos específicos
├── interativos/           # Demonstrações web interativas (8 páginas)
│   ├── index.html        # Portal principal
│   ├── attention.html    # Demo de atenção
│   ├── transformer.html  # Demo de Transformer
│   └── ...               # Outras demos
├── models/                # Modelos e checkpoints
└── data/                  # Datasets e corpus (1005 linhas)
    └── corpus_pt_br.txt  # Corpus em português brasileiro
```

## 🧩 **Componentes Implementados**

### 🎮 **Sistema de Demonstrações Educacionais**

#### **Servidor Web Completo** 🌐
- **Servidor Axum**: Backend robusto com 1880+ linhas de código
- **WebSocket Real-time**: Comunicação bidirecional para demos interativas
- **API REST**: Endpoints para controle de parâmetros dinâmicos
- **Sistema de Integração**: Sincronização entre CLI e interface web
- **Página de Índice**: Portal central com navegação intuitiva
- **Arquivos Estáticos**: CSS, JS e recursos visuais otimizados

#### **Módulos Interativos Disponíveis:**

##### 🧠 **Mecanismo de Atenção** (`attention.html`)
- Visualização interativa do mecanismo de self-attention
- Demonstração de como queries, keys e values interagem
- Animações em tempo real dos pesos de atenção
- Controles para ajustar parâmetros e observar mudanças
- Navegação integrada com botão "Voltar"

##### 🔤 **Sistema de Tokenização** (`tokenizer.html`)
- Demonstração visual do processo de tokenização BPE
- Visualização da construção do vocabulário
- Comparação entre diferentes estratégias de tokenização
- Interface para testar textos personalizados

##### 🏗️ **Arquitetura do Modelo** (`model.html`)
- Visualização da arquitetura completa do Transformer
- Fluxo de dados através das camadas
- Demonstração de forward pass
- Controles interativos para parâmetros do modelo

##### 🔄 **Blocos Transformer** (`transformer.html`)
- Demonstração detalhada de um bloco Transformer
- Visualização de Multi-Head Attention
- Feed-Forward Networks e conexões residuais
- Layer Normalization em ação
- Navegação integrada com botão "Voltar"

##### 🎓 **Sistema de Treinamento** (`training.html`)
- Visualização do processo de treinamento
- Demonstração de backpropagation
- Gráficos de loss e métricas em tempo real
- Controles para hiperparâmetros
- Navegação integrada com botão "Voltar"

##### 🔍 **Sistema de Inferência** (`inference.html`)
- Demonstração do processo de geração de texto
- Visualização step-by-step da inferência
- Controles para temperatura e top-k sampling
- Interface para prompt customizado
- Fluxo de tokenização em tempo real

##### ✂️ **Sistema de Chunking** (`chunking.html`)
- Demonstração de diferentes estratégias de chunking
- Visualização de overlap e tamanhos de chunk
- Comparação de performance entre estratégias
- Interface para testar textos longos
- Navegação integrada com botão "Voltar"

##### 📊 **Demonstração de Chunking** (`sample.html`)
- Exemplo prático de chunking em ação
- Visualização de texto sendo dividido
- Animações de processamento
- Métricas de performance em tempo real
- Scanner visual para análise de chunks

##### 🔗 **Sistema de Embeddings** (`embeddings.html`)
- Visualização de representações vetoriais
- Demonstração de similaridade semântica
- Interface para explorar espaço de embeddings
- Navegação integrada com botão "Voltar"

### 📊 **Sistema de Benchmarks Avançado** (404 linhas)
- **Métricas Temporais**: Latência, throughput, chars/tokens por segundo
- **Métricas de Qualidade**: Densidade de informação, preservação de contexto
- **Métricas de Memória**: Pico de uso, fragmentação, overhead
- **Configuração Flexível**: Múltiplas estratégias e tamanhos de texto
- **Relatórios Detalhados**: Análise estatística completa com desvio padrão
- **Testes de Stress**: Avaliação com diferentes cargas de trabalho
- **Warmup Iterations**: Medições precisas de performance

### ⚡ **Otimizações de Kernel**
- **SIMD**: Operações vetorizadas para performance
- **Paralelização**: Processamento concorrente seguro com Rayon
- **Cache-Friendly**: Estruturas otimizadas para cache L1/L2/L3
- **Memory Layout**: Organização eficiente de dados (AoS vs SoA)
- **Zero-Copy**: Minimização de alocações desnecessárias

### 📝 **Logging Educacional**
- **Logs Estruturados**: Informações detalhadas sobre operações
- **Visualização de Tensores**: Debug visual de matrizes
- **Mapas de Atenção**: Visualização de pesos de atenção
- **Métricas de Treinamento**: Acompanhamento de progresso
- **Performance Profiling**: Análise detalhada de bottlenecks

### 🎯 **Exemplos Educacionais Avançados**
- **Arquitetura Transformer**: Demonstração completa da arquitetura
- **Processo de Tokenização**: BPE step-by-step com visualização
- **Embeddings Explicados**: Representações vetoriais e similaridade
- **Sistema de Treinamento**: Backpropagation e otimização
- **Computação de Gradientes**: Cálculos matemáticos detalhados
- **Técnicas de Otimização**: Adam, SGD, learning rate scheduling
- **Engine de Inferência**: Geração de texto com sampling
- **Gerenciamento de Memória**: Otimizações de baixo nível

### 🌐 **Sistema Web Completo** (2462+ linhas)
- **Servidor Axum**: Backend robusto com 1880+ linhas
- **WebSocket Integration**: Comunicação real-time com 582+ linhas
- **API REST**: Endpoints para controle dinâmico de parâmetros
- **Sistema de Sincronização**: Estado compartilhado entre CLI e web
- **Roteamento Dinâmico**: Servindo interativos automaticamente
- **Arquivos Estáticos**: CSS, JS, imagens otimizados
- **CORS**: Configuração para desenvolvimento
- **Error Handling**: Tratamento elegante de erros
- **Performance Monitoring**: Métricas em tempo real
- **Client Management**: Gerenciamento de conexões WebSocket

## 🛠️ Como Usar

### **Início Rápido** ⚡

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/mini-gpt-rust.git
cd mini-gpt-rust

# Instalar dependências e compilar
cargo build --release

# Iniciar servidor web com todas as demos
cargo run -- web
# Acesse: http://localhost:3000

# Executar demo específico
cargo run -- demo --module transformer_architecture

# Executar benchmarks
cargo run -- chunk-bench
```

### 🌐 **Sistema Web Completo (NOVO!)**
```bash
# Modo básico - Servidor web com interativos educacionais
cargo run -- web --host 127.0.0.1 --port 8080 --dir interativos

# Modo integração - Sistema completo com WebSocket e API REST
cargo run -- web --integration --port 3001

# Acesse no navegador
# Modo básico: http://127.0.0.1:8080
# Modo integração: http://127.0.0.1:3001

# Opções disponíveis:
# --host: Endereço IP do servidor (padrão: 127.0.0.1)
# --port: Porta do servidor (padrão: 8080 básico, 3001 integração)
# --dir: Diretório dos arquivos interativos (padrão: interativos)
# --integration: Ativa modo integração com WebSocket/API REST
```

### 🚀 **Funcionalidades do Modo Integração**
```bash
# Sistema completo com todas as funcionalidades avançadas
cargo run -- web --integration --port 3001

# Funcionalidades disponíveis:
# ✅ WebSocket para comunicação em tempo real
# ✅ API REST para controle de parâmetros
# ✅ Sincronização bidirecional CLI ↔ Web
# ✅ Controles dinâmicos de parâmetros
# ✅ Sistema de presets e configurações
# ✅ Monitoramento de performance em tempo real
# ✅ Visualizações avançadas com Chart.js
# ✅ Execução de demos pela interface web
# ✅ Gerenciamento de estado centralizado
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

#### 🌐 **Sistema Web Completo** (2462+ linhas)
- ✅ **Servidor Axum Robusto**: Backend com 1880+ linhas de código
- ✅ **WebSocket Real-time**: Comunicação bidirecional com 582+ linhas
- ✅ **API REST Completa**: Endpoints para controle dinâmico de parâmetros
- ✅ **8 Interativos Educacionais**: Páginas web totalmente funcionais
- ✅ **Sistema de Navegação**: Botões "Voltar" em todas as páginas
- ✅ **Client Management**: Gerenciamento avançado de conexões WebSocket
- ✅ **Performance Monitoring**: Métricas detalhadas em tempo real

#### 📊 **Sistema de Benchmarks Avançado** (404 linhas)
- ✅ **Métricas Abrangentes**: Tempo, memória, qualidade, densidade
- ✅ **Análise Estatística**: Desvio padrão, médias, percentis
- ✅ **Testes de Stress**: Avaliação com diferentes cargas
- ✅ **Warmup Iterations**: Medições precisas de performance
- ✅ **Relatórios Detalhados**: Análise completa com recomendações

#### ✂️ **Sistema de Chunking Completo**
- ✅ **Múltiplas Estratégias**: Fixed, Semantic, Sliding Window, Overlap
- ✅ **Visualização Interativa**: Interface web com animações
- ✅ **Scanner Visual**: Análise em tempo real de chunks
- ✅ **Benchmarks Específicos**: Comparação detalhada de performance

#### 🔧 **Arquitetura Robusta**
- ✅ **51.114+ Linhas de Código**: Implementação completa e robusta
- ✅ **35 Módulos Rust**: Arquitetura modular bem organizada
- ✅ **15+ Exemplos Educacionais**: Casos de uso práticos
- ✅ **Corpus Educacional**: 1005 linhas em português brasileiro
- ✅ **Sistema de Logging**: Logs estruturados e educacionais

#### 🎮 **Demonstrações Interativas**
- ✅ **Comando `demo` Integrado**: CLI educacional completo
- ✅ **Modo Interativo**: Pausas educacionais para aprendizado
- ✅ **Visualizações Avançadas**: Tensores e mapas de atenção
- ✅ **Sincronização CLI-Web**: Integração bidirecional em tempo real

### 🎯 **Em Desenvolvimento**

#### 1. **⚡ GPU Computing & Aceleração**
   - **Metal GPU**: Integração nativa com candle-core para macOS
   - **CUDA Support**: Aceleração NVIDIA para Linux/Windows
   - **Kernels Customizados**: Operações otimizadas para Transformer
   - **Memory Management**: Otimizações para GPU memory pools
   - **Benchmarks Comparativos**: GPU vs CPU performance analysis

#### 2. **🤖 Modelos Pré-treinados & Integração**
   - **Hugging Face Hub**: Download automático de modelos
   - **SafeTensors Support**: Formato seguro para modelos
   - **GGML Compatibility**: Integração com llama.cpp ecosystem
   - **Fine-tuning Pipeline**: Sistema completo de ajuste fino
   - **Model Zoo**: Biblioteca de modelos educacionais

#### 3. **🔬 Otimizações Avançadas**
   - **Flash Attention**: Implementação memory-efficient
   - **Gradient Checkpointing**: Redução de uso de memória
   - **Mixed Precision**: FP16/BF16 training
   - **Quantização Dinâmica**: INT8/INT4 inference
   - **Kernel Fusion**: Otimizações de operações combinadas

#### 4. **📊 Ferramentas de Análise Avançada**
   - **Profiler Integrado**: Análise detalhada de performance
   - **Bottleneck Detection**: Identificação automática de gargalos
   - **Memory Profiling**: Análise de uso de memória
   - **Flamegraphs**: Visualização de call stacks
   - **Educational Metrics**: Métricas específicas para aprendizado

#### 5. **🌐 Expansões Web**
   - **WebAssembly**: Execução de modelos no navegador
   - **Progressive Web App**: Experiência mobile otimizada
   - **Real-time Collaboration**: Múltiplos usuários simultâneos
   - **Cloud Integration**: Deploy em serviços cloud
   - **API Gateway**: Endpoints RESTful para integração

## 🛠️ **Stack Tecnológico**

### **Core Technologies**
- **🦀 Rust 2021**: Linguagem principal com ownership e zero-cost abstractions
- **🔥 Candle**: Framework de ML nativo em Rust para operações de tensor
- **⚡ Tokio**: Runtime assíncrono para concorrência e I/O não-bloqueante
- **🌐 Axum**: Framework web moderno para APIs REST e WebSocket

### **Performance & Concurrency**
- **🚀 Rayon**: Paralelismo de dados para operações computacionalmente intensivas
- **🔒 DashMap**: HashMap concorrente para estado compartilhado thread-safe
- **⚛️ Crossbeam**: Primitivas de concorrência lock-free
- **📊 Criterion**: Benchmarking estatisticamente rigoroso

### **Web & Serialization**
- **📡 Serde**: Serialização/deserialização type-safe
- **🎯 Tower**: Middleware e abstrações de serviço
- **🔌 Tower-HTTP**: Middleware HTTP (CORS, logging, compression)
- **📝 Askama**: Templates HTML type-safe compilados

### **Development & Tooling**
- **🐛 Tracing**: Logging estruturado e observabilidade
- **🎨 Clap**: CLI parsing com derive macros
- **⚙️ Config**: Gerenciamento de configuração hierárquica
- **🧪 Proptest**: Property-based testing

### **Educational Features**
- **📚 Custom Tokenizer**: Implementação BPE educacional
- **🧠 Transformer Architecture**: Implementação completa from scratch
- **📈 Real-time Visualization**: Gráficos interativos de atenção e embeddings
- **🔍 Performance Profiling**: Métricas detalhadas de CPU, memória e cache

## ⚡ **Requisitos & Performance**

### **Requisitos Mínimos**
- **OS**: macOS 10.15+, Linux (Ubuntu 20.04+), Windows 10+
- **RAM**: 4GB (8GB recomendado para modelos maiores)
- **CPU**: Qualquer arquitetura x86_64 ou ARM64
- **Rust**: 1.70+ (MSRV - Minimum Supported Rust Version)

### **Performance Benchmarks**
```bash
# Tokenização BPE (10k tokens)
CPU (M1 Pro):     ~2.3ms
CPU (Intel i7):   ~4.1ms

# Inferência Transformer (seq_len=512)
CPU (M1 Pro):     ~45ms
CPU (Intel i7):   ~78ms

# Chunking Strategies (1MB texto)
Semantic:          ~12ms
Fixed-size:        ~3ms
Sentence-based:    ~8ms
```

### **Otimizações Implementadas**
- **🚀 SIMD**: Operações vetorizadas para cálculos de embeddings
- **🧠 Cache-friendly**: Layouts de memória otimizados (AoS vs SoA)
- **⚡ Zero-copy**: Minimização de alocações desnecessárias
- **🔄 Parallel**: Processamento paralelo com Rayon
- **📊 Memory pools**: Reutilização de buffers para reduzir GC pressure

## 🤝 Contribuindo

Contribuições são bem-vindas! Áreas de interesse:

- Novos exercícios educacionais
- Otimizações de performance
- Documentação e tutoriais
- Testes e benchmarks
- Exemplos de uso real

### **Áreas de Contribuição**
- 🐛 **Bug fixes**: Correções e melhorias de estabilidade
- ⚡ **Performance**: Otimizações de algoritmos e estruturas de dados
- 📚 **Documentação**: Exemplos, tutoriais e explicações
- 🎨 **UI/UX**: Melhorias na interface web e visualizações
- 🧪 **Testing**: Testes unitários, de integração e property-based
- 🌐 **Internacionalização**: Suporte a múltiplos idiomas

## 📖 Recursos Adicionais

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Paper original do Transformer
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visualização excelente
- [Rust Book](https://doc.rust-lang.org/book/) - Aprendendo Rust
- [Candle](https://github.com/huggingface/candle) - Framework de ML em Rust
- [Rust Performance Book](https://nnethercote.github.io/perf-book/) - Otimizações em Rust
- [Async Rust Book](https://rust-lang.github.io/async-book/) - Programação assíncrona

## 📄 Licença

Este projeto é licenciado sob a MIT License - veja o arquivo LICENSE para detalhes.

---

**Feito com ❤️ e 🦀 por um Rust Systems Architect**

*"Fearless concurrency meets fearless learning"*