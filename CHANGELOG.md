# Changelog - Mini-GPT-Rust

## [2026-07-14] - Treinamento real, GPU centralizada e demonstração web recriada

### 🐛 Correções críticas no core

- **Otimizador real**: o laço de treinamento calculava gradientes e os descartava
  (`loss.backward()` sem nenhum otimizador aplicando a atualização) — o modelo
  nunca aprendia de fato. Agora usa Adam (`candle-optimisers`) de verdade.
- **Carregamento de checkpoint**: `load_from_checkpoint` ignorava os metadados
  reais salvos e usava uma config hardcoded (escala GPT-2), quebrando o
  carregamento de qualquer checkpoint real. Agora lê o sidecar `<checkpoint>.json`.
- **Deadlock no carregamento de pesos**: o `Mutex` do `VarMap` era travado duas
  vezes na mesma thread ao carregar tensores do SafeTensors.
- **Geração via CLI**: `cargo run -- load --prompt ...` era um stub
  ("ainda não implementada"). Agora chama `model.generate()` de verdade, com o
  tokenizer persistido junto do checkpoint (`<checkpoint>.safetensors.tokenizer.json`).
- **`list_checkpoints`**: retornava só o nome do arquivo, não o caminho completo,
  quebrando `--latest`/`--best`.

### 🚀 GPU

- Seleção de dispositivo centralizada em `src/device.rs::resolve_device()`
  (Metal → CPU), usada por toda a CLI e pelo servidor web — antes duplicada
  5x e ausente do caminho web.
- Captura o pânico conhecido do candle-core quando `Device::new_metal` não
  encontra uma sessão de janela válida ([candle#3566](https://github.com/huggingface/candle/issues/3566)),
  tratando como fallback para CPU em vez de derrubar o processo.

### 🌐 API web real

- `src/web_demo_integration.rs` e `src/demo_bridge.rs` (dados 100% simulados/
  hardcoded, rotas que nem batiam com o que o frontend chamava) removidos.
- Novo `src/api.rs`: toda rota chama o modelo/tokenizer/treinador de verdade —
  tokenização BPE real, pesos de atenção reais (com extração não-invasiva via
  `forward_with_attention`), embeddings reais + projeção PCA, geração via SSE,
  treinamento real transmitido via WebSocket (`TrainingEvent`).
- `src/web_server.rs` encolheu de ~2000 linhas (gerava HTML inline em Rust) para
  bootstrap + `ServeDir` + montagem da API.

### 🎨 Demonstração recriada

- Três gerações duplicadas de páginas (`interativos/*.html` original, o conjunto
  "advanced_*" mesclado depois, e um diretório `interativos/interactive/` nunca
  commitado) removidas.
- 5 páginas novas, cada uma ligada à API real, sem `Math.random()`: Corpus &
  Tokenização, Atenção, Embeddings, **Treinamento** (fluxo completo ao vivo:
  corpus → tokenização → batching → forward/loss/backward/update → checkpoint),
  Inferência.
- Design system compartilhado (`interativos/css/style.css`) com paleta validada
  (contraste + CVD) via a skill de dataviz.

---

## [2024-01-28] - Expansão Educacional e Corpus Ampliado

### ✨ Novos Recursos

#### Exemplos Educacionais
- **Arquitetura Transformer** (`examples/educational/transformer_architecture.rs`)
  - Implementação didática completa de um bloco Transformer
  - Atenção multi-head com scaled dot-product attention
  - Feed-forward network com ativação ReLU
  - Demonstração prática de forward pass
  - Explicações detalhadas dos conceitos fundamentais

- **Processo de Tokenização** (`examples/educational/tokenization_process.rs`)
  - Tokenização por palavras com vocabulário dinâmico
  - Implementação de Byte Pair Encoding (BPE)
  - Tratamento de palavras desconhecidas
  - Comparação entre diferentes métodos de tokenização
  - Exemplos práticos de codificação e decodificação

- **Embeddings Explicados** (`examples/educational/embeddings_explained.rs`)
  - Criação de embeddings de tokens e posicionais
  - Cálculo de similaridade semântica usando produto escalar
  - Operações vetoriais fundamentais
  - Análise de relações semânticas entre palavras
  - Visualização de propriedades dos embeddings

#### Corpus Expandido
- **Ampliação Significativa** do `data/corpus_pt_br.txt`
  - **10 novas seções temáticas** adicionadas:
    - Filosofia e Pensamento Crítico
    - Psicologia e Comportamento Humano
    - Matemática e Lógica
    - Física e Universo
    - Química e Transformações
    - Biologia e Vida
    - História Mundial e Civilizações
    - Sociologia e Sociedade
    - Antropologia e Cultura
    - Linguística e Comunicação
    - Neurociência e Mente
    - Sistemas Distribuídos e Arquitetura
    - Criptografia e Segurança
    - Arquiteturas de Software Avançadas

- **Conteúdo Técnico Avançado**:
  - Conceitos de sistemas distribuídos (CAP theorem, consensus algorithms)
  - Criptografia moderna (RSA, ECC, AES, blockchain)
  - Arquiteturas de software (hexagonal, clean architecture, microserviços)
  - Padrões avançados (event sourcing, CQRS, saga pattern)

### 🔧 Melhorias

#### Documentação
- **README.md** atualizado com seção completa sobre exemplos educacionais
- Instruções detalhadas de compilação e execução
- Tabela de conceitos abordados por exemplo
- Objetivos educacionais claramente definidos

#### Configuração do Projeto
- **Cargo.toml** atualizado com novos exemplos
- Configuração adequada para compilação dos exemplos educacionais
- Dependências organizadas e documentadas

### 🐛 Correções
- Corrigido erro de dimensão na multiplicação de matrizes do Transformer
- Resolvido problema de "borrow of moved value" no tokenizador
- Eliminados warnings de variáveis não utilizadas

### 📊 Estatísticas
- **Corpus expandido**: +325 linhas de conteúdo educacional
- **Cobertura temática**: 14 áreas do conhecimento
- **Exemplos funcionais**: 3 exemplos completos e testados
- **Conceitos abordados**: 200+ conceitos técnicos e científicos

### 🎯 Objetivos Educacionais Alcançados
1. **Compreensão da Arquitetura Transformer**: Implementação prática dos componentes fundamentais
2. **Domínio de Tokenização**: Comparação entre métodos e implementação de BPE
3. **Fundamentos de Embeddings**: Criação, manipulação e análise de representações vetoriais
4. **Base de Conhecimento Ampla**: Corpus abrangente para treinamento diversificado

### 🚀 Próximos Passos
- Implementação de exemplos de treinamento
- Otimização de performance com SIMD
- Integração com aceleração GPU (Metal/CUDA)
- Expansão para modelos multimodais

---

**Contribuidores**: Sistema de IA Rust Architect
**Data**: 28 de Janeiro de 2024
**Versão**: 0.2.0-educational