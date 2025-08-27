//! Técnicas de Otimização para LLMs
//! 
//! Este exemplo demonstra técnicas avançadas de otimização para inferência
//! de Large Language Models, incluindo quantização, KV caching e batching.
//! 
//! ## Conceitos Fundamentais
//! 
//! ### Quantização
//! - Redução da precisão numérica (FP32 -> FP16 -> INT8)
//! - Diminui uso de memória e acelera computação
//! - Trade-off entre velocidade e precisão
//! 
//! ### KV Caching
//! - Cache das chaves e valores do mecanismo de atenção
//! - Evita recálculo em gerações sequenciais
//! - Fundamental para eficiência em inferência
//! 
//! ### Batching
//! - Processamento de múltiplas sequências simultaneamente
//! - Melhora utilização de recursos computacionais
//! - Continuous batching para latência otimizada

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Tipos de quantização suportados
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationType {
    /// Precisão completa (32 bits)
    FP32,
    /// Meia precisão (16 bits)
    FP16,
    /// Inteiro de 8 bits
    INT8,
    /// Inteiro de 4 bits (experimental)
    INT4,
}

/// Tensor quantizado com metadados
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Dados quantizados
    pub data: Vec<u8>,
    /// Fator de escala para dequantização
    pub scale: f32,
    /// Offset para quantização
    pub zero_point: u8,
    /// Tipo de quantização
    pub qtype: QuantizationType,
    /// Dimensões do tensor
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Quantiza um tensor FP32 para INT8
    pub fn quantize_int8(data: &[f32], shape: Vec<usize>) -> Self {
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Calcular escala e zero point
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round().clamp(0.0, 255.0) as u8;
        
        // Quantizar dados
        let quantized_data: Vec<u8> = data
            .iter()
            .map(|&x| {
                let quantized = (x / scale + zero_point as f32).round();
                quantized.clamp(0.0, 255.0) as u8
            })
            .collect();
        
        Self {
            data: quantized_data,
            scale,
            zero_point,
            qtype: QuantizationType::INT8,
            shape,
        }
    }
    
    /// Quantiza um tensor FP32 para INT4
    pub fn quantize_int4(data: &[f32], shape: Vec<usize>) -> Self {
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Calcular escala e zero point para 4 bits (0-15)
        let scale = (max_val - min_val) / 15.0;
        let zero_point = (-min_val / scale).round().clamp(0.0, 15.0) as u8;
        
        // Quantizar dados (2 valores por byte)
        let mut quantized_data = Vec::new();
        for chunk in data.chunks(2) {
            let val1 = (chunk[0] / scale + zero_point as f32).round().clamp(0.0, 15.0) as u8;
            let val2 = if chunk.len() > 1 {
                (chunk[1] / scale + zero_point as f32).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };
            
            // Empacotar 2 valores de 4 bits em 1 byte
            quantized_data.push((val1 & 0x0F) | ((val2 & 0x0F) << 4));
        }
        
        Self {
            data: quantized_data,
            scale,
            zero_point,
            qtype: QuantizationType::INT4,
            shape,
        }
    }
    
    /// Dequantiza o tensor de volta para FP32
    pub fn dequantize(&self) -> Vec<f32> {
        match self.qtype {
            QuantizationType::INT8 => {
                self.data
                    .iter()
                    .map(|&x| (x as f32 - self.zero_point as f32) * self.scale)
                    .collect()
            }
            QuantizationType::INT4 => {
                let mut result = Vec::new();
                for &byte in &self.data {
                    // Extrair primeiro valor de 4 bits
                    let val1 = (byte & 0x0F) as f32;
                    result.push((val1 - self.zero_point as f32) * self.scale);
                    
                    // Extrair segundo valor de 4 bits
                    let val2 = ((byte >> 4) & 0x0F) as f32;
                    result.push((val2 - self.zero_point as f32) * self.scale);
                }
                
                // Truncar para o tamanho original
                let original_size = self.shape.iter().product();
                result.truncate(original_size);
                result
            }
            _ => panic!("Tipo de quantização não suportado para dequantização"),
        }
    }
    
    /// Calcula o tamanho em bytes do tensor quantizado
    pub fn size_bytes(&self) -> usize {
        match self.qtype {
            QuantizationType::FP32 => self.shape.iter().product::<usize>() * 4,
            QuantizationType::FP16 => self.shape.iter().product::<usize>() * 2,
            QuantizationType::INT8 => self.data.len(),
            QuantizationType::INT4 => self.data.len(),
        }
    }
    
    /// Calcula a taxa de compressão
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.shape.iter().product::<usize>() * 4; // FP32
        let compressed_size = self.size_bytes();
        original_size as f32 / compressed_size as f32
    }
}

/// Cache de chaves e valores para mecanismo de atenção
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cache das chaves por camada
    pub keys: HashMap<usize, Vec<Vec<f32>>>,
    /// Cache dos valores por camada
    pub values: HashMap<usize, Vec<Vec<f32>>>,
    /// Tamanho máximo do cache
    pub max_length: usize,
    /// Posição atual no cache
    pub current_length: usize,
}

impl KVCache {
    pub fn new(max_length: usize) -> Self {
        Self {
            keys: HashMap::new(),
            values: HashMap::new(),
            max_length,
            current_length: 0,
        }
    }
    
    /// Adiciona chaves e valores ao cache para uma camada
    pub fn add_kv(&mut self, layer_id: usize, keys: Vec<f32>, values: Vec<f32>) {
        // Inicializar cache da camada se necessário
        self.keys.entry(layer_id).or_insert_with(Vec::new);
        self.values.entry(layer_id).or_insert_with(Vec::new);
        
        let layer_keys = self.keys.get_mut(&layer_id).unwrap();
        let layer_values = self.values.get_mut(&layer_id).unwrap();
        
        // Adicionar novos KV
        layer_keys.push(keys);
        layer_values.push(values);
        
        // Manter apenas o tamanho máximo
        if layer_keys.len() > self.max_length {
            layer_keys.remove(0);
            layer_values.remove(0);
        }
        
        self.current_length = layer_keys.len();
    }
    
    /// Obtém todas as chaves de uma camada
    pub fn get_keys(&self, layer_id: usize) -> Option<&Vec<Vec<f32>>> {
        self.keys.get(&layer_id)
    }
    
    /// Obtém todos os valores de uma camada
    pub fn get_values(&self, layer_id: usize) -> Option<&Vec<Vec<f32>>> {
        self.values.get(&layer_id)
    }
    
    /// Limpa o cache
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.current_length = 0;
    }
    
    /// Calcula o uso de memória do cache
    pub fn memory_usage_mb(&self) -> f32 {
        let mut total_elements = 0;
        
        for layer_keys in self.keys.values() {
            for key_vec in layer_keys {
                total_elements += key_vec.len();
            }
        }
        
        for layer_values in self.values.values() {
            for value_vec in layer_values {
                total_elements += value_vec.len();
            }
        }
        
        (total_elements * 4) as f32 / (1024.0 * 1024.0) // 4 bytes por f32
    }
}

/// Requisição de inferência
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: String,
    pub input_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub created_at: Instant,
}

/// Resposta de inferência
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub id: String,
    pub output_tokens: Vec<u32>,
    pub processing_time: Duration,
    pub tokens_per_second: f32,
}

/// Sistema de batching contínuo
#[derive(Debug)]
pub struct ContinuousBatcher {
    /// Fila de requisições pendentes
    pub pending_requests: Vec<InferenceRequest>,
    /// Requisições sendo processadas
    pub active_requests: Vec<InferenceRequest>,
    /// Tamanho máximo do batch
    pub max_batch_size: usize,
    /// Timeout máximo para formar um batch
    pub batch_timeout: Duration,
    /// Estatísticas de processamento
    pub stats: BatchingStats,
}

#[derive(Debug, Clone)]
pub struct BatchingStats {
    pub total_requests: usize,
    pub total_batches: usize,
    pub average_batch_size: f32,
    pub average_wait_time: Duration,
    pub throughput_rps: f32,
}

impl BatchingStats {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            total_batches: 0,
            average_batch_size: 0.0,
            average_wait_time: Duration::from_secs(0),
            throughput_rps: 0.0,
        }
    }
}

impl ContinuousBatcher {
    pub fn new(max_batch_size: usize, batch_timeout: Duration) -> Self {
        Self {
            pending_requests: Vec::new(),
            active_requests: Vec::new(),
            max_batch_size,
            batch_timeout,
            stats: BatchingStats::new(),
        }
    }
    
    /// Adiciona uma nova requisição à fila
    pub fn add_request(&mut self, request: InferenceRequest) {
        self.pending_requests.push(request);
        self.stats.total_requests += 1;
    }
    
    /// Forma um batch de requisições para processamento
    pub fn form_batch(&mut self) -> Vec<InferenceRequest> {
        let now = Instant::now();
        let mut batch = Vec::new();
        
        // Verificar se há requisições prontas para processamento
        let mut i = 0;
        while i < self.pending_requests.len() && batch.len() < self.max_batch_size {
            let request = &self.pending_requests[i];
            
            // Adicionar ao batch se:
            // 1. Batch está vazio (primeira requisição)
            // 2. Timeout foi atingido
            // 3. Batch está cheio
            let should_add = batch.is_empty() || 
                           now.duration_since(request.created_at) >= self.batch_timeout ||
                           batch.len() >= self.max_batch_size;
            
            if should_add {
                batch.push(self.pending_requests.remove(i));
            } else {
                i += 1;
            }
        }
        
        if !batch.is_empty() {
            self.stats.total_batches += 1;
            self.stats.average_batch_size = 
                (self.stats.average_batch_size * (self.stats.total_batches - 1) as f32 + batch.len() as f32) 
                / self.stats.total_batches as f32;
        }
        
        batch
    }
    
    /// Processa um batch de requisições (simulado)
    pub fn process_batch(&mut self, batch: Vec<InferenceRequest>) -> Vec<InferenceResponse> {
        let start_time = Instant::now();
        let mut responses = Vec::new();
        
        for request in batch {
            // Simular processamento
            let processing_time = Duration::from_millis(100 + request.input_tokens.len() as u64 * 10);
            std::thread::sleep(Duration::from_millis(10)); // Simular trabalho real
            
            // Gerar resposta simulada
            let output_tokens: Vec<u32> = (0..request.max_tokens)
                .map(|i| (request.input_tokens.len() + i) as u32)
                .collect();
            
            let tokens_per_second = output_tokens.len() as f32 / processing_time.as_secs_f32();
            
            responses.push(InferenceResponse {
                id: request.id,
                output_tokens,
                processing_time,
                tokens_per_second,
            });
        }
        
        let total_time = start_time.elapsed();
        self.stats.throughput_rps = responses.len() as f32 / total_time.as_secs_f32();
        
        responses
    }
}

/// Simulador de mecanismo de atenção otimizado
#[derive(Debug)]
pub struct OptimizedAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub kv_cache: KVCache,
    pub use_quantization: bool,
    pub quantization_type: QuantizationType,
}

impl OptimizedAttention {
    pub fn new(d_model: usize, num_heads: usize, max_seq_len: usize) -> Self {
        Self {
            d_model,
            num_heads,
            kv_cache: KVCache::new(max_seq_len),
            use_quantization: false,
            quantization_type: QuantizationType::FP32,
        }
    }
    
    pub fn with_quantization(mut self, qtype: QuantizationType) -> Self {
        self.use_quantization = true;
        self.quantization_type = qtype;
        self
    }
    
    /// Executa atenção com otimizações
    pub fn forward(&mut self, 
                   layer_id: usize,
                   query: &[f32], 
                   key: &[f32], 
                   value: &[f32],
                   use_cache: bool) -> Vec<f32> {
        
        let head_dim = self.d_model / self.num_heads;
        
        // Adicionar ao cache se solicitado
        if use_cache {
            self.kv_cache.add_kv(layer_id, key.to_vec(), value.to_vec());
        }
        
        // Obter chaves e valores do cache
        let all_keys = if use_cache {
            self.kv_cache.get_keys(layer_id)
                .map(|keys| keys.iter().flatten().cloned().collect())
                .unwrap_or_else(|| key.to_vec())
        } else {
            key.to_vec()
        };
        
        let all_values = if use_cache {
            self.kv_cache.get_values(layer_id)
                .map(|values| values.iter().flatten().cloned().collect())
                .unwrap_or_else(|| value.to_vec())
        } else {
            value.to_vec()
        };
        
        // Aplicar quantização se habilitada
        let (proc_query, proc_keys, proc_values) = if self.use_quantization {
            let q_query = QuantizedTensor::quantize_int8(query, vec![query.len()]);
            let q_keys = QuantizedTensor::quantize_int8(&all_keys, vec![all_keys.len()]);
            let q_values = QuantizedTensor::quantize_int8(&all_values, vec![all_values.len()]);
            
            (q_query.dequantize(), q_keys.dequantize(), q_values.dequantize())
        } else {
            (query.to_vec(), all_keys, all_values)
        };
        
        // Calcular atenção simplificada (produto escalar)
        let seq_len = proc_keys.len() / head_dim;
        let mut attention_output = vec![0.0; self.d_model];
        
        for head in 0..self.num_heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;
            
            // Calcular scores de atenção
            let mut attention_scores = Vec::new();
            for pos in 0..seq_len {
                let key_start = pos * head_dim;
                let key_end = key_start + head_dim;
                
                if key_end <= proc_keys.len() {
                    let score: f32 = proc_query[head_start..head_end]
                        .iter()
                        .zip(proc_keys[key_start..key_end].iter())
                        .map(|(&q, &k)| q * k)
                        .sum();
                    
                    attention_scores.push(score / (head_dim as f32).sqrt());
                }
            }
            
            // Aplicar softmax
            let max_score = attention_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_scores: Vec<f32> = attention_scores
                .iter()
                .map(|&s| (s - max_score).exp())
                .collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let softmax_scores: Vec<f32> = exp_scores
                .iter()
                .map(|&s| s / sum_exp)
                .collect();
            
            // Aplicar atenção aos valores
            for (pos, &weight) in softmax_scores.iter().enumerate() {
                let value_start = pos * head_dim;
                let value_end = value_start + head_dim;
                
                if value_end <= proc_values.len() {
                    for (i, &v) in proc_values[value_start..value_end].iter().enumerate() {
                        attention_output[head_start + i] += weight * v;
                    }
                }
            }
        }
        
        attention_output
    }
}

/// Função principal demonstrando técnicas de otimização
fn main() {
    println!("⚡ Técnicas de Otimização para LLMs");
    println!("=" * 40);
    
    // Exemplo 1: Quantização de Tensores
    println!("\n📊 Exemplo 1: Quantização de Tensores");
    println!("-" * 40);
    
    // Criar tensor de exemplo
    let original_data: Vec<f32> = (0..1000)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    
    println!("Tensor original: {} elementos", original_data.len());
    println!("Tamanho FP32: {} bytes", original_data.len() * 4);
    
    // Quantização INT8
    let quantized_int8 = QuantizedTensor::quantize_int8(&original_data, vec![1000]);
    println!("\nQuantização INT8:");
    println!("  Tamanho: {} bytes", quantized_int8.size_bytes());
    println!("  Compressão: {:.2}x", quantized_int8.compression_ratio());
    println!("  Escala: {:.6}", quantized_int8.scale);
    println!("  Zero point: {}", quantized_int8.zero_point);
    
    // Quantização INT4
    let quantized_int4 = QuantizedTensor::quantize_int4(&original_data, vec![1000]);
    println!("\nQuantização INT4:");
    println!("  Tamanho: {} bytes", quantized_int4.size_bytes());
    println!("  Compressão: {:.2}x", quantized_int4.compression_ratio());
    
    // Verificar precisão
    let dequantized_int8 = quantized_int8.dequantize();
    let mse_int8: f32 = original_data
        .iter()
        .zip(dequantized_int8.iter())
        .map(|(&orig, &deq)| (orig - deq).powi(2))
        .sum::<f32>() / original_data.len() as f32;
    
    println!("\nPrecisão INT8 - MSE: {:.8}", mse_int8);
    
    // Exemplo 2: KV Caching
    println!("\n🧠 Exemplo 2: KV Caching");
    println!("-" * 30);
    
    let mut kv_cache = KVCache::new(512);
    
    // Simular adição de KV para múltiplas camadas
    for layer in 0..12 {
        for seq_pos in 0..10 {
            let keys: Vec<f32> = (0..64).map(|i| (layer * 100 + seq_pos * 10 + i) as f32).collect();
            let values: Vec<f32> = (0..64).map(|i| (layer * 100 + seq_pos * 10 + i) as f32 * 0.1).collect();
            
            kv_cache.add_kv(layer, keys, values);
        }
    }
    
    println!("Cache criado para {} camadas", kv_cache.keys.len());
    println!("Comprimento atual: {}", kv_cache.current_length);
    println!("Uso de memória: {:.2} MB", kv_cache.memory_usage_mb());
    
    // Exemplo 3: Continuous Batching
    println!("\n🚀 Exemplo 3: Continuous Batching");
    println!("-" * 35);
    
    let mut batcher = ContinuousBatcher::new(4, Duration::from_millis(50));
    
    // Adicionar requisições com diferentes tamanhos
    for i in 0..10 {
        let request = InferenceRequest {
            id: format!("req_{}", i),
            input_tokens: (0..(10 + i * 5)).map(|j| (i * 100 + j) as u32).collect(),
            max_tokens: 20,
            temperature: 0.7,
            created_at: Instant::now(),
        };
        batcher.add_request(request);
        
        // Simular chegada de requisições ao longo do tempo
        if i % 3 == 0 {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    
    println!("Requisições adicionadas: {}", batcher.stats.total_requests);
    
    // Processar batches
    let mut total_responses = 0;
    while !batcher.pending_requests.is_empty() {
        let batch = batcher.form_batch();
        if !batch.is_empty() {
            println!("\nProcessando batch de {} requisições", batch.len());
            let responses = batcher.process_batch(batch);
            total_responses += responses.len();
            
            for response in &responses {
                println!("  {} -> {} tokens em {:.2}ms ({:.1} tok/s)",
                        response.id,
                        response.output_tokens.len(),
                        response.processing_time.as_millis(),
                        response.tokens_per_second);
            }
        }
    }
    
    println!("\nEstatísticas de Batching:");
    println!("  Total de requisições: {}", batcher.stats.total_requests);
    println!("  Total de batches: {}", batcher.stats.total_batches);
    println!("  Tamanho médio do batch: {:.2}", batcher.stats.average_batch_size);
    println!("  Throughput: {:.2} req/s", batcher.stats.throughput_rps);
    
    // Exemplo 4: Atenção Otimizada
    println!("\n🎯 Exemplo 4: Atenção Otimizada");
    println!("-" * 35);
    
    let mut attention = OptimizedAttention::new(512, 8, 128)
        .with_quantization(QuantizationType::INT8);
    
    // Simular sequência de tokens
    let seq_len = 10;
    let d_model = 512;
    
    println!("Configuração:");
    println!("  d_model: {}", d_model);
    println!("  num_heads: {}", attention.num_heads);
    println!("  Quantização: {:?}", attention.quantization_type);
    
    let mut total_time = Duration::from_secs(0);
    
    for pos in 0..seq_len {
        let query: Vec<f32> = (0..d_model).map(|i| (pos * 100 + i) as f32 * 0.01).collect();
        let key: Vec<f32> = (0..d_model).map(|i| (pos * 100 + i) as f32 * 0.01).collect();
        let value: Vec<f32> = (0..d_model).map(|i| (pos * 100 + i) as f32 * 0.01).collect();
        
        let start = Instant::now();
        let output = attention.forward(0, &query, &key, &value, true);
        let elapsed = start.elapsed();
        total_time += elapsed;
        
        if pos < 3 {
            println!("  Posição {}: {:.2}ms, output_sum: {:.3}", 
                    pos, elapsed.as_millis(), output.iter().sum::<f32>());
        }
    }
    
    println!("\nPerformance:");
    println!("  Tempo total: {:.2}ms", total_time.as_millis());
    println!("  Tempo médio por token: {:.2}ms", total_time.as_millis() as f32 / seq_len as f32);
    println!("  Cache de memória: {:.2} MB", attention.kv_cache.memory_usage_mb());
    
    // Exemplo 5: Comparação de Performance
    println!("\n📈 Exemplo 5: Comparação de Performance");
    println!("-" * 45);
    
    let test_data: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.001).sin()).collect();
    
    // Benchmark quantização
    let start = Instant::now();
    let _q_int8 = QuantizedTensor::quantize_int8(&test_data, vec![10000]);
    let quantize_time = start.elapsed();
    
    let start = Instant::now();
    let _dequantized = _q_int8.dequantize();
    let dequantize_time = start.elapsed();
    
    println!("Benchmark Quantização (10k elementos):");
    println!("  Quantização INT8: {:.2}ms", quantize_time.as_millis());
    println!("  Dequantização: {:.2}ms", dequantize_time.as_millis());
    println!("  Economia de memória: {:.1}%", (1.0 - 1.0/_q_int8.compression_ratio()) * 100.0);
    
    println!("\n🎯 Conceitos Fundamentais Demonstrados:");
    println!("=" * 45);
    println!("✅ Quantização INT8 e INT4");
    println!("✅ KV Caching para atenção");
    println!("✅ Continuous Batching");
    println!("✅ Otimização de memória");
    println!("✅ Compressão de modelos");
    println!("✅ Throughput e latência");
    println!("✅ Trade-offs precisão vs velocidade");
    
    println!("\n🚀 Aplicações em LLMs:");
    println!("=" * 25);
    println!("• Inferência em tempo real");
    println!("• Deployment em edge devices");
    println!("• Serving de alta escala");
    println!("• Redução de custos computacionais");
    println!("• Otimização de latência");
    
    println!("\n💡 Exercícios Sugeridos:");
    println!("=" * 25);
    println!("1. Implementar quantização dinâmica");
    println!("2. Adicionar Flash Attention");
    println!("3. Criar sistema de prefill/decode");
    println!("4. Implementar speculative decoding");
    println!("5. Adicionar model sharding");
    println!("6. Criar pipeline parallelism");
    println!("7. Implementar tensor parallelism");
    println!("8. Adicionar memory mapping");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantization_int8() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantized = QuantizedTensor::quantize_int8(&data, vec![5]);
        let dequantized = quantized.dequantize();
        
        // Verificar que a dequantização está próxima do original
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.1, "Erro de quantização muito alto");
        }
    }
    
    #[test]
    fn test_kv_cache() {
        let mut cache = KVCache::new(10);
        
        let keys = vec![1.0, 2.0, 3.0];
        let values = vec![4.0, 5.0, 6.0];
        
        cache.add_kv(0, keys.clone(), values.clone());
        
        let cached_keys = cache.get_keys(0).unwrap();
        let cached_values = cache.get_values(0).unwrap();
        
        assert_eq!(cached_keys.len(), 1);
        assert_eq!(cached_values.len(), 1);
        assert_eq!(cached_keys[0], keys);
        assert_eq!(cached_values[0], values);
    }
    
    #[test]
    fn test_continuous_batcher() {
        let mut batcher = ContinuousBatcher::new(2, Duration::from_millis(100));
        
        let req1 = InferenceRequest {
            id: "test1".to_string(),
            input_tokens: vec![1, 2, 3],
            max_tokens: 10,
            temperature: 0.7,
            created_at: Instant::now(),
        };
        
        let req2 = InferenceRequest {
            id: "test2".to_string(),
            input_tokens: vec![4, 5, 6],
            max_tokens: 10,
            temperature: 0.7,
            created_at: Instant::now(),
        };
        
        batcher.add_request(req1);
        batcher.add_request(req2);
        
        let batch = batcher.form_batch();
        assert_eq!(batch.len(), 2);
    }
    
    #[test]
    fn test_optimized_attention() {
        let mut attention = OptimizedAttention::new(64, 4, 10);
        
        let query = vec![1.0; 64];
        let key = vec![0.5; 64];
        let value = vec![0.1; 64];
        
        let output = attention.forward(0, &query, &key, &value, true);
        
        assert_eq!(output.len(), 64);
        assert!(attention.kv_cache.current_length > 0);
    }
    
    #[test]
    fn test_compression_ratio() {
        let data = vec![1.0; 1000];
        let quantized = QuantizedTensor::quantize_int8(&data, vec![1000]);
        
        let ratio = quantized.compression_ratio();
        assert!(ratio > 3.0); // INT8 deve ter pelo menos 4x compressão vs FP32
    }
}