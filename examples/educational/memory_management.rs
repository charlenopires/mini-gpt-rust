//! # Sistema de Gerenciamento de Memória GPU/CPU
//!
//! Este exemplo demonstra um sistema avançado de gerenciamento de memória
//! para Large Language Models (LLMs) com suporte a:
//! - Alocação eficiente de memória GPU e CPU
//! - Memory pooling para reduzir fragmentação
//! - Transferência otimizada entre GPU e CPU
//! - Garbage collection inteligente
//! - Monitoramento de uso de memória
//!
//! ## Conceitos Fundamentais
//!
//! ### Memory Pooling
//! - Pré-alocação de blocos de memória
//! - Redução de overhead de alocação/dealocação
//! - Prevenção de fragmentação de memória
//!
//! ### GPU Memory Management
//! - Unified Memory para acesso transparente
//! - Pinned memory para transferências rápidas
//! - Memory mapping para grandes datasets
//!
//! ### CPU Memory Optimization
//! - NUMA-aware allocation
//! - Cache-friendly data layouts
//! - Memory prefetching

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::alloc::{GlobalAlloc, Layout};
use std::ptr::NonNull;
use uuid::Uuid;

/// Tipos de memória disponíveis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    CpuSystem,     // Memória RAM do sistema
    CpuPinned,     // Memória CPU pinned (page-locked)
    GpuDevice,     // Memória da GPU (VRAM)
    GpuUnified,    // Unified Memory (acessível por CPU e GPU)
    GpuManaged,    // Managed Memory (migração automática)
}

/// Configuração de memória
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub cpu_pool_size: usize,
    pub gpu_pool_size: usize,
    pub unified_pool_size: usize,
    pub block_sizes: Vec<usize>,
    pub enable_prefetching: bool,
    pub gc_threshold: f64,
    pub numa_policy: NumaPolicy,
}

/// Política NUMA
#[derive(Debug, Clone)]
pub enum NumaPolicy {
    Default,
    Preferred(u32),
    Interleaved,
    Bind(Vec<u32>),
}

/// Bloco de memória
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub id: Uuid,
    pub ptr: usize, // Simplified pointer representation
    pub size: usize,
    pub memory_type: MemoryType,
    pub is_allocated: bool,
    pub allocated_at: Instant,
    pub last_accessed: Instant,
    pub reference_count: u32,
    pub owner_id: Option<Uuid>,
}

/// Pool de memória
#[derive(Debug)]
pub struct MemoryPool {
    memory_type: MemoryType,
    total_size: usize,
    allocated_size: usize,
    free_blocks: VecDeque<MemoryBlock>,
    allocated_blocks: HashMap<Uuid, MemoryBlock>,
    block_sizes: Vec<usize>,
}

/// Estatísticas de memória
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub total_free: usize,
    pub fragmentation_ratio: f64,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub gc_runs: u64,
    pub peak_usage: usize,
    pub average_block_size: f64,
}

/// Gerenciador de memória principal
pub struct MemoryManager {
    pools: HashMap<MemoryType, Arc<Mutex<MemoryPool>>>,
    config: MemoryConfig,
    stats: Arc<RwLock<HashMap<MemoryType, MemoryStats>>>,
    allocator: CustomAllocator,
    gc_scheduler: GarbageCollector,
}

/// Alocador customizado
pub struct CustomAllocator {
    numa_policy: NumaPolicy,
    alignment: usize,
    enable_tracking: bool,
    allocation_map: Arc<Mutex<HashMap<usize, AllocationInfo>>>,
}

/// Informações de alocação
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub size: usize,
    pub layout: Layout,
    pub allocated_at: Instant,
    pub thread_id: std::thread::ThreadId,
}

/// Garbage Collector
pub struct GarbageCollector {
    threshold: f64,
    last_run: Instant,
    min_interval: Duration,
    stats: Arc<RwLock<GcStats>>,
}

/// Estatísticas do GC
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    pub total_runs: u64,
    pub total_freed: usize,
    pub average_duration: Duration,
    pub last_run_duration: Duration,
}

/// Handle para transferência de memória
#[derive(Debug)]
pub struct MemoryTransfer {
    pub id: Uuid,
    pub source_type: MemoryType,
    pub dest_type: MemoryType,
    pub size: usize,
    pub progress: f64,
    pub started_at: Instant,
}

/// Erros de gerenciamento de memória
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Out of memory in {memory_type:?} pool")]
    OutOfMemory { memory_type: MemoryType },
    #[error("Invalid block ID: {id}")]
    InvalidBlockId { id: Uuid },
    #[error("Memory alignment error")]
    AlignmentError,
    #[error("Transfer failed: {reason}")]
    TransferFailed { reason: String },
    #[error("NUMA policy error: {policy:?}")]
    NumaPolicyError { policy: NumaPolicy },
    #[error("GPU memory error: {details}")]
    GpuMemoryError { details: String },
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            cpu_pool_size: 2 * 1024 * 1024 * 1024, // 2GB
            gpu_pool_size: 8 * 1024 * 1024 * 1024, // 8GB
            unified_pool_size: 1024 * 1024 * 1024, // 1GB
            block_sizes: vec![4096, 65536, 1048576, 16777216], // 4KB, 64KB, 1MB, 16MB
            enable_prefetching: true,
            gc_threshold: 0.8, // 80% usage triggers GC
            numa_policy: NumaPolicy::Default,
        }
    }
}

impl MemoryManager {
    /// Cria um novo gerenciador de memória
    pub fn new(config: MemoryConfig) -> Result<Self, MemoryError> {
        let mut pools = HashMap::new();
        let mut stats = HashMap::new();
        
        // Inicializa pools para cada tipo de memória
        for &memory_type in &[MemoryType::CpuSystem, MemoryType::CpuPinned, 
                             MemoryType::GpuDevice, MemoryType::GpuUnified] {
            let pool_size = match memory_type {
                MemoryType::CpuSystem | MemoryType::CpuPinned => config.cpu_pool_size,
                MemoryType::GpuDevice => config.gpu_pool_size,
                MemoryType::GpuUnified => config.unified_pool_size,
                _ => config.cpu_pool_size,
            };
            
            let pool = MemoryPool::new(memory_type, pool_size, config.block_sizes.clone())?;
            pools.insert(memory_type, Arc::new(Mutex::new(pool)));
            stats.insert(memory_type, MemoryStats::default());
        }
        
        Ok(Self {
            pools,
            allocator: CustomAllocator::new(config.numa_policy.clone()),
            gc_scheduler: GarbageCollector::new(config.gc_threshold),
            stats: Arc::new(RwLock::new(stats)),
            config,
        })
    }
    
    /// Aloca um bloco de memória
    pub fn allocate(
        &self, 
        size: usize, 
        memory_type: MemoryType,
        owner_id: Option<Uuid>
    ) -> Result<MemoryBlock, MemoryError> {
        let pool = self.pools.get(&memory_type)
            .ok_or(MemoryError::OutOfMemory { memory_type })?;
        
        let mut pool_guard = pool.lock().unwrap();
        let block = pool_guard.allocate(size, owner_id)?;
        
        // Atualiza estatísticas
        self.update_stats(memory_type, |stats| {
            stats.allocation_count += 1;
            stats.total_allocated += size;
            if stats.total_allocated > stats.peak_usage {
                stats.peak_usage = stats.total_allocated;
            }
        });
        
        // Verifica se precisa executar GC
        if self.should_run_gc(memory_type) {
            self.run_gc(memory_type)?;
        }
        
        Ok(block)
    }
    
    /// Libera um bloco de memória
    pub fn deallocate(&self, block_id: Uuid, memory_type: MemoryType) -> Result<(), MemoryError> {
        let pool = self.pools.get(&memory_type)
            .ok_or(MemoryError::InvalidBlockId { id: block_id })?;
        
        let mut pool_guard = pool.lock().unwrap();
        let freed_size = pool_guard.deallocate(block_id)?;
        
        // Atualiza estatísticas
        self.update_stats(memory_type, |stats| {
            stats.deallocation_count += 1;
            stats.total_allocated = stats.total_allocated.saturating_sub(freed_size);
        });
        
        Ok(())
    }
    
    /// Transfere dados entre tipos de memória
    pub async fn transfer_memory(
        &self,
        source_block: &MemoryBlock,
        dest_type: MemoryType,
    ) -> Result<MemoryBlock, MemoryError> {
        let transfer_id = Uuid::new_v4();
        
        println!("Iniciando transferência {} -> {:?} ({})", 
                transfer_id, dest_type, source_block.size);
        
        // Aloca bloco de destino
        let dest_block = self.allocate(source_block.size, dest_type, source_block.owner_id)?;
        
        // Simula transferência de dados
        let transfer = MemoryTransfer {
            id: transfer_id,
            source_type: source_block.memory_type,
            dest_type,
            size: source_block.size,
            progress: 0.0,
            started_at: Instant::now(),
        };
        
        // Simula transferência progressiva
        for progress in (0..=100).step_by(10) {
            tokio::time::sleep(Duration::from_millis(10)).await;
            println!("Transferência {}: {}%", transfer_id, progress);
        }
        
        let duration = transfer.started_at.elapsed();
        let bandwidth = source_block.size as f64 / duration.as_secs_f64() / (1024.0 * 1024.0);
        
        println!("Transferência {} concluída em {:?} ({:.2} MB/s)", 
                transfer_id, duration, bandwidth);
        
        Ok(dest_block)
    }
    
    /// Executa prefetching de dados
    pub async fn prefetch_data(
        &self,
        blocks: &[Uuid],
        target_type: MemoryType,
    ) -> Result<Vec<MemoryBlock>, MemoryError> {
        if !self.config.enable_prefetching {
            return Ok(Vec::new());
        }
        
        println!("Iniciando prefetch de {} blocos para {:?}", blocks.len(), target_type);
        
        let mut prefetched_blocks = Vec::new();
        
        for &block_id in blocks {
            // Encontra o bloco original
            if let Some(source_block) = self.find_block(block_id)? {
                if source_block.memory_type != target_type {
                    let prefetched = self.transfer_memory(&source_block, target_type).await?;
                    prefetched_blocks.push(prefetched);
                }
            }
        }
        
        println!("Prefetch concluído: {} blocos transferidos", prefetched_blocks.len());
        
        Ok(prefetched_blocks)
    }
    
    /// Encontra um bloco por ID
    fn find_block(&self, block_id: Uuid) -> Result<Option<MemoryBlock>, MemoryError> {
        for pool in self.pools.values() {
            let pool_guard = pool.lock().unwrap();
            if let Some(block) = pool_guard.allocated_blocks.get(&block_id) {
                return Ok(Some(block.clone()));
            }
        }
        Ok(None)
    }
    
    /// Verifica se deve executar garbage collection
    fn should_run_gc(&self, memory_type: MemoryType) -> bool {
        if let Some(pool) = self.pools.get(&memory_type) {
            let pool_guard = pool.lock().unwrap();
            let usage_ratio = pool_guard.allocated_size as f64 / pool_guard.total_size as f64;
            usage_ratio > self.config.gc_threshold
        } else {
            false
        }
    }
    
    /// Executa garbage collection
    fn run_gc(&self, memory_type: MemoryType) -> Result<usize, MemoryError> {
        let start_time = Instant::now();
        
        let pool = self.pools.get(&memory_type)
            .ok_or(MemoryError::OutOfMemory { memory_type })?;
        
        let mut pool_guard = pool.lock().unwrap();
        let freed_bytes = pool_guard.garbage_collect()?;
        
        let duration = start_time.elapsed();
        
        // Atualiza estatísticas do GC
        self.gc_scheduler.update_stats(freed_bytes, duration);
        
        // Atualiza estatísticas de memória
        self.update_stats(memory_type, |stats| {
            stats.gc_runs += 1;
        });
        
        println!("GC executado em {:?}: {} bytes liberados", memory_type, freed_bytes);
        
        Ok(freed_bytes)
    }
    
    /// Atualiza estatísticas
    fn update_stats<F>(&self, memory_type: MemoryType, updater: F)
    where
        F: FnOnce(&mut MemoryStats),
    {
        if let Ok(mut stats_guard) = self.stats.write() {
            if let Some(stats) = stats_guard.get_mut(&memory_type) {
                updater(stats);
                
                // Calcula estatísticas derivadas
                if stats.allocation_count > 0 {
                    stats.average_block_size = stats.total_allocated as f64 / stats.allocation_count as f64;
                }
                
                stats.fragmentation_ratio = self.calculate_fragmentation(memory_type);
            }
        }
    }
    
    /// Calcula fragmentação
    fn calculate_fragmentation(&self, memory_type: MemoryType) -> f64 {
        if let Some(pool) = self.pools.get(&memory_type) {
            let pool_guard = pool.lock().unwrap();
            if pool_guard.total_size > 0 {
                let free_blocks = pool_guard.free_blocks.len();
                let total_blocks = free_blocks + pool_guard.allocated_blocks.len();
                if total_blocks > 0 {
                    return free_blocks as f64 / total_blocks as f64;
                }
            }
        }
        0.0
    }
    
    /// Obtém estatísticas de memória
    pub fn get_stats(&self) -> HashMap<MemoryType, MemoryStats> {
        self.stats.read().unwrap().clone()
    }
    
    /// Obtém estatísticas do GC
    pub fn get_gc_stats(&self) -> GcStats {
        self.gc_scheduler.get_stats()
    }
}

impl MemoryPool {
    /// Cria um novo pool de memória
    pub fn new(
        memory_type: MemoryType, 
        total_size: usize, 
        block_sizes: Vec<usize>
    ) -> Result<Self, MemoryError> {
        let mut free_blocks = VecDeque::new();
        
        // Pré-aloca blocos de diferentes tamanhos
        let mut allocated_size = 0;
        for &block_size in &block_sizes {
            let num_blocks = (total_size / 4) / block_size; // 25% para cada tamanho
            
            for _ in 0..num_blocks {
                if allocated_size + block_size <= total_size {
                    let block = MemoryBlock {
                        id: Uuid::new_v4(),
                        ptr: allocated_size, // Simplified
                        size: block_size,
                        memory_type,
                        is_allocated: false,
                        allocated_at: Instant::now(),
                        last_accessed: Instant::now(),
                        reference_count: 0,
                        owner_id: None,
                    };
                    
                    free_blocks.push_back(block);
                    allocated_size += block_size;
                }
            }
        }
        
        Ok(Self {
            memory_type,
            total_size,
            allocated_size: 0,
            free_blocks,
            allocated_blocks: HashMap::new(),
            block_sizes,
        })
    }
    
    /// Aloca um bloco
    pub fn allocate(&mut self, size: usize, owner_id: Option<Uuid>) -> Result<MemoryBlock, MemoryError> {
        // Encontra o menor bloco que atende ao tamanho
        let mut best_fit_idx = None;
        let mut best_fit_size = usize::MAX;
        
        for (idx, block) in self.free_blocks.iter().enumerate() {
            if block.size >= size && block.size < best_fit_size {
                best_fit_idx = Some(idx);
                best_fit_size = block.size;
            }
        }
        
        if let Some(idx) = best_fit_idx {
            let mut block = self.free_blocks.remove(idx).unwrap();
            block.is_allocated = true;
            block.allocated_at = Instant::now();
            block.last_accessed = Instant::now();
            block.owner_id = owner_id;
            
            self.allocated_blocks.insert(block.id, block.clone());
            self.allocated_size += block.size;
            
            Ok(block)
        } else {
            Err(MemoryError::OutOfMemory { memory_type: self.memory_type })
        }
    }
    
    /// Libera um bloco
    pub fn deallocate(&mut self, block_id: Uuid) -> Result<usize, MemoryError> {
        if let Some(mut block) = self.allocated_blocks.remove(&block_id) {
            block.is_allocated = false;
            block.owner_id = None;
            block.reference_count = 0;
            
            let size = block.size;
            self.allocated_size -= size;
            self.free_blocks.push_back(block);
            
            Ok(size)
        } else {
            Err(MemoryError::InvalidBlockId { id: block_id })
        }
    }
    
    /// Executa garbage collection
    pub fn garbage_collect(&mut self) -> Result<usize, MemoryError> {
        let mut freed_bytes = 0;
        let now = Instant::now();
        let timeout = Duration::from_secs(300); // 5 minutos
        
        // Identifica blocos não utilizados há muito tempo
        let mut blocks_to_free = Vec::new();
        
        for (id, block) in &self.allocated_blocks {
            if block.reference_count == 0 && now.duration_since(block.last_accessed) > timeout {
                blocks_to_free.push(*id);
            }
        }
        
        // Libera blocos identificados
        for block_id in blocks_to_free {
            if let Ok(size) = self.deallocate(block_id) {
                freed_bytes += size;
            }
        }
        
        // Compacta blocos livres adjacentes
        self.compact_free_blocks();
        
        Ok(freed_bytes)
    }
    
    /// Compacta blocos livres adjacentes
    fn compact_free_blocks(&mut self) {
        // Ordena blocos por endereço
        let mut blocks: Vec<_> = self.free_blocks.drain(..).collect();
        blocks.sort_by_key(|b| b.ptr);
        
        let mut compacted = VecDeque::new();
        let mut current_block: Option<MemoryBlock> = None;
        
        for block in blocks {
            match current_block {
                None => current_block = Some(block),
                Some(ref mut current) => {
                    // Verifica se os blocos são adjacentes
                    if current.ptr + current.size == block.ptr {
                        // Mescla os blocos
                        current.size += block.size;
                    } else {
                        // Adiciona o bloco atual e inicia um novo
                        compacted.push_back(current.clone());
                        *current = block;
                    }
                }
            }
        }
        
        // Adiciona o último bloco
        if let Some(block) = current_block {
            compacted.push_back(block);
        }
        
        self.free_blocks = compacted;
    }
}

impl CustomAllocator {
    pub fn new(numa_policy: NumaPolicy) -> Self {
        Self {
            numa_policy,
            alignment: 64, // Cache line alignment
            enable_tracking: true,
            allocation_map: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Aloca memória com política NUMA
    pub fn allocate_numa(&self, layout: Layout) -> Result<NonNull<u8>, MemoryError> {
        // Implementação simplificada - em produção usaria libnuma
        match self.numa_policy {
            NumaPolicy::Default => {
                // Alocação padrão do sistema
                self.allocate_aligned(layout)
            }
            NumaPolicy::Preferred(node) => {
                println!("Alocando em nó NUMA preferido: {}", node);
                self.allocate_aligned(layout)
            }
            NumaPolicy::Interleaved => {
                println!("Alocando com política interleaved");
                self.allocate_aligned(layout)
            }
            NumaPolicy::Bind(ref nodes) => {
                println!("Alocando em nós NUMA específicos: {:?}", nodes);
                self.allocate_aligned(layout)
            }
        }
    }
    
    /// Aloca memória alinhada
    fn allocate_aligned(&self, layout: Layout) -> Result<NonNull<u8>, MemoryError> {
        // Implementação simplificada
        let aligned_layout = layout.align_to(self.alignment)
            .map_err(|_| MemoryError::AlignmentError)?;
        
        // Simula alocação
        let ptr = aligned_layout.size() as *mut u8;
        
        if let Some(non_null) = NonNull::new(ptr) {
            if self.enable_tracking {
                let info = AllocationInfo {
                    size: aligned_layout.size(),
                    layout: aligned_layout,
                    allocated_at: Instant::now(),
                    thread_id: std::thread::current().id(),
                };
                
                self.allocation_map.lock().unwrap()
                    .insert(ptr as usize, info);
            }
            
            Ok(non_null)
        } else {
            Err(MemoryError::OutOfMemory { memory_type: MemoryType::CpuSystem })
        }
    }
}

impl GarbageCollector {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            last_run: Instant::now(),
            min_interval: Duration::from_secs(60),
            stats: Arc::new(RwLock::new(GcStats::default())),
        }
    }
    
    pub fn update_stats(&mut self, freed_bytes: usize, duration: Duration) {
        if let Ok(mut stats) = self.stats.write() {
            stats.total_runs += 1;
            stats.total_freed += freed_bytes;
            stats.last_run_duration = duration;
            
            // Calcula média móvel da duração
            if stats.total_runs == 1 {
                stats.average_duration = duration;
            } else {
                let alpha = 0.1; // Fator de suavização
                let new_avg = Duration::from_secs_f64(
                    alpha * duration.as_secs_f64() + 
                    (1.0 - alpha) * stats.average_duration.as_secs_f64()
                );
                stats.average_duration = new_avg;
            }
        }
        
        self.last_run = Instant::now();
    }
    
    pub fn get_stats(&self) -> GcStats {
        self.stats.read().unwrap().clone()
    }
}

/// Exemplo de uso do sistema de gerenciamento de memória
pub async fn exemplo_gerenciamento_memoria() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Exemplo: Sistema de Gerenciamento de Memória ===");
    
    // Configuração personalizada
    let config = MemoryConfig {
        cpu_pool_size: 512 * 1024 * 1024, // 512MB
        gpu_pool_size: 2 * 1024 * 1024 * 1024, // 2GB
        unified_pool_size: 256 * 1024 * 1024, // 256MB
        block_sizes: vec![4096, 65536, 1048576], // 4KB, 64KB, 1MB
        enable_prefetching: true,
        gc_threshold: 0.75,
        numa_policy: NumaPolicy::Preferred(0),
    };
    
    // Cria gerenciador
    let manager = MemoryManager::new(config)?;
    
    // Aloca blocos em diferentes tipos de memória
    println!("\n--- Alocação de Memória ---");
    
    let cpu_block = manager.allocate(1024 * 1024, MemoryType::CpuSystem, None)?;
    println!("Alocado bloco CPU: {} bytes (ID: {})", cpu_block.size, cpu_block.id);
    
    let gpu_block = manager.allocate(4 * 1024 * 1024, MemoryType::GpuDevice, None)?;
    println!("Alocado bloco GPU: {} bytes (ID: {})", gpu_block.size, gpu_block.id);
    
    let unified_block = manager.allocate(2 * 1024 * 1024, MemoryType::GpuUnified, None)?;
    println!("Alocado bloco Unified: {} bytes (ID: {})", unified_block.size, unified_block.id);
    
    // Transferência entre tipos de memória
    println!("\n--- Transferência de Memória ---");
    
    let transferred_block = manager.transfer_memory(&cpu_block, MemoryType::GpuDevice).await?;
    println!("Transferido CPU -> GPU: {} bytes", transferred_block.size);
    
    // Prefetching
    println!("\n--- Prefetching ---");
    
    let block_ids = vec![cpu_block.id, gpu_block.id];
    let prefetched = manager.prefetch_data(&block_ids, MemoryType::GpuUnified).await?;
    println!("Prefetch concluído: {} blocos", prefetched.len());
    
    // Estatísticas
    println!("\n--- Estatísticas de Memória ---");
    
    let stats = manager.get_stats();
    for (memory_type, stat) in &stats {
        println!("{:?}:", memory_type);
        println!("  Total alocado: {} bytes", stat.total_allocated);
        println!("  Alocações: {}", stat.allocation_count);
        println!("  Fragmentação: {:.2}%", stat.fragmentation_ratio * 100.0);
        println!("  Pico de uso: {} bytes", stat.peak_usage);
    }
    
    let gc_stats = manager.get_gc_stats();
    println!("\nGarbage Collector:");
    println!("  Execuções: {}", gc_stats.total_runs);
    println!("  Total liberado: {} bytes", gc_stats.total_freed);
    println!("  Duração média: {:?}", gc_stats.average_duration);
    
    // Limpeza
    manager.deallocate(cpu_block.id, MemoryType::CpuSystem)?;
    manager.deallocate(gpu_block.id, MemoryType::GpuDevice)?;
    manager.deallocate(unified_block.id, MemoryType::GpuUnified)?;
    manager.deallocate(transferred_block.id, MemoryType::GpuDevice)?;
    
    println!("\nLimpeza concluída!");
    
    Ok(())
}

/// Exemplo de otimização NUMA
pub fn exemplo_otimizacao_numa() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Exemplo: Otimização NUMA ===");
    
    // Diferentes políticas NUMA
    let policies = vec![
        ("Default", NumaPolicy::Default),
        ("Preferred Node 0", NumaPolicy::Preferred(0)),
        ("Interleaved", NumaPolicy::Interleaved),
        ("Bind to Nodes 0,1", NumaPolicy::Bind(vec![0, 1])),
    ];
    
    for (name, policy) in policies {
        println!("\nTestando política: {}", name);
        
        let allocator = CustomAllocator::new(policy);
        let layout = Layout::from_size_align(1024 * 1024, 64)?; // 1MB alinhado a 64 bytes
        
        let start = Instant::now();
        let ptr = allocator.allocate_numa(layout)?;
        let duration = start.elapsed();
        
        println!("  Alocação: {:?} ({:?})", ptr, duration);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_creation() {
        let block_sizes = vec![4096, 65536];
        let pool = MemoryPool::new(MemoryType::CpuSystem, 1024 * 1024, block_sizes).unwrap();
        
        assert_eq!(pool.memory_type, MemoryType::CpuSystem);
        assert_eq!(pool.total_size, 1024 * 1024);
        assert!(!pool.free_blocks.is_empty());
    }
    
    #[test]
    fn test_memory_allocation() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config).unwrap();
        
        let block = manager.allocate(1024, MemoryType::CpuSystem, None).unwrap();
        assert_eq!(block.memory_type, MemoryType::CpuSystem);
        assert!(block.size >= 1024);
        assert!(block.is_allocated);
        
        // Testa liberação
        manager.deallocate(block.id, MemoryType::CpuSystem).unwrap();
    }
    
    #[tokio::test]
    async fn test_memory_transfer() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config).unwrap();
        
        let source_block = manager.allocate(1024, MemoryType::CpuSystem, None).unwrap();
        let dest_block = manager.transfer_memory(&source_block, MemoryType::GpuDevice).await.unwrap();
        
        assert_eq!(dest_block.size, source_block.size);
        assert_eq!(dest_block.memory_type, MemoryType::GpuDevice);
    }
    
    #[test]
    fn test_garbage_collection() {
        let block_sizes = vec![4096];
        let mut pool = MemoryPool::new(MemoryType::CpuSystem, 64 * 1024, block_sizes).unwrap();
        
        // Aloca alguns blocos
        let _block1 = pool.allocate(4096, None).unwrap();
        let _block2 = pool.allocate(4096, None).unwrap();
        
        let initial_allocated = pool.allocated_size;
        
        // Simula blocos antigos (modifica last_accessed)
        for block in pool.allocated_blocks.values_mut() {
            block.last_accessed = Instant::now() - Duration::from_secs(400);
            block.reference_count = 0;
        }
        
        let freed = pool.garbage_collect().unwrap();
        assert!(freed > 0);
        assert!(pool.allocated_size < initial_allocated);
    }
    
    #[test]
    fn test_custom_allocator() {
        let allocator = CustomAllocator::new(NumaPolicy::Default);
        let layout = Layout::from_size_align(1024, 64).unwrap();
        
        let ptr = allocator.allocate_numa(layout).unwrap();
        assert!(!ptr.as_ptr().is_null());
    }
}

/// Função principal para executar os exemplos
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Executa exemplos
    exemplo_gerenciamento_memoria().await?;
    exemplo_otimizacao_numa()?;
    
    Ok(())
}