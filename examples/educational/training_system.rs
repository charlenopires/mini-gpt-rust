//! Sistema de Treinamento com Otimizadores
//! 
//! Este exemplo demonstra como implementar um sistema completo de treinamento
//! para redes neurais, incluindo otimizadores como SGD e Adam.
//! 
//! ## Conceitos Fundamentais
//! 
//! ### Otimizadores
//! - Algoritmos que atualizam os parâmetros do modelo baseado nos gradientes
//! - SGD: Stochastic Gradient Descent - método básico de otimização
//! - Adam: Adaptive Moment Estimation - otimizador adaptativo avançado
//! 
//! ### Loop de Treinamento
//! - Forward pass: cálculo da predição
//! - Loss calculation: cálculo da função de perda
//! - Backward pass: cálculo dos gradientes
//! - Parameter update: atualização dos parâmetros

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

/// Trait para parâmetros treináveis
pub trait Parameter {
    fn value(&self) -> Vec<f64>;
    fn gradient(&self) -> Vec<f64>;
    fn update(&mut self, delta: &[f64]);
    fn zero_grad(&mut self);
    fn name(&self) -> &str;
}

/// Implementação de um parâmetro simples
#[derive(Debug, Clone)]
pub struct SimpleParameter {
    pub name: String,
    pub data: Vec<f64>,
    pub grad: Vec<f64>,
}

impl SimpleParameter {
    pub fn new(name: String, data: Vec<f64>) -> Self {
        let grad = vec![0.0; data.len()];
        Self { name, data, grad }
    }
    
    pub fn random(name: String, size: usize, scale: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self::new(name, data)
    }
}

impl Parameter for SimpleParameter {
    fn value(&self) -> Vec<f64> {
        self.data.clone()
    }
    
    fn gradient(&self) -> Vec<f64> {
        self.grad.clone()
    }
    
    fn update(&mut self, delta: &[f64]) {
        for (i, &d) in delta.iter().enumerate() {
            if i < self.data.len() {
                self.data[i] += d;
            }
        }
    }
    
    fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Trait para otimizadores
pub trait Optimizer {
    fn step(&mut self, parameters: &mut [&mut dyn Parameter]);
    fn zero_grad(&mut self, parameters: &mut [&mut dyn Parameter]);
    fn name(&self) -> &str;
}

/// Otimizador SGD (Stochastic Gradient Descent)
#[derive(Debug)]
pub struct SGD {
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    /// Velocidades para momentum (uma por parâmetro)
    velocities: HashMap<String, Vec<f64>>,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            velocities: HashMap::new(),
        }
    }
    
    pub fn with_momentum(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay: 0.0,
            velocities: HashMap::new(),
        }
    }
    
    pub fn with_weight_decay(learning_rate: f64, momentum: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocities: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [&mut dyn Parameter]) {
        for param in parameters.iter_mut() {
            let param_name = param.name().to_string();
            let grad = param.gradient();
            let mut update = grad.clone();
            
            // Aplicar weight decay se configurado
            if self.weight_decay > 0.0 {
                let values = param.value();
                for (i, &val) in values.iter().enumerate() {
                    if i < update.len() {
                        update[i] += self.weight_decay * val;
                    }
                }
            }
            
            // Aplicar momentum se configurado
            if self.momentum > 0.0 {
                let velocity = self.velocities
                    .entry(param_name)
                    .or_insert_with(|| vec![0.0; grad.len()]);
                
                for (i, &g) in update.iter().enumerate() {
                    if i < velocity.len() {
                        velocity[i] = self.momentum * velocity[i] + g;
                        update[i] = velocity[i];
                    }
                }
            }
            
            // Aplicar learning rate e atualizar parâmetros
            let delta: Vec<f64> = update.iter().map(|&u| -self.learning_rate * u).collect();
            param.update(&delta);
        }
    }
    
    fn zero_grad(&mut self, parameters: &mut [&mut dyn Parameter]) {
        for param in parameters.iter_mut() {
            param.zero_grad();
        }
    }
    
    fn name(&self) -> &str {
        "SGD"
    }
}

/// Otimizador Adam (Adaptive Moment Estimation)
#[derive(Debug)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,  // Decay rate para primeiro momento
    pub beta2: f64,  // Decay rate para segundo momento
    pub epsilon: f64, // Termo para estabilidade numérica
    pub weight_decay: f64,
    /// Primeiro momento (média dos gradientes)
    m: HashMap<String, Vec<f64>>,
    /// Segundo momento (média dos gradientes ao quadrado)
    v: HashMap<String, Vec<f64>>,
    /// Contador de steps para bias correction
    step_count: usize,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
            step_count: 0,
        }
    }
    
    pub fn with_betas(learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
            step_count: 0,
        }
    }
    
    pub fn with_weight_decay(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            step_count: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut [&mut dyn Parameter]) {
        self.step_count += 1;
        
        for param in parameters.iter_mut() {
            let param_name = param.name().to_string();
            let mut grad = param.gradient();
            
            // Aplicar weight decay se configurado
            if self.weight_decay > 0.0 {
                let values = param.value();
                for (i, &val) in values.iter().enumerate() {
                    if i < grad.len() {
                        grad[i] += self.weight_decay * val;
                    }
                }
            }
            
            // Inicializar momentos se necessário
            let m = self.m.entry(param_name.clone())
                .or_insert_with(|| vec![0.0; grad.len()]);
            let v = self.v.entry(param_name)
                .or_insert_with(|| vec![0.0; grad.len()]);
            
            // Atualizar momentos
            for i in 0..grad.len() {
                // Primeiro momento (média exponencial dos gradientes)
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
                
                // Segundo momento (média exponencial dos gradientes ao quadrado)
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            }
            
            // Bias correction
            let m_corrected_scale = 1.0 / (1.0 - self.beta1.powi(self.step_count as i32));
            let v_corrected_scale = 1.0 / (1.0 - self.beta2.powi(self.step_count as i32));
            
            // Calcular update
            let delta: Vec<f64> = m.iter().zip(v.iter())
                .map(|(&m_val, &v_val)| {
                    let m_corrected = m_val * m_corrected_scale;
                    let v_corrected = v_val * v_corrected_scale;
                    -self.learning_rate * m_corrected / (v_corrected.sqrt() + self.epsilon)
                })
                .collect();
            
            param.update(&delta);
        }
    }
    
    fn zero_grad(&mut self, parameters: &mut [&mut dyn Parameter]) {
        for param in parameters.iter_mut() {
            param.zero_grad();
        }
    }
    
    fn name(&self) -> &str {
        "Adam"
    }
}

/// Modelo de rede neural simples
#[derive(Debug)]
pub struct SimpleNeuralNetwork {
    pub weights: SimpleParameter,
    pub bias: SimpleParameter,
}

impl SimpleNeuralNetwork {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Inicialização Xavier/Glorot
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        
        let weights = SimpleParameter::random(
            "weights".to_string(),
            input_size * output_size,
            scale,
        );
        
        let bias = SimpleParameter::new(
            "bias".to_string(),
            vec![0.0; output_size],
        );
        
        Self { weights, bias }
    }
    
    /// Forward pass: y = x * W + b
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let weights = self.weights.value();
        let bias = self.bias.value();
        
        let input_size = input.len();
        let output_size = bias.len();
        
        let mut output = bias.clone();
        
        // Multiplicação matriz-vetor: output = input * weights + bias
        for i in 0..output_size {
            for j in 0..input_size {
                let weight_idx = i * input_size + j;
                if weight_idx < weights.len() {
                    output[i] += input[j] * weights[weight_idx];
                }
            }
        }
        
        output
    }
    
    /// Calcula gradientes manualmente para demonstração
    pub fn backward(&mut self, input: &[f64], grad_output: &[f64]) {
        let input_size = input.len();
        let output_size = grad_output.len();
        
        // Gradiente do bias: dL/db = grad_output
        for (i, &grad) in grad_output.iter().enumerate() {
            if i < self.bias.grad.len() {
                self.bias.grad[i] += grad;
            }
        }
        
        // Gradiente dos weights: dL/dW = input^T * grad_output
        for i in 0..output_size {
            for j in 0..input_size {
                let weight_idx = i * input_size + j;
                if weight_idx < self.weights.grad.len() {
                    self.weights.grad[weight_idx] += input[j] * grad_output[i];
                }
            }
        }
    }
    
    /// Obtém referências mutáveis para os parâmetros
    pub fn parameters(&mut self) -> Vec<&mut dyn Parameter> {
        vec![&mut self.weights, &mut self.bias]
    }
}

/// Função de perda Mean Squared Error
#[derive(Debug)]
pub struct MSELoss;

impl MSELoss {
    pub fn forward(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        let mut loss = 0.0;
        let n = predictions.len().min(targets.len());
        
        for i in 0..n {
            let diff = predictions[i] - targets[i];
            loss += diff * diff;
        }
        
        loss / n as f64
    }
    
    pub fn backward(&self, predictions: &[f64], targets: &[f64]) -> Vec<f64> {
        let n = predictions.len().min(targets.len());
        let mut grad = vec![0.0; predictions.len()];
        
        for i in 0..n {
            grad[i] = 2.0 * (predictions[i] - targets[i]) / n as f64;
        }
        
        grad
    }
}

/// Sistema de treinamento completo
#[derive(Debug)]
pub struct TrainingSystem {
    pub model: SimpleNeuralNetwork,
    pub loss_fn: MSELoss,
    pub optimizer: Box<dyn Optimizer>,
    pub history: TrainingHistory,
}

#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub losses: Vec<f64>,
    pub epochs: Vec<usize>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            losses: Vec::new(),
            epochs: Vec::new(),
        }
    }
    
    pub fn add_loss(&mut self, epoch: usize, loss: f64) {
        self.epochs.push(epoch);
        self.losses.push(loss);
    }
    
    pub fn average_loss(&self, last_n: usize) -> f64 {
        if self.losses.is_empty() {
            return 0.0;
        }
        
        let start = if self.losses.len() > last_n {
            self.losses.len() - last_n
        } else {
            0
        };
        
        let sum: f64 = self.losses[start..].iter().sum();
        sum / (self.losses.len() - start) as f64
    }
}

impl TrainingSystem {
    pub fn new(input_size: usize, output_size: usize, optimizer: Box<dyn Optimizer>) -> Self {
        Self {
            model: SimpleNeuralNetwork::new(input_size, output_size),
            loss_fn: MSELoss,
            optimizer,
            history: TrainingHistory::new(),
        }
    }
    
    /// Executa um step de treinamento
    pub fn train_step(&mut self, input: &[f64], target: &[f64]) -> f64 {
        // Forward pass
        let prediction = self.model.forward(input);
        
        // Calcular loss
        let loss = self.loss_fn.forward(&prediction, target);
        
        // Backward pass
        let grad_output = self.loss_fn.backward(&prediction, target);
        self.model.backward(input, &grad_output);
        
        // Atualizar parâmetros
        let mut params = self.model.parameters();
        self.optimizer.step(&mut params);
        
        loss
    }
    
    /// Treina o modelo por várias épocas
    pub fn train(&mut self, dataset: &[(Vec<f64>, Vec<f64>)], epochs: usize) {
        println!("🚀 Iniciando treinamento com {} épocas...", epochs);
        println!("📊 Otimizador: {}", self.optimizer.name());
        println!("📈 Dataset: {} amostras", dataset.len());
        println!("-" * 50);
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            
            // Zerar gradientes
            let mut params = self.model.parameters();
            self.optimizer.zero_grad(&mut params);
            
            // Treinar em todas as amostras
            for (input, target) in dataset {
                let loss = self.train_step(input, target);
                epoch_loss += loss;
            }
            
            epoch_loss /= dataset.len() as f64;
            self.history.add_loss(epoch, epoch_loss);
            
            // Log do progresso
            if epoch % 10 == 0 || epoch == epochs - 1 {
                println!("Época {}: Loss = {:.6}", epoch, epoch_loss);
            }
        }
        
        println!("✅ Treinamento concluído!");
        println!("📉 Loss final: {:.6}", self.history.losses.last().unwrap_or(&0.0));
        println!("📊 Loss média (últimas 10 épocas): {:.6}", self.history.average_loss(10));
    }
    
    /// Avalia o modelo em um dataset
    pub fn evaluate(&self, dataset: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        let mut total_loss = 0.0;
        
        for (input, target) in dataset {
            let prediction = self.model.forward(input);
            let loss = self.loss_fn.forward(&prediction, target);
            total_loss += loss;
        }
        
        total_loss / dataset.len() as f64
    }
    
    /// Faz predição em uma amostra
    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        self.model.forward(input)
    }
}

/// Função principal demonstrando o sistema de treinamento
fn main() {
    println!("🧠 Sistema de Treinamento com Otimizadores");
    println!("=" * 50);
    
    // Criar dataset sintético: y = 2*x + 1 + ruído
    let mut dataset = Vec::new();
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for _ in 0..100 {
        let x = rng.gen_range(-1.0..1.0);
        let y = 2.0 * x + 1.0 + rng.gen_range(-0.1..0.1); // Adicionar ruído
        dataset.push((vec![x], vec![y]));
    }
    
    println!("📊 Dataset criado: {} amostras", dataset.len());
    println!("🎯 Função alvo: y = 2*x + 1");
    
    // Exemplo 1: Treinamento com SGD
    println!("\n📈 Exemplo 1: Treinamento com SGD");
    println!("-" * 40);
    
    let sgd_optimizer = Box::new(SGD::with_momentum(0.01, 0.9));
    let mut sgd_system = TrainingSystem::new(1, 1, sgd_optimizer);
    
    println!("Parâmetros iniciais:");
    println!("  Weights: {:?}", sgd_system.model.weights.value());
    println!("  Bias: {:?}", sgd_system.model.bias.value());
    
    sgd_system.train(&dataset, 100);
    
    println!("\nParâmetros finais (SGD):");
    println!("  Weights: {:?} (esperado: ~2.0)", sgd_system.model.weights.value());
    println!("  Bias: {:?} (esperado: ~1.0)", sgd_system.model.bias.value());
    
    // Exemplo 2: Treinamento com Adam
    println!("\n📈 Exemplo 2: Treinamento com Adam");
    println!("-" * 40);
    
    let adam_optimizer = Box::new(Adam::new(0.01));
    let mut adam_system = TrainingSystem::new(1, 1, adam_optimizer);
    
    println!("Parâmetros iniciais:");
    println!("  Weights: {:?}", adam_system.model.weights.value());
    println!("  Bias: {:?}", adam_system.model.bias.value());
    
    adam_system.train(&dataset, 100);
    
    println!("\nParâmetros finais (Adam):");
    println!("  Weights: {:?} (esperado: ~2.0)", adam_system.model.weights.value());
    println!("  Bias: {:?} (esperado: ~1.0)", adam_system.model.bias.value());
    
    // Exemplo 3: Comparação de performance
    println!("\n📊 Exemplo 3: Comparação de Performance");
    println!("-" * 45);
    
    let test_input = vec![0.5];
    let expected_output = 2.0 * 0.5 + 1.0; // = 2.0
    
    let sgd_prediction = sgd_system.predict(&test_input);
    let adam_prediction = adam_system.predict(&test_input);
    
    println!("Input de teste: {:?}", test_input);
    println!("Output esperado: {:.3}", expected_output);
    println!("Predição SGD: {:.3}", sgd_prediction[0]);
    println!("Predição Adam: {:.3}", adam_prediction[0]);
    println!("Erro SGD: {:.3}", (sgd_prediction[0] - expected_output).abs());
    println!("Erro Adam: {:.3}", (adam_prediction[0] - expected_output).abs());
    
    // Exemplo 4: Análise de convergência
    println!("\n📉 Exemplo 4: Análise de Convergência");
    println!("-" * 40);
    
    println!("SGD - Últimas 5 losses: {:?}", 
             sgd_system.history.losses.iter().rev().take(5).collect::<Vec<_>>());
    println!("Adam - Últimas 5 losses: {:?}", 
             adam_system.history.losses.iter().rev().take(5).collect::<Vec<_>>());
    
    println!("SGD - Loss média (últimas 10): {:.6}", sgd_system.history.average_loss(10));
    println!("Adam - Loss média (últimas 10): {:.6}", adam_system.history.average_loss(10));
    
    // Exemplo 5: Rede neural multicamada
    println!("\n🧠 Exemplo 5: Rede Neural Multicamada");
    println!("-" * 40);
    
    // Criar dataset para XOR (problema não-linear)
    let xor_dataset = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    println!("📊 Dataset XOR criado: {} amostras", xor_dataset.len());
    println!("🎯 Problema: XOR lógico (não-linear)");
    
    // Treinar rede para XOR (nota: precisa de camadas ocultas para funcionar bem)
    let adam_xor = Box::new(Adam::new(0.1));
    let mut xor_system = TrainingSystem::new(2, 1, adam_xor);
    
    println!("\nTreinando rede para XOR...");
    xor_system.train(&xor_dataset, 1000);
    
    println!("\nResultados XOR:");
    for (input, expected) in &xor_dataset {
        let prediction = xor_system.predict(input);
        println!("  {:?} -> {:.3} (esperado: {:.1})", 
                input, prediction[0], expected[0]);
    }
    
    println!("\n🎯 Conceitos Fundamentais Demonstrados:");
    println!("=" * 45);
    println!("✅ Otimizador SGD (Stochastic Gradient Descent)");
    println!("✅ Otimizador Adam (Adaptive Moment Estimation)");
    println!("✅ Momentum e Weight Decay");
    println!("✅ Bias Correction (Adam)");
    println!("✅ Loop de Treinamento Completo");
    println!("✅ Forward Pass e Backward Pass");
    println!("✅ Função de Perda MSE");
    println!("✅ Inicialização de Parâmetros");
    println!("✅ Histórico de Treinamento");
    println!("✅ Avaliação de Modelo");
    
    println!("\n🚀 Aplicações em LLMs:");
    println!("=" * 25);
    println!("• Treinamento de modelos Transformer");
    println!("• Fine-tuning de modelos pré-treinados");
    println!("• Otimização de hiperparâmetros");
    println!("• Regularização e prevenção de overfitting");
    println!("• Treinamento distribuído e paralelo");
    
    println!("\n💡 Exercícios Sugeridos:");
    println!("=" * 25);
    println!("1. Implementar otimizador AdamW");
    println!("2. Adicionar learning rate scheduling");
    println!("3. Implementar gradient clipping");
    println!("4. Criar sistema de early stopping");
    println!("5. Adicionar regularização L1/L2");
    println!("6. Implementar batch training");
    println!("7. Criar sistema de checkpointing");
    println!("8. Adicionar métricas de avaliação");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sgd_optimizer() {
        let mut param = SimpleParameter::new("test".to_string(), vec![1.0]);
        param.grad = vec![0.5]; // Gradiente positivo
        
        let mut sgd = SGD::new(0.1);
        let mut params: Vec<&mut dyn Parameter> = vec![&mut param];
        
        sgd.step(&mut params);
        
        // Parâmetro deve diminuir: 1.0 - 0.1 * 0.5 = 0.95
        assert!((param.value()[0] - 0.95).abs() < 1e-6);
    }
    
    #[test]
    fn test_adam_optimizer() {
        let mut param = SimpleParameter::new("test".to_string(), vec![1.0]);
        param.grad = vec![0.5];
        
        let mut adam = Adam::new(0.1);
        let mut params: Vec<&mut dyn Parameter> = vec![&mut param];
        
        let initial_value = param.value()[0];
        adam.step(&mut params);
        
        // Parâmetro deve ter mudado
        assert!((param.value()[0] - initial_value).abs() > 1e-6);
    }
    
    #[test]
    fn test_neural_network_forward() {
        let mut nn = SimpleNeuralNetwork::new(2, 1);
        
        // Definir pesos conhecidos
        nn.weights.data = vec![1.0, 2.0]; // [w1, w2]
        nn.bias.data = vec![0.5];          // [b]
        
        let input = vec![1.0, 2.0];
        let output = nn.forward(&input);
        
        // output = [1*1 + 2*2] + [0.5] = [5.5]
        assert!((output[0] - 5.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_mse_loss() {
        let loss_fn = MSELoss;
        
        let predictions = vec![1.0, 2.0];
        let targets = vec![1.5, 1.5];
        
        let loss = loss_fn.forward(&predictions, &targets);
        // MSE = ((1.0-1.5)² + (2.0-1.5)²) / 2 = (0.25 + 0.25) / 2 = 0.25
        assert!((loss - 0.25).abs() < 1e-6);
        
        let grad = loss_fn.backward(&predictions, &targets);
        // dL/dp = 2*(p-t)/n = [2*(1.0-1.5)/2, 2*(2.0-1.5)/2] = [-0.5, 0.5]
        assert!((grad[0] - (-0.5)).abs() < 1e-6);
        assert!((grad[1] - 0.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_training_system() {
        let optimizer = Box::new(SGD::new(0.1));
        let mut system = TrainingSystem::new(1, 1, optimizer);
        
        let dataset = vec![
            (vec![1.0], vec![2.0]),
            (vec![2.0], vec![4.0]),
        ];
        
        let initial_loss = system.evaluate(&dataset);
        system.train(&dataset, 10);
        let final_loss = system.evaluate(&dataset);
        
        // Loss deve ter diminuído
        assert!(final_loss < initial_loss);
    }
}