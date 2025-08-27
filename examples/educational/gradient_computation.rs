//! Sistema de Computação de Gradientes com Computational Graph
//! 
//! Este exemplo demonstra como implementar automatic differentiation (autodiff)
//! e backpropagation em Rust, conceitos fundamentais para o treinamento de LLMs.
//! 
//! ## Conceitos Fundamentais
//! 
//! ### Automatic Differentiation
//! - Técnica para calcular derivadas automaticamente
//! - Constrói um grafo computacional durante a forward pass
//! - Aplica a regra da cadeia durante a backward pass
//! 
//! ### Computational Graph
//! - Representação das operações como nós em um grafo
//! - Permite rastreamento eficiente de dependências
//! - Facilita o cálculo de gradientes via backpropagation

use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::fmt;

/// Identificador único para nós no grafo computacional
type NodeId = usize;

/// Operações suportadas no grafo computacional
#[derive(Debug, Clone)]
pub enum Operation {
    /// Operação de adição: z = x + y
    Add,
    /// Operação de multiplicação: z = x * y
    Mul,
    /// Operação de multiplicação por escalar: z = x * scalar
    MulScalar(f64),
    /// Operação de exponenciação: z = x^power
    Pow(f64),
    /// Operação de função sigmoide: z = 1 / (1 + exp(-x))
    Sigmoid,
    /// Operação de função ReLU: z = max(0, x)
    ReLU,
    /// Operação de função Tanh: z = tanh(x)
    Tanh,
    /// Operação de soma de elementos: z = sum(x)
    Sum,
    /// Operação de produto escalar: z = dot(x, y)
    Dot,
    /// Operação de entrada (leaf node)
    Input,
}

/// Nó no grafo computacional
#[derive(Debug)]
pub struct ComputeNode {
    /// Identificador único do nó
    pub id: NodeId,
    /// Valor atual do nó
    pub value: Vec<f64>,
    /// Gradiente acumulado
    pub gradient: Vec<f64>,
    /// Operação que gerou este nó
    pub operation: Operation,
    /// Nós pais (inputs da operação)
    pub parents: Vec<Weak<RefCell<ComputeNode>>>,
    /// Nós filhos (outputs da operação)
    pub children: Vec<Weak<RefCell<ComputeNode>>>,
    /// Flag indicando se este nó requer gradiente
    pub requires_grad: bool,
}

impl ComputeNode {
    /// Cria um novo nó computacional
    pub fn new(id: NodeId, value: Vec<f64>, operation: Operation, requires_grad: bool) -> Self {
        let gradient = vec![0.0; value.len()];
        Self {
            id,
            value,
            gradient,
            operation,
            parents: Vec::new(),
            children: Vec::new(),
            requires_grad,
        }
    }
    
    /// Zera o gradiente do nó
    pub fn zero_grad(&mut self) {
        self.gradient.fill(0.0);
    }
    
    /// Adiciona gradiente ao nó (acumulação)
    pub fn add_gradient(&mut self, grad: &[f64]) {
        for (i, &g) in grad.iter().enumerate() {
            if i < self.gradient.len() {
                self.gradient[i] += g;
            }
        }
    }
}

/// Tensor com suporte a automatic differentiation
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Referência para o nó no grafo computacional
    pub node: Rc<RefCell<ComputeNode>>,
}

impl Tensor {
    /// Cria um novo tensor a partir de um valor
    pub fn new(value: Vec<f64>, requires_grad: bool) -> Self {
        let id = Self::generate_id();
        let node = Rc::new(RefCell::new(ComputeNode::new(
            id,
            value,
            Operation::Input,
            requires_grad,
        )));
        Self { node }
    }
    
    /// Cria um tensor escalar
    pub fn scalar(value: f64, requires_grad: bool) -> Self {
        Self::new(vec![value], requires_grad)
    }
    
    /// Gera um ID único para o nó
    fn generate_id() -> NodeId {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        COUNTER.fetch_add(1, Ordering::SeqCst)
    }
    
    /// Obtém o valor do tensor
    pub fn value(&self) -> Vec<f64> {
        self.node.borrow().value.clone()
    }
    
    /// Obtém o gradiente do tensor
    pub fn gradient(&self) -> Vec<f64> {
        self.node.borrow().gradient.clone()
    }
    
    /// Zera todos os gradientes no grafo
    pub fn zero_grad(&self) {
        let mut visited = std::collections::HashSet::new();
        self.zero_grad_recursive(&mut visited);
    }
    
    fn zero_grad_recursive(&self, visited: &mut std::collections::HashSet<NodeId>) {
        let node_id = self.node.borrow().id;
        if visited.contains(&node_id) {
            return;
        }
        visited.insert(node_id);
        
        self.node.borrow_mut().zero_grad();
        
        // Recursivamente zera gradientes dos pais
        let parents = self.node.borrow().parents.clone();
        for parent_weak in parents {
            if let Some(parent) = parent_weak.upgrade() {
                let parent_tensor = Tensor { node: parent };
                parent_tensor.zero_grad_recursive(visited);
            }
        }
    }
    
    /// Executa backpropagation a partir deste tensor
    pub fn backward(&self) {
        // Inicializa o gradiente do nó raiz com 1.0
        {
            let mut node = self.node.borrow_mut();
            if node.value.len() == 1 {
                node.gradient[0] = 1.0;
            } else {
                // Para tensores não-escalares, inicializa com vetor de 1s
                node.gradient.fill(1.0);
            }
        }
        
        let mut visited = std::collections::HashSet::new();
        self.backward_recursive(&mut visited);
    }
    
    fn backward_recursive(&self, visited: &mut std::collections::HashSet<NodeId>) {
        let node_id = self.node.borrow().id;
        if visited.contains(&node_id) {
            return;
        }
        visited.insert(node_id);
        
        let (operation, parents, current_gradient, current_value) = {
            let node = self.node.borrow();
            (
                node.operation.clone(),
                node.parents.clone(),
                node.gradient.clone(),
                node.value.clone(),
            )
        };
        
        // Calcula gradientes para os nós pais baseado na operação
        match operation {
            Operation::Add => {
                // d/dx (x + y) = 1, d/dy (x + y) = 1
                for parent_weak in &parents {
                    if let Some(parent) = parent_weak.upgrade() {
                        parent.borrow_mut().add_gradient(&current_gradient);
                    }
                }
            }
            Operation::Mul => {
                // d/dx (x * y) = y, d/dy (x * y) = x
                if parents.len() == 2 {
                    if let (Some(parent1), Some(parent2)) = (
                        parents[0].upgrade(),
                        parents[1].upgrade(),
                    ) {
                        let parent1_value = parent1.borrow().value.clone();
                        let parent2_value = parent2.borrow().value.clone();
                        
                        // Gradiente para parent1: current_gradient * parent2_value
                        let grad1: Vec<f64> = current_gradient
                            .iter()
                            .zip(parent2_value.iter())
                            .map(|(&g, &v)| g * v)
                            .collect();
                        parent1.borrow_mut().add_gradient(&grad1);
                        
                        // Gradiente para parent2: current_gradient * parent1_value
                        let grad2: Vec<f64> = current_gradient
                            .iter()
                            .zip(parent1_value.iter())
                            .map(|(&g, &v)| g * v)
                            .collect();
                        parent2.borrow_mut().add_gradient(&grad2);
                    }
                }
            }
            Operation::MulScalar(scalar) => {
                // d/dx (x * c) = c
                if let Some(parent) = parents.get(0).and_then(|p| p.upgrade()) {
                    let grad: Vec<f64> = current_gradient
                        .iter()
                        .map(|&g| g * scalar)
                        .collect();
                    parent.borrow_mut().add_gradient(&grad);
                }
            }
            Operation::Pow(power) => {
                // d/dx (x^n) = n * x^(n-1)
                if let Some(parent) = parents.get(0).and_then(|p| p.upgrade()) {
                    let parent_value = parent.borrow().value.clone();
                    let grad: Vec<f64> = current_gradient
                        .iter()
                        .zip(parent_value.iter())
                        .map(|(&g, &x)| g * power * x.powf(power - 1.0))
                        .collect();
                    parent.borrow_mut().add_gradient(&grad);
                }
            }
            Operation::Sigmoid => {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                if let Some(parent) = parents.get(0).and_then(|p| p.upgrade()) {
                    let grad: Vec<f64> = current_gradient
                        .iter()
                        .zip(current_value.iter())
                        .map(|(&g, &s)| g * s * (1.0 - s))
                        .collect();
                    parent.borrow_mut().add_gradient(&grad);
                }
            }
            Operation::ReLU => {
                // d/dx ReLU(x) = 1 if x > 0, else 0
                if let Some(parent) = parents.get(0).and_then(|p| p.upgrade()) {
                    let parent_value = parent.borrow().value.clone();
                    let grad: Vec<f64> = current_gradient
                        .iter()
                        .zip(parent_value.iter())
                        .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
                        .collect();
                    parent.borrow_mut().add_gradient(&grad);
                }
            }
            Operation::Tanh => {
                // d/dx tanh(x) = 1 - tanh²(x)
                if let Some(parent) = parents.get(0).and_then(|p| p.upgrade()) {
                    let grad: Vec<f64> = current_gradient
                        .iter()
                        .zip(current_value.iter())
                        .map(|(&g, &t)| g * (1.0 - t * t))
                        .collect();
                    parent.borrow_mut().add_gradient(&grad);
                }
            }
            Operation::Sum => {
                // d/dx sum(x) = 1 para cada elemento
                if let Some(parent) = parents.get(0).and_then(|p| p.upgrade()) {
                    let parent_len = parent.borrow().value.len();
                    let grad = vec![current_gradient[0]; parent_len];
                    parent.borrow_mut().add_gradient(&grad);
                }
            }
            Operation::Input => {
                // Nó folha - não há pais para propagar gradientes
            }
            _ => {
                // Outras operações podem ser implementadas conforme necessário
            }
        }
        
        // Recursivamente propaga gradientes para os pais
        for parent_weak in parents {
            if let Some(parent) = parent_weak.upgrade() {
                let parent_tensor = Tensor { node: parent };
                parent_tensor.backward_recursive(visited);
            }
        }
    }
}

/// Implementação de operações matemáticas para Tensor
impl Tensor {
    /// Adição de tensores: self + other
    pub fn add(&self, other: &Tensor) -> Tensor {
        let result_value: Vec<f64> = self
            .value()
            .iter()
            .zip(other.value().iter())
            .map(|(&a, &b)| a + b)
            .collect();
        
        let requires_grad = self.node.borrow().requires_grad || other.node.borrow().requires_grad;
        let result = Tensor::new(result_value, requires_grad);
        
        // Conecta no grafo computacional
        {
            let mut result_node = result.node.borrow_mut();
            result_node.operation = Operation::Add;
            result_node.parents.push(Rc::downgrade(&self.node));
            result_node.parents.push(Rc::downgrade(&other.node));
        }
        
        // Adiciona como filho dos pais
        self.node.borrow_mut().children.push(Rc::downgrade(&result.node));
        other.node.borrow_mut().children.push(Rc::downgrade(&result.node));
        
        result
    }
    
    /// Multiplicação de tensores: self * other
    pub fn mul(&self, other: &Tensor) -> Tensor {
        let result_value: Vec<f64> = self
            .value()
            .iter()
            .zip(other.value().iter())
            .map(|(&a, &b)| a * b)
            .collect();
        
        let requires_grad = self.node.borrow().requires_grad || other.node.borrow().requires_grad;
        let result = Tensor::new(result_value, requires_grad);
        
        {
            let mut result_node = result.node.borrow_mut();
            result_node.operation = Operation::Mul;
            result_node.parents.push(Rc::downgrade(&self.node));
            result_node.parents.push(Rc::downgrade(&other.node));
        }
        
        self.node.borrow_mut().children.push(Rc::downgrade(&result.node));
        other.node.borrow_mut().children.push(Rc::downgrade(&result.node));
        
        result
    }
    
    /// Multiplicação por escalar: self * scalar
    pub fn mul_scalar(&self, scalar: f64) -> Tensor {
        let result_value: Vec<f64> = self.value().iter().map(|&x| x * scalar).collect();
        
        let requires_grad = self.node.borrow().requires_grad;
        let result = Tensor::new(result_value, requires_grad);
        
        {
            let mut result_node = result.node.borrow_mut();
            result_node.operation = Operation::MulScalar(scalar);
            result_node.parents.push(Rc::downgrade(&self.node));
        }
        
        self.node.borrow_mut().children.push(Rc::downgrade(&result.node));
        
        result
    }
    
    /// Exponenciação: self^power
    pub fn pow(&self, power: f64) -> Tensor {
        let result_value: Vec<f64> = self.value().iter().map(|&x| x.powf(power)).collect();
        
        let requires_grad = self.node.borrow().requires_grad;
        let result = Tensor::new(result_value, requires_grad);
        
        {
            let mut result_node = result.node.borrow_mut();
            result_node.operation = Operation::Pow(power);
            result_node.parents.push(Rc::downgrade(&self.node));
        }
        
        self.node.borrow_mut().children.push(Rc::downgrade(&result.node));
        
        result
    }
    
    /// Função sigmoide: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor {
        let result_value: Vec<f64> = self
            .value()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        
        let requires_grad = self.node.borrow().requires_grad;
        let result = Tensor::new(result_value, requires_grad);
        
        {
            let mut result_node = result.node.borrow_mut();
            result_node.operation = Operation::Sigmoid;
            result_node.parents.push(Rc::downgrade(&self.node));
        }
        
        self.node.borrow_mut().children.push(Rc::downgrade(&result.node));
        
        result
    }
    
    /// Função ReLU: max(0, x)
    pub fn relu(&self) -> Tensor {
        let result_value: Vec<f64> = self.value().iter().map(|&x| x.max(0.0)).collect();
        
        let requires_grad = self.node.borrow().requires_grad;
        let result = Tensor::new(result_value, requires_grad);
        
        {
            let mut result_node = result.node.borrow_mut();
            result_node.operation = Operation::ReLU;
            result_node.parents.push(Rc::downgrade(&self.node));
        }
        
        self.node.borrow_mut().children.push(Rc::downgrade(&result.node));
        
        result
    }
    
    /// Função Tanh: tanh(x)
    pub fn tanh(&self) -> Tensor {
        let result_value: Vec<f64> = self.value().iter().map(|&x| x.tanh()).collect();
        
        let requires_grad = self.node.borrow().requires_grad;
        let result = Tensor::new(result_value, requires_grad);
        
        {
            let mut result_node = result.node.borrow_mut();
            result_node.operation = Operation::Tanh;
            result_node.parents.push(Rc::downgrade(&self.node));
        }
        
        self.node.borrow_mut().children.push(Rc::downgrade(&result.node));
        
        result
    }
    
    /// Soma de todos os elementos: sum(x)
    pub fn sum(&self) -> Tensor {
        let sum_value = self.value().iter().sum();
        let result_value = vec![sum_value];
        
        let requires_grad = self.node.borrow().requires_grad;
        let result = Tensor::new(result_value, requires_grad);
        
        {
            let mut result_node = result.node.borrow_mut();
            result_node.operation = Operation::Sum;
            result_node.parents.push(Rc::downgrade(&self.node));
        }
        
        self.node.borrow_mut().children.push(Rc::downgrade(&result.node));
        
        result
    }
}

/// Implementação de Display para Tensor
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.node.borrow();
        write!(
            f,
            "Tensor(id={}, value={:?}, grad={:?}, requires_grad={})",
            node.id, node.value, node.gradient, node.requires_grad
        )
    }
}

/// Função principal demonstrando o sistema de gradientes
fn main() {
    println!("🧮 Sistema de Computação de Gradientes com Computational Graph");
    println!("=" * 70);
    
    // Exemplo 1: Gradiente simples - f(x) = x²
    println!("\n📊 Exemplo 1: Gradiente de f(x) = x²");
    println!("-" * 40);
    
    let x = Tensor::scalar(3.0, true);
    println!("x = {}", x);
    
    let y = x.pow(2.0);
    println!("y = x² = {}", y);
    
    y.backward();
    println!("dy/dx = {} (esperado: 2*x = 6.0)", x.gradient()[0]);
    
    // Exemplo 2: Gradiente de função composta - f(x) = (x + 1)²
    println!("\n📊 Exemplo 2: Gradiente de f(x) = (x + 1)²");
    println!("-" * 45);
    
    let x = Tensor::scalar(2.0, true);
    let one = Tensor::scalar(1.0, false);
    let x_plus_one = x.add(&one);
    let y = x_plus_one.pow(2.0);
    
    println!("x = {}", x.value()[0]);
    println!("y = (x + 1)² = {}", y.value()[0]);
    
    y.backward();
    println!("dy/dx = {} (esperado: 2*(x+1) = 6.0)", x.gradient()[0]);
    
    // Exemplo 3: Rede neural simples - f(x) = sigmoid(x * w + b)
    println!("\n📊 Exemplo 3: Rede Neural Simples - f(x) = sigmoid(x*w + b)");
    println!("-" * 55);
    
    let x = Tensor::scalar(0.5, false);  // Input
    let w = Tensor::scalar(2.0, true);   // Weight
    let b = Tensor::scalar(-1.0, true);  // Bias
    
    let linear = x.mul(&w).add(&b);      // x*w + b
    let output = linear.sigmoid();        // sigmoid(x*w + b)
    
    println!("x = {}, w = {}, b = {}", x.value()[0], w.value()[0], b.value()[0]);
    println!("linear = x*w + b = {}", linear.value()[0]);
    println!("output = sigmoid(linear) = {}", output.value()[0]);
    
    output.backward();
    println!("dL/dw = {} (gradiente do peso)", w.gradient()[0]);
    println!("dL/db = {} (gradiente do bias)", b.gradient()[0]);
    
    // Exemplo 4: Função de perda MSE - L = (y_pred - y_true)²
    println!("\n📊 Exemplo 4: Função de Perda MSE - L = (y_pred - y_true)²");
    println!("-" * 55);
    
    let y_true = Tensor::scalar(1.0, false);
    let y_pred = Tensor::scalar(0.8, true);
    
    // Calcula a diferença
    let diff = y_pred.add(&y_true.mul_scalar(-1.0));
    // Calcula o erro quadrático
    let loss = diff.pow(2.0);
    
    println!("y_true = {}", y_true.value()[0]);
    println!("y_pred = {}", y_pred.value()[0]);
    println!("loss = (y_pred - y_true)² = {}", loss.value()[0]);
    
    loss.backward();
    println!("dL/dy_pred = {} (esperado: 2*(y_pred - y_true) = -0.4)", y_pred.gradient()[0]);
    
    // Exemplo 5: Operações com vetores
    println!("\n📊 Exemplo 5: Operações com Vetores");
    println!("-" * 35);
    
    let x = Tensor::new(vec![1.0, 2.0, 3.0], true);
    let y = Tensor::new(vec![4.0, 5.0, 6.0], true);
    
    let z = x.mul(&y);  // Multiplicação elemento a elemento
    let loss = z.sum(); // Soma todos os elementos
    
    println!("x = {:?}", x.value());
    println!("y = {:?}", y.value());
    println!("z = x * y = {:?}", z.value());
    println!("loss = sum(z) = {}", loss.value()[0]);
    
    loss.backward();
    println!("dL/dx = {:?} (esperado: y = [4, 5, 6])", x.gradient());
    println!("dL/dy = {:?} (esperado: x = [1, 2, 3])", y.gradient());
    
    // Exemplo 6: Função de ativação ReLU
    println!("\n📊 Exemplo 6: Função de Ativação ReLU");
    println!("-" * 40);
    
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], true);
    let y = x.relu();
    let loss = y.sum();
    
    println!("x = {:?}", x.value());
    println!("y = ReLU(x) = {:?}", y.value());
    println!("loss = sum(y) = {}", loss.value()[0]);
    
    loss.backward();
    println!("dL/dx = {:?} (gradiente: 0 para x<0, 1 para x>0)", x.gradient());
    
    println!("\n🎯 Conceitos Fundamentais Demonstrados:");
    println!("=" * 45);
    println!("✅ Automatic Differentiation (Autodiff)");
    println!("✅ Computational Graph Construction");
    println!("✅ Forward Pass (cálculo de valores)");
    println!("✅ Backward Pass (backpropagation)");
    println!("✅ Chain Rule (regra da cadeia)");
    println!("✅ Gradient Accumulation");
    println!("✅ Operações Matemáticas Diferenciáveis");
    println!("✅ Funções de Ativação (Sigmoid, ReLU, Tanh)");
    println!("✅ Função de Perda (MSE)");
    println!("✅ Operações Vetoriais");
    
    println!("\n🚀 Aplicações em LLMs:");
    println!("=" * 25);
    println!("• Treinamento de redes neurais");
    println!("• Otimização de parâmetros");
    println!("• Cálculo de gradientes para backpropagation");
    println!("• Implementação de algoritmos de otimização (SGD, Adam)");
    println!("• Fine-tuning de modelos pré-treinados");
    
    println!("\n💡 Exercícios Sugeridos:");
    println!("=" * 25);
    println!("1. Implementar operação de convolução");
    println!("2. Adicionar suporte a matrizes 2D");
    println!("3. Implementar função de perda Cross-Entropy");
    println!("4. Criar otimizador SGD usando os gradientes");
    println!("5. Implementar normalização por lotes (BatchNorm)");
    println!("6. Adicionar suporte a operações em GPU");
    println!("7. Implementar mecanismo de atenção diferenciável");
    println!("8. Criar sistema de checkpointing para memória");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_gradient() {
        let x = Tensor::scalar(3.0, true);
        let y = x.pow(2.0);
        y.backward();
        
        // dy/dx = 2x = 6.0
        assert!((x.gradient()[0] - 6.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_chain_rule() {
        let x = Tensor::scalar(2.0, true);
        let one = Tensor::scalar(1.0, false);
        let y = x.add(&one).pow(2.0);
        y.backward();
        
        // dy/dx = 2(x+1) = 6.0
        assert!((x.gradient()[0] - 6.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_sigmoid_gradient() {
        let x = Tensor::scalar(0.0, true);
        let y = x.sigmoid();
        y.backward();
        
        // dy/dx = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        assert!((x.gradient()[0] - 0.25).abs() < 1e-6);
    }
    
    #[test]
    fn test_vector_operations() {
        let x = Tensor::new(vec![1.0, 2.0], true);
        let y = Tensor::new(vec![3.0, 4.0], true);
        let z = x.mul(&y).sum();
        z.backward();
        
        // dz/dx = y = [3.0, 4.0]
        assert!((x.gradient()[0] - 3.0).abs() < 1e-6);
        assert!((x.gradient()[1] - 4.0).abs() < 1e-6);
        
        // dz/dy = x = [1.0, 2.0]
        assert!((y.gradient()[0] - 1.0).abs() < 1e-6);
        assert!((y.gradient()[1] - 2.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_relu_gradient() {
        let x = Tensor::new(vec![-1.0, 0.0, 1.0], true);
        let y = x.relu().sum();
        y.backward();
        
        // ReLU gradient: 0 for x < 0, 1 for x > 0, undefined at x = 0 (we use 0)
        assert!((x.gradient()[0] - 0.0).abs() < 1e-6);
        assert!((x.gradient()[1] - 0.0).abs() < 1e-6);
        assert!((x.gradient()[2] - 1.0).abs() < 1e-6);
    }
}