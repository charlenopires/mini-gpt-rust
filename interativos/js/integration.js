/**
 * 🔗 Sistema de Integração Web-Demo
 * Comunicação em tempo real entre interface web e CLI
 */

class WebDemoIntegration {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.modules = new Map();
        this.performanceData = new Map();
        
        this.init();
    }

    /**
     * 🚀 Inicializa o sistema de integração
     */
    async init() {
        await this.loadModules();
        this.connectWebSocket();
        this.setupEventListeners();
        this.createControlPanel();
    }

    /**
     * 📡 Conecta ao WebSocket
     */
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('🔗 WebSocket conectado');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                
                // Solicita status inicial
                this.sendMessage({
                    type: 'get_status',
                    data: {}
                });
            };
            
            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };
            
            this.ws.onclose = () => {
                console.log('❌ WebSocket desconectado');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('🚨 Erro WebSocket:', error);
            };
            
        } catch (error) {
            console.error('🚨 Erro ao conectar WebSocket:', error);
            this.attemptReconnect();
        }
    }

    /**
     * 🔄 Tenta reconectar ao WebSocket
     */
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`🔄 Tentativa de reconexão ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            
            setTimeout(() => {
                this.connectWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }

    /**
     * 📨 Envia mensagem via WebSocket
     */
    sendMessage(message) {
        if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.warn('⚠️ WebSocket não conectado');
        }
    }

    /**
     * 📥 Processa mensagens recebidas
     */
    handleMessage(message) {
        switch (message.type) {
            case 'status_update':
                this.updateModuleStatus(message.data);
                break;
            case 'parameter_update':
                this.updateParameters(message.data);
                break;
            case 'performance_update':
                this.updatePerformanceMetrics(message.data);
                break;
            case 'demo_data':
                this.updateDemoVisualization(message.data);
                break;
            default:
                console.log('📨 Mensagem recebida:', message);
        }
    }

    /**
     * 📊 Carrega módulos disponíveis
     */
    async loadModules() {
        try {
            const response = await fetch('/api/v1/modules');
            const modules = await response.json();
            
            // A API retorna um array de strings com os nomes dos módulos
            modules.forEach(moduleName => {
                this.modules.set(moduleName, {
                    name: moduleName,
                    status: 'idle',
                    parameters: {}
                });
            });
            
            console.log('📚 Módulos carregados:', this.modules.size);
        } catch (error) {
            console.error('🚨 Erro ao carregar módulos:', error);
        }
    }

    /**
     * 🎛️ Cria painel de controle dinâmico
     */
    createControlPanel() {
        const existingPanel = document.getElementById('integration-panel');
        if (existingPanel) {
            existingPanel.remove();
        }

        const panel = document.createElement('div');
        panel.id = 'integration-panel';
        panel.className = 'fixed top-4 right-4 bg-white shadow-lg rounded-lg p-4 z-50 max-w-sm';
        panel.innerHTML = `
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-semibold text-gray-800">🔗 Integração Web-Demo</h3>
                <div id="connection-status" class="w-3 h-3 rounded-full bg-red-500"></div>
            </div>
            
            <div class="space-y-3">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Módulo Ativo</label>
                    <select id="active-module" class="w-full p-2 border border-gray-300 rounded-md text-sm">
                        <option value="">Selecione um módulo</option>
                    </select>
                </div>
                
                <div id="module-parameters" class="space-y-2">
                    <!-- Parâmetros dinâmicos serão inseridos aqui -->
                </div>
                
                <div class="flex space-x-2">
                    <button id="run-demo" class="flex-1 bg-blue-500 text-white px-3 py-2 rounded-md text-sm hover:bg-blue-600 transition-colors">
                        ▶️ Executar Demo
                    </button>
                    <button id="reset-params" class="flex-1 bg-gray-500 text-white px-3 py-2 rounded-md text-sm hover:bg-gray-600 transition-colors">
                        🔄 Reset
                    </button>
                </div>
                
                <div id="performance-metrics" class="text-xs text-gray-600 space-y-1">
                    <!-- Métricas de performance -->
                </div>
            </div>
        `;
        
        document.body.appendChild(panel);
        this.populateModuleSelect();
    }

    /**
     * 📋 Popula select de módulos
     */
    populateModuleSelect() {
        const select = document.getElementById('active-module');
        if (!select) return;
        
        this.modules.forEach((module, name) => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = `${module.icon || '📦'} ${module.display_name || name}`;
            select.appendChild(option);
        });
    }

    /**
     * 🎛️ Configura event listeners
     */
    setupEventListeners() {
        document.addEventListener('change', (e) => {
            if (e.target.id === 'active-module') {
                this.onModuleChange(e.target.value);
            } else if (e.target.classList.contains('param-input')) {
                this.onParameterChange(e.target);
            }
        });
        
        document.addEventListener('click', (e) => {
            if (e.target.id === 'run-demo') {
                this.runDemo();
            } else if (e.target.id === 'reset-params') {
                this.resetParameters();
            }
        });
    }

    /**
     * 🔄 Atualiza status de conexão
     */
    updateConnectionStatus(connected) {
        const status = document.getElementById('connection-status');
        if (status) {
            status.className = `w-3 h-3 rounded-full ${
                connected ? 'bg-green-500' : 'bg-red-500'
            }`;
            status.title = connected ? 'Conectado' : 'Desconectado';
        }
    }

    /**
     * 📊 Atualiza status dos módulos
     */
    updateModuleStatus(data) {
        console.log('📊 Status atualizado:', data);
        // Implementar atualização visual do status
    }

    /**
     * 🎛️ Atualiza parâmetros
     */
    updateParameters(data) {
        console.log('🎛️ Parâmetros atualizados:', data);
        // Implementar sincronização de parâmetros
    }

    /**
     * 📈 Atualiza métricas de performance
     */
    updatePerformanceMetrics(data) {
        const metricsDiv = document.getElementById('performance-metrics');
        if (!metricsDiv) return;
        
        metricsDiv.innerHTML = `
            <div>⚡ Latência: ${data.latency || 'N/A'}ms</div>
            <div>🧠 Memória: ${data.memory_usage || 'N/A'}MB</div>
            <div>⏱️ Tempo: ${data.execution_time || 'N/A'}ms</div>
        `;
    }

    /**
     * 📊 Atualiza visualização de demo
     */
    updateDemoVisualization(data) {
        console.log('📊 Dados de demo:', data);
        // Implementar visualizações avançadas
    }

    /**
     * 🔄 Mudança de módulo
     */
    async onModuleChange(moduleName) {
        if (!moduleName) return;
        
        try {
            const response = await fetch(`/api/modules/${moduleName}/parameters`);
            const parameters = await response.json();
            
            this.renderParameterControls(parameters);
        } catch (error) {
            console.error('🚨 Erro ao carregar parâmetros:', error);
        }
    }

    /**
     * 🎛️ Renderiza controles de parâmetros
     */
    renderParameterControls(parameters) {
        const container = document.getElementById('module-parameters');
        if (!container) return;
        
        container.innerHTML = '';
        
        Object.entries(parameters).forEach(([key, param]) => {
            const control = this.createParameterControl(key, param);
            container.appendChild(control);
        });
    }

    /**
     * 🎚️ Cria controle de parâmetro
     */
    createParameterControl(key, param) {
        const div = document.createElement('div');
        div.className = 'space-y-1';
        
        const label = document.createElement('label');
        label.className = 'block text-xs font-medium text-gray-700';
        label.textContent = param.display_name || key;
        
        let input;
        
        switch (param.type) {
            case 'number':
                input = document.createElement('input');
                input.type = 'number';
                input.min = param.min || 0;
                input.max = param.max || 100;
                input.step = param.step || 1;
                input.value = param.default || 0;
                break;
                
            case 'range':
                input = document.createElement('input');
                input.type = 'range';
                input.min = param.min || 0;
                input.max = param.max || 100;
                input.step = param.step || 1;
                input.value = param.default || 50;
                break;
                
            case 'boolean':
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = param.default || false;
                break;
                
            case 'select':
                input = document.createElement('select');
                param.options?.forEach(option => {
                    const opt = document.createElement('option');
                    opt.value = option.value;
                    opt.textContent = option.label;
                    input.appendChild(opt);
                });
                break;
                
            default:
                input = document.createElement('input');
                input.type = 'text';
                input.value = param.default || '';
        }
        
        input.className = 'param-input w-full p-1 border border-gray-300 rounded text-xs';
        input.dataset.paramKey = key;
        
        div.appendChild(label);
        div.appendChild(input);
        
        return div;
    }

    /**
     * 🔄 Mudança de parâmetro
     */
    async onParameterChange(input) {
        const moduleName = document.getElementById('active-module')?.value;
        if (!moduleName) return;
        
        const paramKey = input.dataset.paramKey;
        let value = input.value;
        
        if (input.type === 'checkbox') {
            value = input.checked;
        } else if (input.type === 'number' || input.type === 'range') {
            value = parseFloat(value);
        }
        
        try {
            await fetch(`/api/modules/${moduleName}/parameters`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    [paramKey]: value
                })
            });
            
            // Notifica via WebSocket
            this.sendMessage({
                type: 'parameter_change',
                data: {
                    module: moduleName,
                    parameter: paramKey,
                    value: value
                }
            });
            
        } catch (error) {
            console.error('🚨 Erro ao atualizar parâmetro:', error);
        }
    }

    /**
     * ▶️ Executa demonstração
     */
    async runDemo() {
        const moduleName = document.getElementById('active-module')?.value;
        if (!moduleName) {
            alert('⚠️ Selecione um módulo primeiro');
            return;
        }
        
        try {
            const response = await fetch(`/api/demo/${moduleName}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            console.log('✅ Demo executado:', result);
            
            // Notifica via WebSocket
            this.sendMessage({
                type: 'run_demo',
                data: {
                    module: moduleName
                }
            });
            
        } catch (error) {
            console.error('🚨 Erro ao executar demo:', error);
        }
    }

    /**
     * 🔄 Reset de parâmetros
     */
    async resetParameters() {
        const moduleName = document.getElementById('active-module')?.value;
        if (!moduleName) return;
        
        try {
            await fetch(`/api/modules/${moduleName}/reset`, {
                method: 'POST'
            });
            
            // Recarrega parâmetros
            this.onModuleChange(moduleName);
            
        } catch (error) {
            console.error('🚨 Erro ao resetar parâmetros:', error);
        }
    }

    /**
     * 📊 Exibe resultado da demonstração
     */
    displayDemoResult(module, result) {
        const resultContainer = document.getElementById('demo-results') || this.createDemoResultsContainer();
        
        const resultDiv = document.createElement('div');
        resultDiv.className = 'demo-result bg-gray-50 p-4 rounded-lg mb-4';
        resultDiv.innerHTML = `
            <h4 class="text-lg font-semibold mb-2">🎯 Resultado: ${module}</h4>
            <div class="result-metrics flex flex-wrap gap-4 mb-3 text-sm">
                <span class="bg-blue-100 px-2 py-1 rounded">⏱️ Tempo: ${result.execution_time}ms</span>
                <span class="bg-green-100 px-2 py-1 rounded">🧠 CPU: ${result.metrics?.cpu_usage?.toFixed(1) || 'N/A'}%</span>
                <span class="bg-yellow-100 px-2 py-1 rounded">💾 Memória: ${result.metrics?.memory_usage?.toFixed(1) || 'N/A'}MB</span>
                <span class="bg-purple-100 px-2 py-1 rounded">🚀 Throughput: ${result.metrics?.throughput?.toFixed(1) || 'N/A'}</span>
            </div>
            <div class="result-output bg-gray-800 text-green-400 p-3 rounded text-sm overflow-auto max-h-32">
                <pre>${result.output || 'Sem saída'}</pre>
            </div>
            ${result.visualizations?.length > 0 ? `
                <div class="visualizations mt-3">
                    ${result.visualizations.map(viz => `
                        <div class="visualization border rounded p-3 mt-2" data-type="${viz.viz_type}">
                            <h5 class="font-medium mb-2">${viz.title}</h5>
                            <div class="viz-container" id="viz-${module}-${viz.viz_type}"></div>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        `;
        
        resultContainer.appendChild(resultDiv);
        
        // Renderiza visualizações
        if (result.visualizations) {
            result.visualizations.forEach(viz => {
                this.renderVisualization(module, viz);
            });
        }
        
        // Remove resultados antigos (mantém apenas os 5 mais recentes)
        const results = resultContainer.querySelectorAll('.demo-result');
        if (results.length > 5) {
            results[0].remove();
        }
    }

    /**
     * 🎨 Renderiza visualizações
     */
    renderVisualization(module, viz) {
        const container = document.getElementById(`viz-${module}-${viz.viz_type}`);
        if (!container) return;

        switch (viz.viz_type) {
            case 'tokenizer':
                this.renderTokenizerViz(container, viz.data);
                break;
            case 'attention_heatmap':
                this.renderAttentionHeatmap(container, viz.data);
                break;
            case 'performance_chart':
                this.renderPerformanceChart(container, viz.data);
                break;
            default:
                container.innerHTML = `<pre class="text-xs bg-gray-100 p-2 rounded overflow-auto">${JSON.stringify(viz.data, null, 2)}</pre>`;
        }
    }

    /**
     * 🔤 Renderiza visualização de tokenização
     */
    renderTokenizerViz(container, data) {
        container.innerHTML = `
            <div class="tokenizer-viz">
                <div class="original-text mb-2">
                    <strong>Texto Original:</strong> <span class="text-gray-700">${data.text}</span>
                </div>
                <div class="tokens">
                    <strong>Tokens (${data.token_count}):</strong>
                    <div class="token-list flex flex-wrap gap-1 mt-2">
                        ${data.tokens.map((token, i) => `
                            <span class="token px-2 py-1 rounded text-xs border" 
                                  style="background-color: hsl(${i * 360 / data.tokens.length}, 70%, 85%)">
                                ${token}
                            </span>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * 🎯 Renderiza mapa de atenção
     */
    renderAttentionHeatmap(container, data) {
        const size = Math.min(container.clientWidth || 300, 300);
        const cellSize = size / data.seq_len;
        
        let html = `<div class="attention-heatmap relative border" style="width: ${size}px; height: ${size}px;">`;
        
        for (let i = 0; i < data.seq_len; i++) {
            for (let j = 0; j < data.seq_len; j++) {
                const attention = data.matrix[i][j];
                const opacity = Math.min(attention, 1.0);
                html += `
                    <div class="attention-cell absolute border-gray-300" 
                         style="
                             left: ${j * cellSize}px;
                             top: ${i * cellSize}px;
                             width: ${cellSize}px;
                             height: ${cellSize}px;
                             background-color: rgba(255, 0, 0, ${opacity});
                             border: 1px solid #ccc;
                         "
                         title="Atenção [${i},${j}]: ${attention.toFixed(3)}">
                    </div>
                `;
            }
        }
        
        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * 📈 Renderiza gráfico de performance
     */
    renderPerformanceChart(container, data) {
        container.innerHTML = `
            <div class="performance-chart">
                <div class="chart-metrics grid grid-cols-2 gap-2 text-sm">
                    <div class="metric bg-blue-50 p-2 rounded">
                        <div class="font-medium">Latência Média</div>
                        <div class="text-lg">${data.avg_latency?.toFixed(2) || 'N/A'}ms</div>
                    </div>
                    <div class="metric bg-green-50 p-2 rounded">
                        <div class="font-medium">Throughput</div>
                        <div class="text-lg">${data.throughput?.toFixed(1) || 'N/A'}/s</div>
                    </div>
                    <div class="metric bg-yellow-50 p-2 rounded">
                        <div class="font-medium">Uso de CPU</div>
                        <div class="text-lg">${data.cpu_usage?.toFixed(1) || 'N/A'}%</div>
                    </div>
                    <div class="metric bg-purple-50 p-2 rounded">
                        <div class="font-medium">Memória</div>
                        <div class="text-lg">${data.memory_usage?.toFixed(1) || 'N/A'}MB</div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * 📊 Cria container para resultados de demo
     */
    createDemoResultsContainer() {
        const existingContainer = document.getElementById('demo-results');
        if (existingContainer) {
            return existingContainer;
        }

        const container = document.createElement('div');
        container.id = 'demo-results';
        container.className = 'fixed bottom-4 left-4 max-w-2xl max-h-96 overflow-y-auto bg-white shadow-lg rounded-lg p-4 z-40';
        container.innerHTML = '<h3 class="text-lg font-semibold mb-3">📊 Resultados das Demonstrações</h3>';
        
        document.body.appendChild(container);
        return container;
    }

    /**
     * 📈 Atualiza métricas em tempo real
     */
    updateRealTimeMetrics(metrics) {
        // Atualiza métricas no painel de controle
        const metricsDiv = document.getElementById('performance-metrics');
        if (metricsDiv) {
            metricsDiv.innerHTML = `
                <div>⚡ Latência: ${metrics.latency || 'N/A'}ms</div>
                <div>🧠 CPU: ${metrics.cpu_usage?.toFixed(1) || 'N/A'}%</div>
                <div>💾 Memória: ${metrics.memory_usage?.toFixed(1) || 'N/A'}MB</div>
                <div>🚀 Throughput: ${metrics.throughput?.toFixed(1) || 'N/A'}/s</div>
            `;
        }

        // Armazena métricas para análise
        const timestamp = Date.now();
        this.performanceData.set(timestamp, metrics);

        // Mantém apenas os últimos 100 pontos
        if (this.performanceData.size > 100) {
            const oldestKey = Math.min(...this.performanceData.keys());
            this.performanceData.delete(oldestKey);
        }
    }
}

// 🚀 Inicializa quando o DOM estiver pronto
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.webDemoIntegration = new WebDemoIntegration();
    });
} else {
    window.webDemoIntegration = new WebDemoIntegration();
}

// 🔗 Exporta para uso global
window.WebDemoIntegration = WebDemoIntegration;