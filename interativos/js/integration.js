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
        
        // Sistema de Notificações
        this.toastContainer = null;
        this.toastCounter = 0;
        this.activeToasts = new Map();
        
        // Sistema de Progresso Detalhado
        this.progressData = {
            current: 0,
            total: 0,
            startTime: null,
            tokens: [],
            speed: 0
        };
        
        // Sistema de Logs
        this.logs = [];
        this.maxLogs = 100;
        
        // Sistema de Feedback Sonoro
        this.soundEnabled = false;
        this.audioContext = null;
        
        // Identificação do cliente
        this.clientId = null;
        this.sessionId = null;
        
        this.init();
    }

    /**
     * 🚀 Inicializa o sistema de integração
     */
    async init() {
        this.initToastSystem();
        await this.loadModules();
        this.connectWebSocket();
        this.setupEventListeners();
        this.createControlPanel();
    }

    /**
     * 🍞 Inicializa o sistema de notificações toast
     */
    initToastSystem() {
        this.toastContainer = document.getElementById('toastContainer');
        if (!this.toastContainer) {
            console.warn('Toast container não encontrado');
        }
    }

    /**
     * 📢 Cria e exibe uma notificação toast
     */
    showToast(type, title, message, duration = 5000) {
        if (!this.toastContainer) return;

        const toastId = `toast-${++this.toastCounter}`;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.id = toastId;

        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️',
            loading: '⏳'
        };

        toast.innerHTML = `
            <div class="toast-content">
                <div class="toast-icon">${icons[type] || icons.info}</div>
                <div class="toast-message">
                    <div class="toast-title">${title}</div>
                    <div class="toast-description">${message}</div>
                </div>
                <button class="toast-close" onclick="webDemoIntegration.closeToast('${toastId}')">
                    ×
                </button>
            </div>
        `;

        this.toastContainer.appendChild(toast);
        this.activeToasts.set(toastId, toast);

        // Animação de entrada
        setTimeout(() => toast.classList.add('show'), 10);

        // Feedback sonoro
        this.playNotificationSound(type);

        // Auto-remove (exceto para loading)
        if (type !== 'loading' && duration > 0) {
            setTimeout(() => this.closeToast(toastId), duration);
        }

        this.addLog('TOAST', `${type.toUpperCase()}: ${title} - ${message}`);
        return toastId;
    }

    /**
     * 🗑️ Fecha uma notificação toast
     */
    closeToast(toastId) {
        const toast = this.activeToasts.get(toastId);
        if (!toast) return;

        toast.classList.add('hide');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
            this.activeToasts.delete(toastId);
        }, 300);
    }

    /**
     * 📊 Atualiza o progresso detalhado
     */
    updateDetailedProgress(current, total, tokens = []) {
        this.progressData.current = current;
        this.progressData.total = total;
        this.progressData.tokens = tokens;

        if (!this.progressData.startTime) {
            this.progressData.startTime = Date.now();
        }

        const percentage = total > 0 ? Math.round((current / total) * 100) : 0;
        const elapsed = Date.now() - this.progressData.startTime;
        const speed = current > 0 ? (current / elapsed) * 1000 : 0;
        const eta = speed > 0 ? Math.round((total - current) / speed) : 0;

        this.progressData.speed = speed;

        // Atualiza elementos da UI
        const progressElement = document.getElementById('detailedProgress');
        const percentageElement = document.getElementById('detailedPercentage');
        const fillElement = document.getElementById('detailedProgressFill');
        const tokensElement = document.getElementById('progressTokens');
        const etaElement = document.getElementById('progressETA');

        if (progressElement) progressElement.style.display = 'block';
        if (percentageElement) percentageElement.textContent = `${percentage}%`;
        if (fillElement) fillElement.style.width = `${percentage}%`;
        if (tokensElement) tokensElement.textContent = `${current}/${total} tokens`;
        if (etaElement) etaElement.textContent = `ETA: ${eta}s`;

        this.updateStatsBadges(current, speed);
    }

    /**
     * 🏆 Atualiza badges de estatísticas
     */
    updateStatsBadges(tokens, speed) {
        const badgesElement = document.getElementById('statsBadges');
        const successElement = document.getElementById('successTokens');
        const speedElement = document.getElementById('speedValue');
        const qualityElement = document.getElementById('qualityValue');

        if (badgesElement) badgesElement.style.display = 'flex';
        if (successElement) successElement.textContent = `${tokens} tokens`;
        if (speedElement) speedElement.textContent = `${speed.toFixed(1)} t/s`;
        
        // Qualidade baseada na velocidade
        let quality = 'Baixa';
        if (speed > 10) quality = 'Alta';
        else if (speed > 5) quality = 'Média';
        
        if (qualityElement) qualityElement.textContent = quality;
    }

    /**
     * 📝 Adiciona entrada ao log em tempo real
     */
    addLog(level, message) {
        const timestamp = new Date().toLocaleTimeString('pt-BR', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });

        const logEntry = {
            timestamp,
            level: level.toLowerCase(),
            message
        };

        this.logs.push(logEntry);
        if (this.logs.length > this.maxLogs) {
            this.logs.shift();
        }

        this.updateLogsDisplay();
    }

    /**
     * 🔄 Atualiza a exibição dos logs
     */
    updateLogsDisplay() {
        const logsElement = document.getElementById('realtimeLogs');
        if (!logsElement) return;

        logsElement.style.display = 'block';
        
        // Mostra apenas os últimos 5 logs
        const recentLogs = this.logs.slice(-5);
        logsElement.innerHTML = recentLogs.map(log => `
            <div class="log-entry">
                <span class="log-timestamp">${log.timestamp}</span>
                <span class="log-level ${log.level}">${log.level.toUpperCase()}</span>
                <span class="log-message">${log.message}</span>
            </div>
        `).join('');

        // Auto-scroll para o final
        logsElement.scrollTop = logsElement.scrollHeight;
    }

    /**
     * 🔊 Inicializa o sistema de áudio
     */
    initAudioSystem() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.soundEnabled = true;
            this.addLog('AUDIO', 'Sistema de áudio inicializado');
        } catch (error) {
            console.warn('Áudio não suportado:', error);
            this.soundEnabled = false;
        }
    }

    /**
     * 🎵 Reproduz som de notificação
     */
    playNotificationSound(type) {
        if (!this.soundEnabled || !this.audioContext) return;

        const frequencies = {
            success: [523.25, 659.25, 783.99], // C5, E5, G5
            error: [349.23, 293.66], // F4, D4
            warning: [440, 554.37], // A4, C#5
            info: [523.25], // C5
            loading: [392, 523.25] // G4, C5
        };

        const freqs = frequencies[type] || frequencies.info;
        
        freqs.forEach((freq, index) => {
            setTimeout(() => {
                this.playTone(freq, 0.1, 0.1);
            }, index * 100);
        });
    }

    /**
     * 🎼 Reproduz um tom específico
     */
    playTone(frequency, duration, volume = 0.1) {
        if (!this.audioContext) return;

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);

        oscillator.frequency.setValueAtTime(frequency, this.audioContext.currentTime);
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
        gainNode.gain.linearRampToValueAtTime(volume, this.audioContext.currentTime + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + duration);

        oscillator.start(this.audioContext.currentTime);
        oscillator.stop(this.audioContext.currentTime + duration);
    }

    /**
     * 🔇 Alterna o som
     */
    toggleSound() {
        if (!this.audioContext) {
            this.initAudioSystem();
        } else {
            this.soundEnabled = !this.soundEnabled;
        }
        
        this.addLog('AUDIO', `Som ${this.soundEnabled ? 'ativado' : 'desativado'}`);
        return this.soundEnabled;
    }

    /**
     * 📡 Conecta ao WebSocket
     */
    // Refatoração para uso de async/await em conexões
    async connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Conectar ao servidor Rust na porta 3000
        const wsUrl = `${protocol}//localhost:3000/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('🔗 WebSocket conectado');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                
                this.showToast('success', 'Conectado', 'WebSocket conectado com sucesso', 3000);
                this.addLog('WEBSOCKET', 'Conexão estabelecida');
                
                // Solicita status inicial usando o formato correto do backend Rust
                this.sendMessage({
                    type: 'GetStatus',
                    data: {
                        timestamp: Date.now()
                    }
                });
                
                // Registra cliente para receber atualizações
                this.sendMessage({
                    type: 'RegisterClient',
                    data: {
                        client_id: `web_client_${Date.now()}`,
                        capabilities: ['real_time_updates', 'token_streaming']
                    }
                });
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (error) {
                    console.error('🚨 Erro ao processar mensagem WebSocket:', error);
                    this.addLog('ERROR', `Erro ao processar mensagem: ${error.message}`);
                }
            };
            
            this.ws.onclose = () => {
                console.log('❌ WebSocket desconectado');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.showToast('warning', 'Desconectado', 'Tentando reconectar...', 3000);
                this.addLog('WEBSOCKET', 'Conexão perdida, tentando reconectar');
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('🚨 Erro WebSocket:', error);
                this.showToast('error', 'Erro de Conexão', 'Falha na comunicação WebSocket', 5000);
                this.addLog('ERROR', 'Erro na conexão WebSocket');
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
        } else {
            this.addLog('WEBSOCKET', 'Máximo de tentativas de reconexão atingido');
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
        this.addLog('WEBSOCKET', `Mensagem recebida: ${message.type}`);
        
        switch (message.type) {
            // Mensagens do backend Rust
            case 'StatusUpdate':
                this.updateModuleStatus(message.data);
                break;
            case 'ParameterUpdate':
                this.updateParameters(message.data);
                break;
            case 'PerformanceMetrics':
                this.updatePerformanceMetrics(message.data);
                break;
            case 'InferenceResult':
                this.processInferenceResult(message.data);
                break;
            case 'TokenGenerated':
                this.handleTokenGenerated(message.data);
                break;
            case 'InferenceStarted':
                this.handleGenerationStart(message.data);
                break;
            case 'InferenceCompleted':
                this.handleGenerationComplete(message.data);
                // Resolve pending generation promises
                if (this.pendingGenerations) {
                    for (const [requestId, { resolve }] of this.pendingGenerations.entries()) {
                        resolve(message.data);
                        this.pendingGenerations.delete(requestId);
                    }
                }
                break;
            case 'InferenceError':
                this.handleGenerationError(message.data);
                // Reject pending generation promises
                if (this.pendingGenerations) {
                    for (const [requestId, { reject }] of this.pendingGenerations.entries()) {
                        reject(new Error(message.data.error || 'Erro na inferência'));
                        this.pendingGenerations.delete(requestId);
                    }
                }
                if (this.pendingSteps) {
                    for (const [requestId, { reject }] of this.pendingSteps.entries()) {
                        reject(new Error(message.data.error || 'Erro na execução do passo'));
                        this.pendingSteps.delete(requestId);
                    }
                }
                break;
            case 'GenerateTextResult':
                // Resolve specific generation request
                if (this.pendingGenerations && message.data.request_id) {
                    const pending = this.pendingGenerations.get(message.data.request_id);
                    if (pending) {
                        pending.resolve(message.data);
                        this.pendingGenerations.delete(message.data.request_id);
                    }
                }
                break;
            case 'StepResult':
                // Resolve specific step request
                if (this.pendingSteps && message.data.request_id) {
                    const pending = this.pendingSteps.get(message.data.request_id);
                    if (pending) {
                        pending.resolve(message.data);
                        this.pendingSteps.delete(message.data.request_id);
                    }
                }
                break;
            case 'ModuleLoaded':
                this.handleModuleLoaded(message.data);
                break;
            case 'ClientRegistered':
                this.handleClientRegistered(message.data);
                break;
            // Compatibilidade com mensagens antigas
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
            case 'token_generated':
                this.handleTokenGenerated(message.data);
                break;
            case 'generation_start':
                this.handleGenerationStart(message.data);
                break;
            case 'generation_complete':
                this.handleGenerationComplete(message.data);
                break;
            case 'generation_error':
                this.handleGenerationError(message.data);
                break;
            default:
                console.log('📨 Mensagem não reconhecida:', message);
                this.addLog('WEBSOCKET', `Tipo de mensagem desconhecido: ${message.type}`);
        }
    }
    
    /**
     * 📦 Manipula carregamento de módulo
     */
    handleModuleLoaded(data) {
        const { module_name, capabilities, parameters } = data;
        
        this.modules.set(module_name, {
            name: module_name,
            status: 'loaded',
            capabilities: capabilities || [],
            parameters: parameters || {}
        });
        
        this.showToast('success', 'Módulo Carregado', `${module_name} está pronto para uso`, 3000);
        this.addLog('MODULE', `Módulo ${module_name} carregado com sucesso`);
        
        // Atualiza interface
        this.populateModuleSelect();
    }
    
    /**
     * 👤 Manipula registro de cliente
     */
    handleClientRegistered(data) {
        const { client_id, session_id } = data;
        this.clientId = client_id;
        this.sessionId = session_id;
        
        this.addLog('CLIENT', `Cliente registrado: ${client_id}`);
        this.showToast('info', 'Cliente Registrado', 'Conectado ao sistema de inferência', 2000);
    }

    /**
     * 🎯 Manipula início da geração
     */
    handleGenerationStart(data) {
        this.showToast('info', 'Geração Iniciada', `Gerando ${data.max_tokens || 20} tokens...`, 3000);
        this.addLog('GENERATION', 'Início da geração de tokens');
        this.progressData.startTime = Date.now();
        this.updateDetailedProgress(0, data.max_tokens || 20);
    }

    /**
     * 🔤 Manipula token gerado
     */
    handleTokenGenerated(data) {
        const { token, position, total, text } = data;
        this.updateDetailedProgress(position, total, [token]);
        this.addLog('TOKEN', `Token ${position}/${total}: "${token}"`);
        
        // Atualiza métricas em tempo real
        if (position % 5 === 0) { // A cada 5 tokens
            const speed = this.progressData.speed;
            this.updateRealTimeMetrics({
                tokens_generated: position,
                generation_speed: speed,
                perplexity: Math.random() * 2 + 1 // Simulado
            });
        }
    }

    /**
     * ✅ Manipula conclusão da geração
     */
    handleGenerationComplete(data) {
        const { total_tokens, elapsed_time, final_text } = data;
        this.showToast('success', 'Geração Completa', `${total_tokens} tokens gerados em ${elapsed_time}ms`, 5000);
        this.addLog('SUCCESS', `Geração concluída: ${total_tokens} tokens em ${elapsed_time}ms`);
        this.updateDetailedProgress(total_tokens, total_tokens);
    }

    /**
     * ❌ Manipula erro na geração
     */
    handleGenerationError(data) {
        const { error, position } = data;
        this.showToast('error', 'Erro na Geração', `Erro no token ${position}: ${error}`, 7000);
        this.addLog('ERROR', `Erro na geração: ${error}`);
    }

    /**
     * 🎲 Simula progresso de geração (para demonstração)
     */
    simulateTokenGeneration(maxTokens = 20) {
        let currentToken = 0;
        const tokens = ['O', ' gato', ' subiu', ' no', ' telhado', ' para', ' ver', ' a', ' lua', ' brilhar', ' na', ' noite', ' estrelada', ' de', ' verão', '.', ' Fim', ' da', ' história', '.'];
        
        const interval = setInterval(() => {
            if (currentToken >= maxTokens) {
                clearInterval(interval);
                this.handleGenerationComplete({
                    total_tokens: maxTokens,
                    elapsed_time: Date.now() - this.progressData.startTime,
                    final_text: tokens.slice(0, maxTokens).join('')
                });
                return;
            }
            
            this.handleTokenGenerated({
                token: tokens[currentToken] || `token_${currentToken}`,
                position: currentToken + 1,
                total: maxTokens,
                text: tokens.slice(0, currentToken + 1).join('')
            });
            
            currentToken++;
        }, 200 + Math.random() * 300); // Velocidade variável
    }

    /**
     * 📊 Carrega módulos disponíveis
     */
    async loadModules() {
        try {
            // Tenta carregar módulos do backend Rust
            const response = await fetch('/api/v1/modules', {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            if (response.ok) {
                const modules = await response.json();
                
                modules.forEach(module => {
                    this.modules.set(module.name, {
                        name: module.name,
                        display_name: module.display_name || module.name,
                        icon: module.icon || '📦',
                        status: module.status || 'idle',
                        capabilities: module.capabilities || [],
                        parameters: module.parameters || {}
                    });
                });
                
                console.log('📚 Módulos carregados do backend:', this.modules.size);
                this.addLog('MODULES', `${this.modules.size} módulos carregados`);
            } else {
                throw new Error(`Erro ao carregar módulos: ${response.status}`);
            }
        } catch (error) {
            console.warn('⚠️ Usando módulos mock devido ao erro:', error);
            
            // Fallback para dados mock
            const mockModules = [
                { name: 'tokenization', display_name: 'Tokenização', icon: '🔤' },
                { name: 'embedding', display_name: 'Embeddings', icon: '🧮' },
                { name: 'attention', display_name: 'Atenção', icon: '🎯' },
                { name: 'transformer', display_name: 'Transformer', icon: '🤖' },
                { name: 'generation', display_name: 'Geração', icon: '✨' }
            ];
            
            mockModules.forEach(module => {
                this.modules.set(module.name, {
                    name: module.name,
                    display_name: module.display_name,
                    icon: module.icon,
                    status: 'idle',
                    parameters: {
                        temperature: { value: 0.7, min: 0.1, max: 2.0, type: 'range', step: 0.1 },
                        max_tokens: { value: 100, min: 1, max: 1000, type: 'number' },
                        top_p: { value: 0.9, min: 0.1, max: 1.0, type: 'range', step: 0.1 },
                        frequency_penalty: { value: 0.0, min: -2.0, max: 2.0, type: 'range', step: 0.1 }
                    }
                });
            });
            
            console.log('📚 Módulos mock carregados:', this.modules.size);
            this.addLog('MODULES', `${this.modules.size} módulos mock carregados`);
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
        
        if (document.body) {
            document.body.appendChild(panel);
            this.populateModuleSelect();
        } else {
            console.warn('Document body not ready for control panel');
        }
    }

    /**
     * 📋 Popula select de módulos
     */
    populateModuleSelect() {
        const select = document.getElementById('active-module');
        if (!select) return;
        
        // Limpa opções existentes (exceto a primeira)
        while (select.children.length > 1) {
            select.removeChild(select.lastChild);
        }
        
        this.modules.forEach((module, name) => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = `${module.icon || '📦'} ${module.display_name || name}`;
            select.appendChild(option);
        });
        
        this.addLog('UI', `Select atualizado com ${this.modules.size} módulos`);
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
            } else if (e.target.id === 'testNotifications') {
                this.testNotificationSystem();
            } else if (e.target.id === 'toggleSound') {
                this.handleToggleSound();
            }
        });
    }

    /**
     * 🧪 Testa o sistema de notificações
     */
    testNotificationSystem() {
        this.addLog('TEST', 'Iniciando teste do sistema de notificações');
        
        // Sequência de testes
        setTimeout(() => {
            this.showToast('info', 'Teste Iniciado', 'Testando sistema de notificações...', 3000);
        }, 100);
        
        setTimeout(() => {
            this.showToast('success', 'Sucesso', 'Esta é uma notificação de sucesso!', 4000);
        }, 1500);
        
        setTimeout(() => {
            this.showToast('warning', 'Atenção', 'Esta é uma notificação de aviso.', 4000);
        }, 3000);
        
        setTimeout(() => {
            this.showToast('error', 'Erro', 'Esta é uma notificação de erro.', 5000);
        }, 4500);
        
        setTimeout(() => {
            const loadingId = this.showToast('loading', 'Carregando', 'Processando dados...', 0);
            setTimeout(() => {
                this.closeToast(loadingId);
                this.showToast('success', 'Concluído', 'Processamento finalizado!', 3000);
            }, 3000);
        }, 6000);
        
        // Testa progresso detalhado
        setTimeout(() => {
            this.addLog('TEST', 'Testando progresso detalhado');
            this.handleGenerationStart({ max_tokens: 15 });
            this.simulateTokenGeneration(15);
        }, 10000);
    }

    /**
     * 🔊 Gerencia o botão de alternar som
     */
    handleToggleSound() {
        const isEnabled = this.toggleSound();
        const button = document.getElementById('toggleSoundLabel');
        
        if (button) {
            button.textContent = isEnabled ? '🔊 Som Ativado' : '🔇 Som Desativado';
        }
        
        this.showToast('info', 'Configuração de Áudio', 
            `Som ${isEnabled ? 'ativado' : 'desativado'}`, 2000);
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
            const response = await fetch(`/parameters/${moduleName}`);
            const parameters = await response.json();
            
            this.renderParameterControls(parameters);
        } catch (error) {
            console.error('🚨 Erro ao carregar parâmetros:', error);
        }
    }

    /**
     * 🔤 Tokeniza texto usando o backend
     */
    async tokenizeText(text) {
        try {
            // Envia via WebSocket para tokenização em tempo real
            if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'ParameterUpdate',
                    data: {
                        module: 'tokenization',
                        parameter: 'input_text',
                        value: text
                    }
                };
                this.ws.send(JSON.stringify(message));
                
                // Simula tokenização enquanto aguarda resposta do backend
                return this.simulateTokenization(text);
            }
            
            // Fallback para API REST se WebSocket não estiver disponível
            const response = await fetch('/api/v1/modules/tokenization/parameters', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    module: 'tokenization',
                    parameters: {
                        input_text: text,
                        max_tokens: 512
                    },
                    updated_at: Date.now()
                })
            });

            if (!response.ok) {
                throw new Error(`Erro na tokenização: ${response.status}`);
            }

            // Por enquanto, simula a tokenização até o backend estar completo
            return this.simulateTokenization(text);
        } catch (error) {
            console.error('❌ Erro na tokenização:', error);
            this.showToast('error', 'Erro', 'Falha na tokenização do texto', 5000);
            return this.simulateTokenization(text);
        }
    }
    
    /**
     * 🎯 Simula tokenização para demonstração
     */
    simulateTokenization(text) {
        if (!text || text.trim() === '') return [];
        
        // Tokenização simples por palavras e pontuação
        const words = text.split(/\s+/);
        const tokens = [];
        let tokenId = 0;
        
        words.forEach(word => {
            // Separa pontuação
            const matches = word.match(/\w+|[^\w\s]/g) || [];
            matches.forEach(match => {
                tokens.push({
                    id: tokenId++,
                    text: match,
                    type: /\w+/.test(match) ? 'word' : 'punctuation'
                });
            });
        });
        
        return tokens;
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
            // Prioriza WebSocket para atualizações em tempo real
            if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                this.sendMessage({
                    type: 'ParameterUpdate',
                    data: {
                        module: moduleName,
                        parameter: paramKey,
                        value: value,
                        timestamp: Date.now(),
                        client_id: this.clientId
                    }
                });
            } else {
                // Fallback para API REST
                await fetch(`/api/v1/modules/${moduleName}/parameters`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        module: moduleName,
                        parameters: {
                            [paramKey]: value
                        },
                        updated_at: Date.now()
                    })
                });
            }
            
            this.addLog('PARAM', `${moduleName}.${paramKey} = ${value}`);
            
        } catch (error) {
            console.error('🚨 Erro ao atualizar parâmetro:', error);
            this.showToast('error', 'Erro', `Falha ao atualizar ${paramKey}`, 3000);
        }
    }

    /**
     * ▶️ Executa demonstração
     */
    async runDemo() {
        const moduleName = document.getElementById('active-module')?.value;
        if (!moduleName) {
            this.showToast('warning', 'Módulo Necessário', 'Selecione um módulo primeiro', 3000);
            return;
        }
        
        // Notificação de início
        const loadingToastId = this.showToast('loading', 'Carregando inferência...', 'Preparando modelo para inferência...', 0);
        this.addLog('DEMO', `Iniciando demonstração do módulo ${moduleName}`);
        
        // Reset do progresso
        this.progressData.startTime = null;
        this.updateDetailedProgress(0, 20); // Assumindo 20 tokens como padrão
        
        try {
            // Simula geração em tempo real
            this.handleGenerationStart({ max_tokens: 20 });
            
            // Integração real com o backend Rust
            if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                // Envia comando de execução via WebSocket
                this.sendMessage({
                    type: 'RunInference',
                    data: {
                        module: moduleName,
                        parameters: this.getCurrentParameters(),
                        max_tokens: 20,
                        temperature: 0.7
                    }
                });
                
                // Simula enquanto aguarda resposta real
                this.simulateTokenGeneration(20);
            } else {
                // Fallback para API REST
                const response = await fetch('/api/v1/inference/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        module: moduleName,
                        parameters: this.getCurrentParameters(),
                        max_tokens: 20,
                        temperature: 0.7,
                        timestamp: Date.now()
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`Erro na inferência: ${response.status}`);
                }
                
                const result = await response.json();
                console.log('✅ Demo executado:', result);
                
                // Processa resultado da inferência
                this.processInferenceResult(result);
            }
            
            // Fecha toast de loading
            this.closeToast(loadingToastId);
            
        } catch (error) {
            console.error('🚨 Erro ao executar demo:', error);
            
            // Fecha toast de loading
            this.closeToast(loadingToastId);
            
            // Notificação de erro
            this.showToast('error', 'Erro na Geração', `Falha ao executar demo: ${error.message}`, 7000);
            this.addLog('ERROR', `Erro no demo: ${error.message}`);
        }
    }
    
    /**
     * 📊 Obtém parâmetros atuais do módulo
     */
    getCurrentParameters() {
        const parameters = {};
        const inputs = document.querySelectorAll('.param-input');
        
        inputs.forEach(input => {
            const key = input.dataset.paramKey;
            let value = input.value;
            
            if (input.type === 'checkbox') {
                value = input.checked;
            } else if (input.type === 'number' || input.type === 'range') {
                value = parseFloat(value);
            }
            
            parameters[key] = value;
        });
        
        return parameters;
    }
    
    /**
      * 🔄 Processa resultado da inferência
      */
     processInferenceResult(result) {
         if (result.tokens) {
             result.tokens.forEach((token, index) => {
                 setTimeout(() => {
                     this.handleTokenGenerated({
                         token: token.text || token,
                         position: index + 1,
                         total: result.tokens.length,
                         confidence: token.confidence || 1.0
                     });
                 }, index * 100);
             });
             
             setTimeout(() => {
                 this.handleGenerationComplete({
                     total_tokens: result.tokens.length,
                     elapsed_time: result.elapsed_time || 0,
                     final_text: result.generated_text || result.tokens.map(t => t.text || t).join('')
                 });
             }, result.tokens.length * 100 + 500);
         }
         
         // Exibe resultado na interface
         this.displayDemoResult(result.module || 'unknown', result);
     }
     
     /**
      * ✨ Gera texto usando o backend
      */
     async generateText(options) {
         try {
             if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                 // Envia via WebSocket
                 this.sendMessage({
                     type: 'GenerateText',
                     data: {
                         prompt: options.prompt,
                         max_tokens: options.maxTokens || 20,
                         temperature: options.temperature || 0.7,
                         top_p: options.topP || 0.9,
                         top_k: options.topK || 50,
                         timestamp: Date.now(),
                         client_id: this.clientId
                     }
                 });
                 
                 // Retorna promessa que será resolvida quando receber resposta
                 return new Promise((resolve, reject) => {
                     this.pendingGenerations = this.pendingGenerations || new Map();
                     const requestId = `gen_${Date.now()}`;
                     this.pendingGenerations.set(requestId, { resolve, reject });
                     
                     // Timeout após 30 segundos
                     setTimeout(() => {
                         if (this.pendingGenerations.has(requestId)) {
                             this.pendingGenerations.delete(requestId);
                             reject(new Error('Timeout na geração de texto'));
                         }
                     }, 30000);
                 });
             } else {
                 // Fallback para API REST
                 const response = await fetch('/api/v1/inference/generate', {
                     method: 'POST',
                     headers: {
                         'Content-Type': 'application/json',
                     },
                     body: JSON.stringify({
                         prompt: options.prompt,
                         max_tokens: options.maxTokens || 20,
                         temperature: options.temperature || 0.7,
                         top_p: options.topP || 0.9,
                         top_k: options.topK || 50
                     })
                 });
                 
                 if (!response.ok) {
                     throw new Error(`Erro na geração: ${response.status}`);
                 }
                 
                 return await response.json();
             }
         } catch (error) {
             console.error('❌ Erro na geração de texto:', error);
             throw error;
         }
     }
     
     /**
      * 👣 Executa um passo específico da inferência
      */
     async executeStep(options) {
         try {
             if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                 // Envia via WebSocket
                 this.sendMessage({
                     type: 'ExecuteStep',
                     data: {
                         step: options.step,
                         prompt: options.prompt,
                         max_tokens: options.maxTokens || 1,
                         temperature: options.temperature || 0.7,
                         timestamp: Date.now(),
                         client_id: this.clientId
                     }
                 });
                 
                 // Retorna promessa que será resolvida quando receber resposta
                 return new Promise((resolve, reject) => {
                     this.pendingSteps = this.pendingSteps || new Map();
                     const requestId = `step_${options.step}_${Date.now()}`;
                     this.pendingSteps.set(requestId, { resolve, reject });
                     
                     // Timeout após 10 segundos
                     setTimeout(() => {
                         if (this.pendingSteps.has(requestId)) {
                             this.pendingSteps.delete(requestId);
                             reject(new Error(`Timeout no passo ${options.step}`));
                         }
                     }, 10000);
                 });
             } else {
                 // Fallback para API REST
                 const response = await fetch('/api/v1/inference/step', {
                     method: 'POST',
                     headers: {
                         'Content-Type': 'application/json',
                     },
                     body: JSON.stringify({
                         step: options.step,
                         prompt: options.prompt,
                         max_tokens: options.maxTokens || 1,
                         temperature: options.temperature || 0.7
                     })
                 });
                 
                 if (!response.ok) {
                     throw new Error(`Erro no passo ${options.step}: ${response.status}`);
                 }
                 
                 return await response.json();
             }
         } catch (error) {
             console.error(`❌ Erro no passo ${options.step}:`, error);
             throw error;
         }
     }

    /**
     * 🔄 Reset de parâmetros
     */
    async resetParameters() {
        const moduleName = document.getElementById('active-module')?.value;
        if (!moduleName) return;
        
        try {
            // Envia comando de reset via WebSocket
            if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                this.sendMessage({
                    type: 'ResetParameters',
                    data: {
                        module: moduleName,
                        timestamp: Date.now(),
                        client_id: this.clientId
                    }
                });
            } else {
                // Fallback para API REST
                await fetch(`/api/v1/modules/${moduleName}/parameters/reset`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        module: moduleName,
                        reset_to_defaults: true,
                        timestamp: Date.now()
                    })
                });
            }
            
            // Recarrega parâmetros
            this.onModuleChange(moduleName);
            
            this.showToast('success', 'Reset Completo', 'Parâmetros restaurados aos valores padrão', 3000);
            this.addLog('RESET', `Parâmetros do módulo ${moduleName} resetados`);
            
        } catch (error) {
            console.error('🚨 Erro ao resetar parâmetros:', error);
            this.showToast('error', 'Erro', 'Falha ao resetar parâmetros', 3000);
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

// 🚀 Inicializa apenas uma vez
if (!window.webDemoIntegration) {
    window.webDemoIntegration = new WebDemoIntegration();
    window.WebDemoIntegration = WebDemoIntegration;
}

/**
 * 🔧 Função utilitária para debounce
 * Evita execuções excessivas de funções em eventos frequentes
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Comentários educacionais
// Nota: WebSocket fornece comunicação full-duplex de baixa latência, ideal para apps interativos
// Aqui, implementamos reconexão exponencial para resiliência