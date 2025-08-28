/**
 * 🎨 Sistema de Interface de Inferência Aprimorada
 * Gerencia estados visuais, feedback e experiência do usuário
 */

class InferenceUI {
    constructor() {
        this.isGenerating = false;
        this.currentStep = 0;
        this.totalSteps = 5;
        this.generationStartTime = null;
        this.tokensGenerated = 0;
        this.currentText = '';
        this.currentTokens = [];
        this.flowSteps = ['tokenize', 'embed', 'attention', 'forward', 'sample', 'decode'];
        
        this.init();
    }

    /**
     * 🚀 Inicializa a interface de inferência
     */
    init() {
        this.setupEventListeners();
        this.setupValidation();
        this.updateUI();
    }

    /**
     * 🎯 Configura event listeners específicos para inferência
     */
    setupEventListeners() {
        // Botão Gerar Texto
        document.getElementById('generateText')?.addEventListener('click', () => {
            this.handleGenerateText();
        });

        // Botão Gerar Passo a Passo
        document.getElementById('stepGenerate')?.addEventListener('click', () => {
            this.handleStepGenerate();
        });

        // Botão Reset
        document.getElementById('resetGeneration')?.addEventListener('click', () => {
            this.handleReset();
        });

        // Validação em tempo real do prompt
        document.getElementById('promptInput')?.addEventListener('input', async (e) => {
            this.validatePrompt(e.target.value);
            await this.updateTokenization(e.target.value);
        });

        // Atualização de parâmetros em tempo real
        const sliders = ['maxTokens', 'temperature', 'topP', 'topK', 'beamWidth'];
        sliders.forEach(sliderId => {
            const slider = document.getElementById(sliderId);
            if (slider) {
                slider.addEventListener('input', () => this.updateSliderValue(sliderId));
            }
        });
    }

    /**
     * ✅ Configura validação de entrada
     */
    setupValidation() {
        const promptInput = document.getElementById('promptInput');
        if (promptInput) {
            promptInput.setAttribute('maxlength', '500');
            promptInput.setAttribute('required', 'true');
        }
    }

    /**
     * 🔤 Atualiza a tokenização em tempo real
     */
    async updateTokenization(text) {
        if (!text || text.trim().length === 0) {
            this.clearTokenization();
            return;
        }

        // Atualiza o texto original na interface
        const originalTextElement = document.getElementById('originalText');
        if (originalTextElement) {
            originalTextElement.textContent = text;
        }

        // Tokeniza o texto (com integração real ou simulação)
        const tokens = await this.tokenizeText(text);
        this.displayTokens(tokens);
        this.currentTokens = tokens;
    }

    /**
     * 🔧 Tokeniza o texto de entrada
     */
    async tokenizeText(text) {
        if (!text || text.trim() === '') return [];
        
        try {
            // Tenta usar a integração real se disponível
            if (window.integration && window.integration.isConnected) {
                const tokens = await window.integration.tokenizeText(text);
                if (tokens && tokens.length > 0) {
                    return tokens;
                }
            }
            
            // Fallback para simulação local
            return this.simulateTokenization(text);
        } catch (error) {
            console.error('❌ Erro na tokenização:', error);
            return this.simulateTokenization(text);
        }
    }
    
    /**
     * 🎭 Simulação local de tokenização
     */
    simulateTokenization(text) {
        // Tokenização simples para demonstração
        const words = text.toLowerCase()
            .replace(/[^\w\s]/g, ' $& ')
            .split(/\s+/)
            .filter(word => word.trim().length > 0);
        
        return words.map((word, index) => ({
            id: index,
            text: word,
            tokenId: Math.floor(Math.random() * 50000) + 1000
        }));
    }

    /**
     * 🎨 Exibe os tokens na interface
     */
    displayTokens(tokens) {
        const tokensContainer = document.getElementById('tokenizedOutput');
        if (!tokensContainer) return;

        tokensContainer.innerHTML = '';
        
        tokens.forEach((token, index) => {
            const tokenElement = document.createElement('div');
            tokenElement.className = 'token-item';
            tokenElement.innerHTML = `
                <span class="token-text">${token.text}</span>
                <span class="token-id">#${token.tokenId}</span>
            `;
            
            // Animação de aparição
            setTimeout(() => {
                tokenElement.style.opacity = '1';
                tokenElement.style.transform = 'translateY(0)';
            }, index * 50);
            
            tokensContainer.appendChild(tokenElement);
        });
    }

    /**
     * 🧹 Limpa a tokenização
     */
    clearTokenization() {
        const tokensContainer = document.getElementById('tokenizedOutput');
        if (tokensContainer) {
            tokensContainer.innerHTML = '';
        }
        
        const originalTextElement = document.getElementById('originalText');
        if (originalTextElement) {
            originalTextElement.textContent = '';
        }
        
        this.currentTokens = [];
    }

    /**
     * 🌊 Inicia a visualização do fluxo de inferência
     */
    async startInferenceFlow() {
        this.resetInferenceFlow();
        
        for (let i = 0; i < this.flowSteps.length; i++) {
            await this.animateFlowStep(this.flowSteps[i], i);
            await this.delay(800);
        }
    }

    /**
     * 🎬 Anima uma etapa do fluxo
     */
    async animateFlowStep(stepName, stepIndex) {
        const stepElement = document.querySelector(`[data-step="${stepName}"]`);
        if (!stepElement) return;

        // Criar efeito de transição da etapa anterior
        if (stepIndex > 0) {
            const prevStepName = this.flowSteps[stepIndex - 1];
            const prevStepElement = document.querySelector(`[data-step="${prevStepName}"]`);
            if (prevStepElement) {
                this.createTransitionEffect(prevStepElement, stepElement);
            }
        }

        // Ativa a etapa atual
        stepElement.classList.add('active');
        
        // Destacar conexões ativas
        this.highlightActiveConnections();
        
        // Anima os tokens na etapa
        await this.animateTokensInStep(stepName);
        
        // Marca como concluída após um tempo
        setTimeout(() => {
            stepElement.classList.remove('active');
            stepElement.classList.add('completed');
        }, 1500);
    }

    /**
     * 🎭 Anima tokens em uma etapa específica
     */
    async animateTokensInStep(stepName) {
        const tokensContainer = document.getElementById(`${stepName}Tokens`);
        if (!tokensContainer || this.currentTokens.length === 0) return;

        tokensContainer.innerHTML = '';
        
        for (let i = 0; i < this.currentTokens.length; i++) {
            const token = this.currentTokens[i];
            const tokenElement = document.createElement('div');
            tokenElement.className = 'flow-token flowing';
            tokenElement.textContent = this.getTokenRepresentation(token, stepName);
            
            tokensContainer.appendChild(tokenElement);
            
            // Animação de entrada
            setTimeout(() => {
                tokenElement.classList.add('highlighted');
            }, i * 100);
            
            // Add transition animation
            setTimeout(() => {
                tokenElement.classList.add('transitioning');
                tokenElement.classList.remove('highlighted');
            }, i * 100 + 500);
            
            // Remove animations after completion
            setTimeout(() => {
                tokenElement.classList.remove('transitioning', 'flowing');
            }, i * 100 + 2500);
            
            await this.delay(50);
        }
    }

    /**
     * 🔄 Obtém representação do token para cada etapa
     */
    getTokenRepresentation(token, stepName) {
        switch (stepName) {
            case 'tokenization':
                return token.text;
            case 'embeddings':
                return `[${token.tokenId}]`;
            case 'attention':
                return `α${token.id}`;
            case 'forward':
                return `h${token.id}`;
            case 'sampling':
                return `p${token.id}`;
            case 'decoding':
                return token.text;
            default:
                return token.text;
        }
    }

    /**
     * 🔄 Reseta o fluxo de inferência
     */
    resetInferenceFlow() {
        this.flowSteps.forEach(stepName => {
            const stepElement = document.querySelector(`[data-step="${stepName}"]`);
            if (stepElement) {
                stepElement.classList.remove('active', 'completed');
                const tokensContainer = document.getElementById(`${stepName}Tokens`);
                if (tokensContainer) {
                    tokensContainer.innerHTML = '';
                }
            }
        });
    }

    /**
     * ✨ Cria efeitos de partículas durante transições
     */
    createTransitionEffect(fromStep, toStep) {
        const fromRect = fromStep.getBoundingClientRect();
        const toRect = toStep.getBoundingClientRect();
        
        // Criar partículas que se movem entre as etapas
        for (let i = 0; i < 5; i++) {
            const particle = document.createElement('div');
            particle.className = 'transition-particle';
            particle.style.cssText = `
                position: fixed;
                width: 4px;
                height: 4px;
                background: rgba(59, 130, 246, 0.8);
                border-radius: 50%;
                pointer-events: none;
                z-index: 1000;
                left: ${fromRect.left + fromRect.width / 2}px;
                top: ${fromRect.bottom}px;
                transition: all 1s ease-in-out;
            `;
            
            document.body.appendChild(particle);
            
            // Animar partícula para a próxima etapa
            setTimeout(() => {
                particle.style.left = `${toRect.left + toRect.width / 2 + (Math.random() - 0.5) * 20}px`;
                particle.style.top = `${toRect.top}px`;
                particle.style.opacity = '0';
            }, i * 100);
            
            // Remover partícula após animação
            setTimeout(() => {
                document.body.removeChild(particle);
            }, 1200 + i * 100);
        }
    }
    
    /**
     * 🔗 Destaca conexões ativas entre etapas
     */
    highlightActiveConnections() {
         this.flowSteps.forEach((stepName, index) => {
             const stepElement = document.querySelector(`[data-step="${stepName}"]`);
             const connection = stepElement?.querySelector('.flow-connection');
             if (connection && stepElement?.classList.contains('active')) {
                 connection.style.opacity = '1';
                 connection.style.animation = 'connectionPulse 2s ease-in-out infinite';
             } else if (connection) {
                 connection.style.opacity = '0';
                 connection.style.animation = 'none';
             }
         });
     }

    /**
     * 🎬 Manipula geração de texto completa
     */
    async handleGenerateText() {
        if (this.isGenerating) {
            this.showStatusMessage('⚠️ Geração já em andamento', 'warning');
            return;
        }

        const prompt = this.getPromptValue();
        if (!this.validatePrompt(prompt)) {
            return;
        }

        try {
            await this.startGeneration('full');
            await this.startInferenceFlow();
            
            // Tenta usar integração real com o backend
            if (window.integration && window.integration.isConnected) {
                await this.runRealInference(prompt, 'complete');
            } else {
                await this.simulateTextGeneration();
            }
            
            this.completeGeneration();
        } catch (error) {
            this.handleGenerationError(error);
        }
    }

    /**
     * 👣 Manipula geração passo a passo
     */
    async handleStepGenerate() {
        if (this.isGenerating) {
            this.showStatusMessage('⚠️ Geração já em andamento', 'warning');
            return;
        }

        const prompt = this.getPromptValue();
        if (!this.validatePrompt(prompt)) {
            return;
        }

        try {
            await this.startGeneration('step');
            await this.startInferenceFlow();
            
            // Tenta usar integração real com o backend
            if (window.integration && window.integration.isConnected) {
                await this.runRealInference(prompt, 'step');
            } else {
                await this.simulateStepByStepGeneration();
            }
            
            this.completeGeneration();
        } catch (error) {
            this.handleGenerationError(error);
        }
    }

    /**
     * 🔄 Manipula reset da geração
     */
    handleReset() {
        this.isGenerating = false;
        this.currentStep = 0;
        this.tokensGenerated = 0;
        this.currentText = '';
        this.generationStartTime = null;
        
        // Reset da interface
        this.hideGenerationStatus();
        this.resetButtons();
        this.clearGeneratedText();
        this.resetMetrics();
        this.resetSteps();
        this.clearTokenization();
        this.resetInferenceFlow();
        
        this.showStatusMessage('✅ Interface resetada', 'success');
        setTimeout(() => this.hideStatusMessage(), 2000);
    }

    /**
     * 🚀 Inicia processo de geração
     */
    async startGeneration(mode) {
        this.isGenerating = true;
        this.generationStartTime = Date.now();
        this.currentStep = 0;
        
        // Atualiza interface
        this.showGenerationStatus();
        this.setButtonLoading('generateText', mode === 'full');
        this.setButtonLoading('stepGenerate', mode === 'step');
        this.disableButton('resetGeneration', false);
        
        // Mostra status inicial
        this.showStatusMessage('🚀 Iniciando geração...', 'info');
        this.updateProgress(0);
        
        // Simula preparação
        await this.delay(500);
    }

    /**
     * 📝 Simula geração de texto completa
     */
    async simulateTextGeneration() {
        const maxTokens = parseInt(document.getElementById('maxTokens')?.value || 20);
        const steps = ['tokenize', 'embed', 'forward', 'sample', 'decode'];
        
        for (let i = 0; i < steps.length; i++) {
            this.currentStep = i + 1;
            this.highlightStep(steps[i]);
            this.showStatusMessage(`⚙️ ${this.getStepDescription(steps[i])}`, 'info');
            this.updateProgress((i + 1) / steps.length * 50); // 50% para processamento
            
            await this.delay(800);
        }
        
        // Simula geração token por token
        this.showStatusMessage('✍️ Gerando texto...', 'info');
        
        for (let token = 0; token < maxTokens; token++) {
            await this.delay(200);
            this.addToken(this.generateRandomToken());
            this.tokensGenerated++;
            this.updateProgress(50 + (token + 1) / maxTokens * 50); // 50% restante para tokens
            this.updateMetrics();
        }
    }

    /**
     * 👣 Simula geração passo a passo
     */
    async simulateStepByStepGeneration() {
        const steps = ['tokenize', 'embed', 'forward', 'sample', 'decode'];
        
        for (let i = 0; i < steps.length; i++) {
            this.currentStep = i + 1;
            this.highlightStep(steps[i]);
            this.showStatusMessage(`🔍 ${this.getStepDescription(steps[i])} (Passo ${i + 1}/${steps.length})`, 'info');
            this.updateProgress((i + 1) / steps.length * 100);
            
            // Pausa mais longa para visualização
            await this.delay(1500);
            
            // Adiciona token após cada passo completo
            if (i === steps.length - 1) {
                this.addToken(this.generateRandomToken());
                this.tokensGenerated++;
                this.updateMetrics();
            }
        }
    }

    /**
     * ✅ Completa processo de geração
     */
    completeGeneration() {
        this.isGenerating = false;
        
        // Atualiza interface
        this.setButtonSuccess('generateText');
        this.setButtonSuccess('stepGenerate');
        this.showStatusMessage('🎉 Geração concluída com sucesso!', 'success');
        this.updateProgress(100);
        
        // Reset automático após 3 segundos
        setTimeout(() => {
            this.resetButtons();
            this.hideGenerationStatus();
        }, 3000);
    }

    /**
     * ❌ Manipula erros de geração
     */
    handleGenerationError(error) {
        this.isGenerating = false;
        
        console.error('Erro na geração:', error);
        
        this.setButtonError('generateText');
        this.setButtonError('stepGenerate');
        this.showStatusMessage(`❌ Erro na geração: ${error.message}`, 'error');
        
        // Reset automático após 5 segundos
        setTimeout(() => {
            this.resetButtons();
            this.hideGenerationStatus();
        }, 5000);
    }

    /**
     * ✅ Valida prompt de entrada
     */
    validatePrompt(prompt) {
        if (!prompt || prompt.trim().length === 0) {
            this.showStatusMessage('⚠️ Por favor, digite um prompt', 'warning');
            return false;
        }
        
        if (prompt.length < 3) {
            this.showStatusMessage('⚠️ Prompt muito curto (mínimo 3 caracteres)', 'warning');
            return false;
        }
        
        if (prompt.length > 500) {
            this.showStatusMessage('⚠️ Prompt muito longo (máximo 500 caracteres)', 'warning');
            return false;
        }
        
        return true;
    }

    /**
     * 📊 Atualiza valores dos sliders
     */
    updateSliderValue(sliderId) {
        const slider = document.getElementById(sliderId);
        const valueSpan = document.getElementById(sliderId + 'Value');
        
        if (!slider || !valueSpan) return;
        
        let displayValue = slider.value;
        
        switch (sliderId) {
            case 'temperature':
                displayValue = (slider.value / 100).toFixed(1);
                break;
            case 'topP':
                displayValue = (slider.value / 100).toFixed(1);
                break;
            case 'maxTokens':
                displayValue = `${slider.value} tokens`;
                break;
            case 'beamWidth':
                displayValue = `${slider.value} beams`;
                break;
        }
        
        valueSpan.textContent = displayValue;
    }

    // === Métodos de Interface ===

    showGenerationStatus() {
        const status = document.getElementById('generationStatus');
        if (status) status.style.display = 'block';
    }

    hideGenerationStatus() {
        const status = document.getElementById('generationStatus');
        if (status) status.style.display = 'none';
    }

    updateProgress(percentage) {
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
        }
    }

    showStatusMessage(message, type = 'info') {
        const statusMessage = document.getElementById('statusMessage');
        if (!statusMessage) return;
        
        statusMessage.className = `status-message status-${type}`;
        statusMessage.querySelector('span').textContent = message;
        statusMessage.style.display = 'flex';
    }

    hideStatusMessage() {
        const statusMessage = document.getElementById('statusMessage');
        if (statusMessage) statusMessage.style.display = 'none';
    }

    setButtonLoading(buttonId, isLoading) {
        const button = document.getElementById(buttonId);
        const label = document.getElementById(buttonId + 'Label');
        const spinner = document.getElementById(buttonId.replace('Text', 'Spinner').replace('Generate', 'Spinner'));
        
        if (!button) return;
        
        if (isLoading) {
            button.classList.add('btn-loading');
            button.disabled = true;
            if (label) label.style.display = 'none';
            if (spinner) spinner.style.display = 'block';
        } else {
            button.classList.remove('btn-loading');
            button.disabled = false;
            if (label) label.style.display = 'block';
            if (spinner) spinner.style.display = 'none';
        }
    }

    setButtonSuccess(buttonId) {
        const button = document.getElementById(buttonId);
        if (button) {
            button.classList.remove('btn-loading', 'btn-error');
            button.classList.add('btn-success');
            button.disabled = false;
        }
    }

    setButtonError(buttonId) {
        const button = document.getElementById(buttonId);
        if (button) {
            button.classList.remove('btn-loading', 'btn-success');
            button.classList.add('btn-error');
            button.disabled = false;
        }
    }

    resetButtons() {
        ['generateText', 'stepGenerate'].forEach(buttonId => {
            const button = document.getElementById(buttonId);
            const label = document.getElementById(buttonId + 'Label');
            const spinner = document.getElementById(buttonId.replace('Text', 'Spinner').replace('Generate', 'Spinner'));
            
            if (button) {
                button.classList.remove('btn-loading', 'btn-success', 'btn-error');
                button.disabled = false;
            }
            if (label) label.style.display = 'block';
            if (spinner) spinner.style.display = 'none';
        });
    }

    disableButton(buttonId, disabled) {
        const button = document.getElementById(buttonId);
        if (button) button.disabled = disabled;
    }

    highlightStep(stepName) {
        // Remove highlight de todos os passos
        document.querySelectorAll('.flow-step').forEach(step => {
            step.classList.remove('active');
        });
        
        // Adiciona highlight ao passo atual
        const currentStep = document.querySelector(`[data-step="${stepName}"]`);
        if (currentStep) {
            currentStep.classList.add('active');
        }
    }

    resetSteps() {
        document.querySelectorAll('.flow-step').forEach(step => {
            step.classList.remove('active', 'completed');
        });
    }

    addToken(token) {
        const generatedText = document.getElementById('generatedText');
        if (!generatedText) return;
        
        // Remove placeholder se existir
        if (generatedText.querySelector('.text-gray-400')) {
            generatedText.innerHTML = '';
        }
        
        // Adiciona novo token com animação
        const tokenSpan = document.createElement('span');
        tokenSpan.className = 'token generated';
        tokenSpan.textContent = token;
        
        generatedText.appendChild(tokenSpan);
        this.currentText += token;
        
        // Scroll para o final
        generatedText.scrollTop = generatedText.scrollHeight;
    }

    clearGeneratedText() {
        const generatedText = document.getElementById('generatedText');
        if (generatedText) {
            generatedText.innerHTML = '<span class="text-gray-400">O texto gerado aparecerá aqui...</span>';
        }
        this.currentText = '';
    }

    updateMetrics() {
        // Tokens gerados
        const tokensElement = document.getElementById('tokensGenerated');
        if (tokensElement) {
            tokensElement.textContent = this.tokensGenerated;
        }
        
        // Velocidade de geração
        const speedElement = document.getElementById('generationSpeed');
        if (speedElement && this.generationStartTime) {
            const elapsed = (Date.now() - this.generationStartTime) / 1000;
            const speed = elapsed > 0 ? (this.tokensGenerated / elapsed).toFixed(1) : '0.0';
            speedElement.textContent = speed;
        }
        
        // Perplexidade simulada
        const perplexityElement = document.getElementById('perplexity');
        if (perplexityElement) {
            const perplexity = (Math.random() * 50 + 10).toFixed(1);
            perplexityElement.textContent = perplexity;
        }
    }

    resetMetrics() {
        document.getElementById('tokensGenerated')?.textContent && (document.getElementById('tokensGenerated').textContent = '0');
        document.getElementById('generationSpeed')?.textContent && (document.getElementById('generationSpeed').textContent = '0');
        document.getElementById('perplexity')?.textContent && (document.getElementById('perplexity').textContent = '0.0');
    }

    // === Métodos Utilitários ===

    getPromptValue() {
        const promptInput = document.getElementById('promptInput');
        return promptInput ? promptInput.value.trim() : '';
    }

    getStepDescription(step) {
        const descriptions = {
            'tokenize': 'Convertendo texto em tokens',
            'embed': 'Gerando embeddings dos tokens',
            'forward': 'Processando através do modelo',
            'sample': 'Selecionando próximo token',
            'decode': 'Decodificando token para texto'
        };
        return descriptions[step] || 'Processando...';
    }

    generateRandomToken() {
        const tokens = [' mundo', ' belo', ' dia', ' sol', ' lua', ' estrela', ' casa', ' jardim', ' flor', ' árvore', ' rio', ' montanha', ' cidade', ' pessoa', ' criança', ' livro', ' música', ' arte', ' vida', ' amor'];
        return tokens[Math.floor(Math.random() * tokens.length)];
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * 🔗 Executa inferência real usando integração com backend
     */
    async runRealInference(prompt, mode) {
        try {
            const parameters = this.getInferenceParameters();
            
            if (mode === 'step') {
                await this.runStepByStepInference(prompt, parameters);
            } else {
                await this.runCompleteInference(prompt, parameters);
            }
        } catch (error) {
            console.error('❌ Erro na inferência real:', error);
            // Fallback para simulação
            if (mode === 'step') {
                await this.simulateStepByStepGeneration();
            } else {
                await this.simulateTextGeneration();
            }
        }
    }

    /**
     * 📊 Obtém parâmetros de inferência da interface
     */
    getInferenceParameters() {
        return {
            maxTokens: parseInt(document.getElementById('maxTokens')?.value || 20),
            temperature: parseFloat(document.getElementById('temperature')?.value || 70) / 100,
            topP: parseFloat(document.getElementById('topP')?.value || 90) / 100,
            topK: parseInt(document.getElementById('topK')?.value || 50),
            beamWidth: parseInt(document.getElementById('beamWidth')?.value || 1)
        };
    }

    /**
     * 🔄 Executa inferência completa com backend
     */
    async runCompleteInference(prompt, parameters) {
        this.showStatusMessage('🔗 Conectando com o modelo...', 'info');
        
        const response = await window.integration.generateText({
            prompt: prompt,
            ...parameters
        });
        
        if (response && response.text) {
            // Simula a adição token por token para visualização
            const tokens = response.text.split(' ');
            for (let i = 0; i < tokens.length; i++) {
                await this.delay(100);
                this.addToken(i === 0 ? tokens[i] : ' ' + tokens[i]);
                this.tokensGenerated++;
                this.updateProgress(50 + (i + 1) / tokens.length * 50);
                this.updateMetrics();
            }
        }
    }

    /**
     * 👣 Executa inferência passo a passo com backend
     */
    async runStepByStepInference(prompt, parameters) {
        this.showStatusMessage('🔗 Executando inferência passo a passo...', 'info');
        
        const steps = ['tokenize', 'embed', 'forward', 'sample', 'decode'];
        
        for (let i = 0; i < steps.length; i++) {
            this.currentStep = i + 1;
            this.highlightStep(steps[i]);
            this.showStatusMessage(`🔍 ${this.getStepDescription(steps[i])} (Passo ${i + 1}/${steps.length})`, 'info');
            this.updateProgress((i + 1) / steps.length * 100);
            
            // Chama o backend para cada passo
            try {
                const stepResult = await window.integration.executeStep({
                    step: steps[i],
                    prompt: prompt,
                    ...parameters
                });
                
                if (stepResult && stepResult.token && i === steps.length - 1) {
                    this.addToken(stepResult.token);
                    this.tokensGenerated++;
                    this.updateMetrics();
                }
            } catch (stepError) {
                console.warn(`⚠️ Erro no passo ${steps[i]}, continuando...`, stepError);
            }
            
            await this.delay(1500);
        }
    }

    updateUI() {
        // Inicializa valores dos sliders
        ['maxTokens', 'temperature', 'topP', 'topK', 'beamWidth'].forEach(sliderId => {
            this.updateSliderValue(sliderId);
        });
    }
}

// Inicialização automática
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.inferenceUI = new InferenceUI();
    });
} else {
    window.inferenceUI = new InferenceUI();
}

// Exporta para uso global
window.InferenceUI = InferenceUI;