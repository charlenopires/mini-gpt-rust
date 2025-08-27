/**
 * 📊 Visualizações Avançadas para Mini-GPT Rust
 * 
 * Este módulo implementa visualizações interativas usando Chart.js e D3.js
 * para demonstrar conceitos de transformers, atenção e embeddings.
 */

class AdvancedVisualizations {
    constructor() {
        this.charts = new Map();
        this.d3Visualizations = new Map();
        this.animationSpeed = 1000;
        this.colorSchemes = {
            attention: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            embeddings: ['#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', '#E17055'],
            performance: ['#00B894', '#00CEC9', '#0984E3', '#6C5CE7', '#A29BFE']
        };
        this.initializeVisualizations();
    }

    /**
     * Inicializa todas as visualizações
     */
    initializeVisualizations() {
        this.createAttentionHeatmap();
        this.createEmbeddingSpace();
        this.createTransformerArchitecture();
        this.createTokenFlowVisualization();
        this.createPerformanceMetrics();
        this.createLossLandscape();
    }

    /**
     * 🎯 Mapa de calor de atenção
     */
    createAttentionHeatmap() {
        const container = d3.select('#attention-heatmap')
            .style('width', '100%')
            .style('height', '400px');

        if (container.empty()) {
            console.warn('Container #attention-heatmap não encontrado');
            return;
        }

        const margin = { top: 50, right: 50, bottom: 50, left: 50 };
        const width = 500 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const svg = container.append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom);

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Dados de exemplo para atenção
        const tokens = ['[CLS]', 'O', 'gato', 'subiu', 'no', 'telhado', '[SEP]'];
        const attentionMatrix = this.generateAttentionMatrix(tokens.length);

        const colorScale = d3.scaleSequential(d3.interpolateBlues)
            .domain([0, 1]);

        const cellSize = Math.min(width, height) / tokens.length;

        // Criar células do mapa de calor
        const cells = g.selectAll('.attention-cell')
            .data(attentionMatrix.flat())
            .enter().append('rect')
            .attr('class', 'attention-cell')
            .attr('x', (d, i) => (i % tokens.length) * cellSize)
            .attr('y', (d, i) => Math.floor(i / tokens.length) * cellSize)
            .attr('width', cellSize - 1)
            .attr('height', cellSize - 1)
            .attr('fill', d => colorScale(d.value))
            .style('cursor', 'pointer')
            .on('mouseover', function(event, d) {
                d3.select(this).style('stroke', '#333').style('stroke-width', 2);
                
                // Tooltip
                const tooltip = d3.select('body').append('div')
                    .attr('class', 'attention-tooltip')
                    .style('position', 'absolute')
                    .style('background', 'rgba(0,0,0,0.8)')
                    .style('color', 'white')
                    .style('padding', '8px')
                    .style('border-radius', '4px')
                    .style('font-size', '12px')
                    .style('pointer-events', 'none')
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px')
                    .text(`${tokens[d.from]} → ${tokens[d.to]}: ${d.value.toFixed(3)}`);
            })
            .on('mouseout', function() {
                d3.select(this).style('stroke', 'none');
                d3.selectAll('.attention-tooltip').remove();
            });

        // Labels dos tokens
        g.selectAll('.token-label-x')
            .data(tokens)
            .enter().append('text')
            .attr('class', 'token-label-x')
            .attr('x', (d, i) => i * cellSize + cellSize / 2)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .text(d => d);

        g.selectAll('.token-label-y')
            .data(tokens)
            .enter().append('text')
            .attr('class', 'token-label-y')
            .attr('x', -10)
            .attr('y', (d, i) => i * cellSize + cellSize / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .style('font-size', '12px')
            .text(d => d);

        this.d3Visualizations.set('attention-heatmap', { svg, update: this.updateAttentionHeatmap.bind(this) });
    }

    /**
     * 🌌 Espaço de embeddings 3D
     */
    createEmbeddingSpace() {
        const container = d3.select('#embedding-space')
            .style('width', '100%')
            .style('height', '400px');

        if (container.empty()) {
            console.warn('Container #embedding-space não encontrado');
            return;
        }

        const width = 500;
        const height = 400;

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height);

        // Simular embeddings 2D (projeção de alta dimensão)
        const embeddings = this.generateEmbeddings(50);
        
        const xScale = d3.scaleLinear()
            .domain(d3.extent(embeddings, d => d.x))
            .range([50, width - 50]);

        const yScale = d3.scaleLinear()
            .domain(d3.extent(embeddings, d => d.y))
            .range([height - 50, 50]);

        const colorScale = d3.scaleOrdinal(this.colorSchemes.embeddings);

        // Pontos de embedding
        const points = svg.selectAll('.embedding-point')
            .data(embeddings)
            .enter().append('circle')
            .attr('class', 'embedding-point')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 0)
            .attr('fill', d => colorScale(d.cluster))
            .style('opacity', 0.7)
            .style('cursor', 'pointer')
            .on('mouseover', function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', 8)
                    .style('opacity', 1);
                
                // Mostrar palavra
                svg.append('text')
                    .attr('class', 'embedding-label')
                    .attr('x', xScale(d.x))
                    .attr('y', yScale(d.y) - 15)
                    .attr('text-anchor', 'middle')
                    .style('font-size', '12px')
                    .style('font-weight', 'bold')
                    .style('fill', '#333')
                    .text(d.word);
            })
            .on('mouseout', function() {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', 5)
                    .style('opacity', 0.7);
                
                svg.selectAll('.embedding-label').remove();
            });

        // Animação de entrada
        points.transition()
            .duration(1000)
            .delay((d, i) => i * 50)
            .attr('r', 5);

        this.d3Visualizations.set('embedding-space', { svg, update: this.updateEmbeddingSpace.bind(this) });
    }

    /**
     * 🏗️ Arquitetura do Transformer
     */
    createTransformerArchitecture() {
        const container = d3.select('#transformer-architecture')
            .style('width', '100%')
            .style('height', '500px');

        if (container.empty()) {
            console.warn('Container #transformer-architecture não encontrado');
            return;
        }

        const width = 600;
        const height = 500;

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height);

        // Camadas do transformer
        const layers = [
            { name: 'Input Embeddings', y: 450, color: '#FF6B6B' },
            { name: 'Positional Encoding', y: 400, color: '#4ECDC4' },
            { name: 'Multi-Head Attention', y: 350, color: '#45B7D1' },
            { name: 'Add & Norm', y: 300, color: '#96CEB4' },
            { name: 'Feed Forward', y: 250, color: '#FFEAA7' },
            { name: 'Add & Norm', y: 200, color: '#96CEB4' },
            { name: 'Output Layer', y: 150, color: '#DDA0DD' },
            { name: 'Softmax', y: 100, color: '#98D8C8' }
        ];

        // Desenhar camadas
        const layerGroups = svg.selectAll('.layer-group')
            .data(layers)
            .enter().append('g')
            .attr('class', 'layer-group')
            .style('cursor', 'pointer')
            .on('click', (event, d) => this.highlightLayer(d.name));

        layerGroups.append('rect')
            .attr('x', 100)
            .attr('y', d => d.y - 20)
            .attr('width', 400)
            .attr('height', 35)
            .attr('fill', d => d.color)
            .attr('stroke', '#333')
            .attr('stroke-width', 2)
            .attr('rx', 5)
            .style('opacity', 0)
            .transition()
            .duration(1000)
            .delay((d, i) => i * 200)
            .style('opacity', 0.8);

        layerGroups.append('text')
            .attr('x', 300)
            .attr('y', d => d.y)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .style('fill', '#333')
            .text(d => d.name)
            .style('opacity', 0)
            .transition()
            .duration(1000)
            .delay((d, i) => i * 200 + 500)
            .style('opacity', 1);

        // Setas conectoras
        for (let i = 0; i < layers.length - 1; i++) {
            svg.append('path')
                .attr('d', `M 300 ${layers[i].y - 20} L 300 ${layers[i + 1].y + 15}`)
                .attr('stroke', '#666')
                .attr('stroke-width', 2)
                .attr('marker-end', 'url(#arrowhead)')
                .style('opacity', 0)
                .transition()
                .duration(500)
                .delay(i * 200 + 1000)
                .style('opacity', 1);
        }

        // Definir marcador de seta
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 8)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M 0,-5 L 10,0 L 0,5')
            .attr('fill', '#666');

        this.d3Visualizations.set('transformer-architecture', { svg });
    }

    /**
     * 🌊 Fluxo de tokens
     */
    createTokenFlowVisualization() {
        const container = d3.select('#token-flow')
            .style('width', '100%')
            .style('height', '300px');

        if (container.empty()) {
            console.warn('Container #token-flow não encontrado');
            return;
        }

        const width = 800;
        const height = 300;

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height);

        const tokens = ['O', 'gato', 'subiu', 'no', 'telhado'];
        const tokenWidth = 120;
        const tokenHeight = 40;
        const startX = 50;
        const y = height / 2;

        // Desenhar tokens
        const tokenGroups = svg.selectAll('.token-group')
            .data(tokens)
            .enter().append('g')
            .attr('class', 'token-group')
            .attr('transform', (d, i) => `translate(${startX + i * (tokenWidth + 20)}, ${y - tokenHeight/2})`);

        tokenGroups.append('rect')
            .attr('width', tokenWidth)
            .attr('height', tokenHeight)
            .attr('fill', '#E3F2FD')
            .attr('stroke', '#1976D2')
            .attr('stroke-width', 2)
            .attr('rx', 8)
            .style('opacity', 0)
            .transition()
            .duration(800)
            .delay((d, i) => i * 200)
            .style('opacity', 1);

        tokenGroups.append('text')
            .attr('x', tokenWidth / 2)
            .attr('y', tokenHeight / 2)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .style('fill', '#1976D2')
            .text(d => d)
            .style('opacity', 0)
            .transition()
            .duration(800)
            .delay((d, i) => i * 200 + 400)
            .style('opacity', 1);

        // Animação de processamento
        this.animateTokenProcessing(svg, tokens.length, tokenWidth, y);

        this.d3Visualizations.set('token-flow', { svg });
    }

    /**
     * 📈 Métricas de performance em tempo real
     */
    createPerformanceMetrics() {
        const ctx = document.getElementById('performance-metrics-chart');
        if (!ctx) {
            console.warn('Canvas #performance-metrics-chart não encontrado');
            return;
        }

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Latência (ms)',
                        data: [],
                        borderColor: '#FF6B6B',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'CPU (%)',
                        data: [],
                        borderColor: '#4ECDC4',
                        backgroundColor: 'rgba(78, 205, 196, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Memória (MB)',
                        data: [],
                        borderColor: '#45B7D1',
                        backgroundColor: 'rgba(69, 183, 209, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y2'
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Tempo'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Latência (ms)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'CPU (%)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    },
                    y2: {
                        type: 'linear',
                        display: false,
                        position: 'right',
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: '📊 Métricas de Performance em Tempo Real'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });

        this.charts.set('performance-metrics', chart);
    }

    /**
     * 🏔️ Paisagem de loss
     */
    createLossLandscape() {
        const container = d3.select('#loss-landscape')
            .style('width', '100%')
            .style('height', '400px');

        if (container.empty()) {
            console.warn('Container #loss-landscape não encontrado');
            return;
        }

        const width = 500;
        const height = 400;
        const margin = { top: 20, right: 20, bottom: 40, left: 40 };

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height);

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Gerar dados de loss
        const lossData = this.generateLossData(100);
        
        const xScale = d3.scaleLinear()
            .domain([0, lossData.length - 1])
            .range([0, width - margin.left - margin.right]);

        const yScale = d3.scaleLinear()
            .domain(d3.extent(lossData))
            .range([height - margin.top - margin.bottom, 0]);

        const line = d3.line()
            .x((d, i) => xScale(i))
            .y(d => yScale(d))
            .curve(d3.curveMonotoneX);

        // Área sob a curva
        const area = d3.area()
            .x((d, i) => xScale(i))
            .y0(height - margin.top - margin.bottom)
            .y1(d => yScale(d))
            .curve(d3.curveMonotoneX);

        // Desenhar área
        g.append('path')
            .datum(lossData)
            .attr('fill', 'rgba(255, 107, 107, 0.3)')
            .attr('d', area)
            .style('opacity', 0)
            .transition()
            .duration(2000)
            .style('opacity', 1);

        // Desenhar linha
        const path = g.append('path')
            .datum(lossData)
            .attr('fill', 'none')
            .attr('stroke', '#FF6B6B')
            .attr('stroke-width', 3)
            .attr('d', line);

        // Animação da linha
        const totalLength = path.node().getTotalLength();
        path
            .attr('stroke-dasharray', totalLength + ' ' + totalLength)
            .attr('stroke-dashoffset', totalLength)
            .transition()
            .duration(2000)
            .ease(d3.easeLinear)
            .attr('stroke-dashoffset', 0);

        // Eixos
        g.append('g')
            .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
            .call(d3.axisBottom(xScale))
            .append('text')
            .attr('x', (width - margin.left - margin.right) / 2)
            .attr('y', 35)
            .attr('fill', 'black')
            .style('text-anchor', 'middle')
            .text('Época');

        g.append('g')
            .call(d3.axisLeft(yScale))
            .append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', -25)
            .attr('x', -(height - margin.top - margin.bottom) / 2)
            .attr('fill', 'black')
            .style('text-anchor', 'middle')
            .text('Loss');

        this.d3Visualizations.set('loss-landscape', { svg });
    }

    // Métodos auxiliares

    generateAttentionMatrix(size) {
        const matrix = [];
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                matrix.push({
                    from: i,
                    to: j,
                    value: Math.random() * 0.8 + 0.1 // 0.1 a 0.9
                });
            }
        }
        return matrix;
    }

    generateEmbeddings(count) {
        const embeddings = [];
        const clusters = ['substantivos', 'verbos', 'adjetivos', 'preposições', 'artigos'];
        const words = [
            'gato', 'casa', 'livro', 'mesa', 'carro',
            'correr', 'pular', 'ler', 'escrever', 'dormir',
            'grande', 'pequeno', 'azul', 'rápido', 'bonito',
            'em', 'sobre', 'para', 'com', 'de',
            'o', 'a', 'um', 'uma', 'os'
        ];

        for (let i = 0; i < count; i++) {
            embeddings.push({
                x: (Math.random() - 0.5) * 10,
                y: (Math.random() - 0.5) * 10,
                word: words[i % words.length],
                cluster: clusters[Math.floor(i / 10) % clusters.length]
            });
        }
        return embeddings;
    }

    generateLossData(epochs) {
        const data = [];
        let loss = 2.5;
        for (let i = 0; i < epochs; i++) {
            loss = loss * 0.98 + (Math.random() - 0.5) * 0.1;
            data.push(Math.max(0.1, loss));
        }
        return data;
    }

    animateTokenProcessing(svg, tokenCount, tokenWidth, y) {
        const processingIndicator = svg.append('circle')
            .attr('cx', 50)
            .attr('cy', y + 60)
            .attr('r', 8)
            .attr('fill', '#FF6B6B')
            .style('opacity', 0.8);

        function animate() {
            processingIndicator
                .transition()
                .duration(2000)
                .ease(d3.easeLinear)
                .attr('cx', 50 + tokenCount * (tokenWidth + 20))
                .transition()
                .duration(100)
                .attr('cx', 50)
                .on('end', animate);
        }

        setTimeout(animate, 2000);
    }

    highlightLayer(layerName) {
        console.log(`🎯 Camada destacada: ${layerName}`);
        // Implementar destaque visual da camada
    }

    updateAttentionHeatmap(newData) {
        // Implementar atualização do mapa de atenção
        console.log('🔄 Atualizando mapa de atenção:', newData);
    }

    updateEmbeddingSpace(newEmbeddings) {
        // Implementar atualização do espaço de embeddings
        console.log('🔄 Atualizando espaço de embeddings:', newEmbeddings);
    }

    updatePerformanceMetrics(metrics) {
        const chart = this.charts.get('performance-metrics');
        if (!chart) return;

        const now = new Date().toLocaleTimeString();
        
        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(metrics.latency || 0);
        chart.data.datasets[1].data.push(metrics.cpu_usage || 0);
        chart.data.datasets[2].data.push(metrics.memory_usage || 0);

        // Manter apenas os últimos 20 pontos
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(dataset => dataset.data.shift());
        }

        chart.update('none');
    }

    // API pública

    /**
     * Atualiza todas as visualizações com novos dados
     */
    updateAll(data) {
        if (data.attention) this.updateAttentionHeatmap(data.attention);
        if (data.embeddings) this.updateEmbeddingSpace(data.embeddings);
        if (data.metrics) this.updatePerformanceMetrics(data.metrics);
    }

    /**
     * Redimensiona todas as visualizações
     */
    resize() {
        this.charts.forEach(chart => chart.resize());
        // Redimensionar visualizações D3 se necessário
    }

    /**
     * Limpa todas as visualizações
     */
    clear() {
        this.charts.forEach(chart => {
            chart.data.labels = [];
            chart.data.datasets.forEach(dataset => dataset.data = []);
            chart.update();
        });
    }
}

// Instância global
window.advancedViz = new AdvancedVisualizations();

// Exportar para uso em módulos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedVisualizations;
}