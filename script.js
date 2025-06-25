document.addEventListener('DOMContentLoaded', () => {
    let modelData = [];
    let hardwareData = [];
    let memoryChart;
    const DEFAULT_MAX_SEQLEN = 32768;

    const dom = {
        presetModeBtn: document.getElementById('preset-mode-btn'),
        customModeBtn: document.getElementById('custom-mode-btn'),
        presetSelectGroup: document.getElementById('preset-model-select-group'),
        presetSelect: document.getElementById('preset-model-select'),
        model: {
            type: document.getElementById('model-type'),
            hidden_size: document.getElementById('hidden-size'),
            num_hidden_layers: document.getElementById('num-hidden-layers'),
            num_attention_heads: document.getElementById('num-attention-heads'),
            num_key_value_heads: document.getElementById('num-key-value-heads'),
            vocab_size: document.getElementById('vocab-size'),
            model_total_params: document.getElementById('model-total-params'),
            moe_total_params: document.getElementById('moe-total-params'),
            expert_params_per_expert: document.getElementById('expert-params-per-expert'),
            intermediate_size: document.getElementById('intermediate-size'),
            num_local_experts: document.getElementById('num-local-experts'),
            num_experts_per_tok: document.getElementById('num-experts-per-tok'),
            has_vision_modal: document.getElementById('has-vision-modal'),
            vision_params: document.getElementById('vision-params'),
            has_audio_modal: document.getElementById('has-audio-modal'),
            audio_params: document.getElementById('audio-params'),
        },
        multimodal_inference: {
            image_input_size: document.getElementById('image-input-size'),
            patch_size: document.getElementById('patch-size'),
            audio_input_length: document.getElementById('audio-input-length'),
            audio_sample_rate: document.getElementById('audio-sample-rate'),
            multimodal_config: document.getElementById('multimodal-inference-config'),
            vision_config: document.getElementById('vision-inference-config'),
            audio_config: document.getElementById('audio-inference-config'),
        },
        denseParamsDiv: document.getElementById('dense-params'),
        moeParamsDiv: document.getElementById('moe-params'),
        multimodalParamsDiv: document.getElementById('multimodal-params'),
        visionModalParams: document.getElementById('vision-modal-params'),
        audioModalParams: document.getElementById('audio-modal-params'),
        deployment: {
            quant_weights: document.getElementById('quant-weights'),
            quant_kvcache: document.getElementById('quant-kvcache'),
            quant_activations: document.getElementById('quant-activations'),
            input_length: document.getElementById('input-length'),
            output_length: document.getElementById('output-length'),
            batch_size: document.getElementById('batch-size'),
            overhead_ratio: document.getElementById('overhead-ratio'),
        },
        seqLengthValue: document.getElementById('seq-length-value'),
        batchSizeValue: document.getElementById('batch-size-value'),
        throughput: document.getElementById('throughput'),
        ttft: document.getElementById('ttft'),
        tpot: document.getElementById('tpot'),
        recommendHwBtn: document.getElementById('recommend-hw-btn'),
        customHwBtn: document.getElementById('custom-hw-btn'),
        recommendHwView: document.getElementById('recommend-hw-view'),
        customHwView: document.getElementById('custom-hw-view'),
        hwRecommendations: document.getElementById('hardware-recommendations'),
        specificGpuSelect: document.getElementById('specific-gpu-select'),
        specificGpuCards: document.getElementById('specific-gpu-cards'),
        gpuUtilizationRecommend: document.getElementById('gpu-utilization-recommend'),
        gpuUtilizationCustom: document.getElementById('gpu-utilization-custom'),
        customVram: document.getElementById('custom-vram'),
        customMemoryBw: document.getElementById('custom-memory-bw'),
        customFp16: document.getElementById('custom-fp16'),
        customBf16: document.getElementById('custom-bf16'),
        customFp8: document.getElementById('custom-fp8'),
        customInt8: document.getElementById('custom-int8'),
        customInt4: document.getElementById('custom-int4'),
        customInterconnectBw: document.getElementById('custom-interconnect-bw'),
        customCardsResult: document.getElementById('custom-cards-result'),
        startSimBtn: document.getElementById('start-sim-btn'),
        resetSimBtn: document.getElementById('reset-sim-btn'),
        simulationOutput: document.getElementById('simulation-output'),
        simulationSpeedDisplay: document.getElementById('simulation-speed-display'),
        memoryLegend: document.getElementById('memory-legend'),
        inputLengthValue: document.getElementById('input-length-value'),
        outputLengthValue: document.getElementById('output-length-value'),
    };

    async function initialize() {
        Chart.register(ChartDataLabels);
        await loadData();
        populateSelectors();
        setupEventListeners();
        initMemoryChart();
        switchModelMode('preset');
        updateConditionalFields();
        renderSliderLabels(dom.deployment.input_length, document.getElementById('input-length-ticks'), document.getElementById('input-length-labels'));
        renderSliderLabels(dom.deployment.output_length, document.getElementById('output-length-ticks'), document.getElementById('output-length-labels'));
        renderSliderLabels(dom.deployment.batch_size, document.getElementById('concurrency-ticks'), document.getElementById('batch-size-labels'));
        updateInputOutputSliderLimits();
        updateSpeedDisplay();
    }

    async function loadData() {
        try {
            const [modelsRes, hardwareRes] = await Promise.all([
                fetch('data/model.json'),
                fetch('data/gpu.json')
            ]);
            if (!modelsRes.ok || !hardwareRes.ok) {
                 throw new Error(`HTTP error! status: models=${modelsRes.status}, hardware=${hardwareRes.status}`);
            }
            modelData = await modelsRes.json();
            hardwareData = await hardwareRes.json();
        } catch (error) {
            console.error("数据加载失败:", error);
            alert("无法加载配置文件，请检查 data/model.json 和 data/gpu.json 是否存在且格式正确。");
        }
    }

    function populateSelectors() {
        dom.presetSelect.innerHTML = '';
        modelData.forEach((model, index) => {
            const option = new Option(model.model_name, index);
            dom.presetSelect.add(option);
        });
        dom.specificGpuSelect.innerHTML = '';
        hardwareData.forEach(gpu => {
            const option = new Option(`${gpu.vendor} ${gpu.model} (${gpu.vram}GB)`, JSON.stringify({vendor: gpu.vendor, model: gpu.model}));
            dom.specificGpuSelect.add(option);
        });
    }
    
    function setupEventListeners() {
        dom.presetModeBtn.addEventListener('click', () => switchModelMode('preset'));
        dom.customModeBtn.addEventListener('click', () => switchModelMode('custom'));
        dom.recommendHwBtn.addEventListener('click', () => switchHardwareMode('recommend'));
        dom.customHwBtn.addEventListener('click', () => switchHardwareMode('custom'));
        
        // 添加输入验证
        setupInputValidation();
        
        dom.model.type.addEventListener('change', () => {
            updateConditionalFields();
            updateAllCalculations();
        });
        dom.presetSelect.addEventListener('change', populatePresetData);
        
        dom.deployment.input_length.addEventListener('input', () => {
            updateInputOutputSliderLimits();
            updateAllCalculations();
        });
        dom.deployment.output_length.addEventListener('input', () => {
            updateInputOutputSliderLimits();
            updateAllCalculations();
        });
        dom.deployment.batch_size.addEventListener('input', () => {
            dom.batchSizeValue.textContent = dom.deployment.batch_size.value;
        });

        // 多模态复选框事件监听
        dom.model.has_vision_modal.addEventListener('change', () => {
            updateMultimodalFields();
            updateAllCalculations();
        });
        
        dom.model.has_audio_modal.addEventListener('change', () => {
            updateMultimodalFields();
            updateAllCalculations();
        });

        // 添加自定义硬件计算按钮的事件监听
        document.getElementById('calculate-custom-btn').addEventListener('click', () => {
            const vramUsage = calculateVram();
            if (vramUsage) {
                const computingPower = calculateComputingPower();
                updateCustomCardCount(vramUsage.total, computingPower);
            } else {
                dom.customCardsResult.textContent = '请先输入模型参数';
            }
        });

        const allInputs = [ 
            ...Object.values(dom.model), 
            ...Object.values(dom.deployment), 
            dom.gpuUtilizationRecommend,
            dom.throughput, 
            dom.ttft, 
            dom.tpot 
        ];
        allInputs.forEach(input => {
            if(input) {
               input.addEventListener('change', updateAllCalculations);
               input.addEventListener('input', updateAllCalculations);
            }
        });
        
        dom.throughput.addEventListener('input', updateSpeedDisplay);
        dom.specificGpuSelect.addEventListener('change', updateAllCalculations);
        dom.startSimBtn.addEventListener('click', startSimulation);
        dom.resetSimBtn.addEventListener('click', resetSimulation);
    }
    
    function switchModelMode(mode) {
        const isPreset = mode === 'preset';
        dom.presetModeBtn.classList.toggle('active', isPreset);
        dom.customModeBtn.classList.toggle('active', !isPreset);
        dom.presetSelectGroup.style.display = isPreset ? 'flex' : 'none';
        
        Object.values(dom.model).forEach(input => {
            if (input && (input.tagName === 'SELECT' || input.tagName === 'INPUT')) {
                input.disabled = isPreset;
            }
        });
        
        if (isPreset) {
            populatePresetData();
        } else {
            dom.deployment.input_length.max = DEFAULT_MAX_SEQLEN;
            dom.deployment.output_length.max = DEFAULT_MAX_SEQLEN;
            renderSliderLabels(dom.deployment.input_length, document.getElementById('input-length-ticks'), document.getElementById('input-length-labels'));
            renderSliderLabels(dom.deployment.output_length, document.getElementById('output-length-ticks'), document.getElementById('output-length-labels'));
            renderSliderLabels(dom.deployment.batch_size, document.getElementById('concurrency-ticks'), document.getElementById('batch-size-labels'));
            updateInputOutputSliderLimits();
            updateAllCalculations();
        }
    }

    function switchHardwareMode(mode) {
        const isRecommend = mode === 'recommend';
        dom.recommendHwBtn.classList.toggle('active', isRecommend);
        dom.customHwBtn.classList.toggle('active', !isRecommend);
        dom.recommendHwView.classList.toggle('hidden', !isRecommend);
        dom.customHwView.classList.toggle('hidden', isRecommend);
        
        // 如果切换到自定义硬件模式，填充默认值并初始化一次计算
        if (!isRecommend) {
            // 如果没有填写过值，预填写A100的值
            if (!dom.customVram.value) {
                // 使用A100 SXM的参数
                dom.customVram.value = '80';
                dom.customMemoryBw.value = '2039';
                dom.customFp16.value = '312';
                dom.customBf16.value = '312';
                dom.customFp8.value = '0';
                dom.customInt8.value = '624';
                dom.customInt4.value = '1248';
                dom.customInterconnectBw.value = '600';
            }
            
            // 获取当前的计算结果
            const vramUsage = calculateVram();
            if (vramUsage) {
                const computingPower = calculateComputingPower();
                // 使用当前结果更新自定义卡数
                updateCustomCardCount(vramUsage.total, computingPower);
            } else {
                dom.customCardsResult.textContent = '请先输入模型参数';
            }
        }
    }

    function populatePresetData() {
        const selectedIndex = dom.presetSelect.value;
        const data = modelData[selectedIndex];
        if (!data) return;
        const params = data.parameters;
        dom.model.type.value = data.model_type;
        dom.model.hidden_size.value = params.hidden_size || '';
        dom.model.num_hidden_layers.value = params.num_hidden_layers || '';
        dom.model.num_attention_heads.value = params.num_attention_heads || '';
        dom.model.num_key_value_heads.value = params.num_key_value_heads || '';
        dom.model.vocab_size.value = params.vocab_size || '';
        
        if (data.model_type === 'Dense') {
            dom.model.model_total_params.value = params.model_total_params / 1e9 || '';
        } else if (data.model_type === 'MoE') {
            dom.model.moe_total_params.value = params.model_total_params / 1e9 || '';
            dom.model.expert_params_per_expert.value = params.expert_params_per_expert / 1e9 || '';
            dom.model.intermediate_size.value = params.intermediate_size || '';
            dom.model.num_local_experts.value = params.num_local_experts || '';
            dom.model.num_experts_per_tok.value = params.num_experts_per_tok || '';
            // 存储shared_params到DOM元素的数据属性中，以便在计算时使用
            if (params.shared_params) {
                dom.model.moe_total_params.dataset.sharedParams = params.shared_params / 1e9;
            } else {
                delete dom.model.moe_total_params.dataset.sharedParams;
            }
        } else if (data.model_type === 'Multimodal') {
            dom.model.model_total_params.value = params.llm_base_params / 1e9 || '';
            
            // 设置视觉模态参数
            if (params.has_vision_modal !== undefined) {
                dom.model.has_vision_modal.checked = params.has_vision_modal;
            } else {
                dom.model.has_vision_modal.checked = true; // 默认启用视觉模态
            }
            dom.model.vision_params.value = params.vision_params / 1e9 || '';
            dom.multimodal_inference.image_input_size.value = params.image_input_size || '';
            dom.multimodal_inference.patch_size.value = params.patch_size || '';
            
            // 设置语音模态参数
            if (params.has_audio_modal !== undefined) {
                dom.model.has_audio_modal.checked = params.has_audio_modal;
            } else {
                dom.model.has_audio_modal.checked = false; // 默认禁用语音模态
            }
            dom.model.audio_params.value = params.audio_params / 1e9 || '';
            dom.multimodal_inference.audio_input_length.value = params.audio_input_length || '';
            dom.multimodal_inference.audio_sample_rate.value = params.audio_sample_rate || '';
            
            // 更新多模态字段的显示/隐藏状态
            updateMultimodalFields();
        } else {
            dom.model.model_total_params.value = '';
        }
        
        const max_len = params.max_position_embeddings || DEFAULT_MAX_SEQLEN;
        dom.deployment.input_length.max = max_len;
        dom.deployment.output_length.max = max_len;
        dom.deployment.input_length.value = params.input_length || 512;
        dom.deployment.output_length.value = params.output_length || 512;
        renderSliderLabels(dom.deployment.input_length, document.getElementById('input-length-ticks'), document.getElementById('input-length-labels'));
        renderSliderLabels(dom.deployment.output_length, document.getElementById('output-length-ticks'), document.getElementById('output-length-labels'));
        updateInputOutputSliderLimits();
        updateConditionalFields();
        updateAllCalculations();
    }
    
    function renderSliderLabels(slider, datalist, labelContainer) {
        const sliderMin = parseFloat(slider.min);
        const sliderMax = parseFloat(slider.max);
        const range = sliderMax - sliderMin;
        
        labelContainer.innerHTML = '';
        if (range <= 0) return;

        const options = datalist.getElementsByTagName('option');
        for (let i = 0; i < options.length; i++) {
            const value = parseFloat(options[i].value);
            if (value >= sliderMin && value <= sliderMax) {
                const percent = ((value - sliderMin) / range) * 100;
                const label = document.createElement('span');
                label.textContent = value >= 1024 ? `${value / 1024}k` : value;
                label.style.left = `${percent}%`;
                labelContainer.appendChild(label);
            }
        }
    }
    
    function updateConditionalFields() {
        const selectedType = dom.model.type.value;
        dom.denseParamsDiv.classList.toggle('hidden', selectedType === 'MoE');
        dom.moeParamsDiv.classList.toggle('hidden', selectedType !== 'MoE');
        dom.multimodalParamsDiv.classList.toggle('hidden', selectedType !== 'Multimodal');
        
        // 多模态推理配置的显示/隐藏
        dom.multimodal_inference.multimodal_config.classList.toggle('hidden', selectedType !== 'Multimodal');
        
        if (selectedType === 'Multimodal') {
            dom.denseParamsDiv.classList.remove('hidden');
            dom.denseParamsDiv.querySelector('label').textContent = 'LLM基础参数(B)';
            updateMultimodalFields();
        } else {
            dom.denseParamsDiv.querySelector('label').textContent = '总/基础参数(B)';
        }
    }

    function updateMultimodalFields() {
        // 处理视觉模态参数的显示/隐藏
        const hasVisionModal = dom.model.has_vision_modal.checked;
        dom.visionModalParams.style.display = hasVisionModal ? 'block' : 'none';
        dom.multimodal_inference.vision_config.classList.toggle('hidden', !hasVisionModal);
        
        // 处理语音模态参数的显示/隐藏
        const hasAudioModal = dom.model.has_audio_modal.checked;
        dom.audioModalParams.classList.toggle('hidden', !hasAudioModal);
        dom.multimodal_inference.audio_config.classList.toggle('hidden', !hasAudioModal);
    }

    function updateAllCalculations() {
        const vramUsage = calculateVram();
        if (vramUsage) {
            updateMemoryChartAndLegend(vramUsage);
            const computingPower = calculateComputingPower();
            updateHardwareRecommendations(vramUsage.total, computingPower);
            updateSpecificGpuCardCount(vramUsage.total, computingPower);
            // 注意：这里不再自动更新自定义硬件的卡数，而是等待用户点击计算按钮
        }
    }

    function calculateVram() {
        const p = {};
        Object.keys(dom.model).forEach(key => {
            if (key === 'has_vision_modal' || key === 'has_audio_modal') {
                p[key] = dom.model[key].checked;
            } else {
                p[key] = parseFloat(dom.model[key].value) || 0;
            }
        });
        Object.keys(dom.deployment).forEach(key => p[key] = parseFloat(dom.deployment[key].value) || 0);
        const seq_length = (parseFloat(dom.deployment.input_length.value) || 0) + (parseFloat(dom.deployment.output_length.value) || 0);
        const modelType = dom.model.type.value;
        
        // 如果关键参数缺失，返回null
        if (seq_length <= 0 || p.batch_size <= 0) {
            return null;
        }
        
        // 初始化各种内存使用变量
        let model_weights_bytes = 0, kv_cache_bytes = 0, activations_bytes = 0;
        let shared_weights_bytes = 0, expert_weights_bytes = 0;
        let base_weights_bytes = 0, vision_weights_bytes = 0, audio_weights_bytes = 0;
        
        // 量化精度相关常量
        const bytes_per_weight = parseFloat(p.quant_weights) || 2;
        const bytes_per_kv = parseFloat(p.quant_kvcache) || 2;
        const bytes_per_activation = parseFloat(p.quant_activations) || 2;
        
        // 激活值经验因子
        const DENSE_ACTIVATION_FACTOR = 18;  // Dense模型激活值因子
        const LLM_ACTIVATION_FACTOR = 24;    // 多模态中LLM部分激活值因子
        const MOE_SHARED_ACTIVATION_FACTOR = 4;  // MoE共享部分激活值因子
        const MOE_EXPERT_ACTIVATION_FACTOR = 2;  // MoE专家部分激活值因子
        const VISION_ACTIVATION_FACTOR = 4.0;    // 视觉模态激活值因子
        const AUDIO_ACTIVATION_FACTOR = 3.0;     // 音频模态激活值因子
        
        // 多模态token因子
        const VISION_TOKEN_FACTOR = 1.0;  // 视觉token对KV Cache的影响因子
        const AUDIO_TOKEN_FACTOR = 0.8;   // 音频token对KV Cache的影响因子
        
        // 计算head维度
        const head_dim = (p.num_attention_heads > 0) ? p.hidden_size / p.num_attention_heads : 0;

        switch (modelType) {
            case 'Dense':
                // Dense模型的权重计算
                model_weights_bytes = p.model_total_params * 1e9 * bytes_per_weight;
                
                // KV Cache计算：batch_size * seq_length * num_layers * 2(K和V) * kv_heads * head_dim * 每个元素的字节数
                // head_dim = hidden_size / attention_heads
                if (p.hidden_size > 0 && p.num_attention_heads > 0 && p.num_hidden_layers > 0) {
                    const kv_heads = Math.max(p.num_key_value_heads, 1); // 确保kv_heads至少为1
                    kv_cache_bytes = p.batch_size * seq_length * p.num_hidden_layers * 2 * kv_heads * head_dim * bytes_per_kv;
                }
                
                // 激活值计算：使用经验因子，通常为hidden_size的12-24倍
                // DENSE_ACTIVATION_FACTOR设为18，表示激活值大约是hidden_size的18倍
                activations_bytes = p.batch_size * seq_length * p.hidden_size * DENSE_ACTIVATION_FACTOR * bytes_per_activation;
                break;
                
            case 'MoE':
                // 使用预设的总参数和单个专家参数
                if (p.moe_total_params > 0 && p.expert_params_per_expert > 0 && p.num_local_experts > 0) {
                    // 如果提供了shared_params，直接使用
                    if (dom.model.moe_total_params.dataset.sharedParams) {
                        shared_weights_bytes = parseFloat(dom.model.moe_total_params.dataset.sharedParams) * 1e9 * bytes_per_weight;
                        expert_weights_bytes = p.expert_params_per_expert * 1e9 * p.num_local_experts * bytes_per_weight;
                    } else {
                        // 计算专家总参数
                        expert_weights_bytes = p.expert_params_per_expert * 1e9 * p.num_local_experts * bytes_per_weight;
                        // 计算共享参数 = 总参数 - 专家总参数
                        shared_weights_bytes = (p.moe_total_params * 1e9 - expert_weights_bytes);
                        
                        // 确保共享参数不为负数
                        if (shared_weights_bytes < 0) {
                            shared_weights_bytes = 0;
                            expert_weights_bytes = p.moe_total_params * 1e9;
                        }
                        
                        // 应用量化
                        shared_weights_bytes = shared_weights_bytes * bytes_per_weight;
                        expert_weights_bytes = expert_weights_bytes * bytes_per_weight;
                    }
                } else {
                    // 如果没有预设参数，使用估算方法
                    const approx_shared_params = (p.vocab_size * p.hidden_size * 2) + p.num_hidden_layers * (4 * p.hidden_size * p.hidden_size);
                    const expert_params = p.num_hidden_layers * p.num_local_experts * (2 * p.hidden_size * p.intermediate_size);
                    
                    shared_weights_bytes = approx_shared_params * bytes_per_weight;
                    expert_weights_bytes = expert_params * bytes_per_weight;
                }
                
                // MoE模型总权重 = 共享权重 + 专家权重
                model_weights_bytes = shared_weights_bytes + expert_weights_bytes;
                
                // KV Cache计算与Dense模型相同
                if (p.hidden_size > 0 && p.num_attention_heads > 0 && p.num_hidden_layers > 0) {
                    const kv_heads = Math.max(p.num_key_value_heads, 1); // 确保kv_heads至少为1
                    kv_cache_bytes = p.batch_size * seq_length * p.num_hidden_layers * 2 * kv_heads * head_dim * bytes_per_kv;
                }
                
                // MoE激活值计算分两部分：
                // 1. 共享部分激活值：与Dense模型类似，但因子较小
                // 2. 专家部分激活值：考虑每个token激活的专家数量和专家网络的中间层维度
                
                activations_bytes = p.batch_size * seq_length * p.hidden_size * MOE_SHARED_ACTIVATION_FACTOR * bytes_per_activation + 
                                   p.batch_size * seq_length * Math.max(p.num_experts_per_tok, 1) * Math.max(p.intermediate_size, p.hidden_size) * MOE_EXPERT_ACTIVATION_FACTOR * bytes_per_activation;
                break;
                
            case 'Multimodal':
                // 获取推理配置中的多模态参数
                const mm_params = {
                    image_input_size: parseFloat(dom.multimodal_inference.image_input_size.value) || 0,
                    patch_size: parseFloat(dom.multimodal_inference.patch_size.value) || 0,
                    audio_input_length: parseFloat(dom.multimodal_inference.audio_input_length.value) || 0,
                    audio_sample_rate: parseFloat(dom.multimodal_inference.audio_sample_rate.value) || 0
                };
                
                // 使用全局定义的经验因子
                
                // 基础LLM权重
                base_weights_bytes = Math.max(p.model_total_params, 0.1) * 1e9 * bytes_per_weight;
                
                // 视觉模态权重（如果启用）
                if (p.has_vision_modal) {
                    vision_weights_bytes = Math.max(p.vision_params, 0.01) * 1e9 * bytes_per_weight;
                }
                
                // 语音模态权重（如果启用）
                if (p.has_audio_modal) {
                    audio_weights_bytes = Math.max(p.audio_params, 0.01) * 1e9 * bytes_per_weight;
                }
                
                // 总模型权重
                model_weights_bytes = base_weights_bytes + vision_weights_bytes + audio_weights_bytes;
                
                // KV Cache 计算
                let total_seq_len = seq_length;
                
                // 如果启用视觉模态，添加图像patch序列长度
                if (p.has_vision_modal && mm_params.patch_size > 0 && mm_params.image_input_size > 0) {
                    const num_image_patches = Math.ceil((mm_params.image_input_size / mm_params.patch_size) ** 2);
                    total_seq_len += num_image_patches * VISION_TOKEN_FACTOR;
                }
                
                // 如果启用语音模态，添加音频序列长度（简化估算）
                if (p.has_audio_modal && mm_params.audio_input_length > 0) {
                    // 假设每秒音频产生约20个token
                    const audio_tokens = mm_params.audio_input_length * 20;
                    total_seq_len += audio_tokens * AUDIO_TOKEN_FACTOR;
                }
                
                if (p.hidden_size > 0 && p.num_attention_heads > 0 && p.num_hidden_layers > 0) {
                    const kv_heads = Math.max(p.num_key_value_heads, 1); // 确保kv_heads至少为1
                    kv_cache_bytes = p.batch_size * total_seq_len * p.num_hidden_layers * 2 * kv_heads * head_dim * bytes_per_kv;
                }
                
                // 激活值计算
                let total_activations_bytes = 0;
                
                // 基础LLM激活值
                const llm_activations_bytes = p.batch_size * seq_length * p.hidden_size * LLM_ACTIVATION_FACTOR * bytes_per_activation;
                total_activations_bytes += llm_activations_bytes;
                
                // 视觉模态激活值（如果启用）
                if (p.has_vision_modal && mm_params.patch_size > 0 && mm_params.image_input_size > 0) {
                    const num_image_patches = Math.ceil((mm_params.image_input_size / mm_params.patch_size) ** 2);
                    const vision_activations_bytes = p.batch_size * num_image_patches * p.hidden_size * VISION_ACTIVATION_FACTOR * bytes_per_activation;
                    total_activations_bytes += vision_activations_bytes;
                }
                
                // 语音模态激活值（如果启用）
                if (p.has_audio_modal && mm_params.audio_input_length > 0) {
                    // 假设每秒音频产生约20个token
                    const audio_tokens = mm_params.audio_input_length * 20;
                    const audio_activations_bytes = p.batch_size * audio_tokens * p.hidden_size * AUDIO_ACTIVATION_FACTOR * bytes_per_activation;
                    total_activations_bytes += audio_activations_bytes;
                }
                
                activations_bytes = total_activations_bytes;
                break;
                
            default:
                return null;
        }
        
        const modelWeightsGB = model_weights_bytes / (1024 ** 3);
        const sharedWeightsGB = shared_weights_bytes / (1024 ** 3);
        const expertWeightsGB = expert_weights_bytes / (1024 ** 3);
        const baseWeightsGB = base_weights_bytes / (1024 ** 3);
        const visionWeightsGB = vision_weights_bytes / (1024 ** 3);
        const audioWeightsGB = audio_weights_bytes / (1024 ** 3);
        const kvCacheGB = kv_cache_bytes / (1024 ** 3);
        const activationsGB = activations_bytes / (1024 ** 3);
        const overheadGB = (modelWeightsGB + kvCacheGB + activationsGB) * p.overhead_ratio;
        const total = modelWeightsGB + kvCacheGB + activationsGB + overheadGB;
        
        // 确保返回值不为NaN或无穷大
        if (isNaN(total) || !isFinite(total) || total <= 0) {
            return null;
        }
        
        if (modelType === 'MoE') {
            return { 
                modelType,
                sharedWeightsGB, 
                expertWeightsGB, 
                kvCacheGB, 
                activationsGB, 
                overheadGB, 
                total 
            };
        } else if (modelType === 'Multimodal') {
            return {
                modelType,
                baseWeightsGB,
                visionWeightsGB,
                audioWeightsGB,
                has_vision_modal: p.has_vision_modal,
                has_audio_modal: p.has_audio_modal,
                kvCacheGB,
                activationsGB,
                overheadGB,
                total
            };
        } else {
            return { 
                modelType,
                modelWeightsGB, 
                kvCacheGB, 
                activationsGB, 
                overheadGB, 
                total 
            };
        }
    }

    function initMemoryChart() {
        const ctx = document.getElementById('memory-pie-chart').getContext('2d');
        memoryChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['模型权重', 'KV Cache', '激活值', '额外开销'],
                datasets: [{
                    data: [1, 1, 1, 1],
                    backgroundColor: [
                        getComputedStyle(document.documentElement).getPropertyValue('--chart-color-1').trim(),
                        getComputedStyle(document.documentElement).getPropertyValue('--chart-color-2').trim(),
                        getComputedStyle(document.documentElement).getPropertyValue('--chart-color-3').trim(),
                        getComputedStyle(document.documentElement).getPropertyValue('--chart-color-4').trim()
                    ],
                    borderColor: '#fff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    datalabels: {
                        formatter: (value, ctx) => {
                            const total = ctx.chart.getDatasetMeta(0).total;
                            if (value <= 0.01 || total <= 0) return null;
                            const percentage = (value / total * 100);
                            if (percentage < 5) return null;
                            return percentage.toFixed(1) + '%';
                        },
                        color: '#fff',
                        font: { weight: 'bold' }
                    }
                }
            }
        });
    }

    function updateMemoryChartAndLegend(vramData) {
        const { modelType } = vramData;
        
        if (modelType === 'MoE') {
            const { sharedWeightsGB, expertWeightsGB, kvCacheGB, activationsGB, overheadGB, total } = vramData;
            
            // 更新图表标签和数据
            memoryChart.data.labels = ['共享权重', '专家权重', 'KV Cache', '激活值', '额外开销'];
            memoryChart.data.datasets[0].data = [sharedWeightsGB, expertWeightsGB, kvCacheGB, activationsGB, overheadGB];
            memoryChart.data.datasets[0].backgroundColor = [
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-1').trim(),
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-5').trim(),
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-2').trim(),
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-3').trim(),
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-4').trim()
            ];
            memoryChart.update();
            
            // 更新图例
            const legendData = [
                { label: '共享权重', value: sharedWeightsGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-1').trim() },
                { label: '专家权重', value: expertWeightsGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-5').trim() },
                { label: 'KV Cache', value: kvCacheGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-2').trim() },
                { label: '激活值', value: activationsGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-3').trim() },
                { label: '额外开销', value: overheadGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-4').trim() },
            ];
            
            dom.memoryLegend.innerHTML = '';
            const totalLegendHtml = `<div class="legend-item total"><div class="legend-label">总计</div><div class="legend-value">${total.toFixed(2)} GB</div></div>`;
            
            legendData.forEach(item => {
                if (item.value > 0) {
                    const percentage = total > 0 ? (item.value / total * 100).toFixed(1) : 0;
                    const legendHtml = `<div class="legend-item"><div class="legend-color-box" style="background-color: ${item.color};"></div><div class="legend-label">${item.label}</div><div class="legend-value">${item.value.toFixed(2)} GB (${percentage}%)</div></div>`;
                    dom.memoryLegend.innerHTML += legendHtml;
                }
            });
            
            dom.memoryLegend.innerHTML += totalLegendHtml;
        } else if (modelType === 'Multimodal') {
            const { baseWeightsGB, visionWeightsGB, audioWeightsGB, has_vision_modal, has_audio_modal, kvCacheGB, activationsGB, overheadGB, total } = vramData;
            
            // 准备标签、数据和颜色
            const labels = ['基础权重'];
            const data = [baseWeightsGB];
            const colors = [getComputedStyle(document.documentElement).getPropertyValue('--chart-color-1').trim()];
            
            // 如果有视觉模态
            if (has_vision_modal) {
                labels.push('视觉权重');
                data.push(visionWeightsGB);
                colors.push(getComputedStyle(document.documentElement).getPropertyValue('--chart-color-6').trim());
            }
            
            // 如果有语音模态
            if (has_audio_modal) {
                labels.push('语音权重');
                data.push(audioWeightsGB);
                colors.push(getComputedStyle(document.documentElement).getPropertyValue('--chart-color-7').trim());
            }
            
            // 添加其他组件
            labels.push('KV Cache', '激活值', '额外开销');
            data.push(kvCacheGB, activationsGB, overheadGB);
            colors.push(
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-2').trim(),
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-3').trim(),
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-4').trim()
            );
            
            // 更新图表
            memoryChart.data.labels = labels;
            memoryChart.data.datasets[0].data = data;
            memoryChart.data.datasets[0].backgroundColor = colors;
            memoryChart.update();
            
            // 构建图例数据
            const legendData = [
                { label: '基础权重', value: baseWeightsGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-1').trim() }
            ];
            
            if (has_vision_modal) {
                legendData.push({ 
                    label: '视觉权重', 
                    value: visionWeightsGB, 
                    color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-6').trim() 
                });
            }
            
            if (has_audio_modal) {
                legendData.push({ 
                    label: '语音权重', 
                    value: audioWeightsGB, 
                    color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-7').trim() 
                });
            }
            
            legendData.push(
                { label: 'KV Cache', value: kvCacheGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-2').trim() },
                { label: '激活值', value: activationsGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-3').trim() },
                { label: '额外开销', value: overheadGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-4').trim() }
            );
            
            // 更新图例HTML
            dom.memoryLegend.innerHTML = '';
            const totalLegendHtml = `<div class="legend-item total"><div class="legend-label">总计</div><div class="legend-value">${total.toFixed(2)} GB</div></div>`;
            
            legendData.forEach(item => {
                if (item.value > 0) {
                    const percentage = total > 0 ? (item.value / total * 100).toFixed(1) : 0;
                    const legendHtml = `<div class="legend-item"><div class="legend-color-box" style="background-color: ${item.color};"></div><div class="legend-label">${item.label}</div><div class="legend-value">${item.value.toFixed(2)} GB (${percentage}%)</div></div>`;
                    dom.memoryLegend.innerHTML += legendHtml;
                }
            });
            
            dom.memoryLegend.innerHTML += totalLegendHtml;
        } else {
            // 常规模型显示
            const { modelWeightsGB, kvCacheGB, activationsGB, overheadGB, total } = vramData;
            
            // 更新图表标签和数据
            memoryChart.data.labels = ['模型权重', 'KV Cache', '激活值', '额外开销'];
            memoryChart.data.datasets[0].data = [modelWeightsGB, kvCacheGB, activationsGB, overheadGB];
            memoryChart.data.datasets[0].backgroundColor = [
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-1').trim(),
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-2').trim(),
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-3').trim(),
                getComputedStyle(document.documentElement).getPropertyValue('--chart-color-4').trim()
            ];
            memoryChart.update();
            
            // 更新图例
            const legendData = [
                { label: '模型权重', value: modelWeightsGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-1').trim() },
                { label: 'KV Cache', value: kvCacheGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-2').trim() },
                { label: '激活值', value: activationsGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-3').trim() },
                { label: '额外开销', value: overheadGB, color: getComputedStyle(document.documentElement).getPropertyValue('--chart-color-4').trim() },
            ];
            
            dom.memoryLegend.innerHTML = '';
            const totalLegendHtml = `<div class="legend-item total"><div class="legend-label">总计</div><div class="legend-value">${total.toFixed(2)} GB</div></div>`;
            
            legendData.forEach(item => {
                if (item.value > 0) {
                    const percentage = total > 0 ? (item.value / total * 100).toFixed(1) : 0;
                    const legendHtml = `<div class="legend-item"><div class="legend-color-box" style="background-color: ${item.color};"></div><div class="legend-label">${item.label}</div><div class="legend-value">${item.value.toFixed(2)} GB (${percentage}%)</div></div>`;
                    dom.memoryLegend.innerHTML += legendHtml;
                }
            });
            
            dom.memoryLegend.innerHTML += totalLegendHtml;
        }
    }

    function calculateComputingPower() {
        const p = {};
        Object.keys(dom.model).forEach(key => {
            if (key === 'has_vision_modal' || key === 'has_audio_modal') {
                p[key] = dom.model[key].checked;
            } else {
                p[key] = parseFloat(dom.model[key].value) || 0;
            }
        });
        Object.keys(dom.deployment).forEach(key => p[key] = parseFloat(dom.deployment[key].value) || 0);
        const throughput = parseFloat(dom.throughput.value) || 50; // 吞吐量 tokens/s
        const batch_size = p.batch_size;
        const seq_length = (parseFloat(dom.deployment.input_length.value) || 0) + (parseFloat(dom.deployment.output_length.value) || 0);
        const modelType = dom.model.type.value;
        
        // 量化精度对应的算力类型
        const quantWeights = dom.deployment.quant_weights.value;
        
        // 算力单位转换因子 (TFLOPS/TOPS)
        const TFLOPS_CONVERSION = 1e12; // 1 TFLOPS = 10^12 FLOPS
        
        // 模型架构复杂度因子 - 调低以减少算力需求
        const DENSE_COMPLEXITY_FACTOR = 2.0;  // 原值: 4.0
        const MOE_SHARED_COMPLEXITY_FACTOR = 1.2;  // 原值: 2.5
        const MOE_EXPERT_COMPLEXITY_FACTOR = 1.5;  // 原值: 3.0
        const VISION_COMPLEXITY_FACTOR = 1.0;  // 原值: 2.0
        const AUDIO_COMPLEXITY_FACTOR = 0.8;  // 原值: 1.5
        
        // 注意力机制计算复杂度因子 - 调低
        const ATTENTION_COMPLEXITY_FACTOR = 1.0;  // 原值: 2.0
        
        // 序列长度非线性增长因子 - 降低对长序列的惩罚
        const SEQ_LENGTH_FACTOR = 1.0 + Math.log10(Math.max(seq_length, 10)) / 20;  // 原值: /10
        
        // 批处理效率因子 - 提高批处理效率
        const BATCH_EFFICIENCY_FACTOR = Math.max(0.8, 1.0 - Math.log10(Math.max(batch_size, 1)) / 15);  // 原值: 0.7, /10
        
        // 吞吐量阈值 - 只有当吞吐量超过阈值时才考虑算力限制
        const THROUGHPUT_THRESHOLD = 100;  // 新增: 吞吐量阈值
        const throughputFactor = Math.max(0.2, Math.min(1.0, throughput / THROUGHPUT_THRESHOLD));  // 新增: 吞吐量因子
        
        // 吞吐量转换为每秒FLOPs
        let required_flops = 0;
        
        switch (modelType) {
            case 'Dense':
                // 对于Dense模型，算力需求与隐藏层大小、层数、头数和吞吐量相关
                const dense_params_factor = p.hidden_size * p.num_hidden_layers / 2000;  // 原值: /1000
                const attention_factor = p.num_attention_heads * ATTENTION_COMPLEXITY_FACTOR / Math.max(p.num_key_value_heads, 1);
                
                required_flops = throughput * dense_params_factor * attention_factor * 
                                DENSE_COMPLEXITY_FACTOR * SEQ_LENGTH_FACTOR * BATCH_EFFICIENCY_FACTOR * 
                                (p.model_total_params * 1e9 / 1e12) * throughputFactor; // 转换为TFLOPS
                break;
                
            case 'MoE':
                // 对于MoE模型，分别计算共享部分和专家部分的算力需求
                let shared_params = 0;
                if (dom.model.moe_total_params.dataset.sharedParams) {
                    shared_params = parseFloat(dom.model.moe_total_params.dataset.sharedParams) * 1e9;
                } else if (p.moe_total_params > 0 && p.expert_params_per_expert > 0 && p.num_local_experts > 0) {
                    shared_params = p.moe_total_params * 1e9 - p.expert_params_per_expert * 1e9 * p.num_local_experts;
                }
                
                // 确保shared_params不为负数
                shared_params = Math.max(0, shared_params);
                
                const expert_params = p.expert_params_per_expert * 1e9 * p.num_local_experts;
                const expert_activation_ratio = p.num_experts_per_tok / Math.max(p.num_local_experts, 1);
                
                // 共享部分算力
                const shared_flops = throughput * (shared_params / 1e12) * MOE_SHARED_COMPLEXITY_FACTOR * 
                                    SEQ_LENGTH_FACTOR * BATCH_EFFICIENCY_FACTOR;
                
                // 专家部分算力 (考虑只有部分专家被激活)
                const expert_flops = throughput * (expert_params / 1e12) * expert_activation_ratio * 
                                    MOE_EXPERT_COMPLEXITY_FACTOR * SEQ_LENGTH_FACTOR * BATCH_EFFICIENCY_FACTOR;
                
                required_flops = (shared_flops + expert_flops) * throughputFactor;
                break;
                
            case 'Multimodal':
                // 对于多模态模型，分别计算LLM基础部分和各模态部分的算力需求
                const llm_params = Math.max(p.model_total_params, 0.1) * 1e9;  // 确保至少有一些参数
                const llm_flops = throughput * (llm_params / 1e12) * DENSE_COMPLEXITY_FACTOR * 
                                SEQ_LENGTH_FACTOR * BATCH_EFFICIENCY_FACTOR;
                
                let modal_flops = 0;
                
                // 视觉模态算力 (如果启用)
                if (p.has_vision_modal) {
                    const vision_params = Math.max(p.vision_params, 0.01) * 1e9;  // 确保至少有一些参数
                    // 视觉模态处理是一次性的，与吞吐量关系不大，但与批处理大小相关
                    modal_flops += (vision_params / 1e12) * VISION_COMPLEXITY_FACTOR * 
                                    batch_size * BATCH_EFFICIENCY_FACTOR;
                }
                
                // 音频模态算力 (如果启用)
                if (p.has_audio_modal) {
                    const audio_params = Math.max(p.audio_params, 0.01) * 1e9;  // 确保至少有一些参数
                    // 音频模态处理也是一次性的
                    modal_flops += (audio_params / 1e12) * AUDIO_COMPLEXITY_FACTOR * 
                                    batch_size * BATCH_EFFICIENCY_FACTOR;
                }
                
                required_flops = (llm_flops + modal_flops) * throughputFactor;
                break;
                
            default:
                return 0;
        }
        
        // 考虑系统开销和其他因素
        const overhead_ratio = p.overhead_ratio;
        required_flops *= (1 + overhead_ratio);
        
        // 确保返回值不为NaN或无穷大
        if (isNaN(required_flops) || !isFinite(required_flops)) {
            return 0;
        }
        
        return required_flops;
    }

    function updateHardwareRecommendations(requiredVram, requiredComputing) {
        dom.hwRecommendations.innerHTML = '';
        if (requiredVram <= 0) { 
            dom.hwRecommendations.innerHTML = `<tr><td colspan="5">请先输入模型参数</td></tr>`; 
            return; 
        }
        
        const gpuUtilization = parseFloat(dom.gpuUtilizationRecommend.value) || 0.8;
        const quantWeights = dom.deployment.quant_weights.value;
        
        // 根据量化精度选择对应的算力类型
        let perfField;
        if (quantWeights === "2") {
            // FP16/BF16
            perfField = "perf_bf16";
        } else if (quantWeights === "1" && dom.deployment.quant_weights.selectedOptions[0].text === "FP8") {
            perfField = "perf_fp8";
        } else if (quantWeights === "1" && dom.deployment.quant_weights.selectedOptions[0].text === "INT8") {
            perfField = "perf_int8";
        } else if (quantWeights === "0.5") {
            perfField = "perf_int4";
        } else {
            perfField = "perf_bf16"; // 默认使用BF16
        }
        
        // 计算每种GPU所需的卡数，分别基于显存和算力
        const gpuWithCards = hardwareData.map(gpu => {
            const vramCards = Math.ceil(requiredVram / gpu.vram);
            const computingCards = Math.ceil(requiredComputing / (gpu[perfField] * gpuUtilization));
            
            // 取两者的最大值
            const totalCards = Math.max(vramCards, computingCards);
            
            // 检查是否支持选择的量化精度
            const supportsQuantization = gpu[perfField] > 0;
            
            return { 
                ...gpu, 
                vramCards,
                computingCards,
                totalCards,
                supportsQuantization,
                perfField,
                perfValue: gpu[perfField]
            };
        });
        
        // 排序并获取前5个推荐
        gpuWithCards
            .filter(gpu => gpu.supportsQuantization)
            .sort((a, b) => a.totalCards - b.totalCards || b.vram - a.vram)
            .slice(0, 5)
            .forEach(gpu => {
                const limitingFactor = gpu.vramCards >= gpu.computingCards ? "显存" : "算力";
                const limitingClass = limitingFactor === "显存" ? "memory" : "computing";
                const row = `<tr>
                    <td>${gpu.vendor}</td>
                    <td>${gpu.model}</td>
                    <td>${gpu.vram} GB</td>
                    <td>${gpu.memory_bw} GB/s</td>
                    <td>${gpu.totalCards}</td>
                </tr>`;
                dom.hwRecommendations.innerHTML += row;
            });
        
        // 如果没有支持当前量化精度的GPU
        if (!dom.hwRecommendations.innerHTML) {
            dom.hwRecommendations.innerHTML = `<tr><td colspan="5" class="not-supported">没有找到支持当前量化精度的GPU</td></tr>`;
        }
    }
    
    function updateSpecificGpuCardCount(requiredVram, requiredComputing) {
        const selectedModelValue = dom.specificGpuSelect.value;
        if (!selectedModelValue) {
            dom.specificGpuCards.textContent = 'N/A';
            return;
        }
        
        try {
            const selectedGpuInfo = JSON.parse(selectedModelValue);
            const gpu = hardwareData.find(g => g.vendor === selectedGpuInfo.vendor && g.model === selectedGpuInfo.model);
            const gpuUtilization = parseFloat(dom.gpuUtilizationRecommend.value) || 0.8;
            
            if (gpu && requiredVram > 0) {
                const quantWeights = dom.deployment.quant_weights.value;
                
                // 根据量化精度选择对应的算力类型
                let perfField;
                if (quantWeights === "2") {
                    // FP16/BF16
                    perfField = "perf_bf16";
                } else if (quantWeights === "1" && dom.deployment.quant_weights.selectedOptions[0].text === "FP8") {
                    perfField = "perf_fp8";
                } else if (quantWeights === "1" && dom.deployment.quant_weights.selectedOptions[0].text === "INT8") {
                    perfField = "perf_int8";
                } else if (quantWeights === "0.5") {
                    perfField = "perf_int4";
                } else {
                    perfField = "perf_bf16"; // 默认使用BF16
                }
                
                // 检查是否支持选择的量化精度
                if (gpu[perfField] <= 0) {
                    dom.specificGpuCards.textContent = '不支持此精度';
                    return;
                }
                
                const vramCards = Math.ceil(requiredVram / gpu.vram);
                const computingCards = Math.ceil(requiredComputing / (gpu[perfField] * gpuUtilization));
                
                // 取两者的最大值
                const totalCards = Math.max(vramCards, computingCards);
                
                dom.specificGpuCards.innerHTML = `${totalCards} `;
            } else { 
                dom.specificGpuCards.textContent = 'N/A'; 
            }
        } catch (e) {
            console.error("解析GPU信息出错:", e);
            dom.specificGpuCards.textContent = 'N/A';
        }
    }
    
    function updateCustomCardCount(requiredVram, requiredComputing) {
        // 获取输入值
        const vram = parseFloat(dom.customVram.value);
        const gpuUtilization = parseFloat(dom.gpuUtilizationCustom.value) || 0.8;
        
        // 检查是否输入了显存大小
        if (!vram || vram <= 0) {
            dom.customCardsResult.textContent = '请输入有效的显存大小';
            return;
        }
        
        // 检查是否计算了所需显存
        if (!requiredVram || requiredVram <= 0) {
            dom.customCardsResult.textContent = '无法计算所需显存';
            return;
        }
        
        // 获取量化精度
        const quantWeights = dom.deployment.quant_weights.value;
        const quantText = dom.deployment.quant_weights.selectedOptions[0].text;
        
        // 根据量化精度选择对应的输入元素
        let customPerf;
        if (quantWeights === "2") {
            // FP16/BF16
            customPerf = parseFloat(quantText === "FP16" ? dom.customFp16.value : dom.customBf16.value);
        } else if (quantWeights === "1" && quantText === "FP8") {
            customPerf = parseFloat(dom.customFp8.value);
        } else if (quantWeights === "1" && quantText === "INT8") {
            customPerf = parseFloat(dom.customInt8.value);
        } else if (quantWeights === "0.5") {
            customPerf = parseFloat(dom.customInt4.value);
        } else {
            customPerf = parseFloat(dom.customBf16.value); // 默认使用BF16
        }
        
        // 计算所需卡数
        const vramCards = Math.ceil(requiredVram / vram);
        
        // 如果没有填写算力或算力为0，只显示显存限制的卡数
        if (isNaN(customPerf) || customPerf <= 0) {
            dom.customCardsResult.innerHTML = `${vramCards} 张 (不支持${quantText}精度)`;
            return;
        }
        
        // 计算算力限制的卡数
        const computingCards = Math.ceil(requiredComputing / (customPerf * gpuUtilization));
        
        // 取两者的最大值
        const totalCards = Math.max(vramCards, computingCards);
        
        dom.customCardsResult.innerHTML = `${totalCards} 张`;
        
        // 调试信息
        console.log({
            requiredVram,
            requiredComputing,
            vram,
            customPerf,
            gpuUtilization,
            vramCards,
            computingCards,
            totalCards,
            quantWeights,
            quantText
        });
    }
    
    let simulationInterval = null;
    const sampleText = "AICompass 是一个强大的模型算力评估工具。它通过对模型结构、部署配置和服务指标的综合分析，精确计算出所需的显存资源，并推荐合适的硬件配置...";
    
    function startSimulation() {
        if (simulationInterval) clearInterval(simulationInterval);
        resetSimulation();
        
        // 使用TTFT和TPOT进行模拟
        let ttft = parseInt(dom.ttft.value);
        let tpot = parseInt(dom.tpot.value);
        
        // 验证TTFT和TPOT的值
        if (isNaN(ttft) || ttft < 0) {
            ttft = 500; // 默认值
            dom.ttft.value = ttft;
        }
        
        if (isNaN(tpot) || tpot <= 0) {
            tpot = 50; // 默认值
            dom.tpot.value = tpot;
        }
        
        // 更新速度显示
        updateSpeedDisplay();
        
        // 首先模拟TTFT的等待时间
        dom.simulationOutput.value = "正在思考...";
        
        setTimeout(() => {
            dom.simulationOutput.value = "";
            let currentIndex = 0;
            
            // 然后按照TPOT的速率输出文本
            simulationInterval = setInterval(() => {
                if (currentIndex < sampleText.length) {
                    dom.simulationOutput.value += sampleText[currentIndex++];
                    dom.simulationOutput.scrollTop = dom.simulationOutput.scrollHeight;
                } else {
                    clearInterval(simulationInterval);
                }
            }, tpot);
        }, ttft);
    }
    
    function resetSimulation() {
        if (simulationInterval) clearInterval(simulationInterval);
        dom.simulationOutput.value = '';
    }

    function updateSpeedDisplay() {
        const ttft = dom.ttft.value;
        const tpot = dom.tpot.value;
        
        if (ttft && tpot) {
            dom.simulationSpeedDisplay.textContent = `(TTFT: ${ttft}ms, TPOT: ${tpot}ms)`;
        } else if (ttft) {
            dom.simulationSpeedDisplay.textContent = `(TTFT: ${ttft}ms)`;
        } else if (tpot) {
            dom.simulationSpeedDisplay.textContent = `(TPOT: ${tpot}ms)`;
        } else {
            dom.simulationSpeedDisplay.textContent = '';
        }
    }

    // 工具函数：更新输入/输出长度滑块的label，并保证两者之和不超过最大序列长度
    function updateInputOutputSliderLimits() {
        // 获取最大序列长度
        const maxSeqLen = parseInt(dom.deployment.input_length.max) || 32768;
        // 当前输入/输出长度
        let inputVal = parseInt(dom.deployment.input_length.value) || 0;
        let outputVal = parseInt(dom.deployment.output_length.value) || 0;
        // 如果两者之和超过最大序列长度，则当前操作的滑块自动回调
        // 判断事件来源
        const activeElement = document.activeElement;
        if (activeElement === dom.deployment.input_length) {
            if (inputVal + outputVal > maxSeqLen) {
                inputVal = maxSeqLen - outputVal;
                dom.deployment.input_length.value = inputVal;
            }
        } else if (activeElement === dom.deployment.output_length) {
            if (inputVal + outputVal > maxSeqLen) {
                outputVal = maxSeqLen - inputVal;
                dom.deployment.output_length.value = outputVal;
            }
        } else {
            // 初始化或其他情况也做一次兜底
            if (inputVal + outputVal > maxSeqLen) {
                outputVal = maxSeqLen - inputVal;
                dom.deployment.output_length.value = outputVal;
            }
        }
        // label动态显示"当前值/最大值"
        dom.inputLengthValue.textContent = `${dom.deployment.input_length.value} / ${dom.deployment.input_length.max}`;
        dom.outputLengthValue.textContent = `${dom.deployment.output_length.value} / ${dom.deployment.output_length.max}`;
    }

    // 设置输入验证
    function setupInputValidation() {
        // 获取所有数值输入框
        const numericInputs = document.querySelectorAll('input[type="number"]');
        
        numericInputs.forEach(input => {
            // 设置最小值为0，确保不能输入负数
            if (!input.hasAttribute('min')) {
                input.setAttribute('min', '0');
            }
            
            // 添加输入事件监听器，验证输入内容
            input.addEventListener('input', function() {
                // 移除非数字字符（保留小数点和负号）
                let value = this.value;
                
                // 如果允许负数（有min属性且min<0），则保留负号
                const minValue = parseFloat(this.getAttribute('min') || '0');
                const allowNegative = minValue < 0;
                
                if (!allowNegative) {
                    // 不允许负数，移除负号
                    value = value.replace(/-/g, '');
                } else {
                    // 允许负数，但只允许开头有一个负号
                    if (value.startsWith('-')) {
                        value = '-' + value.substring(1).replace(/-/g, '');
                    } else {
                        value = value.replace(/-/g, '');
                    }
                }
                
                // 确保只有一个小数点
                const parts = value.split('.');
                if (parts.length > 2) {
                    value = parts[0] + '.' + parts.slice(1).join('');
                }
                
                // 移除非数字字符（已经处理过小数点和负号）
                value = value.replace(/[^\d.-]/g, '');
                
                // 更新输入框的值
                if (this.value !== value) {
                    this.value = value;
                }
                
                // 检查是否超出最大值或最小值
                const numValue = parseFloat(value);
                if (!isNaN(numValue)) {
                    if (this.hasAttribute('max') && numValue > parseFloat(this.getAttribute('max'))) {
                        this.value = this.getAttribute('max');
                    }
                    if (this.hasAttribute('min') && numValue < parseFloat(this.getAttribute('min'))) {
                        this.value = this.getAttribute('min');
                    }
                }
            });
            
            // 添加失焦事件监听器，处理空值
            input.addEventListener('blur', function() {
                // 如果值为空且有默认值属性，则使用默认值
                if (this.value === '' && this.hasAttribute('data-default')) {
                    this.value = this.getAttribute('data-default');
                }
                
                // 如果值不是有效数字，则清空
                const numValue = parseFloat(this.value);
                if (isNaN(numValue)) {
                    this.value = '';
                }
            });
        });
        
        // 特殊处理GPU利用率输入框
        const utilizationInputs = [dom.gpuUtilizationRecommend, dom.gpuUtilizationCustom];
        utilizationInputs.forEach(input => {
            if (input) {
                input.addEventListener('blur', function() {
                    const value = parseFloat(this.value);
                    if (!isNaN(value)) {
                        // 确保GPU利用率在0.1到1之间
                        if (value < 0.1) this.value = '0.1';
                        if (value > 1) this.value = '1';
                    } else {
                        this.value = '0.8'; // 默认值
                    }
                });
            }
        });
        
        // 为自定义硬件部分的输入框添加特殊验证
        const customHwInputs = [
            dom.customVram, 
            dom.customMemoryBw, 
            dom.customFp16, 
            dom.customBf16, 
            dom.customFp8, 
            dom.customInt8, 
            dom.customInt4, 
            dom.customInterconnectBw
        ];
        
        customHwInputs.forEach(input => {
            if (input) {
                // 确保自定义硬件输入框的值为正数
                input.addEventListener('blur', function() {
                    const value = parseFloat(this.value);
                    if (!isNaN(value) && value <= 0) {
                        this.value = ''; // 如果是0或负数，则清空
                    }
                });
            }
        });
    }

    // Initial calls
    initialize();
});