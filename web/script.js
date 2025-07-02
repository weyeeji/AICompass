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
            max_position_embeddings: document.getElementById('max-position-embeddings'),
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
        throughputValue: document.getElementById('throughput-value'),
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
        setupMobileSupport();
        initMemoryChart();
        switchModelMode('preset');
        updateConditionalFields();
        
        // 确保在初始化时渲染所有滑块标签
        renderSliderLabels(dom.deployment.input_length, document.getElementById('input-length-ticks'), document.getElementById('input-length-labels'));
        renderSliderLabels(dom.deployment.output_length, document.getElementById('output-length-ticks'), document.getElementById('output-length-labels'));
        renderSliderLabels(dom.deployment.batch_size, document.getElementById('concurrency-ticks'), document.getElementById('batch-size-labels'));
        
        // 在初始化时填充预设模型数据
        if (dom.presetSelect.options.length > 0) {
            populatePresetData();
        } else {
            updateInputOutputSliderLimits();
        }
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
        
        // 添加对max_position_embeddings的监听
        dom.model.max_position_embeddings.addEventListener('change', () => {
            updateInputOutputSliderLimits();
            updateAllCalculations();
        });
        
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
            updateAllCalculations();
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
        dom.startSimBtn.addEventListener('click', () => {
            // 防止连续快速点击
            if (!simulationInProgress) {
                startSimulation();
            }
        });
        dom.resetSimBtn.addEventListener('click', resetSimulation);
        
        // 添加窗口大小变化事件监听，重新渲染滑块标签
        window.addEventListener('resize', debounce(() => {
            renderSliderLabels(dom.deployment.input_length, document.getElementById('input-length-ticks'), document.getElementById('input-length-labels'));
            renderSliderLabels(dom.deployment.output_length, document.getElementById('output-length-ticks'), document.getElementById('output-length-labels'));
            renderSliderLabels(dom.deployment.batch_size, document.getElementById('concurrency-ticks'), document.getElementById('batch-size-labels'));
        }, 250));
    }
    
    function switchModelMode(mode) {
        const isPreset = mode === 'preset';
        dom.presetModeBtn.classList.toggle('active', isPreset);
        dom.customModeBtn.classList.toggle('active', !isPreset);
        dom.presetSelectGroup.style.display = isPreset ? 'flex' : 'none';
        
        // 设置输入框的禁用状态
        Object.values(dom.model).forEach(input => {
            if (input && (input.tagName === 'SELECT' || input.tagName === 'INPUT')) {
                input.disabled = isPreset;
            }
        });
        
        if (isPreset) {
            populatePresetData();
        } else {
            // 清空所有输入框
            Object.keys(dom.model).forEach(key => {
                if (key === 'has_vision_modal' || key === 'has_audio_modal') {
                    dom.model[key].checked = key === 'has_vision_modal';
                } else if (dom.model[key] && dom.model[key].value !== undefined) {
                    dom.model[key].value = '';
                }
            });
            
            // 设置默认值
            dom.model.type.value = 'Dense';
            dom.model.max_position_embeddings.value = DEFAULT_MAX_SEQLEN;
            
            updateConditionalFields();
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
        dom.model.max_position_embeddings.value = params.max_position_embeddings || '';
        
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
        
        // 设置输入输出长度的初始值，如果预设值过大则使用默认值1024
        dom.deployment.input_length.value = (params.input_length && params.input_length <= 16384) ? params.input_length : 1024;
        dom.deployment.output_length.value = (params.output_length && params.output_length <= 16384) ? params.output_length : 1024;
        
        // 更新界面
        updateInputOutputSliderLimits();
        updateConditionalFields();
        updateAllCalculations();
    }
    
    function renderSliderLabels(slider, datalist, labelContainer) {
        const sliderMin = parseFloat(slider.min);
        const sliderMax = parseFloat(slider.max);
        
        // 清除现有标签
        labelContainer.innerHTML = '';
        
        // 如果范围无效，直接返回
        if (sliderMax <= sliderMin) return;
        
        // 获取datalist中的所有选项值
        const allOptions = Array.from(datalist.getElementsByTagName('option'))
            .map(option => parseFloat(option.value))
            .filter(value => !isNaN(value) && value >= sliderMin && value <= sliderMax)
            .sort((a, b) => a - b);
        
        // 如果没有选项，则使用最小值和最大值
        if (allOptions.length === 0) {
            addLabel(sliderMin, 0);
            addLabel(sliderMax, 100);
            return;
        }
        
        // 始终显示最小值
        addLabel(sliderMin, 0);
        
        // 选择中间的4个标签值
        // 优先选择2的幂次值，并且是常用值
        const commonValues = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
        
        // 过滤出范围内的常用值
        const validCommonValues = commonValues.filter(val => 
            val > sliderMin && val < sliderMax && allOptions.includes(val)
        );
        
        // 如果常用值不够4个，则使用所有可用的常用值
        let selectedValues = [];
        
        if (validCommonValues.length <= 4) {
            selectedValues = validCommonValues;
        } else {
            // 从最大值往左选择4个值
            // 首先找到小于最大值的最大常用值的索引
            const maxValidValueIndex = validCommonValues.findIndex(val => val >= sliderMax) - 1;
            const startIndex = maxValidValueIndex >= 0 ? maxValidValueIndex : validCommonValues.length - 1;
            
            // 从这个索引开始，选择4个或更少的值
            const count = Math.min(4, validCommonValues.length);
            const endIndex = Math.max(0, startIndex - count + 1);
            
            for (let i = startIndex; i >= endIndex; i--) {
                selectedValues.unshift(validCommonValues[i]);
            }
            
            // 如果选择的值少于4个，尝试从更小的值中选择
            if (selectedValues.length < count && endIndex > 0) {
                const remainingCount = count - selectedValues.length;
                for (let i = endIndex - 1; i >= 0 && selectedValues.length < count; i--) {
                    selectedValues.unshift(validCommonValues[i]);
                }
            }
        }
        
        // 添加选中的中间值
        selectedValues.forEach(value => {
            const percent = ((value - sliderMin) / (sliderMax - sliderMin)) * 100;
            addLabel(value, percent);
        });
        
        // 始终显示最大值
        addLabel(sliderMax, 100);
        
        // 辅助函数：添加标签
        function addLabel(value, percent) {
            const label = document.createElement('span');
            
            // 格式化显示值
            let formattedValue;
            if (value >= 1000000) {
                formattedValue = `${Math.round(value / 1000000)}M`;
            } else if (value >= 1000) {
                formattedValue = `${Math.round(value / 1000)}K`;
            } else {
                formattedValue = Math.round(value);
            }
            
            label.textContent = formattedValue;
            label.style.left = `${percent}%`;
            
            // 特殊处理第一个和最后一个标签的位置
            if (percent === 0) {
                label.style.left = '0%';
                label.style.transform = 'none';
                label.style.textAlign = 'left';
            } else if (percent === 100) {
                label.style.left = '100%';
                label.style.transform = 'translateX(-100%)';
                label.style.textAlign = 'right';
            }
            
            labelContainer.appendChild(label);
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
        // 确保先更新吞吐量，它依赖于TPOT
        updateSpeedDisplay();
        
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
        const DENSE_ACTIVATION_FACTOR = 4;  // Dense模型激活值因子
        const LLM_ACTIVATION_FACTOR = 1;    // 多模态中LLM部分激活值因子
        const MOE_SHARED_ACTIVATION_FACTOR = 4;  // MoE共享部分激活值因子
        const MOE_EXPERT_ACTIVATION_FACTOR = 4;  // MoE专家部分激活值因子
        const VISION_ACTIVATION_FACTOR = 1;    // 视觉模态激活值因子
        const AUDIO_ACTIVATION_FACTOR = 1;     // 音频模态激活值因子
        
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
        
        // 获取TTFT和TPOT
        const ttft = parseFloat(dom.ttft.value) || 500; // 默认500ms
        const tpot = parseFloat(dom.tpot.value) || 50; // 默认50ms
        const throughput = parseFloat(dom.throughput.value) || 20; // 吞吐量 tokens/s (由TPOT计算得到)
        
        const batch_size = p.batch_size;
        const input_length = parseFloat(dom.deployment.input_length.value) || 0;
        const output_length = parseFloat(dom.deployment.output_length.value) || 0;
        const seq_length = input_length + output_length;
        const modelType = dom.model.type.value;
        
        // 量化精度对应的算力类型和内存效率
        const quantWeights = parseFloat(dom.deployment.quant_weights.value) || 2;
        const quantActivations = parseFloat(dom.deployment.quant_activations.value) || 2;
        
        // 基于业界共识的计算方法
        // 参考：https://arxiv.org/abs/2203.15556, https://arxiv.org/abs/2104.04473
        
        // 每个token的FLOPs计算基准 - 基于Transformer架构的共识估算
        // 对于前向推理，每个token大约需要2 * 6 * h^2 的FLOPs，其中h是隐藏层大小
        // 2表示矩阵乘法的乘加操作，6表示Transformer中的主要矩阵乘法数量
        const BASE_FLOPS_PER_TOKEN = 12; // 基础系数
        
        // 模型规模系数 - 根据隐藏层大小和层数进行缩放
        const MODEL_SCALE_FACTOR = (p.hidden_size * p.num_hidden_layers) / 1e6;
        
        // 注意力机制效率系数 - 考虑MHA vs MQA/GQA的差异
        const ATTENTION_EFFICIENCY = Math.max(p.num_key_value_heads, 1) / Math.max(p.num_attention_heads, 1);
        
        // 批处理效率 - 考虑批处理大小带来的并行效率提升
        const BATCH_EFFICIENCY = 0.7 + 0.3 * Math.log10(Math.max(batch_size, 1)) / Math.log10(128);
        
        // 量化精度系数 - 不同量化精度对算力的影响
        const QUANT_COMPUTE_FACTOR = Math.pow(quantWeights, 0.5); // 平方根关系
        
        let required_flops = 0;
        
        switch (modelType) {
            case 'Dense':
                // 计算Dense模型的算力需求
                const dense_params = p.model_total_params * 1e9; // 参数数量，转换为十亿
                
                // 1. Prefill阶段(TTFT)算力
                // 在Prefill阶段，需要处理所有输入token，计算量与输入长度成正比
                const prefill_flops = (
                    BASE_FLOPS_PER_TOKEN * 
                    MODEL_SCALE_FACTOR * 
                    input_length * 
                    dense_params * 
                    (1 / ATTENTION_EFFICIENCY) * 
                    batch_size
                ) / 1e12; // 转换为TFLOPS
                
                // Prefill阶段的计算速率 (TFLOPS) = 计算量 / 时间
                const prefill_compute_rate = prefill_flops / (ttft / 1000); // ttft从毫秒转为秒
                
                // 2. Decode阶段(TPOT)算力
                // 在Decode阶段，每生成一个token都需要处理整个序列，但有缓存优化
                const decode_flops_per_token = (
                    BASE_FLOPS_PER_TOKEN * 
                    MODEL_SCALE_FACTOR * 
                    (1 + 0.1 * Math.log10(seq_length)) * // 序列长度的对数影响
                    dense_params * 
                    (1 / ATTENTION_EFFICIENCY) *
                    0.5 * // Decode阶段有KV缓存，计算量约为Prefill的一半
                    batch_size
                ) / 1e12; // 转换为TFLOPS
                
                // Decode阶段的计算速率 (TFLOPS) = 每token计算量 * 吞吐量
                const decode_compute_rate = decode_flops_per_token * throughput;
                
                // 总算力需求是两个阶段的最大值，因为它们通常不会同时达到峰值
                required_flops = Math.max(prefill_compute_rate, decode_compute_rate) * QUANT_COMPUTE_FACTOR;
                break;
                
            case 'MoE':
                // 计算MoE模型的算力需求
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
                
                // MoE模型的有效参数量 = 共享参数 + 激活的专家参数
                const effective_params = shared_params + expert_params * expert_activation_ratio;
                
                // MoE模型的计算与Dense类似，但需要考虑专家激活比例
                // 1. Prefill阶段(TTFT)算力
                const moe_prefill_flops = (
                    BASE_FLOPS_PER_TOKEN * 
                    MODEL_SCALE_FACTOR * 
                    input_length * 
                    (effective_params / 1e9) * // 转换为十亿参数
                    (1 / ATTENTION_EFFICIENCY) * 
                    batch_size * 
                    1.2 // MoE路由开销
                ) / 1e12; // 转换为TFLOPS
                
                const moe_prefill_compute_rate = moe_prefill_flops / (ttft / 1000); // ttft从毫秒转为秒
                
                // 2. Decode阶段(TPOT)算力
                const moe_decode_flops_per_token = (
                    BASE_FLOPS_PER_TOKEN * 
                    MODEL_SCALE_FACTOR * 
                    (1 + 0.1 * Math.log10(seq_length)) * // 序列长度的对数影响
                    (effective_params / 1e9) * // 转换为十亿参数
                    (1 / ATTENTION_EFFICIENCY) *
                    0.5 * // Decode阶段有KV缓存
                    batch_size * 
                    1.2 // MoE路由开销
                ) / 1e12; // 转换为TFLOPS
                
                const moe_decode_compute_rate = moe_decode_flops_per_token * throughput;
                
                required_flops = Math.max(moe_prefill_compute_rate, moe_decode_compute_rate) * QUANT_COMPUTE_FACTOR;
                break;
                
            case 'Multimodal':
                // 计算多模态模型的算力需求
                const llm_params = Math.max(p.model_total_params, 0.1) * 1e9;
                
                // 1. LLM部分的Prefill阶段(TTFT)算力
                const mm_prefill_flops = (
                    BASE_FLOPS_PER_TOKEN * 
                    MODEL_SCALE_FACTOR * 
                    input_length * 
                    (llm_params / 1e9) * // 转换为十亿参数
                    (1 / ATTENTION_EFFICIENCY) * 
                    batch_size * 
                    1.3 // 多模态处理额外开销
                ) / 1e12; // 转换为TFLOPS
                
                const mm_prefill_compute_rate = mm_prefill_flops / (ttft / 1000); // ttft从毫秒转为秒
                
                // 2. LLM部分的Decode阶段(TPOT)算力
                const mm_decode_flops_per_token = (
                    BASE_FLOPS_PER_TOKEN * 
                    MODEL_SCALE_FACTOR * 
                    (1 + 0.1 * Math.log10(seq_length)) * // 序列长度的对数影响
                    (llm_params / 1e9) * // 转换为十亿参数
                    (1 / ATTENTION_EFFICIENCY) *
                    0.5 * // Decode阶段有KV缓存
                    batch_size
                ) / 1e12; // 转换为TFLOPS
                
                const mm_decode_compute_rate = mm_decode_flops_per_token * throughput;
                
                // 3. 模态处理算力 (一次性处理，不随token生成而变化)
                let modal_compute_rate = 0;
                
                // 视觉模态算力 (如果启用)
                if (p.has_vision_modal) {
                    const vision_params = Math.max(p.vision_params, 0.01) * 1e9;
                    // 视觉模型的计算量与参数量和批处理大小成正比
                    modal_compute_rate += (vision_params / 1e12) * 0.05 * batch_size;
                }
                
                // 音频模态算力 (如果启用)
                if (p.has_audio_modal) {
                    const audio_params = Math.max(p.audio_params, 0.01) * 1e9;
                    // 音频模型的计算量与参数量和批处理大小成正比
                    modal_compute_rate += (audio_params / 1e12) * 0.03 * batch_size;
                }
                
                // 多模态总算力 = max(Prefill算力, Decode算力) + 模态处理算力
                required_flops = (Math.max(mm_prefill_compute_rate, mm_decode_compute_rate) + modal_compute_rate) * QUANT_COMPUTE_FACTOR;
                break;
                
            default:
                return 0;
        }
        
        // 考虑显存带宽限制对算力的影响
        // 显存带宽通常是高端GPU的瓶颈，特别是对于大型模型
        const MEMORY_BW_FACTOR = 0.9; // 显存带宽利用率
        
        // 考虑系统开销和其他因素
        const overhead_ratio = p.overhead_ratio;
        required_flops *= (1 + overhead_ratio);
        
        // 应用批处理效率
        required_flops *= BATCH_EFFICIENCY;
        
        // 确保返回值不为NaN或无穷大
        if (isNaN(required_flops) || !isFinite(required_flops)) {
            return 0;
        }
        
        // 应用一个全局校准因子，确保结果在合理范围内
        // 这个因子是基于实际部署经验调整的
        const GLOBAL_CALIBRATION = 0.01; // 全局校准因子
        return required_flops * GLOBAL_CALIBRATION;
    }

    function updateHardwareRecommendations(requiredVram, requiredComputing) {
        dom.hwRecommendations.innerHTML = '';
        if (requiredVram <= 0) { 
            dom.hwRecommendations.innerHTML = `<tr><td colspan="5">请先输入模型参数</td></tr>`; 
            return; 
        }
        
        const gpuUtilization = parseFloat(dom.gpuUtilizationRecommend.value) || 0.8;
        const quantWeights = dom.deployment.quant_weights.value;
        
        // 检查当前选择的模型是否是盘古大模型
        let isPanguModel = false;
        const selectedIndex = dom.presetSelect.value;
        if (selectedIndex !== '' && modelData[selectedIndex]) {
            const selectedModel = modelData[selectedIndex];
            isPanguModel = selectedModel.model_name && selectedModel.model_name.includes('盘古');
        }
        
        // 根据量化精度选择对应的算力类型
        let perfField;
        if (quantWeights === "2") {
            // 区分FP16和BF16
            const selectedText = dom.deployment.quant_weights.selectedOptions[0].text;
            if (selectedText === "FP16") {
                perfField = "perf_fp16";
            } else if (selectedText === "BF16") {
                perfField = "perf_bf16";
            } else {
                perfField = "perf_bf16"; // 默认使用BF16
            }
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
        
        // 定义推荐的GPU型号优先级（优先推荐高性能和主流GPU）
        const getRecommendationPriority = (gpu) => {
            const key = `${gpu.vendor}_${gpu.model}`;
            
            // 如果是盘古模型，优先推荐昇腾显卡
            if (isPanguModel) {
                const panguPriorityMap = {
                    'Huawei_Atlas 300I Duo 96GB': 1,
                    'Huawei_Atlas 300I Duo 48GB': 2,
                    'Huawei_Atlas 300I (Model 3010)': 3,
                    'Huawei_Atlas 300T Pro': 4,
                    'NVIDIA_H100 PCIe': 5  // 最后加一个英伟达显卡
                };
                return panguPriorityMap[key] || 999;
            }
            
            // 默认优先级
            const priorityMap = {
                'NVIDIA_A100 PCIe': 1,
                'NVIDIA_H100 PCIe': 2,
                'AMD_Instinct MI300X': 3,
                'Huawei_Atlas 300I Duo 96GB': 4,
                '壁仞科技_BR100': 5,
                '天数智芯_BI-V200': 6,
                '寒武纪_思元590': 7,
                'AMD_Instinct MI250X': 8,
                'Huawei_Atlas 300I Duo 48GB': 9,
                '摩尔线程_MTT S4000': 10
            };
            return priorityMap[key] || 999;  // 其他GPU排在最后
        };
        
        // 混合推荐策略：优先推荐指定的GPU型号
        const recommendedGpus = gpuWithCards
            .filter(gpu => gpu.supportsQuantization)
            .sort((a, b) => {
                const priorityA = getRecommendationPriority(a);
                const priorityB = getRecommendationPriority(b);
                
                // 如果都是优先推荐的GPU，按优先级排序
                if (priorityA !== 999 && priorityB !== 999) {
                    return priorityA - priorityB;
                }
                
                // 如果只有一个是优先推荐的GPU
                if (priorityA !== 999) return -1;
                if (priorityB !== 999) return 1;
                
                // 都不是优先推荐的GPU，按厂商和性能排序
                const getVendorPriority = (vendor) => {
                    if (vendor === 'NVIDIA') return 0;
                    if (vendor === 'AMD') return 1;
                    if (vendor === 'Huawei') return 2;
                    if (['壁仞科技', '天数智芯', '寒武纪', '摩尔线程', '海光'].includes(vendor)) return 3;
                    return 4;
                };
                const vendorPriorityA = getVendorPriority(a.vendor);
                const vendorPriorityB = getVendorPriority(b.vendor);
                const vendorDiff = vendorPriorityA - vendorPriorityB;
                if (vendorDiff !== 0) return vendorDiff;
                
                const cardsDiff = a.totalCards - b.totalCards;
                if (cardsDiff !== 0) return cardsDiff;
                
                return b.vram - a.vram;
            })
            .slice(0, 5);
        
        recommendedGpus
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
                    // 区分FP16和BF16
                    const selectedText = dom.deployment.quant_weights.selectedOptions[0].text;
                    if (selectedText === "FP16") {
                        perfField = "perf_fp16";
                    } else if (selectedText === "BF16") {
                        perfField = "perf_bf16";
                    } else {
                        perfField = "perf_bf16"; // 默认使用BF16
                    }
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
            // 区分FP16和BF16
            if (quantText === "FP16") {
                customPerf = parseFloat(dom.customFp16.value);
            } else if (quantText === "BF16") {
                customPerf = parseFloat(dom.customBf16.value);
            } else {
                // 默认使用BF16
                customPerf = parseFloat(dom.customBf16.value);
            }
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
    let simulationInProgress = false; // 添加标志变量，跟踪模拟是否正在进行
    const sampleText = "AICompass 是一个强大的模型算力评估工具。它通过对模型结构、部署配置和服务指标的综合分析，精确计算出所需的显存资源，并推荐合适的硬件配置...";
    
    function startSimulation() {
        // 如果模拟正在进行中，先停止它
        if (simulationInProgress) {
            resetSimulation();
        }
        
        simulationInProgress = true; // 设置模拟进行中标志
        
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
        
        // 禁用开始按钮，防止重复点击
        dom.startSimBtn.disabled = true;
        
        // 首先模拟TTFT的等待时间
        dom.simulationOutput.value = "正在思考...";
        
        let ttftTimeout = setTimeout(() => {
            dom.simulationOutput.value = "";
            let currentIndex = 0;
            
            // 然后按照TPOT的速率输出文本
            let tokenInterval = setInterval(() => {
                if (currentIndex < sampleText.length) {
                    dom.simulationOutput.value += sampleText[currentIndex++];
                    dom.simulationOutput.scrollTop = dom.simulationOutput.scrollHeight;
                } else {
                    clearInterval(tokenInterval);
                    simulationInterval = null;
                    simulationInProgress = false; // 模拟完成
                    dom.startSimBtn.disabled = false; // 重新启用开始按钮
                }
            }, tpot);
            
            // 更新simulationInterval引用
            simulationInterval = {
                ttftTimeout: null, // 已经执行完毕
                interval: tokenInterval
            };
        }, ttft);
        
        // 保存timeout引用，以便在需要时清除
        simulationInterval = {
            ttftTimeout: ttftTimeout,
            interval: null
        };
    }
    
    function resetSimulation() {
        // 清除所有定时器
        if (simulationInterval) {
            if (simulationInterval.ttftTimeout) {
                clearTimeout(simulationInterval.ttftTimeout);
            }
            if (simulationInterval.interval) {
                clearInterval(simulationInterval.interval);
            }
            simulationInterval = null;
        }
        
        dom.simulationOutput.value = '';
        simulationInProgress = false; // 重置模拟状态
        dom.startSimBtn.disabled = false; // 确保开始按钮可用
    }

    function updateSpeedDisplay() {
        const ttft = dom.ttft.value;
        const tpot = dom.tpot.value;
        
        // 更新吞吐量，由TPOT计算得到 (1/TPOT×1000)
        if (tpot && tpot > 0) {
            const throughput = Math.round(1000 / tpot);
            dom.throughput.value = throughput;
            // 更新吞吐量显示
            dom.throughputValue.textContent = throughput;
        } else {
            dom.throughput.value = 20; // 默认值
            dom.throughputValue.textContent = 20;
        }
        
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
        // 获取最大序列长度，优先使用模型的max_position_embeddings
        const maxPositionEmbeddings = parseInt(dom.model.max_position_embeddings.value) || DEFAULT_MAX_SEQLEN;
        
        // 更新滑块的最大值
        dom.deployment.input_length.max = maxPositionEmbeddings;
        dom.deployment.output_length.max = maxPositionEmbeddings;
        
        // 当前输入/输出长度
        let inputVal = parseInt(dom.deployment.input_length.value) || 0;
        let outputVal = parseInt(dom.deployment.output_length.value) || 0;
        
        // 如果当前值超过新的最大值，则调整
        if (inputVal > maxPositionEmbeddings) {
            inputVal = maxPositionEmbeddings;
            dom.deployment.input_length.value = inputVal;
        }
        
        if (outputVal > maxPositionEmbeddings) {
            outputVal = maxPositionEmbeddings;
            dom.deployment.output_length.value = outputVal;
        }
        
        // 如果两者之和超过最大序列长度，则当前操作的滑块自动回调
        // 判断事件来源
        const activeElement = document.activeElement;
        if (activeElement === dom.deployment.input_length) {
            if (inputVal + outputVal > maxPositionEmbeddings) {
                inputVal = maxPositionEmbeddings - outputVal;
                dom.deployment.input_length.value = inputVal;
            }
        } else if (activeElement === dom.deployment.output_length) {
            if (inputVal + outputVal > maxPositionEmbeddings) {
                outputVal = maxPositionEmbeddings - inputVal;
                dom.deployment.output_length.value = outputVal;
            }
        } else {
            // 初始化或其他情况也做一次兜底
            if (inputVal + outputVal > maxPositionEmbeddings) {
                outputVal = maxPositionEmbeddings - inputVal;
                dom.deployment.output_length.value = outputVal;
            }
        }
        
        // 立即更新滑块标签显示
        renderSliderLabels(dom.deployment.input_length, document.getElementById('input-length-ticks'), document.getElementById('input-length-labels'));
        renderSliderLabels(dom.deployment.output_length, document.getElementById('output-length-ticks'), document.getElementById('output-length-labels'));
        
        // label动态显示"当前值/最大值"
        dom.inputLengthValue.textContent = `${dom.deployment.input_length.value} / ${maxPositionEmbeddings}`;
        dom.outputLengthValue.textContent = `${dom.deployment.output_length.value} / ${maxPositionEmbeddings}`;
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

    function setupMobileSupport() {
        // 为所有问号图标添加点击事件，在移动设备上支持点击显示提示
        const helpIcons = document.querySelectorAll('.help-icon');
        helpIcons.forEach(icon => {
            icon.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                
                // 先移除所有其他激活的问号
                helpIcons.forEach(otherIcon => {
                    if (otherIcon !== icon) {
                        otherIcon.classList.remove('active');
                    }
                });
                
                // 切换当前问号的激活状态
                icon.classList.toggle('active');
            });
        });
        
        // 点击页面其他地方关闭所有提示
        document.addEventListener('click', function() {
            helpIcons.forEach(icon => {
                icon.classList.remove('active');
            });
        });
        
        // 优化移动端滑块体验
        const sliders = document.querySelectorAll('input[type="range"]');
        sliders.forEach(slider => {
            // 添加触摸结束事件，确保值更新
            slider.addEventListener('touchend', function() {
                const event = new Event('change');
                slider.dispatchEvent(event);
            });
            
            // 添加触摸移动事件，实时更新显示值
            slider.addEventListener('touchmove', function() {
                if (slider.id === 'input-length') {
                    dom.inputLengthValue.textContent = slider.value;
                } else if (slider.id === 'output-length') {
                    dom.outputLengthValue.textContent = slider.value;
                } else if (slider.id === 'batch-size') {
                    dom.batchSizeValue.textContent = slider.value;
                }
            });
        });
        
        // 检测是否为移动设备
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        if (isMobile) {
            // 调整表单元素大小以适应移动设备
            document.querySelectorAll('select, input[type="number"]').forEach(el => {
                el.style.fontSize = '16px'; // 避免iOS缩放
            });
            
            // 调整表格布局以适应小屏幕
            const table = document.querySelector('.hardware-table');
            if (table) {
                table.classList.add('mobile-table');
            }
        }
    }

    // 防抖函数，避免频繁触发事件
    function debounce(func, wait) {
        let timeout;
        return function() {
            const context = this;
            const args = arguments;
            clearTimeout(timeout);
            timeout = setTimeout(() => {
                func.apply(context, args);
            }, wait);
        };
    }

    // Initial calls
    initialize();
});