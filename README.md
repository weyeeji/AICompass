# AICompass：模型对算力等效建模评估工具

## 项目简介

AICompass是一个用于大语言模型(LLM)算力和显存需求评估的工具，支持Dense、MoE和多模态模型的评估。通过输入模型参数和推理配置，可以快速估算所需的显存和计算资源，并推荐合适的硬件配置。

## 计算方法

### 显存占用计算

#### 稠密模型(Dense)显存计算

稠密模型的显存占用主要包括模型权重、KV缓存、激活值和额外开销四部分：

**模型权重显存**：
- 中文：模型权重显存 = 模型总参数量 × 权重量化精度
- 英文：Model Weights Memory = Total Parameters × Weight Quantization Precision
- 公式：$M_{weights} = P_{total} \times Q_{weight}$

**KV缓存显存**：
- 中文：KV缓存显存 = 批处理大小 × 序列长度 × 层数 × 2 × KV头数 × 头维度 × KV缓存量化精度
- 英文：KV Cache Memory = Batch Size × Sequence Length × Layers × 2 × KV Heads × Head Dimension × KV Cache Quantization Precision
- 公式：$M_{kv} = B \times S \times L \times 2 \times H_{kv} \times D_h \times Q_{kv}$

**激活值显存**：
- 中文：激活值显存 = 批处理大小 × 序列长度 × 隐藏层维度 × 激活因子 × 激活值量化精度
- 英文：Activations Memory = Batch Size × Sequence Length × Hidden Size × Activation Factor × Activations Quantization Precision
- 公式：$M_{act} = B \times S \times H \times F_{act} \times Q_{act}$

**额外开销显存**：
- 中文：额外开销显存 = (模型权重显存 + KV缓存显存 + 激活值显存) × 额外开销比例
- 英文：Overhead Memory = (Model Weights Memory + KV Cache Memory + Activations Memory) × Overhead Ratio
- 公式：$M_{overhead} = (M_{weights} + M_{kv} + M_{act}) \times R_{overhead}$

**总显存**：
- 中文：总显存 = 模型权重显存 + KV缓存显存 + 激活值显存 + 额外开销显存
- 英文：Total Memory = Model Weights Memory + KV Cache Memory + Activations Memory + Overhead Memory
- 公式：$M_{total} = M_{weights} + M_{kv} + M_{act} + M_{overhead}$

#### 混合专家模型(MoE)显存计算

MoE模型的显存占用包括共享权重、专家权重、KV缓存、激活值和额外开销：

**共享权重显存**：
- 中文：共享权重显存 = 共享参数量 × 权重量化精度
- 英文：Shared Weights Memory = Shared Parameters × Weight Quantization Precision
- 公式：$M_{shared} = P_{shared} \times Q_{weight}$

**专家权重显存**：
- 中文：专家权重显存 = 单个专家参数量 × 专家数量 × 权重量化精度
- 英文：Expert Weights Memory = Parameters Per Expert × Number of Experts × Weight Quantization Precision
- 公式：$M_{expert} = P_{expert} \times E \times Q_{weight}$

**KV缓存显存**：与Dense模型相同

**激活值显存**：
- 中文：激活值显存 = 共享激活值 + 专家激活值
  - 共享激活值 = 批处理大小 × 序列长度 × 隐藏层维度 × 共享激活因子 × 激活值量化精度
  - 专家激活值 = 批处理大小 × 序列长度 × 激活专家数 × 专家中间层维度 × 专家激活因子 × 激活值量化精度
- 英文：
  - Shared Activations = Batch Size × Sequence Length × Hidden Size × Shared Activation Factor × Activations Quantization Precision
  - Expert Activations = Batch Size × Sequence Length × Active Experts × Intermediate Size × Expert Activation Factor × Activations Quantization Precision
- 公式：
  - $M_{act\_shared} = B \times S \times H \times F_{shared} \times Q_{act}$
  - $M_{act\_expert} = B \times S \times E_{active} \times I \times F_{expert} \times Q_{act}$
  - $M_{act} = M_{act\_shared} + M_{act\_expert}$

**额外开销显存**：
- 中文：额外开销显存 = (共享权重显存 + 专家权重显存 + KV缓存显存 + 激活值显存) × 额外开销比例
- 英文：Overhead Memory = (Shared Weights Memory + Expert Weights Memory + KV Cache Memory + Activations Memory) × Overhead Ratio
- 公式：$M_{overhead} = (M_{shared} + M_{expert} + M_{kv} + M_{act}) \times R_{overhead}$

**总显存**：
- 中文：总显存 = 共享权重显存 + 专家权重显存 + KV缓存显存 + 激活值显存 + 额外开销显存
- 英文：Total Memory = Shared Weights Memory + Expert Weights Memory + KV Cache Memory + Activations Memory + Overhead Memory
- 公式：$M_{total} = M_{shared} + M_{expert} + M_{kv} + M_{act} + M_{overhead}$

#### 多模态模型(Multimodal)显存计算

多模态模型的显存占用包括基础LLM权重、视觉模态权重、音频模态权重、KV缓存、激活值和额外开销：

**基础LLM权重显存**：
- 中文：基础LLM权重显存 = LLM基础参数量 × 权重量化精度
- 英文：Base LLM Weights Memory = LLM Base Parameters × Weight Quantization Precision
- 公式：$M_{base} = P_{base} \times Q_{weight}$

**视觉模态权重显存**：
- 中文：视觉模态权重显存 = 视觉模态参数量 × 权重量化精度
- 英文：Vision Modal Weights Memory = Vision Modal Parameters × Weight Quantization Precision
- 公式：$M_{vision} = P_{vision} \times Q_{weight}$

**音频模态权重显存**：
- 中文：音频模态权重显存 = 音频模态参数量 × 权重量化精度
- 英文：Audio Modal Weights Memory = Audio Modal Parameters × Weight Quantization Precision
- 公式：$M_{audio} = P_{audio} \times Q_{weight}$

**KV缓存显存**：
- 中文：KV缓存显存 = 批处理大小 × 总序列长度 × 层数 × 2 × KV头数 × 头维度 × KV缓存量化精度
  - 总序列长度 = 文本序列长度 + 图像patch序列长度 × 视觉token因子 + 音频序列长度 × 音频token因子
- 英文：KV Cache Memory = Batch Size × Total Sequence Length × Layers × 2 × KV Heads × Head Dimension × KV Cache Quantization Precision
  - Total Sequence Length = Text Sequence Length + Image Patch Sequence Length × Vision Token Factor + Audio Sequence Length × Audio Token Factor
- 公式：
  - $S_{total} = S_{text} + S_{image} \times F_{vision} + S_{audio} \times F_{audio}$
  - $M_{kv} = B \times S_{total} \times L \times 2 \times H_{kv} \times D_h \times Q_{kv}$

**激活值显存**：
- 中文：激活值显存 = LLM激活值 + 视觉激活值 + 音频激活值
  - LLM激活值 = 批处理大小 × 文本序列长度 × 隐藏层维度 × LLM激活因子 × 激活值量化精度
  - 视觉激活值 = 批处理大小 × 图像patch数 × 隐藏层维度 × 视觉激活因子 × 激活值量化精度
  - 音频激活值 = 批处理大小 × 音频序列长度 × 隐藏层维度 × 音频激活因子 × 激活值量化精度
- 英文：
  - LLM Activations = Batch Size × Text Sequence Length × Hidden Size × LLM Activation Factor × Activations Quantization Precision
  - Vision Activations = Batch Size × Image Patches × Hidden Size × Vision Activation Factor × Activations Quantization Precision
  - Audio Activations = Batch Size × Audio Sequence Length × Hidden Size × Audio Activation Factor × Activations Quantization Precision
- 公式：
  - $M_{act\_llm} = B \times S_{text} \times H \times F_{llm} \times Q_{act}$
  - $M_{act\_vision} = B \times S_{image} \times H \times F_{vision\_act} \times Q_{act}$
  - $M_{act\_audio} = B \times S_{audio} \times H \times F_{audio\_act} \times Q_{act}$
  - $M_{act} = M_{act\_llm} + M_{act\_vision} + M_{act\_audio}$

**额外开销显存**：
- 中文：额外开销显存 = (基础LLM权重显存 + 视觉模态权重显存 + 音频模态权重显存 + KV缓存显存 + 激活值显存) × 额外开销比例
- 英文：Overhead Memory = (Base LLM Weights Memory + Vision Modal Weights Memory + Audio Modal Weights Memory + KV Cache Memory + Activations Memory) × Overhead Ratio
- 公式：$M_{overhead} = (M_{base} + M_{vision} + M_{audio} + M_{kv} + M_{act}) \times R_{overhead}$

**总显存**：
- 中文：总显存 = 基础LLM权重显存 + 视觉模态权重显存 + 音频模态权重显存 + KV缓存显存 + 激活值显存 + 额外开销显存
- 英文：Total Memory = Base LLM Weights Memory + Vision Modal Weights Memory + Audio Modal Weights Memory + KV Cache Memory + Activations Memory + Overhead Memory
- 公式：$M_{total} = M_{base} + M_{vision} + M_{audio} + M_{kv} + M_{act} + M_{overhead}$

### 计算能力需求评估

#### 稠密模型(Dense)算力计算

- 中文：所需算力 = max(Prefill阶段计算速率, Decode阶段计算速率) × 量化精度系数 × 批处理效率 × 全局校准因子
  - Prefill阶段计算速率 = Prefill阶段计算量 / TTFT
    - Prefill阶段计算量 = 基础FLOPs系数 × 模型规模系数 × 输入长度 × 模型参数量 × 注意力效率系数 × 批处理大小
  - Decode阶段计算速率 = 单Token计算量 × 吞吐量
    - 单Token计算量 = 基础FLOPs系数 × 模型规模系数 × 序列长度对数影响 × 模型参数量 × 注意力效率系数 × KV缓存优化系数 × 批处理大小
- 英文：Required Computing Power = max(Prefill Computing Rate, Decode Computing Rate) × Quantization Factor × Batch Efficiency × Global Calibration
  - Prefill Computing Rate = Prefill Computing / TTFT
    - Prefill Computing = Base FLOPs × Model Scale × Input Length × Parameters × Attention Efficiency × Batch Size
  - Decode Computing Rate = Per-Token Computing × Throughput
    - Per-Token Computing = Base FLOPs × Model Scale × Sequence Length Log Effect × Parameters × Attention Efficiency × KV Cache Factor × Batch Size
- 公式：
  - $C_{prefill} = \frac{F_{base} \times S_{model} \times L_{in} \times P_{total} \times (1/E_{attn}) \times B}{TTFT}$
  - $C_{token} = F_{base} \times S_{model} \times (1 + 0.1\log_{10}(L_{seq})) \times P_{total} \times (1/E_{attn}) \times 0.5 \times B$
  - $C_{decode} = C_{token} \times T$
  - $C = \max(C_{prefill}, C_{decode}) \times F_{quant} \times E_{batch} \times G_{cal}$

#### 混合专家模型(MoE)算力计算

- 中文：所需算力 = max(Prefill阶段计算速率, Decode阶段计算速率) × 量化精度系数 × 批处理效率 × 全局校准因子
  - 有效参数量 = 共享参数量 + 专家参数量 × 专家激活比例
  - Prefill阶段计算速率 = Prefill阶段计算量 / TTFT
    - Prefill阶段计算量 = 基础FLOPs系数 × 模型规模系数 × 输入长度 × 有效参数量 × 注意力效率系数 × 批处理大小 × MoE路由开销
  - Decode阶段计算速率 = 单Token计算量 × 吞吐量
    - 单Token计算量 = 基础FLOPs系数 × 模型规模系数 × 序列长度对数影响 × 有效参数量 × 注意力效率系数 × KV缓存优化系数 × 批处理大小 × MoE路由开销
- 英文：Required Computing Power = max(Prefill Computing Rate, Decode Computing Rate) × Quantization Factor × Batch Efficiency × Global Calibration
  - Effective Parameters = Shared Parameters + Expert Parameters × Expert Activation Ratio
  - Prefill Computing Rate = Prefill Computing / TTFT
    - Prefill Computing = Base FLOPs × Model Scale × Input Length × Effective Parameters × Attention Efficiency × Batch Size × MoE Routing Factor
  - Decode Computing Rate = Per-Token Computing × Throughput
    - Per-Token Computing = Base FLOPs × Model Scale × Sequence Length Log Effect × Effective Parameters × Attention Efficiency × KV Cache Factor × Batch Size × MoE Routing Factor
- 公式：
  - $P_{effective} = P_{shared} + P_{expert} \times R_{expert}$
  - $C_{prefill} = \frac{F_{base} \times S_{model} \times L_{in} \times P_{effective} \times (1/E_{attn}) \times B \times F_{moe}}{TTFT}$
  - $C_{token} = F_{base} \times S_{model} \times (1 + 0.1\log_{10}(L_{seq})) \times P_{effective} \times (1/E_{attn}) \times 0.5 \times B \times F_{moe}$
  - $C_{decode} = C_{token} \times T$
  - $C = \max(C_{prefill}, C_{decode}) \times F_{quant} \times E_{batch} \times G_{cal}$

#### 多模态模型(Multimodal)算力计算

- 中文：所需算力 = (max(Prefill阶段计算速率, Decode阶段计算速率) + 模态处理算力) × 量化精度系数 × 批处理效率 × 全局校准因子
  - Prefill阶段计算速率 = Prefill阶段计算量 / TTFT
    - Prefill阶段计算量 = 基础FLOPs系数 × 模型规模系数 × 输入长度 × LLM参数量 × 注意力效率系数 × 批处理大小 × 多模态处理开销
  - Decode阶段计算速率 = 单Token计算量 × 吞吐量
    - 单Token计算量 = 基础FLOPs系数 × 模型规模系数 × 序列长度对数影响 × LLM参数量 × 注意力效率系数 × KV缓存优化系数 × 批处理大小
  - 模态处理算力 = 视觉模态算力 + 音频模态算力
    - 视觉模态算力 = 视觉参数量 × 视觉计算系数 × 批处理大小
    - 音频模态算力 = 音频参数量 × 音频计算系数 × 批处理大小
- 英文：Required Computing Power = (max(Prefill Computing Rate, Decode Computing Rate) + Modal Computing) × Quantization Factor × Batch Efficiency × Global Calibration
  - Prefill Computing Rate = Prefill Computing / TTFT
    - Prefill Computing = Base FLOPs × Model Scale × Input Length × LLM Parameters × Attention Efficiency × Batch Size × Multimodal Factor
  - Decode Computing Rate = Per-Token Computing × Throughput
    - Per-Token Computing = Base FLOPs × Model Scale × Sequence Length Log Effect × LLM Parameters × Attention Efficiency × KV Cache Factor × Batch Size
  - Modal Computing = Vision Computing + Audio Computing
    - Vision Computing = Vision Parameters × Vision Computing Factor × Batch Size
    - Audio Computing = Audio Parameters × Audio Computing Factor × Batch Size
- 公式：
  - $C_{prefill} = \frac{F_{base} \times S_{model} \times L_{in} \times P_{base} \times (1/E_{attn}) \times B \times F_{mm}}{TTFT}$
  - $C_{token} = F_{base} \times S_{model} \times (1 + 0.1\log_{10}(L_{seq})) \times P_{base} \times (1/E_{attn}) \times 0.5 \times B$
  - $C_{decode} = C_{token} \times T$
  - $C_{vision} = P_{vision} \times F_{vision} \times B$
  - $C_{audio} = P_{audio} \times F_{audio} \times B$
  - $C_{modal} = C_{vision} + C_{audio}$
  - $C = (\max(C_{prefill}, C_{decode}) + C_{modal}) \times F_{quant} \times E_{batch} \times G_{cal}$

### 硬件需求计算

#### 显卡数量计算

- 中文：所需显卡数量 = max(显存限制显卡数, 算力限制显卡数)
  - 显存限制显卡数 = ⌈所需总显存 / 单卡显存⌉
  - 算力限制显卡数 = ⌈所需算力 / (单卡算力 × GPU利用率)⌉
- 英文：
  - Required GPU Count = max(Memory Limited GPU Count, Computing Limited GPU Count)
  - Memory Limited GPU Count = ⌈Required Total Memory / Single GPU Memory⌉
  - Computing Limited GPU Count = ⌈Required Computing Power / (Single GPU Computing Power × GPU Utilization)⌉
- 公式：
  - $N_{mem} = \lceil M_{total} / M_{gpu} \rceil$
  - $N_{comp} = \lceil C / (C_{gpu} \times U_{gpu}) \rceil$
  - $N = \max(N_{mem}, N_{comp})$

## 变量说明

### 模型参数

| 中文名称 | 英文名称 | 英文缩写 | 说明 |
|---------|---------|---------|------|
| 隐藏层维度 | Hidden Size | H | 模型中隐藏层的维度大小 |
| 模型层数 | Number of Hidden Layers | L | 模型中Transformer Block的层数 |
| 注意力头数 | Number of Attention Heads | H_attn | 多头注意力机制中的头的数量 |
| KV头数 | Number of Key-Value Heads | H_kv | 用于GQA/MQA的Key/Value头的数量 |
| 词汇表大小 | Vocabulary Size | V | 模型词汇表的大小 |
| 头维度 | Head Dimension | D_h | 每个注意力头的维度，等于隐藏层维度/注意力头数 |
| 总参数量 | Total Parameters | P_total | 模型的总参数量(十亿) |
| 共享参数量 | Shared Parameters | P_shared | MoE模型中的共享参数量 |
| 单个专家参数量 | Parameters Per Expert | P_expert | 每个专家的参数量(十亿) |
| 专家中间层维度 | Intermediate Size | I | MoE专家网络中FFN的中间层维度 |
| 专家总数 | Number of Local Experts | E | 每个MoE层包含的专家总数 |
| 激活专家数 | Number of Experts Per Token | E_active | 每个Token激活的专家数量 |
| LLM基础参数量 | LLM Base Parameters | P_base | 多模态模型中LLM部分的参数量 |
| 视觉模态参数量 | Vision Modal Parameters | P_vision | 视觉模态的总参数量 |
| 音频模态参数量 | Audio Modal Parameters | P_audio | 音频模态的总参数量 |

### 推理配置

| 中文名称 | 英文名称 | 英文缩写 | 说明 |
|---------|---------|---------|------|
| 权重量化精度 | Weight Quantization Precision | Q_weight | 模型权重的量化精度(字节/参数) |
| KV缓存量化精度 | KV Cache Quantization Precision | Q_kv | KV Cache的量化精度(字节/元素) |
| 激活值量化精度 | Activations Quantization Precision | Q_act | 中间激活值的量化精度(字节/元素) |
| 输入长度 | Input Length | S_in | 输入序列的长度(token数) |
| 输出长度 | Output Length | S_out | 输出序列的长度(token数) |
| 序列长度 | Sequence Length | S | 输入长度+输出长度 |
| 批处理大小 | Batch Size | B | 并发处理的请求数量 |
| 额外开销比例 | Overhead Ratio | R_overhead | 用于估算碎片、CUDA上下文等额外开销的比例 |
| 图像输入尺寸 | Image Input Size | I_size | 输入图像的分辨率(像素) |
| 图像Patch尺寸 | Patch Size | P_size | 图像切块(Patch)的大小(像素) |
| 音频输入长度 | Audio Input Length | A_len | 输入音频的时长(秒) |
| 音频采样率 | Audio Sample Rate | A_rate | 音频采样率(Hz) |

### 服务指标

| 中文名称 | 英文名称 | 英文缩写 | 说明 |
|---------|---------|---------|------|
| 首token时间 | Time To First Token | TTFT | 接收请求后，生成首个Token的时间(ms)，主要反映Prefill阶段性能 |
| 单token时间 | Time Per Output Token | TPOT | 生成后续每个Token的平均时间(ms)，主要反映Decode阶段性能 |
| 吞吐量 | Throughput | T | 系统每秒能够生成的Token总数，由TPOT计算得到(1/TPOT×1000) |

### 计算因子

| 中文名称 | 英文名称 | 英文缩写 | 说明 |
|---------|---------|---------|------|
| 稠密激活因子 | Dense Activation Factor | F_act | Dense模型激活值因子(默认18) |
| LLM激活因子 | LLM Activation Factor | F_llm | 多模态中LLM部分激活值因子(默认24) |
| 共享激活因子 | MoE Shared Activation Factor | F_shared | MoE共享部分激活值因子(默认4) |
| 专家激活因子 | MoE Expert Activation Factor | F_expert | MoE专家部分激活值因子(默认2) |
| 视觉激活因子 | Vision Activation Factor | F_vision_act | 视觉模态激活值因子(默认4.0) |
| 音频激活因子 | Audio Activation Factor | F_audio_act | 音频模态激活值因子(默认3.0) |
| 视觉token因子 | Vision Token Factor | F_vision | 视觉token对KV Cache的影响因子(默认1.0) |
| 音频token因子 | Audio Token Factor | F_audio | 音频token对KV Cache的影响因子(默认0.8) |
| 基础FLOPs系数 | Base FLOPs Coefficient | F_base | 每个token的基础计算量系数(默认12) |
| 模型规模系数 | Model Scale Factor | S_model | 基于隐藏层大小和层数的模型规模系数 |
| 注意力效率系数 | Attention Efficiency | E_attn | 注意力机制效率系数(KV头数/注意力头数) |
| 视觉计算系数 | Vision Computing Factor | F_vision | 视觉模态计算系数(默认0.05) |
| 音频计算系数 | Audio Computing Factor | F_audio | 音频模态计算系数(默认0.03) |
| MoE路由开销 | MoE Routing Factor | F_moe | MoE模型路由开销系数(默认1.2) |
| 多模态处理开销 | Multimodal Factor | F_mm | 多模态处理额外开销系数(默认1.3) |
| 批处理效率 | Batch Efficiency | E_batch | 批处理大小对计算效率的影响 |
| 量化精度系数 | Quantization Factor | F_quant | 量化精度对算力需求的影响系数 |
| 全局校准因子 | Global Calibration | G_cal | 基于实际部署经验的全局校准因子(默认0.01) |
| 批处理大小 | Batch Size | B | 并发处理的请求数量 |
| 输入长度 | Input Length | L_in | 输入序列的长度(token数) |
| 序列总长度 | Sequence Length | L_seq | 输入长度+输出长度 |
| 吞吐量 | Throughput | T | 系统每秒能够生成的Token总数(tokens/s) |

### 硬件参数

| 中文名称 | 英文名称 | 英文缩写 | 说明 |
|---------|---------|---------|------|
| 显存大小 | VRAM Size | M_gpu | GPU显存大小(GB) |
| 显存带宽 | Memory Bandwidth | BW_mem | GPU显存带宽(GB/s) |
| FP16算力 | FP16 Computing Power | C_fp16 | GPU的FP16算力(TFLOPS) |
| BF16算力 | BF16 Computing Power | C_bf16 | GPU的BF16算力(TFLOPS) |
| FP8算力 | FP8 Computing Power | C_fp8 | GPU的FP8算力(TFLOPS) |
| INT8算力 | INT8 Computing Power | C_int8 | GPU的INT8算力(TOPS) |
| INT4算力 | INT4 Computing Power | C_int4 | GPU的INT4算力(TOPS) |
| 互联带宽 | Interconnect Bandwidth | BW_ic | GPU间互联带宽(GB/s) |
| GPU利用率 | GPU Utilization | U_gpu | GPU计算资源的实际利用率(0.1-1.0) |

## 使用方法

1. 打开 `index.html` 文件
2. 选择预设模型或自定义模型参数
3. 调整推理配置和服务指标
4. 查看显存占用分析和推荐硬件
5. 可选：使用自定义硬件计算所需卡数
6. 可选：启动推理速度模拟

## 项目结构

```
AICompass/
  - data/                # 数据文件目录
    - gpu.json           # GPU硬件参数数据
    - logo.png           # 项目logo
    - model.json         # 预设模型参数数据
  - index.html           # 主页面HTML
  - script.js            # 主要JavaScript逻辑
  - style.css            # 样式表
  - README.md            # 项目说明文档