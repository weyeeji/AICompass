# AICompass：模型对算力等效建模评估工具

## 项目简介

AICompass是一个用于大语言模型(LLM)算力和显存需求评估的工具，支持Dense、MoE和多模态模型的评估。通过输入模型参数和推理配置，可以快速估算所需的显存和计算资源，并推荐合适的硬件配置。

## 计算方法

### 显存占用计算

#### 稠密模型(Dense)显存计算

稠密模型的显存占用主要包括模型权重、KV缓存和激活值三部分：

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

#### 混合专家模型(MoE)显存计算

MoE模型的显存占用包括共享权重、专家权重、KV缓存和激活值：

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

#### 多模态模型(Multimodal)显存计算

多模态模型的显存占用包括基础LLM权重、视觉模态权重、音频模态权重、KV缓存和激活值：

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
  - $M_{act\_vision} = B \times S_{image} \times H \times F_{vision} \times Q_{act}$
  - $M_{act\_audio} = B \times S_{audio} \times H \times F_{audio} \times Q_{act}$
  - $M_{act} = M_{act\_llm} + M_{act\_vision} + M_{act\_audio}$

### 计算能力需求评估

#### 稠密模型(Dense)算力计算

- 中文：所需算力 = 吞吐量 × 参数因子 × 注意力因子 × 复杂度因子 × 序列长度因子 × 批处理效率因子 × 模型参数量 × 吞吐量因子
- 英文：Required Computing Power = Throughput × Parameter Factor × Attention Factor × Complexity Factor × Sequence Length Factor × Batch Efficiency Factor × Model Parameters × Throughput Factor
- 公式：$C = T \times F_{params} \times F_{attn} \times F_{complex} \times F_{seq} \times F_{batch} \times P_{total} \times F_{throughput}$

#### 混合专家模型(MoE)算力计算

- 中文：所需算力 = 共享部分算力 + 专家部分算力
  - 共享部分算力 = 吞吐量 × 共享参数量 × 共享复杂度因子 × 序列长度因子 × 批处理效率因子
  - 专家部分算力 = 吞吐量 × 专家参数量 × 专家激活比例 × 专家复杂度因子 × 序列长度因子 × 批处理效率因子
- 英文：
  - Shared Computing = Throughput × Shared Parameters × Shared Complexity Factor × Sequence Length Factor × Batch Efficiency Factor
  - Expert Computing = Throughput × Expert Parameters × Expert Activation Ratio × Expert Complexity Factor × Sequence Length Factor × Batch Efficiency Factor
- 公式：
  - $C_{shared} = T \times P_{shared} \times F_{shared\_complex} \times F_{seq} \times F_{batch}$
  - $C_{expert} = T \times P_{expert} \times R_{expert} \times F_{expert\_complex} \times F_{seq} \times F_{batch}$
  - $C = (C_{shared} + C_{expert}) \times F_{throughput}$

#### 多模态模型(Multimodal)算力计算

- 中文：所需算力 = LLM算力 + 模态算力
  - LLM算力 = 吞吐量 × LLM参数量 × 复杂度因子 × 序列长度因子 × 批处理效率因子
  - 模态算力 = 视觉算力 + 音频算力
    - 视觉算力 = 视觉参数量 × 视觉复杂度因子 × 批处理大小 × 批处理效率因子
    - 音频算力 = 音频参数量 × 音频复杂度因子 × 批处理大小 × 批处理效率因子
- 英文：
  - LLM Computing = Throughput × LLM Parameters × Complexity Factor × Sequence Length Factor × Batch Efficiency Factor
  - Modal Computing = Vision Computing + Audio Computing
    - Vision Computing = Vision Parameters × Vision Complexity Factor × Batch Size × Batch Efficiency Factor
    - Audio Computing = Audio Parameters × Audio Complexity Factor × Batch Size × Batch Efficiency Factor
- 公式：
  - $C_{llm} = T \times P_{base} \times F_{complex} \times F_{seq} \times F_{batch}$
  - $C_{vision} = P_{vision} \times F_{vision\_complex} \times B \times F_{batch}$
  - $C_{audio} = P_{audio} \times F_{audio\_complex} \times B \times F_{batch}$
  - $C = (C_{llm} + C_{vision} + C_{audio}) \times F_{throughput}$

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
| 吞吐量 | Throughput | T | 系统每秒能够生成的Token总数 |
| 首token时间 | Time To First Token | TTFT | 接收请求后，生成首个Token的时间(ms) |
| 单token时间 | Time Per Output Token | TPOT | 生成后续每个Token的平均时间(ms) |

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
| 稠密复杂度因子 | Dense Complexity Factor | F_complex | 模型架构复杂度因子(默认2.0) |
| 共享复杂度因子 | MoE Shared Complexity Factor | F_shared_complex | MoE共享部分复杂度因子(默认1.2) |
| 专家复杂度因子 | MoE Expert Complexity Factor | F_expert_complex | MoE专家部分复杂度因子(默认1.5) |
| 视觉复杂度因子 | Vision Complexity Factor | F_vision_complex | 视觉模态复杂度因子(默认1.0) |
| 音频复杂度因子 | Audio Complexity Factor | F_audio_complex | 音频模态复杂度因子(默认0.8) |
| 注意力复杂度因子 | Attention Complexity Factor | F_attn | 注意力机制计算复杂度因子(默认1.0) |
| 序列长度因子 | Sequence Length Factor | F_seq | 序列长度非线性增长因子 |
| 批处理效率因子 | Batch Efficiency Factor | F_batch | 批处理效率因子 |
| 吞吐量因子 | Throughput Factor | F_throughput | 吞吐量对算力需求的影响因子 |

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

1. 打开 `web/index.html` 文件
2. 选择预设模型或自定义模型参数
3. 调整推理配置和服务指标
4. 查看显存占用分析和推荐硬件
5. 可选：使用自定义硬件计算所需卡数
6. 可选：启动推理速度模拟

## 项目结构

```
AICompass/
  - README.md        # 项目说明文档
  - web/
    - data/
      - gpu.json     # GPU硬件数据
      - logo.png     # 项目Logo
      - model.json   # 预设模型数据
    - index.html     # 主页面
    - script.js      # 主要计算逻辑
    - style.css      # 样式表