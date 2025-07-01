import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import __version__ as transformers_version
try:
    from transformers import LlavaForConditionalGeneration
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False
    print(f"警告: 无法导入LlavaForConditionalGeneration，当前transformers版本: {transformers_version}")
    print("可能需要更新transformers库或安装llava相关依赖")

try:
    from transformers import AutoModelForVision2Seq
    VISION2SEQ_AVAILABLE = True
except ImportError:
    VISION2SEQ_AVAILABLE = False

import gc
import time
import os
import argparse
import csv
import datetime
from tqdm import tqdm

# --- 跨设备显存统计辅助函数 ---
def get_total_memory_allocated():
    """获取所有GPU设备的总显存占用"""
    total = 0
    for i in range(torch.cuda.device_count()):
        total += torch.cuda.memory_allocated(i)
    return total

def get_max_memory_allocated():
    """获取所有GPU设备的峰值显存占用"""
    total = 0
    for i in range(torch.cuda.device_count()):
        total += torch.cuda.max_memory_allocated(i)
    return total

def reset_peak_memory_all_devices():
    """重置所有GPU设备的峰值显存统计"""
    for i in range(torch.cuda.device_count()):
        # 确保只重置实际存在的设备
        if i < torch.cuda.device_count():
            torch.cuda.reset_peak_memory_stats(i)

# --- 测试配置参数 ---
# 模型路径
MODEL_PATH = "/home/dataset/llava-1.5-7b-hf"
# 批处理大小列表
BATCH_SIZES = [4, 8]
# BATCH_SIZES = [1, 2, 4, 8, 16, 32]
# 输入长度列表 - 短/长
SHORT_INPUT_LENGTH = 256
LONG_INPUT_LENGTH = 2048
# 输出长度列表 - 短/长
SHORT_OUTPUT_LENGTH = 256
LONG_OUTPUT_LENGTH = 2048
# 结果输出目录
OUTPUT_DIR = "results"
# 是否打印详细信息
VERBOSE = False

# --- 显存计算相关参数 ---
# 权重量化精度 (bytes/parameter)
WEIGHT_QUANTIZATION_PRECISION = 2  # BF16/FP16: 2 bytes
# KV缓存量化精度 (bytes/element)
KV_CACHE_QUANTIZATION_PRECISION = 2  # BF16/FP16: 2 bytes
# 激活值量化精度 (bytes/element)
ACTIVATIONS_QUANTIZATION_PRECISION = 2  # BF16/FP16: 2 bytes
# LLM激活因子
LLM_ACTIVATION_FACTOR = 24
# 视觉激活因子
VISION_ACTIVATION_FACTOR = 4.0
# 音频激活因子
AUDIO_ACTIVATION_FACTOR = 3.0
# 视觉token因子
VISION_TOKEN_FACTOR = 1.0
# 音频token因子
AUDIO_TOKEN_FACTOR = 0.8
# 额外开销比例
OTHER_OVERHEAD_RATIO = 0.02
# 图像输入尺寸 - 将在读取config时更新
IMAGE_SIZE = 336  # 默认图像尺寸
# 图像Patch尺寸 - 将在读取config时更新
PATCH_SIZE = 14  # 默认patch尺寸

def run_inference_test(model, tokenizer, processor, batch_size, input_length, output_length, verbose=False):
    """
    运行单次推理测试并返回结果
    
    参数:
    - model: 模型
    - tokenizer: 分词器
    - processor: 处理器（用于处理图像输入）
    - batch_size: 批处理大小
    - input_length: 输入长度(token数)
    - output_length: 输出长度(token数)
    - verbose: 是否打印详细信息
    
    返回:
    - 包含测试结果的字典
    """
    # 重置所有设备的峰值统计
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(i)
        except:
            pass
    
    # 准备输入数据 - 文本部分
    # 根据目标token长度生成输入文本
    words_per_token = 0.75  # 估计值：平均每个token约0.75个单词
    chars_per_word = 5  # 估计值：平均每个单词约5个字符
    target_chars = int(input_length / words_per_token * chars_per_word)
    
    # 生成足够长的输入文本
    input_text = "Hello, how are you today? This is a test input for the language model. " * (target_chars // 100 + 1)
    
    # 使用分词器处理输入文本
    inputs = tokenizer(
        [input_text] * batch_size,
        return_tensors="pt",
        max_length=input_length,
        truncation=True,
        padding="max_length"
    ).to("cuda")
    
    actual_input_length = inputs['input_ids'].shape[1]
    
    if verbose:
        print(f"\n开始进行推理，批次大小 {batch_size}，输入序列长度 {actual_input_length}，生成 {output_length} 个Token...")
    
    # 记录初始显存
    initial_memory = get_total_memory_allocated() / (1024**3)
    
    # 性能指标测量
    start_time = time.perf_counter()
    with torch.no_grad():
        # 检查模型类型并使用适当的生成方法
        try:
            # 对于任何支持generate方法的模型
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=output_length,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=None,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
        except Exception as e:
            print(f"生成时出错: {e}")
            print("尝试不同的参数组合...")
            try:
                # 尝试更简单的参数组合
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=output_length,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
            except Exception as e2:
                print(f"简化参数后仍然出错: {e2}")
                raise e2
    end_time = time.perf_counter()
    total_inference_time = end_time - start_time
    
    if verbose:
        print(f"推理完成。总耗时: {total_inference_time:.2f} 秒")
    
    # 计算吞吐量
    throughput = (batch_size * output_length) / total_inference_time
    
    # 获取推理过程中所有设备的总峰值显存占用
    peak_memory_actual = get_max_memory_allocated() / (1024**3)
    
    # 模型权重实测显存
    model_weights_actual = initial_memory
    
    # KV Cache 实测
    kv_cache_actual_bytes = 0
    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        for layer_kv in outputs.past_key_values:
            for tensor in layer_kv:
                if tensor is not None and torch.is_floating_point(tensor):
                    kv_cache_actual_bytes += tensor.numel() * tensor.element_size()
    
    kv_cache_actual = kv_cache_actual_bytes / (1024**3)
    
    # 其他实测显存（总显存减去权重和KV缓存）
    other_actual = peak_memory_actual - model_weights_actual - kv_cache_actual
    
    # 计算理论值
    # 尝试从不同的配置属性获取LLM配置
    if hasattr(model, 'config'):
        if hasattr(model.config, 'text_config'):
            llm_config = model.config.text_config
        else:
            llm_config = model.config

    # 获取必要的配置值，提供默认值以防配置中不存在
    hidden_size = getattr(llm_config, 'hidden_size', 4096)  # 默认值为Vicuna-7B的隐藏层大小
    num_attention_heads = getattr(llm_config, 'num_attention_heads', 32)  # 默认值
    num_hidden_layers = getattr(llm_config, 'num_hidden_layers', 32)  # 默认值
    num_key_value_heads = getattr(llm_config, 'num_key_value_heads', num_attention_heads)

    # 计算头维度
    head_dim = hidden_size // num_attention_heads

    # 计算基础LLM、视觉和音频参数量
    base_params_count = base_parameters_count
    vision_params_count = vision_parameters_count
    audio_params_count = 0  # LLaVA没有音频模态

    # 1. 基础LLM权重显存
    base_weights_theoretical_bytes = base_params_count * WEIGHT_QUANTIZATION_PRECISION
    base_weights_theoretical = base_weights_theoretical_bytes / (1024**3)

    # 2. 视觉模态权重显存
    vision_weights_theoretical_bytes = vision_params_count * WEIGHT_QUANTIZATION_PRECISION
    vision_weights_theoretical = vision_weights_theoretical_bytes / (1024**3)

    # 3. 音频模态权重显存
    audio_weights_theoretical_bytes = audio_params_count * WEIGHT_QUANTIZATION_PRECISION
    audio_weights_theoretical = audio_weights_theoretical_bytes / (1024**3)

    # 计算图像序列长度（patch数量）
    image_patch_count = (IMAGE_SIZE // PATCH_SIZE) ** 2

    # 4. KV缓存显存
    # 总序列长度 = 文本序列长度 + 图像patch序列长度 × 视觉token因子 + 音频序列长度 × 音频token因子
    total_sequence_length = actual_input_length + (image_patch_count * VISION_TOKEN_FACTOR)

    kv_cache_theoretical_bytes = (
        batch_size
        * total_sequence_length
        * num_hidden_layers
        * 2  # K和V
        * num_key_value_heads
        * head_dim
        * KV_CACHE_QUANTIZATION_PRECISION
    )
    kv_cache_theoretical = kv_cache_theoretical_bytes / (1024**3)

    # 5. 激活值显存
    # LLM激活值
    llm_activations_theoretical_bytes = (
        batch_size 
        * actual_input_length 
        * hidden_size 
        * LLM_ACTIVATION_FACTOR 
        * ACTIVATIONS_QUANTIZATION_PRECISION
    )

    # 视觉激活值
    vision_activations_theoretical_bytes = (
        batch_size 
        * image_patch_count
        * hidden_size 
        * VISION_ACTIVATION_FACTOR 
        * ACTIVATIONS_QUANTIZATION_PRECISION
    )

    # 音频激活值 (LLaVA没有音频模态，这里为0)
    audio_activations_theoretical_bytes = 0

    # 总激活值
    llm_activations_theoretical = llm_activations_theoretical_bytes / (1024**3)
    vision_activations_theoretical = vision_activations_theoretical_bytes / (1024**3)
    audio_activations_theoretical = audio_activations_theoretical_bytes / (1024**3)
    activations_theoretical = llm_activations_theoretical + vision_activations_theoretical + audio_activations_theoretical

    # 6. 额外开销显存
    other_theoretical = (
        base_weights_theoretical 
        + vision_weights_theoretical 
        + audio_weights_theoretical 
        + kv_cache_theoretical 
        + activations_theoretical
    ) * OTHER_OVERHEAD_RATIO

    # 7. 总显存
    total_theoretical = (
        base_weights_theoretical
        + vision_weights_theoretical
        + audio_weights_theoretical
        + kv_cache_theoretical
        + activations_theoretical
        + other_theoretical
    )
    
    # 返回结果
    return {
        'batch_size': batch_size,
        'input_length': actual_input_length,
        'output_length': output_length,
        'total_sequence_length': actual_input_length + output_length,
        'inference_time': total_inference_time,
        'throughput': throughput,
        'peak_memory_actual': peak_memory_actual,
        'model_weights_actual': model_weights_actual,
        'kv_cache_actual': kv_cache_actual,
        'other_actual': other_actual,
        'base_weights_theoretical': base_weights_theoretical,
        'vision_weights_theoretical': vision_weights_theoretical,
        'audio_weights_theoretical': audio_weights_theoretical,
        'kv_cache_theoretical': kv_cache_theoretical,
        'activations_theoretical': activations_theoretical,
        'other_theoretical': other_theoretical,
        'total_theoretical': total_theoretical
    }

def main():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 设置全局配置参数
    global base_parameters_count
    global vision_parameters_count
    global IMAGE_SIZE
    global PATCH_SIZE
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("错误：未检测到GPU。请确保您的系统安装了CUDA并配置正确。")
        exit()
    
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个可用GPU设备")
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 读取模型配置
    try:
        with open(f"{MODEL_PATH}/config.json", 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到 {MODEL_PATH}/config.json 文件。请检查路径是否正确。")
        exit()
    
    # 从配置文件获取模型参数量
    if 'model_type' in config and config['model_type'].lower() == 'llava':
        # 对于LLaVA模型，从text_config和vision_config中获取参数
        text_config = config.get('text_config', {})
        vision_config = config.get('vision_config', {})
        
        # 更新图像尺寸和Patch尺寸
        if 'image_size' in vision_config:
            IMAGE_SIZE = vision_config['image_size']
            print(f"从配置中读取图像尺寸: {IMAGE_SIZE}")
        
        if 'patch_size' in vision_config:
            PATCH_SIZE = vision_config['patch_size']
            print(f"从配置中读取Patch尺寸: {PATCH_SIZE}")
        
        # 从text_config获取LLM参数
        if text_config:
            hidden_size = text_config.get('hidden_size', 4096)  # Vicuna-7B默认值
            num_hidden_layers = text_config.get('num_hidden_layers', 32)  # Vicuna-7B默认值
            vocab_size = text_config.get('vocab_size', 32000)
            intermediate_size = text_config.get('intermediate_size', 11008)  # Vicuna-7B默认值
            
            # 估算LLM基础参数量
            # 嵌入层参数
            embedding_params = hidden_size * vocab_size
            # 每层Transformer参数
            attention_params_per_layer = 4 * hidden_size * hidden_size  # Q, K, V, O
            ffn_params_per_layer = 2 * hidden_size * intermediate_size  # Up, Down
            layer_params = attention_params_per_layer + ffn_params_per_layer
            # 所有层参数
            all_layers_params = layer_params * num_hidden_layers
            # 输出层参数
            output_params = hidden_size * vocab_size
            
            # LLM基础参数量
            base_parameters_count = embedding_params + all_layers_params + output_params
        else:
            # 如果没有text_config，使用默认值
            base_parameters_count = 7_000_000_000  
        
        # 从vision_config获取视觉模型参数
        if vision_config:
            vision_hidden_size = vision_config.get('hidden_size', 1024)
            vision_num_hidden_layers = vision_config.get('num_hidden_layers', 24)
            vision_intermediate_size = vision_config.get('intermediate_size', 4096)
            
            # 估算视觉模型参数量
            # 计算ViT模型的参数量
            patch_count = (IMAGE_SIZE // PATCH_SIZE) ** 2
            # 嵌入层
            vision_embedding_params = vision_hidden_size * (3 * PATCH_SIZE * PATCH_SIZE + 1)  # RGB通道 + 位置嵌入
            # 每层Transformer参数
            vision_attention_params_per_layer = 4 * vision_hidden_size * vision_hidden_size  # Q, K, V, O
            vision_ffn_params_per_layer = 2 * vision_hidden_size * vision_intermediate_size  # Up, Down
            vision_layer_params = vision_attention_params_per_layer + vision_ffn_params_per_layer
            # 所有层参数
            vision_all_layers_params = vision_layer_params * vision_num_hidden_layers
            # 投影层参数
            projection_dim = vision_config.get('projection_dim', 768)
            vision_projection_params = vision_hidden_size * projection_dim
            
            # 视觉模型总参数量
            vision_parameters_count = vision_embedding_params + vision_all_layers_params + vision_projection_params
        else:
            # 如果没有vision_config，使用默认值
            vision_parameters_count = 300_000_000  
    else:
        # 如果不是llava模型，使用默认值
        base_parameters_count = 7_000_000_000  
        vision_parameters_count = 300_000_000  
    
    # 获取LLM配置参数（尝试从text_config获取，如果不存在则从主config获取）
    if 'text_config' in config:
        llm_config = config['text_config']
    else:
        llm_config = config
    
    print(f"\n--- 模型配置参数 ---")
    print(f"模型路径: {MODEL_PATH}")
    print(f"模型类型: {config.get('model_type', 'unknown')}")
    print(f"隐藏层大小 (hidden_size): {llm_config.get('hidden_size', 'unknown')}")
    print(f"注意力头数 (num_attention_heads): {llm_config.get('num_attention_heads', 'unknown')}")
    print(f"KV 头数 (num_key_value_heads): {llm_config.get('num_key_value_heads', llm_config.get('num_attention_heads', 'unknown'))}")
    print(f"层数 (num_hidden_layers): {llm_config.get('num_hidden_layers', 'unknown')}")
    print(f"图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Patch尺寸: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"基础LLM参数量: {base_parameters_count / 1_000_000_000:.2f}B")
    print(f"视觉模型参数量: {vision_parameters_count / 1_000_000:.2f}M")
    print(f"总参数量: {(base_parameters_count + vision_parameters_count) / 1_000_000_000:.2f}B")
    print("--------------------")
    
    # 加载模型和分词器
    print("\n加载模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer的pad_token已设置为eos_token: {tokenizer.eos_token}")
        tokenizer.padding_side = 'left'
        print(f"Tokenizer的padding_side已设置为: {tokenizer.padding_side}")
        
        # 加载处理器（用于处理图像输入）
        try:
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            print("成功加载图像处理器")
        except:
            processor = None
            print("未找到图像处理器，将使用文本模式测试")
        
        # 加载模型 - 尝试多种模型类
        model = None
        
        # 1. 尝试使用LlavaForConditionalGeneration
        if LLAVA_AVAILABLE:
            try:
                model = LlavaForConditionalGeneration.from_pretrained(
                    MODEL_PATH, 
                    torch_dtype=torch.bfloat16, 
                    device_map="auto"
                )
                print("成功使用LlavaForConditionalGeneration加载模型")
            except Exception as e:
                print(f"使用LlavaForConditionalGeneration加载失败: {e}")
        
        # 2. 尝试使用AutoModelForVision2Seq
        if model is None and VISION2SEQ_AVAILABLE:
            try:
                print("尝试使用AutoModelForVision2Seq...")
                model = AutoModelForVision2Seq.from_pretrained(
                    MODEL_PATH, 
                    torch_dtype=torch.bfloat16, 
                    device_map="auto"
                )
                print("成功使用AutoModelForVision2Seq加载模型")
            except Exception as e:
                print(f"使用AutoModelForVision2Seq加载失败: {e}")
        
        # 3. 尝试使用AutoModel作为最后的备选方案
        if model is None:
            try:
                print("尝试使用AutoModel...")
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    MODEL_PATH, 
                    torch_dtype=torch.bfloat16, 
                    device_map="auto"
                )
                print("成功使用AutoModel加载模型")
            except Exception as e:
                print(f"使用AutoModel加载失败: {e}")
                raise e
        
        model.config.use_cache = True
        
        # 显示模型在各设备上的分布
        if hasattr(model, 'hf_device_map') and isinstance(model.hf_device_map, dict):
            print("\n--- 模型设备分布 ---")
            for layer, device in model.hf_device_map.items():
                print(f"{layer}: {device}")
        else:
            print("\n模型未使用设备映射（可能是单卡）")
        
    except Exception as e:
        print(f"加载模型或分词器时出错：{e}")
        exit()
    
    # 准备CSV文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(MODEL_PATH)
    csv_filename = os.path.join(OUTPUT_DIR, f"{model_name}_memory_benchmark_{timestamp}.csv")
    
    # CSV表头
    csv_header = [
        'batch_size', 'input_length', 'output_length', 'total_sequence_length',
        'inference_time', 'throughput', 'peak_memory_actual',
        'model_weights_actual', 'kv_cache_actual', 'other_actual',
        'base_weights_theoretical', 'vision_weights_theoretical', 'audio_weights_theoretical',
        'kv_cache_theoretical', 'activations_theoretical', 'other_theoretical', 'total_theoretical',
        'scenario'
    ]
    
    # 打开CSV文件
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        
        # 定义测试场景
        scenarios = [
            {"name": "short_input_short_output", "input_length": SHORT_INPUT_LENGTH, "output_length": SHORT_OUTPUT_LENGTH},
            {"name": "short_input_long_output", "input_length": SHORT_INPUT_LENGTH, "output_length": LONG_OUTPUT_LENGTH},
            {"name": "long_input_short_output", "input_length": LONG_INPUT_LENGTH, "output_length": SHORT_OUTPUT_LENGTH},
            {"name": "long_input_long_output", "input_length": LONG_INPUT_LENGTH, "output_length": LONG_OUTPUT_LENGTH}
        ]
        
        # 创建测试组合
        test_combinations = []
        for scenario in scenarios:
            for bs in BATCH_SIZES:
                test_combinations.append({
                    "batch_size": bs,
                    "input_length": scenario["input_length"],
                    "output_length": scenario["output_length"],
                    "scenario": scenario["name"]
                })
        
        # 运行所有测试
        print(f"\n开始运行 {len(test_combinations)} 组测试...")
        for test in tqdm(test_combinations, desc="测试进度"):
            bs = test["batch_size"]
            il = test["input_length"]
            ol = test["output_length"]
            scenario = test["scenario"]
            
            try:
                # 运行测试
                result = run_inference_test(model, tokenizer, processor, bs, il, ol, VERBOSE)
                
                # 添加场景信息
                result["scenario"] = scenario
                
                # 写入CSV
                writer.writerow(result)
                csvfile.flush()  # 确保数据立即写入文件
                
                # 如果是详细模式，打印结果
                if VERBOSE:
                    print(f"\n场景: {scenario}, 批次大小: {bs}")
                    print(f"推理时间: {result['inference_time']:.2f}秒, 吞吐量: {result['throughput']:.2f} tokens/s")
                    print(f"峰值显存: {result['peak_memory_actual']:.2f} GB, 理论显存: {result['total_theoretical']:.2f} GB")
                
                # 清理缓存
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"测试失败 (场景: {scenario}, batch_size={bs}): {e}")
                # 尝试记录错误
                error_result = {
                    'batch_size': bs,
                    'input_length': il,
                    'output_length': ol,
                    'total_sequence_length': il + ol,
                    'inference_time': -1,
                    'throughput': -1,
                    'peak_memory_actual': -1,
                    'model_weights_actual': -1,
                    'kv_cache_actual': -1,
                    'other_actual': -1,
                    'base_weights_theoretical': -1,
                    'vision_weights_theoretical': -1,
                    'audio_weights_theoretical': -1,
                    'kv_cache_theoretical': -1,
                    'activations_theoretical': -1,
                    'other_theoretical': -1,
                    'total_theoretical': -1,
                    'scenario': scenario
                }
                writer.writerow(error_result)
                csvfile.flush()
                
                # 清理缓存
                torch.cuda.empty_cache()
                gc.collect()
    
    print(f"\n测试完成！结果已保存到 {csv_filename}")
    
    # 释放资源
    del model
    del tokenizer
    if processor is not None:
        del processor
    torch.cuda.empty_cache()
    gc.collect()
    print("\n模型和显存已清理。")

if __name__ == "__main__":
    main() 