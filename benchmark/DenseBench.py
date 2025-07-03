import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
MODEL_PATH = "/home/dataset/Llama-2-13b"
# 批处理大小列表
BATCH_SIZES = [1, 2, 4, 8,16, 32]
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
# 激活因子 (Dense模型)
DENSE_ACTIVATION_FACTOR = 18
# 额外开销比例
OTHER_OVERHEAD_RATIO = 0.02

def run_inference_test(model, tokenizer, batch_size, input_length, output_length, verbose=False):
    """
    运行单次推理测试并返回结果
    
    参数:
    - model: 模型
    - tokenizer: 分词器
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
    
    # 准备输入数据
    words_per_token = 0.75
    chars_per_word = 5
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
        outputs = model.generate(
            inputs["input_ids"],
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
    # 头维度
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    
    # 1. 模型权重显存
    model_weights_theoretical_bytes = model_parameters_count * WEIGHT_QUANTIZATION_PRECISION
    model_weights_theoretical = model_weights_theoretical_bytes / (1024**3)
    
    # 2. KV缓存显存
    sequence_length = actual_input_length + output_length
    kv_cache_theoretical_bytes = (
        batch_size
        * sequence_length
        * model.config.num_hidden_layers
        * 2  # K和V
        * num_key_value_heads
        * head_dim
        * KV_CACHE_QUANTIZATION_PRECISION
    )
    kv_cache_theoretical = kv_cache_theoretical_bytes / (1024**3)
    
    # 3. 激活值显存
    activations_theoretical_bytes = (
        batch_size 
        * sequence_length 
        * model.config.hidden_size 
        * DENSE_ACTIVATION_FACTOR 
        * ACTIVATIONS_QUANTIZATION_PRECISION
    )
    activations_theoretical = activations_theoretical_bytes / (1024**3)
    
    # 4. 额外开销显存
    other_theoretical = (model_weights_theoretical + kv_cache_theoretical + activations_theoretical) * OTHER_OVERHEAD_RATIO
    
    # 5. 总显存
    total_theoretical = (
        model_weights_theoretical
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
        'model_weights_theoretical': model_weights_theoretical,
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
    global model_parameters_count
    
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
    if 'model_type' in config:
        model_type = config['model_type'].lower()
        if 'llama' in model_type:
            # 对于LLaMA模型，估算参数量
            hidden_size = config['hidden_size']
            num_hidden_layers = config['num_hidden_layers']
            vocab_size = config['vocab_size']
            intermediate_size = config.get('intermediate_size', 4 * hidden_size)
            
            # 估算参数量
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
            
            # 总参数量
            model_parameters_count = embedding_params + all_layers_params + output_params
        else:
            # 默认情况下使用n_params字段
            model_parameters_count = config.get('n_params', 0) * 1_000_000_000
    else:
        # 如果没有model_type，尝试使用n_params字段
        model_parameters_count = config.get('n_params', 0) * 1_000_000_000
    
    # 如果无法从配置中获取参数量，使用默认值
    if model_parameters_count == 0:
        print("警告：无法从配置中获取模型参数量，使用默认值8B")
        model_parameters_count = 8_000_000_000
    
    print(f"\n--- 模型配置参数 ---")
    print(f"模型路径: {MODEL_PATH}")
    print(f"隐藏层大小 (hidden_size): {config['hidden_size']}")
    print(f"注意力头数 (num_attention_heads): {config['num_attention_heads']}")
    print(f"KV 头数 (num_key_value_heads): {config.get('num_key_value_heads', config['num_attention_heads'])}")
    print(f"层数 (num_hidden_layers): {config['num_hidden_layers']}")
    print(f"模型参数量: {model_parameters_count / 1_000_000_000:.2f}B")
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
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        model.config.use_cache = True
        
        # 显示模型在各设备上的分布
        if hasattr(model, 'hf_device_map'):
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
        'model_weights_theoretical', 'kv_cache_theoretical', 'activations_theoretical', 
        'other_theoretical', 'total_theoretical',
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
                result = run_inference_test(model, tokenizer, bs, il, ol, VERBOSE)
                
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
                    'model_weights_theoretical': -1,
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
    torch.cuda.empty_cache()
    gc.collect()
    print("\n模型和显存已清理。")

if __name__ == "__main__":
    main()