import torch
import platform
import numpy as np
from functools import partial

# =================================================================================
# 1. 全局配置 (Global Configuration)
# =================================================================================

# 基准测试运行次数，用于计算均值，提高结果稳定性
NUM_BENCHMARK_RUNS = 3
# 单次运行内的预热和测试迭代次数
WARMUP_ITERS = 20
TEST_ITERS = 100

# GEMM (矩阵乘法) 测试尺寸
GEMM_M, GEMM_K, GEMM_N = 4096, 4096, 4096
# LayerNorm 测试尺寸
LN_BATCH, LN_SEQ_LEN, LN_HIDDEN_DIM = 32, 4096, 4096


# =================================================================================
# 2. 核心测试与性能计算 (Core Benchmarking & Performance Calculation)
# =================================================================================

def benchmark_op(op_name, prepare_func, run_func, dtype, device):
    """
    通用算子性能测试函数。

    Args:
        op_name (str): 算子名称，用于日志输出。
        prepare_func (callable): 输入数据准备函数。
        run_func (callable): 待测试的目标算子函数。
        dtype (torch.dtype): 测试的数据类型。
        device (str): 'cuda' 或 'cpu'。

    Returns:
        float: 性能指标 (TFLOPS 或 GB/s)，发生错误时返回 None。
    """
    # 打印当前正在测试的项
    print(f"--- Testing: {op_name} ({str(dtype).split('.')[-1]}) ---")
    try:
        # 准备输入张量
        inputs = prepare_func(dtype=dtype, device=device)

        # Warmup: 编译并缓存CUDA核, 稳定GPU时钟频率
        for _ in range(WARMUP_ITERS):
            run_func(*inputs)

        # 使用CUDA Event进行异步GPU操作的精确计时
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(TEST_ITERS):
            run_func(*inputs)
        end_event.record()

        # 等待所有CUDA核心完成任务，确保时间准确性
        torch.cuda.synchronize(device)

        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_s = elapsed_time_ms / 1000 / TEST_ITERS

        # 计算并返回性能
        return calculate_performance(op_name, avg_time_s, *inputs)

    except Exception as e:
        # 捕获并报告执行过程中的任何错误 (如OOM, 不支持的类型等)
        print(f"Error during '{op_name}' with {str(dtype).split('.')[-1]}: {e}\n")
        return None

def calculate_performance(op_name, time_s, *inputs):
    """根据算子类型和耗时计算性能指标。"""
    if "GEMM" in op_name:
        # GEMM 是计算密集型, 衡量算力 (TFLOPS/TOPS)
        # 计算量公式: 2 * M * K * N (每个输出元素需要K次乘法和K-1次加法，约等于2*K)
        M, K = inputs[0].shape
        _, N = inputs[1].shape
        ops = 2 * M * K * N
        performance = ops / time_s / 1e12 # 转换为 TFLOPS/TOPS
        print(f"Avg Time: {time_s * 1000:.4f} ms | Performance: {performance:.2f} TFLOPS/TOPS\n")
        return performance

    if "Layer Normalization" in op_name:
        # LayerNorm 是访存密集型, 衡量有效显存带宽 (GB/s)
        # 简化模型: 完整读取一次输入，完整写入一次输出
        x = inputs[0]
        bytes_processed = 2 * x.numel() * x.element_size()
        performance = bytes_processed / time_s / 1e9 # 转换为 GB/s
        print(f"Avg Time: {time_s * 1000:.4f} ms | Performance: {performance:.2f} GB/s\n")
        return performance
    return None

# =================================================================================
# 3. 算子定义 (Operator Definitions)
# =================================================================================

# --- 矩阵乘法 (GEMM) ---
def prepare_matmul_inputs(dtype, device, m, k, n):
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((k, n), dtype=dtype, device=device)
    return a, b

def run_matmul(a, b):
    torch.matmul(a, b)

# --- Layer Normalization ---
def prepare_layernorm_inputs(dtype, device, batch, seq_len, hidden_dim):
    x = torch.randn((batch * seq_len, hidden_dim), dtype=dtype, device=device)
    norm_layer = torch.nn.LayerNorm(hidden_dim, device=device, dtype=dtype)
    return x, norm_layer

def run_layernorm(x, norm_layer):
    norm_layer(x)

# =================================================================================
# 4. 分离的测试流程 (Decoupled Benchmark Flows)
# =================================================================================

def run_compute_benchmarks(dtypes, device):
    """执行所有计算密集型算子的基准测试。"""
    print("\n" + "="*60)
    print("[Section 1: Compute-Bound Performance (GEMM)]")
    print("="*60)
    
    results = {}
    
    prepare_func = partial(prepare_matmul_inputs, m=GEMM_M, k=GEMM_K, n=GEMM_N)

    for name, dtype in dtypes.items():
        key = (name, 'gemm')
        results[key] = []
        
        # 对低精度类型进行特殊处理
        if dtype in [torch.int8, torch.float8_e4m3fn] or name == "int4":
            print(f"--- Testing: GEMM ({name}) ---")
            print("Skipping: Standard torch.matmul lacks support for this dtype.")
            print("Requires specialized kernels (e.g., from Transformer Engine or AO quantization).\n")
            results[key] = "N/A"
            continue

        # 对支持的类型，进行多次测试取平均
        for run_idx in range(NUM_BENCHMARK_RUNS):
            print(f"Run {run_idx+1}/{NUM_BENCHMARK_RUNS} for {name}...")
            perf = benchmark_op("GEMM", prepare_func, run_matmul, dtype, device)
            if perf is not None:
                results[key].append(perf)

    return results

def run_memory_benchmarks(dtypes, device):
    """执行所有访存密集型算子的基准测试。"""
    print("\n" + "="*60)
    print("[Section 2: Memory-Bound Performance (LayerNorm)]")
    print("="*60)
    
    results = {}
    
    prepare_func = partial(prepare_layernorm_inputs, batch=LN_BATCH, seq_len=LN_SEQ_LEN, hidden_dim=LN_HIDDEN_DIM)
    
    for name, dtype in dtypes.items():
        key = (name, 'layernorm')
        results[key] = []

        if 'int' in name or 'fp8' in name:
            print(f"--- Testing: Layer Normalization ({name}) ---")
            print(f"Skipping: LayerNorm is not typically supported for {name} dtype.\n")
            results[key] = "N/A"
            continue

        # 对支持的类型，进行多次测试取平均
        for run_idx in range(NUM_BENCHMARK_RUNS):
            print(f"Run {run_idx+1}/{NUM_BENCHMARK_RUNS} for {name}...")
            perf = benchmark_op("Layer Normalization", prepare_func, run_layernorm, dtype, device)
            if perf is not None:
                results[key].append(perf)
                
    return results


# =================================================================================
# 5. 主程序入口 (Main Execution Block)
# =================================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA-enabled GPU.")

    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Configuration: Runs={NUM_BENCHMARK_RUNS}, Warmup={WARMUP_ITERS}, Test={TEST_ITERS}")
    
    # 定义所有需要测试的精度类型
    dtypes_to_test = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3": getattr(torch, 'float8_e4m3fn', "N/A"),
        "int8": torch.int8,
        "int4": "N/A", # torch中无原生int4, 作为占位符
    }

    # 分别执行计算和访存测试
    compute_results = run_compute_benchmarks(dtypes_to_test, device)
    memory_results = run_memory_benchmarks(dtypes_to_test, device)

    # 合并所有结果
    all_results = {**compute_results, **memory_results}

    # --- 结果汇总和展示 ---
    print("\n" + "="*60)
    print("Benchmark Summary (Average Performance)")
    print("="*60)
    print(f"{'Precision':<12} | {'GEMM (TFLOPS/TOPS)':<25} | {'LayerNorm (GB/s)':<20}")
    print("-" * 65)

    # 按照定义的顺序打印结果
    for precision_name in dtypes_to_test.keys():
        gemm_perfs = all_results.get((precision_name, 'gemm'), [])
        ln_perfs = all_results.get((precision_name, 'layernorm'), [])
        
        gemm_str = f"{np.mean(gemm_perfs):.2f}" if isinstance(gemm_perfs, list) and gemm_perfs else "N/A"
        ln_str = f"{np.mean(ln_perfs):.2f}" if isinstance(ln_perfs, list) and ln_perfs else "N/A"
        
        print(f"{precision_name:<12} | {gemm_str:<25} | {ln_str:<20}")