(triton_env) ubuntu@ip-172-31-11-210:~/middleend$ cat gpt_tuner_pipeline_v5.py
# gpt_tuner_pipeline.py
import torch
import torch.nn as nn
import torch.fx as fx
import triton
import triton.language as tl
import openai
import time
import re
import os

# -----------------------------
# GPT 配置
# -----------------------------
#openai.api_key = "your-api-key-here"
openai.api_key = "your own api key"



# -----------------------------
# Step 1: 更复杂任务 - 矩阵乘法模型
# -----------------------------
class MyMatmulModel(nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)

model = MyMatmulModel()
traced = fx.symbolic_trace(model)
print("\n🧠 FX Graph:")
print(traced.graph)

# -----------------------------
# Step 2: Prompt 构造
# -----------------------------
ir_metadata = {
    "op_type": ["matmul"],
    "shape": [2048, 2048],
    "dtype": "float32",
    "device": "cuda"
}

prompt_template = f"""
I am compiling a Triton GPU kernel for the following workload:
- Operation: matmul
- Input shape: {ir_metadata['shape']}
- Dtype: {ir_metadata['dtype']}
- Device: {ir_metadata['device']}
Please suggest BLOCK_SIZE, NUM_WARPS, and NUM_STAGES that maximize performance under shared memory constraints.
NOTE: Shared memory must be below 100KB.
"""

# -----------------------------
# GPT 参数提取
# -----------------------------
def parse_gpt_reply(text):
    block_size = re.search(r"BLOCK_SIZE\s*[:=]\s*(\d+)", text)
    num_warps = re.search(r"NUM_WARPS\s*[:=]\s*(\d+)", text)
    num_stages = re.search(r"NUM_STAGES\s*[:=]\s*(\d+)", text)
    config = {
        "BLOCK_SIZE": int(block_size.group(1)) if block_size else 128,
        "NUM_WARPS": int(num_warps.group(1)) if num_warps else 4,
        "NUM_STAGES": int(num_stages.group(1)) if num_stages else 2,
    }
    # 强制约束 BLOCK_SIZE 不超过 64
    config["BLOCK_SIZE"] = min(config["BLOCK_SIZE"], 64)
    return config

# -----------------------------
# Matmul kernel
# -----------------------------
@triton.jit
def matmul_kernel(A, B, C, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE):
        a = tl.load(A + offs_m[:, None] * K + (k + tl.arange(0, BLOCK_SIZE))[None, :],
                    mask=(offs_m[:, None] < M) & ((k + tl.arange(0, BLOCK_SIZE))[None, :] < K))
        b = tl.load(B + (k + tl.arange(0, BLOCK_SIZE))[:, None] * N + offs_n[None, :],
                    mask=((k + tl.arange(0, BLOCK_SIZE))[:, None] < K) & (offs_n[None, :] < N))
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# -----------------------------
# Benchmark with cache clear
# -----------------------------
M, K, N = 2048, 2048, 2048
A = torch.randn((M, K), device='cuda')
B = torch.randn((K, N), device='cuda')

SHARED_MEM_LIMIT = 101376  # bytes


def benchmark_kernel(block_size, num_warps):
    shared_mem_required = 2 * block_size * block_size * 4  # 2 tiles, float32 (4 bytes)
    if shared_mem_required > SHARED_MEM_LIMIT:
        print(f"⚠️ Skipping ({block_size}, {num_warps}) — requires {shared_mem_required} bytes shared memory, exceeds limit {SHARED_MEM_LIMIT}.")
        return float('inf')

    C = torch.empty((M, N), device='cuda')
    grid = lambda META: (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    matmul_kernel[grid](A, B, C, M, N, K, BLOCK_SIZE=block_size, num_warps=num_warps)
    torch.cuda.synchronize()
    os.system("rm -rf /tmp/torchinductor*")

    start = time.time()
    C = torch.empty((M, N), device='cuda')
    matmul_kernel[grid](A, B, C, M, N, K, BLOCK_SIZE=block_size, num_warps=num_warps)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) * 1000

# -----------------------------
# Baseline + GPT 对比
# -----------------------------
results = []

baseline_configs = [
    {"label": "Baseline (64, 4)", "BLOCK_SIZE": 64, "NUM_WARPS": 4},
    {"label": "Baseline (64, 8)", "BLOCK_SIZE": 64, "NUM_WARPS": 8},
]

for config in baseline_configs:
    t = benchmark_kernel(config["BLOCK_SIZE"], config["NUM_WARPS"])
    results.append({"label": config["label"], "time": t})

for i in range(3):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_template}]
        )
        gpt_reply = response['choices'][0]['message']['content']
        config = parse_gpt_reply(gpt_reply)
        t = benchmark_kernel(config["BLOCK_SIZE"], config["NUM_WARPS"])
        label = f"GPT Suggestion #{i+1} ({config['BLOCK_SIZE']}, {config['NUM_WARPS']})"
        results.append({"label": label, "time": t})
        print(f"\n📬 GPT Suggestion #{i+1}:\n{gpt_reply}")
    except Exception as e:
        print(f"\n❌ GPT Suggestion #{i+1} failed: {e}")

# -----------------------------
# 表格输出
# -----------------------------
print("\n📊 Execution Time Summary (ms):")
print("{:<35} {:>10}".format("Configuration", "Time (ms)"))
print("-" * 50)
for r in results:
    print("{:<35} {:>10.3f}".format(r["label"], r["time"]))
