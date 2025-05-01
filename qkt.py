import torch
import torch.nn as nn
import torch.fx as fx
import triton
import triton.language as tl
import openai
import time
import re
import os

openai.api_key = "your own key"


# -----------------------------
# Step 1: Transformer-style QK^T softmax matmul 模型
# -----------------------------
class AttentionMatmulModel(nn.Module):
    def forward(self, q, k):
        return torch.matmul(q, k.transpose(-1, -2))

model = AttentionMatmulModel()
traced = fx.symbolic_trace(model)
print("\n🧠 FX Graph:")
print(traced.graph)

# -----------------------------
# Step 2: Prompt 构造
# -----------------------------
batch = 8
n_heads = 8
seq_len = 4096
head_dim = 128

ir_metadata = {
    "op_type": ["attention_matmul"],
    "shape": [batch, n_heads, seq_len, head_dim],
    "dtype": "float32",
    "device": "cuda"
}

prompt_template = f"""
I am compiling a Triton GPU kernel for the following workload:
- Operation: QK^T matrix multiplication (from self-attention)
- Input shape: [batch, n_heads, seq_len, head_dim] = {ir_metadata['shape']}
- Dtype: {ir_metadata['dtype']}
- Device: {ir_metadata['device']}
Please suggest BLOCK_SIZE, NUM_WARPS, and NUM_STAGES that maximize performance under shared memory constraints (≤ 100KB).
"""

# -----------------------------
# GPT 参数提取
# -----------------------------
def parse_gpt_reply(text):
    block_size = re.search(r"BLOCK_SIZE\s*[:=]\s*(\d+)", text)
    num_warps = re.search(r"NUM_WARPS\s*[:=]\s*(\d+)", text)
    num_stages = re.search(r"NUM_STAGES\s*[:=]\s*(\d+)", text)
    config = {
        "BLOCK_SIZE": int(block_size.group(1)) if block_size else 64,
        "NUM_WARPS": int(num_warps.group(1)) if num_warps else 4,
        "NUM_STAGES": int(num_stages.group(1)) if num_stages else 2,
    }
    config["BLOCK_SIZE"] = min(config["BLOCK_SIZE"], 64)
    return config

# -----------------------------
# Triton matmul kernel（用于 attention）
# -----------------------------
@triton.jit
def qk_matmul_kernel(Q, K, Out, M, N, K_dim, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, K_dim, BLOCK_SIZE):
        q = tl.load(Q + offs_m[:, None] * K_dim + (k + tl.arange(0, BLOCK_SIZE))[None, :],
                    mask=(offs_m[:, None] < M) & ((k + tl.arange(0, BLOCK_SIZE))[None, :] < K_dim))
        k_t = tl.load(K + offs_n[None, :] * K_dim + (k + tl.arange(0, BLOCK_SIZE))[:, None],
                      mask=(offs_n[None, :] < N) & ((k + tl.arange(0, BLOCK_SIZE))[:, None] < K_dim))
        acc += tl.dot(q, k_t)

    tl.store(Out + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# -----------------------------
# Benchmark 测试
# -----------------------------
M = N = seq_len
K_dim = head_dim
Q = torch.randn((batch * n_heads, M, K_dim), device='cuda')
K = torch.randn((batch * n_heads, N, K_dim), device='cuda')

SHARED_MEM_LIMIT = 101376

def benchmark_kernel(block_size, num_warps):
    shared_mem_required = 2 * block_size * block_size * 4
    if shared_mem_required > SHARED_MEM_LIMIT:
        print(f"⚠️ Skipping ({block_size}, {num_warps}) — shared memory exceeds limit.")
        return float('inf')

    total_time = 0.0
    for _ in range(5):
        Out = torch.empty((batch * n_heads, M, N), device='cuda')
        grid = lambda META: (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
        qk_matmul_kernel[grid](Q, K, Out, M, N, K_dim, BLOCK_SIZE=block_size, num_warps=num_warps)
        torch.cuda.synchronize()
        start = time.time()
        qk_matmul_kernel[grid](Q, K, Out, M, N, K_dim, BLOCK_SIZE=block_size, num_warps=num_warps)
        torch.cuda.synchronize()
        total_time += (time.time() - start)
    return total_time * 1000 / 5

# -----------------------------
# Baseline + GPT 结果比较
# -----------------------------
results = []

baseline_configs = [
    {"label": "Baseline (16, 2)", "BLOCK_SIZE": 16, "NUM_WARPS": 2},
    {"label": "Baseline (32, 2)", "BLOCK_SIZE": 32, "NUM_WARPS": 2},
    {"label": "Baseline (64, 4)", "BLOCK_SIZE": 64, "NUM_WARPS": 4},
    {"label": "Baseline (64, 8)", "BLOCK_SIZE": 64, "NUM_WARPS": 8},
    {"label": "Baseline (128, 4)", "BLOCK_SIZE": 128, "NUM_WARPS": 4},
    {"label": "Baseline (128, 8)", "BLOCK_SIZE": 128, "NUM_WARPS": 8},
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
        print(f"\n📬 GPT Suggestion #{i+1}:", gpt_reply)
    except Exception as e:
        print(f"\n❌ GPT Suggestion #{i+1} failed: {e}")

# -----------------------------
# 打印性能对比结果
# -----------------------------
print("\n📊 Execution Time Summary (ms):")
print("{:<35} {:>10}".format("Configuration", "Time (ms)"))
print("-" * 50)
for r in results:
    print("{:<35} {:>10.3f}".format(r["label"], r["time"]))
