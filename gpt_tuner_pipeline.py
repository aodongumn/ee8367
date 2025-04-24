import torch
import torch.nn as nn
import torch.fx as fx
import triton
import triton.language as tl
import openai
import time
import re

# -----------------------------
# GPT 配置
# -----------------------------
openai.api_key = "your own key" 

# -----------------------------
# Step 1: 定义 PyTorch 模型
# -----------------------------
class MyModel(nn.Module):
    def forward(self, x):
        return torch.relu(torch.sin(x))

model = MyModel()
traced = fx.symbolic_trace(model)
print("\n🧠 FX Graph:")
print(traced.graph)

# -----------------------------
# Step 2: 构造 GPT Prompt
# -----------------------------
ir_metadata = {
    "op_type": ["sin", "relu"],
    "shape": [1024, 1024],
    "dtype": "float32",
    "device": "cuda"
}

prompt = f"""
I am compiling a Triton GPU kernel for the following workload:

- Operation sequence: {' → '.join(ir_metadata['op_type'])}
- Input tensor shape: {ir_metadata['shape']}
- Data type: {ir_metadata['dtype']}
- Device: {ir_metadata['device']}

Please suggest a good Triton kernel configuration in terms of:
- BLOCK_SIZE
- NUM_WARPS
- NUM_STAGES

The kernel performs elementwise computation and should prioritize latency hiding and efficient warp scheduling.
"""

# -----------------------------
# Step 3: 请求 GPT 推荐参数
# -----------------------------
def parse_gpt_reply(text):
    block_size = re.search(r"BLOCK_SIZE\s*[:=]\s*(\d+)", text)
    num_warps = re.search(r"NUM_WARPS\s*[:=]\s*(\d+)", text)
    num_stages = re.search(r"NUM_STAGES\s*[:=]\s*(\d+)", text)
    return {
        "BLOCK_SIZE": int(block_size.group(1)) if block_size else 128,
        "NUM_WARPS": int(num_warps.group(1)) if num_warps else 4,
        "NUM_STAGES": int(num_stages.group(1)) if num_stages else 2,
    }

try:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    gpt_reply = response['choices'][0]['message']['content']
    print("\n📬 GPT Suggestion:")
    print(gpt_reply)
    config = parse_gpt_reply(gpt_reply)
except Exception as e:
    print("\n⚠️ GPT 调用失败，使用默认参数。\n错误信息:", e)
    config = {"BLOCK_SIZE": 128, "NUM_WARPS": 4, "NUM_STAGES": 2}

BLOCK_SIZE = config["BLOCK_SIZE"]
NUM_WARPS = config["NUM_WARPS"]

# -----------------------------
# Step 4: 使用 GPT 建议的配置运行 Triton kernel
# -----------------------------
@triton.jit
def relu_sin_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.sin(x)
    y = tl.maximum(y, 0)
    tl.store(Y + offsets, y, mask=mask)

N = 1024 * 1024
x = torch.randn(N, device='cuda')
y = torch.empty_like(x)

grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
start = time.time()
relu_sin_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
torch.cuda.synchronize()
end = time.time()

print("\n✅ Triton Kernel Output (first 5):", y[:5])
print(f"🕒 Execution Time: {(end - start) * 1000:.3f} ms")
