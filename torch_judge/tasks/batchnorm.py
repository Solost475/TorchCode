"""BatchNorm implementation task."""

TASK = {
    "title": "批归一化（BatchNorm）",
    "difficulty": "Medium",
    "function_name": "my_batch_norm",
    "hint": (
        "实现支持训练和推理的 BatchNorm：训练时沿 dim=0 计算批次统计量，并使用 momentum 更新 running_mean/running_var；推理时仅使用运行统计量进行归一化。"
    ),
    "tests": [
        {
            "name": "训练模式：每个特征均值为零",
            "code": """
import torch
x = torch.randn(8, 4)
gamma = torch.ones(4)
beta = torch.zeros(4)
running_mean = torch.zeros(4)
running_var = torch.ones(4)
out = {fn}(x, gamma, beta, running_mean, running_var, training=True)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
col_means = out.mean(dim=0)
assert torch.allclose(col_means, torch.zeros(4), atol=1e-5), f'Column means not zero: {col_means}'
""",
        },
        {
            "name": "训练模式：数值正确并更新运行统计量",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(16, 8)
gamma = torch.randn(8)
beta = torch.randn(8)
running_mean = torch.zeros(8)
running_var = torch.ones(8)
momentum = 0.1
out = {fn}(x, gamma, beta, running_mean, running_var, momentum=momentum, training=True)

# Reference using batch stats
mean = x.mean(dim=0)
var = x.var(dim=0, unbiased=False)
ref = gamma * (x - mean) / torch.sqrt(var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch'

# Running stats should have moved toward batch stats
expected_mean = (1 - momentum) * torch.zeros_like(mean) + momentum * mean
expected_var = (1 - momentum) * torch.ones_like(var) + momentum * var
assert torch.allclose(running_mean, expected_mean, atol=1e-6), 'running_mean not updated correctly'
assert torch.allclose(running_var, expected_var, atol=1e-6), 'running_var not updated correctly'
""",
        },
        {
            "name": "推理模式：使用运行统计量",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 8)
gamma = torch.randn(8)
beta = torch.randn(8)

# Pretend these came from previous training
running_mean = torch.randn(8)
running_var = torch.rand(8) + 0.5  # positive

out = {fn}(x, gamma, beta, running_mean.clone(), running_var.clone(), training=False)
ref = gamma * (x - running_mean) / torch.sqrt(running_var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-4), 'Inference should use running stats'
""",
        },
        {
            "name": "输入与仿射参数的梯度传播",
            "code": """
import torch
x = torch.randn(4, 8, requires_grad=True)
gamma = torch.ones(8, requires_grad=True)
beta = torch.zeros(8, requires_grad=True)
running_mean = torch.zeros(8)
running_var = torch.ones(8)
out = {fn}(x, gamma, beta, running_mean, running_var, training=True)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gamma.grad is not None, 'gamma.grad is None'
assert beta.grad is not None, 'beta.grad is None'
""",
        },
    ],
}
