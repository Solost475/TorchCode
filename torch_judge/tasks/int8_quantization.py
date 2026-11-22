"""INT8 Quantized Linear task."""

TASK = {
    "title": "INT8 量化线性层",
    "difficulty": "Hard",
    "function_name": "Int8Linear",
    "hint": "按输出通道计算 scale = abs(weight).max(dim=1) / 127。量化：round(weight/scale).clamp(-128,127).to(int8)。前向计算时先反量化，再进行矩阵乘法。",
    "tests": [
        {
            "name": "权重类型为 int8",
            "code": "\nimport torch, torch.nn as nn\nw = torch.randn(32, 16)\nq = {fn}(w)\nassert isinstance(q, nn.Module)\nassert q.weight_int8.dtype == torch.int8, f'dtype: {q.weight_int8.dtype}'\n"
        },
        {
            "name": "数值位于 [-128, 127]",
            "code": "\nimport torch\nq = {fn}(torch.randn(64, 32) * 10)\nassert q.weight_int8.min() >= -128 and q.weight_int8.max() <= 127\n"
        },
        {
            "name": "反量化结果接近原始值",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw = torch.randn(16, 8)\nq = {fn}(w)\nw_recon = q.weight_int8.float() * q.scale\nassert (w - w_recon).abs().max() < 0.1, 'Quantization error too large'\n"
        },
        {
            "name": "前向输出形状",
            "code": "\nimport torch\nq = {fn}(torch.randn(8, 4), torch.randn(8))\nout = q(torch.randn(2, 4))\nassert out.shape == (2, 8), f'Shape: {out.shape}'\n"
        },
        {
            "name": "权重保存为缓冲区而非参数",
            "code": "\nimport torch\nq = {fn}(torch.randn(4, 4))\nparam_names = [n for n, _ in q.named_parameters()]\nassert 'weight_int8' not in param_names, 'weight_int8 should be a buffer'\nassert 'scale' not in param_names, 'scale should be a buffer'\n"
        }
    ]
}
