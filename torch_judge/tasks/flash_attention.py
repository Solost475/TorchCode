"""Flash Attention (Tiled) task."""

TASK = {
    "title": "分块注意力（Flash Attention）",
    "difficulty": "Hard",
    "function_name": "flash_attention",
    "hint": "将 Q 分块，对每个 Q 块遍历 K/V 块。使用在线 softmax，维护当前最大值与累加和，并在最大值变化时重新缩放累加器。output = acc / row_sum。",
    "tests": [
        {
            "name": "与标准注意力一致",
            "code": "\nimport torch, math\ntorch.manual_seed(0)\nB, S, D = 2, 16, 8\nQ = torch.randn(B, S, D)\nK = torch.randn(B, S, D)\nV = torch.randn(B, S, D)\nout = {fn}(Q, K, V, block_size=4)\nscores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)\nref = torch.bmm(torch.softmax(scores, dim=-1), V)\nassert torch.allclose(out, ref, atol=1e-4), f'Max diff: {(out-ref).abs().max():.6f}'\n"
        },
        {
            "name": "非整除的分块大小",
            "code": "\nimport torch, math\ntorch.manual_seed(42)\nB, S, D = 1, 7, 4\nQ, K, V = torch.randn(B,S,D), torch.randn(B,S,D), torch.randn(B,S,D)\nout = {fn}(Q, K, V, block_size=3)\nscores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)\nref = torch.bmm(torch.softmax(scores, dim=-1), V)\nassert torch.allclose(out, ref, atol=1e-4), 'Mismatch with non-aligned block size'\n"
        },
        {
            "name": "结果不受分块大小影响",
            "code": "\nimport torch\ntorch.manual_seed(0)\nQ, K, V = torch.randn(1,12,8), torch.randn(1,12,8), torch.randn(1,12,8)\nout4 = {fn}(Q, K, V, block_size=4)\nout6 = {fn}(Q, K, V, block_size=6)\nassert torch.allclose(out4, out6, atol=1e-4), 'Different block sizes should give same result'\n"
        },
        {
            "name": "梯度传播",
            "code": "\nimport torch\nQ = torch.randn(1, 8, 4, requires_grad=True)\nK = torch.randn(1, 8, 4, requires_grad=True)\nV = torch.randn(1, 8, 4, requires_grad=True)\n{fn}(Q, K, V, block_size=4).sum().backward()\nassert Q.grad is not None, 'Q.grad is None'\n"
        }
    ]
}
