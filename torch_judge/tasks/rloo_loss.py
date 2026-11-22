"""RLOO (REINFORCE Leave-One-Out) Loss task."""

TASK = {
    "title": "留一法策略梯度损失（RLOO）",
    "difficulty": "Hard",
    "function_name": "rloo_loss",
    "hint": (
        "若 `logps` 为二维 `(B, T)`，先按词元维求和（如有 `mask` 则对有效词元求和）得到序列对数概率 `seq_logps`。"
        "在每个组 `g` 内计算留一基线 `b_i = (sum(r_g) - r_i) / (K_g - 1)`，优势 `A_i = r_i - b_i`（当 `K_g == 1` 时 `A_i = 0`）。"
        "将 `A_i` 从计算图中分离后返回 `-mean(A_i * seq_logps)`。"
    ),
    "tests": [
        {
            "name": "基本形状与类型",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "logps = torch.randn(6, requires_grad=True)\n"
            "rewards = torch.randn(6)\n"
            "group_ids = torch.tensor([0, 0, 0, 1, 1, 1])\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be a scalar Tensor'\n",
        },
        {
            "name": "与留一基线解析解对照（区分普通均值基线）",
            "code": "\n"
            "import torch\n"
            "# Group 0 (K=2): r=[1.0, 0.0] -> LOO baseline=[0.0, 1.0], A=[+1.0, -1.0]\n"
            "#   (naive group-mean baseline would give [+0.5, -0.5], off by K/(K-1)=2x)\n"
            "# Group 1 (K=2): r=[2.0, 0.5] -> LOO baseline=[0.5, 2.0], A=[+1.5, -1.5]\n"
            "logps = torch.tensor([-0.5, -1.5, -0.2, -0.8], requires_grad=True)\n"
            "rewards = torch.tensor([1.0, 0.0, 2.0, 0.5])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "# -A * logps = [0.5, -1.5, 0.3, -1.2], mean = -1.9 / 4 = -0.475\n"
            "expected = torch.tensor(-0.475)\n"
            "assert torch.allclose(loss, expected, atol=1e-6), (\n"
            "    f'Expected {expected.item():.6f}, got {loss.item():.6f} '\n"
            "    '(check leave-one-out denominator K-1 and negative sign)'\n"
            ")\n",
        },
        {
            "name": "平移不变性与奖励尺度保留（区分 GRPO）",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([-0.5, -1.2, -0.3, -0.9], requires_grad=True)\n"
            "rewards = torch.tensor([1.0, 0.2, 0.5, -0.5])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss_base = {fn}(logps, rewards, group_ids)\n"
            "\n"
            "# 1. Per-group constant reward shift must not change RLOO advantages or loss\n"
            "shifts = torch.tensor([10.0, 10.0, -5.0, -5.0])\n"
            "loss_shifted = {fn}(logps, rewards + shifts, group_ids)\n"
            "assert torch.allclose(loss_shifted, loss_base, atol=1e-6), (\n"
            "    f'Per-group reward shift should not change loss: {loss_shifted.item():.6f} vs {loss_base.item():.6f}'\n"
            ")\n"
            "\n"
            "# 2. Unlike GRPO, RLOO does NOT divide by group std; scaling rewards by 3x scales loss by 3x\n"
            "loss_scaled = {fn}(logps, rewards * 3.0, group_ids)\n"
            "assert torch.allclose(loss_scaled, 3.0 * loss_base, atol=1e-6), (\n"
            "    f'RLOO should preserve reward scale (expected {(3.0 * loss_base).item():.6f}, got {loss_scaled.item():.6f}); '\n"
            "    'do not normalize by group std like GRPO'\n"
            ")\n",
        },
        {
            "name": "非连续组编号、变长组与单样本组",
            "code": "\n"
            "import torch\n"
            "# Interleaved group_ids with sizes: group 0 -> K=2, group 1 -> K=1, group 2 -> K=3\n"
            "group_ids = torch.tensor([2, 0, 2, 1, 0, 2])\n"
            "rewards = torch.tensor([3.0, 2.0, 1.0, 5.0, 0.0, -1.0])\n"
            "logps = torch.tensor([-0.3, -0.4, -0.5, -1.2, -0.8, -0.9], requires_grad=True)\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "assert torch.isfinite(loss), 'Loss must be finite even when a group has K=1 sample'\n"
            "loss.backward()\n"
            "# Expected advantages:\n"
            "#   idx 0 (g=2): 3 - (1 + -1)/2 = 3.0\n"
            "#   idx 1 (g=0): 2 - 0 = 2.0\n"
            "#   idx 2 (g=2): 1 - (3 + -1)/2 = 0.0\n"
            "#   idx 3 (g=1, K=1): 0.0\n"
            "#   idx 4 (g=0): 0 - 2 = -2.0\n"
            "#   idx 5 (g=2): -1 - (3 + 1)/2 = -3.0\n"
            "expected_adv = torch.tensor([3.0, 2.0, 0.0, 0.0, -2.0, -3.0])\n"
            "expected_loss = -(expected_adv * logps.detach()).mean()\n"
            "expected_grad = -expected_adv / logps.numel()\n"
            "assert torch.allclose(loss, expected_loss, atol=1e-6), (\n"
            "    f'Expected {expected_loss.item():.6f}, got {loss.item():.6f}'\n"
            ")\n"
            "assert torch.allclose(logps.grad, expected_grad, atol=1e-6), (\n"
            "    'Gradients mismatch on interleaved variable-size groups (K=1 sample must have 0 advantage)'\n"
            ")\n",
        },
        {
            "name": "词元级对数概率与掩码聚合",
            "code": "\n"
            "import torch\n"
            "logps_2d = torch.tensor([\n"
            "    [-0.2, -0.3, -50.0],\n"
            "    [-0.5, -1.0, -0.5],\n"
            "    [-0.1, -100.0, -100.0],\n"
            "    [-0.4, -0.4, -20.0],\n"
            "], requires_grad=True)\n"
            "mask = torch.tensor([\n"
            "    [1.0, 1.0, 0.0],\n"
            "    [1.0, 1.0, 1.0],\n"
            "    [1.0, 0.0, 0.0],\n"
            "    [1.0, 1.0, 0.0],\n"
            "])\n"
            "rewards = torch.tensor([1.0, 0.0, 0.5, 1.5])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss_masked = {fn}(logps_2d, rewards, group_ids, mask=mask)\n"
            "loss_masked.backward()\n"
            "\n"
            "# Sequence log-prob in RLOO is the SUM over valid response tokens, not token mean\n"
            "seq_logps = (logps_2d.detach() * mask).sum(dim=-1)\n"
            "loss_1d = {fn}(seq_logps, rewards, group_ids)\n"
            "assert torch.allclose(loss_masked, loss_1d, atol=1e-6), (\n"
            "    f'2D masked loss {loss_masked.item():.6f} should match 1D sequence-sum loss {loss_1d.item():.6f}'\n"
            ")\n"
            "assert torch.all(logps_2d.grad[mask == 0] == 0), 'Masked padding tokens must receive zero gradient'\n"
            "\n"
            "# Also verify 2D logps with mask=None sums across all tokens\n"
            "clean_2d = torch.tensor([[-0.2, -0.3], [-0.5, -1.0]], requires_grad=True)\n"
            "loss_nomask = {fn}(clean_2d, torch.tensor([1.0, 0.0]), torch.tensor([0, 0]))\n"
            "loss_sum1d = {fn}(clean_2d.detach().sum(dim=-1), torch.tensor([1.0, 0.0]), torch.tensor([0, 0]))\n"
            "assert torch.allclose(loss_nomask, loss_sum1d, atol=1e-6), '2D logps with mask=None should sum over dim=-1'\n",
        },
        {
            "name": "梯度仅通过 logps 传播",
            "code": "\n"
            "import torch\n"
            "logps = torch.randn(6, requires_grad=True)\n"
            "rewards = torch.randn(6, requires_grad=True)\n"
            "group_ids = torch.tensor([0, 0, 0, 1, 1, 1])\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "loss.backward()\n"
            "assert logps.grad is not None and torch.isfinite(logps.grad).all(), 'Gradients should flow through logps'\n"
            "assert rewards.grad is None, 'Gradients must not flow through rewards (detach advantages)'\n",
        },
        {
            "name": "组内奖励相同时优势与梯度为零",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([-0.4, -1.1, -0.2, -0.7, -1.5], requires_grad=True)\n"
            "rewards = torch.tensor([1.0, 1.0, -2.0, -2.0, -2.0])\n"
            "group_ids = torch.tensor([0, 0, 1, 1, 1])\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "loss.backward()\n"
            "assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), f'Expected 0 loss, got {loss.item():.6f}'\n"
            "assert torch.allclose(logps.grad, torch.zeros_like(logps), atol=1e-6), 'Expected zero gradients when within-group rewards are equal'\n",
        },
    ],
}
