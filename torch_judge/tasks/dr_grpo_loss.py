"""Dr. GRPO (Group Relative Policy Optimization Done Right) de-biased loss task."""

TASK = {
    "title": "去偏组相对策略优化损失（Dr. GRPO）",
    "difficulty": "Hard",
    "function_name": "dr_grpo_loss",
    "hint": (
        "两处关键去偏：(1) 组内优势仅减均值 `A_i = r_i - mean_g`（不除以 `std_g`，也不乘 `K/(K-1)`），并从计算图 `detach`；"
        "(2) 词元级对称裁剪代理目标对有效词元（`mask == 1`）求和后，统一除以固定常数 `B * L_max`（`L_max = max_tokens or new_logps.size(1)`），不除以每句长度 `|o_i|` 或有效词元总数。"
    ),
    "tests": [
        {
            "name": "基本形状与类型",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "new_logps = torch.randn(4, 5, requires_grad=True)\n"
            "old_logps = torch.randn(4, 5)\n"
            "rewards = torch.tensor([1.0, 0.0, 0.2, 0.8])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be a scalar Tensor'\n",
        },
        {
            "name": "去难度偏差：仅减组均值而不除标准差（区分 GRPO 与 RLOO）",
            "code": "\n"
            "import torch\n"
            "# Group 0 (K=2): rewards=[1.0, 0.0] -> mean=0.5, Dr.GRPO A=[+0.5, -0.5]\n"
            "#   (Standard GRPO dividing by std=0.5 gives [+1.0, -1.0]; RLOO K/(K-1) also gives [+1.0, -1.0]!)\n"
            "old_logps = torch.zeros(2, 2)\n"
            "new_logps = torch.tensor([[0.1, 0.0], [-0.1, 0.0]], requires_grad=True)\n"
            "rewards = torch.tensor([1.0, 0.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "loss_base = {fn}(new_logps, old_logps, rewards, group_ids, clip_ratio=0.2)\n"
            "\n"
            "r0 = torch.exp(new_logps.detach()[0]) * 0.5\n"
            "r1 = torch.exp(new_logps.detach()[1]) * (-0.5)\n"
            "expected = -(r0.sum() + r1.sum()) / (2 * 2)\n"
            "assert torch.allclose(loss_base, expected, atol=1e-6), (\n"
            "    f'Expected {expected.item():.6f}, got {loss_base.item():.6f} '\n"
            "    '(Dr. GRPO uses A_i = r_i - mean_g without std division or K/(K-1) scaling)'\n"
            ")\n"
            "\n"
            "# Scaling rewards by 3x must scale Dr. GRPO loss by 3x (unlike std-normalized GRPO!)\n"
            "loss_scaled = {fn}(new_logps, old_logps, rewards * 3.0, group_ids, clip_ratio=0.2)\n"
            "assert torch.allclose(loss_scaled, 3.0 * loss_base, atol=1e-6), (\n"
            "    'Dr. GRPO must preserve reward scale (do not divide by group std)'\n"
            ")\n"
            "\n"
            "# Shifting rewards by a constant within group must leave loss unchanged\n"
            "loss_shifted = {fn}(new_logps, old_logps, rewards + 5.0, group_ids, clip_ratio=0.2)\n"
            "assert torch.allclose(loss_shifted, loss_base, atol=1e-6), 'Per-group constant shift should not change loss'\n",
        },
        {
            "name": "去长度偏差：固定常数分母 B * max_tokens（区分 GRPO 与 DAPO）",
            "code": "\n"
            "import torch\n"
            "# Group 0: Seq 0 is correct (+1), Seq 1 is short wrong (1 token), Seq 2 is rambling wrong (4 tokens)\n"
            "# Rewards=[1.0, -0.5, -0.5] -> mean=0.0 -> A=[1.0, -0.5, -0.5]\n"
            "# At new_logps == old_logps, every valid token in Seq 1 and Seq 2 has equal A=-0.5,\n"
            "# so Dr. GRPO gives IDENTICAL per-token penalty gradient regardless of response length!\n"
            "old_logps = torch.zeros(3, 4)\n"
            "new_logps = torch.zeros(3, 4, requires_grad=True)\n"
            "mask = torch.tensor([\n"
            "    [1.0, 1.0, 0.0, 0.0],\n"
            "    [1.0, 0.0, 0.0, 0.0],\n"
            "    [1.0, 1.0, 1.0, 1.0],\n"
            "])\n"
            "rewards = torch.tensor([1.0, -0.5, -0.5])\n"
            "group_ids = torch.tensor([0, 0, 0])\n"
            "loss_default = {fn}(new_logps, old_logps, rewards, group_ids, mask=mask)\n"
            "loss_default.backward()\n"
            "\n"
            "# Valid token gradients on short wrong Seq 1 (len 1) and long wrong Seq 2 (len 4) MUST be equal!\n"
            "assert torch.allclose(new_logps.grad[1, 0], new_logps.grad[2, 0], atol=1e-6), (\n"
            "    f'Per-token gradient should not be diluted by sequence length: {new_logps.grad[1, 0].item()} vs {new_logps.grad[2, 0].item()}'\n"
            ")\n"
            "expected_default = -(2 * 1.0 + 1 * (-0.5) + 4 * (-0.5)) / (3 * 4)  # 0.5 / 12\n"
            "assert torch.allclose(loss_default, torch.tensor(expected_default), atol=1e-6), (\n"
            "    f'Expected {expected_default:.6f} with default max_tokens=T=4, got {loss_default.item():.6f} '\n"
            "    '(divide by B * L_max, NOT active token count sum(mask) or per-seq |o_i|)'\n"
            ")\n"
            "\n"
            "# Explicit max_tokens=8 must halve the loss compared to max_tokens=4\n"
            "loss_max8 = {fn}(new_logps.detach(), old_logps, rewards, group_ids, mask=mask, max_tokens=8)\n"
            "assert torch.allclose(loss_max8, 0.5 * loss_default, atol=1e-6), (\n"
            "    'Passing explicit max_tokens=8 should normalize by B * 8'\n"
            ")\n",
        },
        {
            "name": "词元级对称裁剪与零梯度验证",
            "code": "\n"
            "import torch\n"
            "ratios = torch.tensor([[1.35, 1.10], [0.65, 0.90]])\n"
            "old_logps = torch.zeros(2, 2)\n"
            "new_logps = ratios.log().clone().requires_grad_(True)\n"
            "rewards = torch.tensor([1.0, -1.0])  # mean=0 -> A=[+1.0, -1.0]\n"
            "group_ids = torch.tensor([0, 0])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids, clip_ratio=0.2, max_tokens=2)\n"
            "loss.backward()\n"
            "expected = torch.tensor(-(1.20 + 1.10 - 0.80 - 0.90) / 4.0)  # -0.15\n"
            "assert torch.allclose(loss, expected, atol=1e-6), f'Expected {expected.item():.6f}, got {loss.item():.6f}'\n"
            "assert torch.allclose(new_logps.grad[0, 0], torch.tensor(0.0), atol=1e-6), 'Upper-clipped token must have 0 grad'\n"
            "assert torch.allclose(new_logps.grad[1, 0], torch.tensor(0.0), atol=1e-6), 'Lower-clipped token must have 0 grad'\n",
        },
        {
            "name": "掩码隔离填充词元极端值",
            "code": "\n"
            "import torch\n"
            "old_logps = torch.tensor([[-0.3, -100.0], [-0.5, 100.0]])\n"
            "new_logps = torch.tensor([[-0.2, 100.0], [-0.6, -100.0]], requires_grad=True)\n"
            "mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])\n"
            "rewards = torch.tensor([1.0, -1.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids, mask=mask, max_tokens=2)\n"
            "assert torch.isfinite(loss), 'Masked padding positions with extreme logps must not overflow'\n"
            "loss.backward()\n"
            "assert torch.all(new_logps.grad[mask == 0] == 0), 'Masked positions must receive zero gradient'\n",
        },
        {
            "name": "非连续组编号、单样本组与同奖励组",
            "code": "\n"
            "import torch\n"
            "# Group 0: [2.0, 0.0] -> A=[+1.0, -1.0]\n"
            "# Group 1 (K=1): [5.0] -> A=[0.0]\n"
            "# Group 2 (equal rewards): [1.0, 1.0] -> A=[0.0, 0.0]\n"
            "old_logps = torch.zeros(5, 3)\n"
            "new_logps = torch.full((5, 3), 0.05, requires_grad=True)\n"
            "rewards = torch.tensor([1.0, 2.0, 5.0, 1.0, 0.0])\n"
            "group_ids = torch.tensor([2, 0, 1, 2, 0])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids)\n"
            "assert torch.isfinite(loss), 'Loss must be finite on K=1 and zero-std groups'\n"
            "loss.backward()\n"
            "assert torch.all(new_logps.grad[0] == 0) and torch.all(new_logps.grad[2] == 0) and torch.all(new_logps.grad[3] == 0), (\n"
            "    'Single-sample and equal-reward groups must have zero advantage and zero gradient'\n"
            ")\n",
        },
        {
            "name": "梯度仅通过 new_logps 传播",
            "code": "\n"
            "import torch\n"
            "new_logps = torch.randn(4, 3, requires_grad=True)\n"
            "old_logps = torch.randn(4, 3, requires_grad=True)\n"
            "rewards = torch.tensor([1.0, 0.0, 0.2, 0.9], requires_grad=True)\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids)\n"
            "loss.backward()\n"
            "assert new_logps.grad is not None and torch.isfinite(new_logps.grad).all(), 'new_logps must receive finite gradients'\n"
            "assert old_logps.grad is None, 'old_logps must be detached'\n"
            "assert rewards.grad is None, 'rewards must be detached'\n",
        },
    ],
}
