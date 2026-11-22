"""DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) Loss task."""

TASK = {
    "title": "解耦裁剪与动态采样策略优化损失（DAPO）",
    "difficulty": "Hard",
    "function_name": "dapo_loss",
    "hint": (
        "按组计算均值与总体标准差 `std_g = r_g.std(unbiased=False)`：若 `std_g == 0` 则由动态采样剔除该组，否则 `A_i = (r_i - mean_g) / (std_g + eps)`。"
        "计算词元级重要性比率 `r = exp(new_logps - old_logps.detach())`，使用非对称区间 `clamp(r, 1 - clip_low, 1 + clip_high)` 构造裁剪代理目标，"
        "仅对有效组且 `mask == 1` 的词元全局求和并除以有效词元总数，取负号返回（全被剔除时返回 `new_logps.sum() * 0.0`）。"
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
            "name": "解耦非对称裁剪 Clip-Higher（区分对称裁剪）",
            "code": "\n"
            "import math\n"
            "import torch\n"
            "# Group 0 (K=2): rewards=[1.0, -1.0], eps=0.0 -> A=[+1.0, -1.0]\n"
            "# Ratios: row 0 (A=+1) has [1.25, 1.40]; row 1 (A=-1) has [0.75, 0.85]\n"
            "# With clip_low=0.20, clip_high=0.28 -> bounds [0.80, 1.28]:\n"
            "#   (0,0): r=1.25 < 1.28 (unclipped!) -> +1.25\n"
            "#   (0,1): r=1.40 > 1.28 (clipped at 1.28) -> +1.28\n"
            "#   (1,0): r=0.75 < 0.80 (clipped at 0.80 for A=-1) -> -0.80\n"
            "#   (1,1): r=0.85 > 0.80 (unclipped) -> -0.85\n"
            "ratios = torch.tensor([[1.25, 1.40], [0.75, 0.85]])\n"
            "old_logps = torch.zeros(2, 2)\n"
            "new_logps = ratios.log().clone().requires_grad_(True)\n"
            "rewards = torch.tensor([1.0, -1.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids, clip_low=0.20, clip_high=0.28, eps=0.0)\n"
            "loss.backward()\n"
            "expected = torch.tensor(-(1.25 + 1.28 - 0.80 - 0.85) / 4.0)  # -0.22\n"
            "assert torch.allclose(loss, expected, atol=1e-6), (\n"
            "    f'Expected {expected.item():.6f}, got {loss.item():.6f} '\n"
            "    '(check asymmetric bounds [1 - clip_low, 1 + clip_high])'\n"
            ")\n"
            "# Unclipped positions (0,0) and (1,1) must have non-zero grad; clipped (0,1) and (1,0) must have 0 grad\n"
            "assert new_logps.grad[0, 0].abs() > 1e-4 and new_logps.grad[1, 1].abs() > 1e-4, 'Unclipped tokens should receive gradient'\n"
            "assert torch.allclose(new_logps.grad[0, 1], torch.tensor(0.0), atol=1e-6), 'Upper-clipped token should have 0 gradient'\n"
            "assert torch.allclose(new_logps.grad[1, 0], torch.tensor(0.0), atol=1e-6), 'Lower-clipped token should have 0 gradient'\n",
        },
        {
            "name": "词元级全局归一化与填充掩码（区分 GRPO 序列级平均）",
            "code": "\n"
            "import torch\n"
            "# Seq 0 has 1 valid token; Seq 1 has 4 valid tokens (total = 5 active tokens)\n"
            "old_logps = torch.tensor([\n"
            "    [-0.5, -50.0, -50.0, -50.0],\n"
            "    [-0.2, -0.3, -0.4, -0.5],\n"
            "])\n"
            "new_logps = torch.tensor([\n"
            "    [-0.4, 50.0, 50.0, 50.0],\n"
            "    [-0.25, -0.25, -0.45, -0.45],\n"
            "], requires_grad=True)\n"
            "mask = torch.tensor([\n"
            "    [1.0, 0.0, 0.0, 0.0],\n"
            "    [1.0, 1.0, 1.0, 1.0],\n"
            "])\n"
            "rewards = torch.tensor([1.0, -1.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids, mask=mask, eps=0.0)\n"
            "loss.backward()\n"
            "\n"
            "r0 = torch.exp(torch.tensor(-0.4 - (-0.5))) * 1.0\n"
            "r1 = torch.exp(new_logps.detach()[1] - old_logps[1]) * (-1.0)\n"
            "# Global token-level denominator is 5 (not per-sequence mean of 1 and 4!)\n"
            "expected = -(r0 + r1.sum()) / 5.0\n"
            "assert torch.allclose(loss, expected, atol=1e-6), (\n"
            "    f'Expected token-level normalized loss {expected.item():.6f}, got {loss.item():.6f} '\n"
            "    '(normalize by total valid tokens across batch, not sequence-level mean)'\n"
            ")\n"
            "assert torch.all(new_logps.grad[mask == 0] == 0), 'Masked padding tokens must have zero gradient'\n",
        },
        {
            "name": "动态采样：剔除零方差组且不稀释有效分母",
            "code": "\n"
            "import torch\n"
            "# Group 0: valid (rewards [1.0, 0.0], std > 0)\n"
            "# Group 1: all correct [1.0, 1.0] (std == 0 -> filtered out)\n"
            "# Group 2: all wrong [0.0, 0.0] (std == 0 -> filtered out)\n"
            "torch.manual_seed(42)\n"
            "old_logps = torch.randn(6, 3)\n"
            "new_logps = (old_logps + 0.05 * torch.randn(6, 3)).requires_grad_(True)\n"
            "rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])\n"
            "group_ids = torch.tensor([0, 0, 1, 1, 2, 2])\n"
            "\n"
            "loss_all = {fn}(new_logps, old_logps, rewards, group_ids)\n"
            "loss_all.backward()\n"
            "loss_valid_only = {fn}(new_logps.detach()[:2], old_logps[:2], rewards[:2], group_ids[:2])\n"
            "assert torch.allclose(loss_all, loss_valid_only, atol=1e-6), (\n"
            "    f'Zero-variance groups must not dilute token denominator: got {loss_all.item():.6f} vs {loss_valid_only.item():.6f}'\n"
            ")\n"
            "assert torch.allclose(new_logps.grad[2:], torch.zeros_like(new_logps.grad[2:]), atol=1e-7), (\n"
            "    'Filtered zero-variance groups must receive zero gradient'\n"
            ")\n",
        },
        {
            "name": "全批次均为零方差组时不产生 NaN",
            "code": "\n"
            "import torch\n"
            "new_logps = torch.randn(4, 3, requires_grad=True)\n"
            "old_logps = torch.randn(4, 3)\n"
            "rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids)\n"
            "assert torch.isfinite(loss) and torch.allclose(loss, torch.tensor(0.0), atol=1e-7), (\n"
            "    f'Expected finite 0.0 loss when all groups have zero std, got {loss}'\n"
            ")\n"
            "loss.backward()\n"
            "assert new_logps.grad is not None and torch.all(new_logps.grad == 0), 'Expected all-zero gradients on new_logps'\n",
        },
        {
            "name": "非连续组编号与变长组一致性",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(7)\n"
            "old_logps = torch.randn(5, 4)\n"
            "new_logps = (old_logps + 0.1 * torch.randn(5, 4)).requires_grad_(True)\n"
            "rewards = torch.tensor([2.0, 1.0, 0.0, -1.0, 1.0])\n"
            "group_ids = torch.tensor([1, 0, 1, 0, 1])\n"
            "\n"
            "perm = torch.tensor([1, 3, 0, 2, 4])\n"
            "new_perm = new_logps.detach()[perm].requires_grad_(True)\n"
            "loss1 = {fn}(new_logps, old_logps, rewards, group_ids)\n"
            "loss2 = {fn}(new_perm, old_logps[perm], rewards[perm], group_ids[perm])\n"
            "loss1.backward()\n"
            "loss2.backward()\n"
            "assert torch.allclose(loss1, loss2, atol=1e-6), 'Row permutation should not change DAPO loss'\n"
            "assert torch.allclose(new_logps.grad[perm], new_perm.grad, atol=1e-6), 'Gradients must match under permutation'\n",
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
            "assert new_logps.grad is not None and torch.isfinite(new_logps.grad).all(), 'Gradients should flow through new_logps'\n"
            "assert old_logps.grad is None, 'Gradients must not flow through old_logps'\n"
            "assert rewards.grad is None, 'Gradients must not flow through rewards'\n",
        },
    ],
}
