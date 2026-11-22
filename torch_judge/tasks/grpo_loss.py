"""GRPO (Group Relative Policy Optimization) token-level clipped loss task."""

TASK = {
    "title": "组相对策略优化损失（GRPO）",
    "difficulty": "Hard",
    "function_name": "grpo_loss",
    "hint": (
        "按组归一化优势 `A_i = (r_i - mean_g) / (std_g + eps)`（`unbiased=False`）并 `detach`。"
        "计算词元比率 `r = exp(new_logps - old_logps.detach())` 与裁剪代理目标 `s = min(r * A, clamp(r, 1 - clip_ratio, 1 + clip_ratio) * A)`。"
        "若提供 `ref_logps` 且 `kl_coef > 0`，计算无偏 k3 KL：`kl = exp(ref - new) - (ref - new) - 1`，每词元损失为 `-(s - kl_coef * kl)`。"
        "先按每条回复的有效词元数 `|o_i|` 求均值，再对批次 `B` 求均值。"
    ),
    "tests": [
        {
            "name": "基本形状与类型",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "new_logps = torch.randn(4, 5, requires_grad=True)\n"
            "old_logps = torch.randn(4, 5)\n"
            "rewards = torch.tensor([1.0, 0.2, 0.0, 0.8])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be a scalar Tensor'\n",
        },
        {
            "name": "词元级对称裁剪代理目标数值对照",
            "code": "\n"
            "import torch\n"
            "ratios = torch.tensor([[1.30, 1.10], [0.70, 0.90]])\n"
            "old_logps = torch.zeros(2, 2)\n"
            "new_logps = ratios.log().clone().requires_grad_(True)\n"
            "rewards = torch.tensor([1.0, -1.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids, clip_ratio=0.2, eps=0.0)\n"
            "loss.backward()\n"
            "# Row 0 (A=+1): [min(1.3, 1.2), 1.1] -> mean = 1.15\n"
            "# Row 1 (A=-1): [min(-0.7, -0.8), -0.9] -> mean = -0.85\n"
            "# Batch loss = -(1.15 + (-0.85)) / 2 = -0.15\n"
            "expected = torch.tensor(-0.15)\n"
            "assert torch.allclose(loss, expected, atol=1e-6), f'Expected {expected.item():.6f}, got {loss.item():.6f}'\n"
            "assert torch.allclose(new_logps.grad[0, 0], torch.tensor(0.0), atol=1e-6), 'Clipped token (0,0) should have 0 grad'\n"
            "assert torch.allclose(new_logps.grad[1, 0], torch.tensor(0.0), atol=1e-6), 'Clipped token (1,0) should have 0 grad'\n"
            "assert new_logps.grad[0, 1].abs() > 1e-4 and new_logps.grad[1, 1].abs() > 1e-4, 'Unclipped tokens must receive gradient'\n",
        },
        {
            "name": "按每条回复有效长度归一化（区分 DAPO 与 Dr. GRPO）",
            "code": "\n"
            "import torch\n"
            "# Seq 0 has length 1; Seq 1 has length 4. Standard GRPO averages within each sequence first!\n"
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
            "seq0_mean = torch.exp(torch.tensor(-0.4 - (-0.5))) * 1.0 / 1.0\n"
            "seq1_mean = (torch.exp(new_logps.detach()[1] - old_logps[1]) * (-1.0)).sum() / 4.0\n"
            "expected = -(seq0_mean + seq1_mean) / 2.0\n"
            "assert torch.allclose(loss, expected, atol=1e-6), (\n"
            "    f'Expected sequence-averaged GRPO loss {expected.item():.6f}, got {loss.item():.6f} '\n"
            "    '(average over |o_i| per sequence first, then mean over B)'\n"
            ")\n"
            "assert torch.all(new_logps.grad[mask == 0] == 0), 'Masked padding tokens must receive zero gradient'\n",
        },
        {
            "name": "组内标准差归一化（区分 Dr. GRPO）",
            "code": "\n"
            "import torch\n"
            "old_logps = torch.zeros(4, 3)\n"
            "new_logps = torch.zeros(4, 3, requires_grad=True)\n"
            "# Group 0 has diff 1 ([0, 1]), Group 1 has diff 20 ([10, 30]) -> identical z-scored advantages!\n"
            "rewards = torch.tensor([0.0, 1.0, 10.0, 30.0])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids, eps=0.0)\n"
            "loss.backward()\n"
            "assert torch.allclose(new_logps.grad[:2], new_logps.grad[2:], atol=1e-6), (\n"
            "    'Standard GRPO normalizes by within-group std, so [0, 1] and [10, 30] must produce identical gradients'\n"
            ")\n",
        },
        {
            "name": "无偏 k3 KL 散度正则项",
            "code": "\n"
            "import torch\n"
            "old_logps = torch.tensor([[-0.5, -0.8], [-0.6, -0.4]])\n"
            "new_logps = torch.tensor([[-0.4, -1.0], [-0.5, -0.3]], requires_grad=True)\n"
            "ref_logps = torch.tensor([[-0.7, -0.6], [-0.9, -0.2]])\n"
            "# Equal rewards -> surrogate is 0, so loss comes purely from kl_coef * k3_kl!\n"
            "rewards = torch.tensor([1.0, 1.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "kl_coef = 0.05\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids, ref_logps=ref_logps, kl_coef=kl_coef)\n"
            "loss.backward()\n"
            "\n"
            "delta = ref_logps - new_logps.detach()\n"
            "expected_kl = (torch.exp(delta) - delta - 1.0).mean(dim=-1).mean()\n"
            "expected_loss = kl_coef * expected_kl\n"
            "assert torch.allclose(loss, expected_loss, atol=1e-6), (\n"
            "    f'Expected k3 KL loss {expected_loss.item():.6f}, got {loss.item():.6f} '\n"
            "    '(use exp(ref - new) - (ref - new) - 1)'\n"
            ")\n"
            "assert new_logps.grad is not None and new_logps.grad.abs().sum() > 1e-4, (\n"
            "    'KL penalty must backpropagate gradients through new_logps'\n"
            ")\n",
        },
        {
            "name": "非连续组编号与置换不变性",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(3)\n"
            "old_logps = torch.randn(5, 3)\n"
            "new_logps = (old_logps + 0.05 * torch.randn(5, 3)).requires_grad_(True)\n"
            "rewards = torch.tensor([2.0, 0.0, 1.0, 1.0, -1.0])\n"
            "group_ids = torch.tensor([1, 0, 1, 0, 1])\n"
            "perm = torch.tensor([1, 3, 0, 2, 4])\n"
            "new_perm = new_logps.detach()[perm].requires_grad_(True)\n"
            "l1 = {fn}(new_logps, old_logps, rewards, group_ids)\n"
            "l2 = {fn}(new_perm, old_logps[perm], rewards[perm], group_ids[perm])\n"
            "assert torch.allclose(l1, l2, atol=1e-6), 'Loss should be invariant to batch row permutation'\n",
        },
        {
            "name": "梯度仅通过 new_logps 传播",
            "code": "\n"
            "import torch\n"
            "new_logps = torch.randn(4, 3, requires_grad=True)\n"
            "old_logps = torch.randn(4, 3, requires_grad=True)\n"
            "ref_logps = torch.randn(4, 3, requires_grad=True)\n"
            "rewards = torch.tensor([1.0, 0.0, 0.3, 0.9], requires_grad=True)\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(new_logps, old_logps, rewards, group_ids, ref_logps=ref_logps, kl_coef=0.04)\n"
            "loss.backward()\n"
            "assert new_logps.grad is not None and torch.isfinite(new_logps.grad).all(), 'new_logps must receive finite gradients'\n"
            "assert old_logps.grad is None, 'old_logps must be detached'\n"
            "assert ref_logps.grad is None, 'ref_logps must be detached'\n"
            "assert rewards.grad is None, 'rewards must be detached'\n",
        },
    ],
}
