"""优化器与学习率调度组件（可开关，默认不影响旧行为）。

- Muon：对 2D 隐层权重做 Newton-Schulz 正交化动量更新（Keller Jordan et al.）
- fused AdamW：CUDA 融合实现
- 调度：cosine（仓库现状，无 warmup） / warmup+cosine / WSD(warmup-stable-decay)
"""
import math
import torch
from torch.optim import AdamW


def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
    """Newton-Schulz 迭代求 (G Gᵀ)^(-1/2) G 的近似（5 步系数来自 Muon 参考实现）"""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """只接受 2D 参数；1D(范数/偏置)、embedding、lm_head、router 交给 AdamW"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError('Muon 只支持 2D 参数')
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                update = g.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
                u = zeropower_via_newtonschulz5(update, steps=group['ns_steps'])
                # 形状缩放：不同形状矩阵的更新尺度对齐（Muon 参考实现）
                scale = max(1.0, g.size(0) / g.size(1)) ** 0.5
                if group['weight_decay']:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(u.to(p.dtype), alpha=-group['lr'] * scale)
        return loss


def zeropower_via_newtonschulz5_batched(G, steps: int = 5, eps: float = 1e-7):
    """批量版：G 形状 [B, m, n]，对 B 个矩阵同时做 NS 迭代（少 10x kernel 启动）"""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    return X


class MuonBatched(Muon):
    """按形状分组 + 批量 Newton-Schulz 的 Muon（数值上与原版等价，kernel 数少一个量级）"""
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    groups.setdefault(tuple(p.shape), []).append((p, group))
        for shape, items in groups.items():
            bufs = []
            for p, _ in items:
                st = self.state[p]
                if 'momentum_buffer' not in st:
                    st['momentum_buffer'] = torch.zeros_like(p)
                bufs.append(st['momentum_buffer'])
            g = torch.stack([p.grad for p, _ in items])
            buf = torch.stack(bufs)
            grp = items[0][1]
            buf.mul_(grp['momentum']).add_(g)
            upd = g.add(buf, alpha=grp['momentum']) if grp['nesterov'] else buf.clone()
            u = zeropower_via_newtonschulz5_batched(upd, steps=grp['ns_steps'])
            scale = max(1.0, shape[0] / shape[1]) ** 0.5
            for i, (p, g_i) in enumerate(items):
                if g_i['weight_decay']:
                    p.mul_(1 - g_i['lr'] * g_i['weight_decay'])
                p.add_(u[i].to(p.dtype), alpha=-g_i['lr'] * scale)
                bufs[i].copy_(buf[i])
        return loss


def split_params_for_muon(model, min_dim: int = 32):
    """Muon 只用于"足够方的"隐层 2D 矩阵；embedding/head/router/过薄矩阵给 AdamW"""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_hidden_2d = (p.ndim == 2
                        and name.endswith('.weight')
                        and 'embed_tokens' not in name and 'lm_head' not in name
                        and '.gate.' not in name            # router 只有 num_experts 列，交给 AdamW
                        and min(p.shape) >= min_dim)        # 过薄矩阵(如 index_weight 768x1)也交给 AdamW
        (muon if is_hidden_2d else adamw).append(p)
    return muon, adamw


def mark_base_lr(opt):
    """记录每组的基础 lr，便于用乘子统一缩放（Muon/AdamW 两组基础 lr 不同）"""
    for g in opt.param_groups:
        g['base_lr'] = g['lr']


def set_lr_mult(opt, mult):
    for g in opt.param_groups:
        g['lr'] = g['base_lr'] * mult


def build_optimizer(model, kind='adamw', lr=5e-4, muon_lr=0.02, weight_decay=0.0,
                    adamw_lr_1d=None, min_dim=32, ns_steps=5):
    """统一构造入口，返回 (optimizer, 说明字符串)"""
    adamw_lr_1d = lr if adamw_lr_1d is None else adamw_lr_1d
    if kind == 'adamw':
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        mark_base_lr(opt)
        return opt, f'AdamW(lr={lr}, wd={weight_decay})'
    if kind == 'adamw_fused':
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                    weight_decay=weight_decay, fused=True)
        mark_base_lr(opt)
        return opt, f'AdamW-fused(lr={lr}, wd={weight_decay})'
    if kind in ('muon', 'muon_batched'):
        muon_params, adamw_params = split_params_for_muon(model, min_dim=min_dim)
        cls = MuonBatched if kind == 'muon_batched' else Muon
        opt_muon = cls(muon_params, lr=muon_lr, weight_decay=weight_decay, ns_steps=ns_steps)
        opt_adam = AdamW(adamw_params, lr=adamw_lr_1d, weight_decay=weight_decay, fused=True)
        opt = _MultiOpt([opt_muon, opt_adam])
        mark_base_lr(opt)
        return opt, \
            f'Muon(lr={muon_lr}, {sum(p.numel() for p in muon_params)/1e6:.1f}M) + AdamW(lr={adamw_lr_1d}, {sum(p.numel() for p in adamw_params)/1e6:.1f}M)'
    raise ValueError(f'未知优化器: {kind}')


class _MultiOpt:
    """把 Muon + AdamW 组合成一个带 param_groups 接口的优化器（可接调度器）"""
    def __init__(self, opts):
        self.opts = opts
        self.param_groups = [g for o in opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for o in self.opts:
            o.step(closure)

    def state_dict(self):
        return {'opts': [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd['opts']):
            o.load_state_dict(s)


def build_lr_lambda(kind='cosine', warmup_steps=0, total_steps=1000, decay_ratio=0.2, min_ratio=0.1):
    """返回 lr_multiplier(step)"""
    def cosine(step):
        p = min(step / max(total_steps, 1), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    def wsd(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        decay_start = int(total_steps * (1 - decay_ratio))
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / max(total_steps - decay_start, 1)
        return max(min_ratio, 1.0 - p)

    def warmup_cosine(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    # 仓库现状：无 warmup 的 cosine（从峰值 lr 直接起步）
    def repo_cosine(step):
        return 0.1 + 0.45 * (1 + math.cos(math.pi * step / max(total_steps, 1)))

    return {'cosine': repo_cosine, 'warmup_cosine': warmup_cosine, 'wsd': wsd,
            'const': lambda step: 1.0}[kind]
