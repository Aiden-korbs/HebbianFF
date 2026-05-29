from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import availability is environment-specific.
    triton = None
    tl = None


TERNARY_RUNTIME_MODES = {"dense_debug", "packed_fallback", "triton_gemv", "hybrid", "auto"}
TERNARY_PREFILL_MODES = {"dense_debug", "packed_fallback", "temp_dense", "triton_gemv"}
TERNARY_DECODE_MODES = {"triton_gemv", "packed_fallback"}
TERNARY_AUTO_PREFILL_MODES = {"dense_if_possible", "temp_dense", "triton", "packed"}
TERNARY_PRESETS = {"low_vram", "balanced", "speed", "manual"}
_PROFILE: dict[tuple[str, str], dict[str, float | int]] = {}
_PROFILE_REGISTERED = False
_AUTO_DENSE_USED_MIB = 0.0
_PROFILE_MODULE_CACHE: dict[str, list[str]] = {}


def current_preset() -> str:
    preset = os.environ.get("TERNARY_PRESET", "manual").strip().lower()
    if preset not in TERNARY_PRESETS:
        raise ValueError(f"TERNARY_PRESET={preset!r} is not one of {sorted(TERNARY_PRESETS)}")
    return preset


def current_ternary_runtime() -> str:
    preset = current_preset()
    if preset == "low_vram":
        mode = "hybrid"
    elif preset in {"balanced", "speed"}:
        mode = "auto"
    else:
        mode = os.environ.get("TERNARY_RUNTIME", "packed_fallback").strip().lower()
    if mode not in TERNARY_RUNTIME_MODES:
        raise ValueError(f"TERNARY_RUNTIME={mode!r} is not one of {sorted(TERNARY_RUNTIME_MODES)}")
    return mode


def current_prefill_runtime() -> str:
    preset = current_preset()
    if preset == "low_vram":
        mode = "temp_dense"
    else:
        mode = os.environ.get("TERNARY_PREFILL_RUNTIME", "temp_dense").strip().lower()
    if mode not in TERNARY_PREFILL_MODES:
        raise ValueError(f"TERNARY_PREFILL_RUNTIME={mode!r} is not one of {sorted(TERNARY_PREFILL_MODES)}")
    return mode


def current_decode_runtime() -> str:
    preset = current_preset()
    if preset in {"low_vram", "balanced", "speed"}:
        mode = "triton_gemv"
    else:
        mode = os.environ.get("TERNARY_DECODE_RUNTIME", "triton_gemv").strip().lower()
    if mode not in TERNARY_DECODE_MODES:
        raise ValueError(f"TERNARY_DECODE_RUNTIME={mode!r} is not one of {sorted(TERNARY_DECODE_MODES)}")
    return mode


def dense_merge_lora_enabled() -> bool:
    if current_preset() in {"balanced", "speed"}:
        return True
    return os.environ.get("TERNARY_DENSE_MERGE_LORA", "0").strip().lower() in {"1", "true", "yes", "on"}


def profile_enabled() -> bool:
    return os.environ.get("TERNARY_PROFILE_MODULES", "0").strip().lower() in {"1", "true", "yes", "on"}


def selective_dense_enabled() -> bool:
    if current_preset() in {"balanced", "speed"}:
        return True
    if current_preset() == "low_vram":
        return False
    return os.environ.get("TERNARY_SELECTIVE_DENSE_CACHE", "0").strip().lower() in {"1", "true", "yes", "on"}


def selective_dense_modules() -> set[str]:
    explicit = os.environ.get("TERNARY_SELECTIVE_DENSE_MODULES", "")
    modules = {x.strip() for x in explicit.split(",") if x.strip()}
    if modules:
        return modules
    topk = selective_dense_topk()
    if topk <= 0:
        return set()
    profile_path = os.environ.get("TERNARY_SELECTIVE_DENSE_PROFILE", "")
    if not profile_path:
        return set()
    return set(load_profile_modules(profile_path)[:topk])


def selective_dense_topk() -> int:
    preset = current_preset()
    if preset == "balanced":
        return 8
    if preset == "speed":
        return 16
    return int(os.environ.get("TERNARY_SELECTIVE_DENSE_TOPK", "0"))


def auto_prefill_mode() -> str:
    preset = current_preset()
    if preset in {"balanced", "speed"}:
        mode = "temp_dense"
    else:
        mode = os.environ.get("TERNARY_AUTO_PREFILL", "dense_if_possible").strip().lower()
    if mode not in TERNARY_AUTO_PREFILL_MODES:
        raise ValueError(f"TERNARY_AUTO_PREFILL={mode!r} is not one of {sorted(TERNARY_AUTO_PREFILL_MODES)}")
    return mode


def auto_max_dense_prefill_tokens() -> int:
    return int(os.environ.get("TERNARY_AUTO_MAX_DENSE_PREFILL_TOKENS", "256"))


def auto_max_extra_vram_mib() -> float:
    return float(os.environ.get("TERNARY_AUTO_MAX_EXTRA_VRAM_MIB", "512"))


def reset_ternary_profile() -> None:
    _PROFILE.clear()


def reset_auto_dense_budget() -> None:
    global _AUTO_DENSE_USED_MIB
    _AUTO_DENSE_USED_MIB = 0.0


def load_profile_modules(profile_path: str) -> list[str]:
    cached = _PROFILE_MODULE_CACHE.get(profile_path)
    if cached is not None:
        return cached
    path = Path(profile_path)
    if not path.exists():
        _PROFILE_MODULE_CACHE[profile_path] = []
        return []
    data = json.loads(path.read_text())
    rows = data.get("ranked_modules") or data.get("profile_top20") or []
    modules = []
    for row in rows:
        if isinstance(row, str):
            modules.append(row)
        elif isinstance(row, dict) and isinstance(row.get("module"), str):
            modules.append(row["module"])
    _PROFILE_MODULE_CACHE[profile_path] = modules
    return modules


def save_profile_json(profile_path: str, metadata: dict | None = None, topk: int = 64) -> None:
    rows = ternary_profile_summary(topk)
    if not profile_path or not rows:
        return
    data = {
        "ranked_modules": rows,
        "recommended_top8": [str(row["module"]) for row in rows[:8]],
        "recommended_top16": [str(row["module"]) for row in rows[:16]],
        "metadata": metadata or {},
    }
    path = Path(profile_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    _PROFILE_MODULE_CACHE[str(path)] = [str(row["module"]) for row in rows]


def ternary_profile_summary(topk: int = 20) -> list[dict[str, float | int | str]]:
    rows = []
    for (name, path), stats in _PROFILE.items():
        calls = int(stats["calls"])
        total_ms = float(stats["total_ms"])
        rows.append({
            "module": name,
            "path": path,
            "calls": calls,
            "total_ms": total_ms,
            "avg_ms": total_ms / max(calls, 1),
        })
    rows.sort(key=lambda x: float(x["total_ms"]), reverse=True)
    return rows[:topk]


def print_ternary_profile(topk: int = 20) -> None:
    rows = ternary_profile_summary(topk)
    if not rows:
        return
    print("[TERNARY PROFILE] top adapted modules", flush=True)
    for row in rows:
        print(
            f"[TERNARY PROFILE] {row['module']} path={row['path']} "
            f"calls={row['calls']} total_ms={row['total_ms']:.3f} avg_ms={row['avg_ms']:.4f}",
            flush=True,
        )
    profile_path = os.environ.get("TERNARY_SELECTIVE_DENSE_PROFILE", "")
    if profile_path:
        save_profile_json(profile_path, {"source": "atexit_profile"}, topk=max(topk, 64))


def resolved_runtime_config() -> dict[str, object]:
    modules = selective_dense_modules() if selective_dense_enabled() else set()
    return {
        "preset": current_preset(),
        "runtime": current_ternary_runtime(),
        "prefill_runtime": current_prefill_runtime(),
        "decode_runtime": current_decode_runtime(),
        "auto_prefill": auto_prefill_mode(),
        "dense_merge_lora": dense_merge_lora_enabled(),
        "selective_dense_cache": selective_dense_enabled(),
        "selective_dense_topk": selective_dense_topk(),
        "selective_dense_modules": sorted(modules),
        "selective_dense_profile": os.environ.get("TERNARY_SELECTIVE_DENSE_PROFILE", ""),
        "auto_max_dense_prefill_tokens": auto_max_dense_prefill_tokens(),
        "auto_max_extra_vram_mib": auto_max_extra_vram_mib(),
    }


def _register_profile_atexit() -> None:
    global _PROFILE_REGISTERED
    if not _PROFILE_REGISTERED:
        atexit.register(print_ternary_profile)
        _PROFILE_REGISTERED = True


def unpack_2bit_codes(packed: torch.Tensor, numel: int) -> torch.Tensor:
    codes = torch.empty(packed.numel() * 4, device=packed.device, dtype=torch.uint8)
    codes[0::4] = packed & 0x03
    codes[1::4] = (packed >> 2) & 0x03
    codes[2::4] = (packed >> 4) & 0x03
    codes[3::4] = (packed >> 6) & 0x03
    return codes[:numel]


def codes_to_bitplanes_cpu(packed: torch.Tensor, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = shape
    codes = unpack_2bit_codes(packed.cpu().contiguous(), out_features * in_features).view(out_features, in_features)
    bits_per_word = 31
    words = (in_features + bits_per_word - 1) // bits_per_word
    pos = torch.zeros((out_features, words), dtype=torch.int32)
    neg = torch.zeros((out_features, words), dtype=torch.int32)
    cols = torch.arange(in_features, dtype=torch.int64)
    shifts = (cols % bits_per_word).to(torch.int32).view(1, -1)
    word_ids = (cols // bits_per_word).view(1, -1).expand(out_features, -1)
    row_ids = torch.arange(out_features, dtype=torch.int64).view(-1, 1).expand(-1, in_features)
    pos_bits = ((codes == 2).to(torch.int32) << shifts).contiguous()
    neg_bits = ((codes == 0).to(torch.int32) << shifts).contiguous()
    pos.index_put_((row_ids.reshape(-1), word_ids.reshape(-1)), pos_bits.reshape(-1), accumulate=True)
    neg.index_put_((row_ids.reshape(-1), word_ids.reshape(-1)), neg_bits.reshape(-1), accumulate=True)
    return pos, neg


def dense_from_2bit(packed: torch.Tensor, scales: torch.Tensor, shape: tuple[int, int], group_size: int, dtype: torch.dtype) -> torch.Tensor:
    codes = unpack_2bit_codes(packed, shape[0] * shape[1])
    ternary = codes.to(torch.int8).sub_(1).view(shape).to(dtype=dtype)
    expanded_scales = scales.to(dtype=dtype).repeat_interleave(group_size, dim=1)[:, : shape[1]]
    return ternary * expanded_scales


if triton is not None:
    @triton.jit
    def _ternary_gemv_kernel(
        x_ptr,
        pos_ptr,
        neg_ptr,
        scales_ptr,
        bias_ptr,
        y_ptr,
        n_size: tl.constexpr,
        k_size: tl.constexpr,
        words_per_row: tl.constexpr,
        bits_per_word: tl.constexpr,
        groups_per_row: tl.constexpr,
        group_size: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N,), tl.float32)

        for k0 in range(0, k_size, BLOCK_K):
            k = k0 + offs_k
            x = tl.load(x_ptr + k, mask=k < k_size, other=0.0).to(tl.float32)
            word_ids = k // bits_per_word
            bit_ids = k % bits_per_word
            mask = (offs_n[:, None] < n_size) & (k[None, :] < k_size)
            pos_words = tl.load(pos_ptr + offs_n[:, None] * words_per_row + word_ids[None, :], mask=mask, other=0)
            neg_words = tl.load(neg_ptr + offs_n[:, None] * words_per_row + word_ids[None, :], mask=mask, other=0)
            bit = 1 << bit_ids
            pos = ((pos_words & bit[None, :]) != 0).to(tl.float32)
            neg = ((neg_words & bit[None, :]) != 0).to(tl.float32)
            ternary = pos - neg
            group_ids = k // group_size
            scale = tl.load(scales_ptr + offs_n[:, None] * groups_per_row + group_ids[None, :], mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(ternary * scale * x[None, :], axis=1)

        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0).to(tl.float32)
            acc += bias
        tl.store(y_ptr + offs_n, acc, mask=offs_n < n_size)


    @triton.jit
    def _ternary_gemm_kernel(
        x_ptr,
        pos_ptr,
        neg_ptr,
        scales_ptr,
        bias_ptr,
        y_ptr,
        m_size: tl.constexpr,
        n_size: tl.constexpr,
        k_size: tl.constexpr,
        words_per_row: tl.constexpr,
        bits_per_word: tl.constexpr,
        groups_per_row: tl.constexpr,
        group_size: tl.constexpr,
        has_bias: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_y_m: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k0 in range(0, k_size, BLOCK_K):
            k = k0 + offs_k
            x = tl.load(x_ptr + offs_m[:, None] * stride_xm + k[None, :], mask=(offs_m[:, None] < m_size) & (k[None, :] < k_size), other=0.0).to(tl.bfloat16)
            word_ids = k // bits_per_word
            bit_ids = k % bits_per_word
            word_mask = (offs_n[:, None] < n_size) & (k[None, :] < k_size)
            pos_words = tl.load(pos_ptr + offs_n[:, None] * words_per_row + word_ids[None, :], mask=word_mask, other=0)
            neg_words = tl.load(neg_ptr + offs_n[:, None] * words_per_row + word_ids[None, :], mask=word_mask, other=0)
            bit = 1 << bit_ids
            pos = ((pos_words & bit[None, :]) != 0).to(tl.float32)
            neg = ((neg_words & bit[None, :]) != 0).to(tl.float32)
            ternary = (pos - neg).to(tl.bfloat16)
            group_ids = k // group_size
            scale = tl.load(scales_ptr + offs_n[:, None] * groups_per_row + group_ids[None, :], mask=word_mask, other=0.0).to(tl.bfloat16)
            w = tl.trans(ternary * scale)
            acc += tl.dot(x, w, input_precision="tf32")

        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0).to(tl.float32)
            acc += bias[None, :]
        tl.store(y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :], acc, mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size))


def _triton_ternary_linear(
    x_2d: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_features: int,
    in_features: int,
    group_size: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("TERNARY_RUNTIME=triton_gemv requested but triton is not available")
    if x_2d.device.type != "cuda":
        raise RuntimeError("triton ternary runtime requires CUDA inputs")
    m_size = int(x_2d.shape[0])
    y = torch.empty((m_size, out_features), device=x_2d.device, dtype=x_2d.dtype)
    words_per_row = int(pos_mask.shape[1])
    bits_per_word = 31
    groups_per_row = int(scales.shape[1])
    if m_size == 1:
        block_n = 32
        _ternary_gemv_kernel[(triton.cdiv(out_features, block_n),)](
            x_2d,
            pos_mask,
            neg_mask,
            scales,
            bias if bias is not None else x_2d,
            y,
            out_features,
            in_features,
            words_per_row,
            bits_per_word,
            groups_per_row,
            group_size,
            bias is not None,
            BLOCK_N=block_n,
            BLOCK_K=128,
            num_warps=4,
        )
        return y
    block_m = 16
    block_n = 32
    grid = (triton.cdiv(m_size, block_m), triton.cdiv(out_features, block_n))
    _ternary_gemm_kernel[grid](
        x_2d,
        pos_mask,
        neg_mask,
        scales,
        bias if bias is not None else x_2d,
        y,
        m_size,
        out_features,
        in_features,
        words_per_row,
        bits_per_word,
        groups_per_row,
        group_size,
        bias is not None,
        x_2d.stride(0),
        y.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=4,
    )
    return y


class PackedTernaryLoRALinear(nn.Module):
    def __init__(self, state: dict, device: str, dtype: torch.dtype, runtime: Optional[str] = None, module_name: str = ""):
        super().__init__()
        runtime = current_ternary_runtime() if runtime is None else runtime
        if runtime not in TERNARY_RUNTIME_MODES:
            raise ValueError(f"unknown ternary runtime {runtime!r}")
        base = state["base"]
        self.module_name = module_name
        self.runtime = runtime
        self.prefill_runtime = current_prefill_runtime()
        self.decode_runtime = current_decode_runtime()
        self.profile = profile_enabled()
        if self.profile:
            _register_profile_atexit()
        self.selective_dense = selective_dense_enabled() and module_name in selective_dense_modules()
        self.dense_weight_has_lora = False
        self.in_features = int(state["in_features"])
        self.out_features = int(state["out_features"])
        self.rank = int(state["rank"])
        self.shape = tuple(int(x) for x in base["shape"])
        self.group_size = int(base["group_size"])
        self.code_numel = self.shape[0] * self.shape[1]
        self._expanded_scale_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

        packed_cpu = base["packed_codes"].contiguous()
        scales = base["scales"].to(device=device, dtype=dtype)
        self.register_buffer("packed_codes", packed_cpu.to(device=device, dtype=torch.uint8), persistent=True)
        self.register_buffer("scales", scales, persistent=True)
        self.register_buffer("a", state["a"].to(device=device, dtype=dtype), persistent=True)
        self.register_buffer("b", state["b"].to(device=device, dtype=dtype), persistent=True)
        self.register_buffer("alpha", state["alpha"].to(device=device, dtype=dtype), persistent=True)
        if state["bias"] is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", state["bias"].to(device=device, dtype=dtype), persistent=True)

        dense_for_runtime = (
            runtime == "dense_debug"
            or self.selective_dense
            or (runtime == "hybrid" and self.prefill_runtime == "dense_debug")
            or (runtime == "auto" and self.auto_wants_dense_prefill())
        )
        if dense_for_runtime:
            dense = dense_from_2bit(self.packed_codes, self.scales, self.shape, self.group_size, dtype)
            merge_dense = self.selective_dense or dense_merge_lora_enabled()
            if merge_dense:
                dense = dense + self.alpha * (self.a @ self.b)
                self.dense_weight_has_lora = True
            self.register_buffer("dense_weight", dense.contiguous(), persistent=True)
        else:
            self.register_buffer("dense_weight", None, persistent=True)

        needs_triton = (
            runtime == "triton_gemv"
            or runtime == "auto"
            or (runtime == "hybrid" and self.decode_runtime == "triton_gemv")
        )
        if needs_triton:
            pos_cpu, neg_cpu = codes_to_bitplanes_cpu(packed_cpu, self.shape)
            self.register_buffer("pos_mask", pos_cpu.to(device=device), persistent=True)
            self.register_buffer("neg_mask", neg_cpu.to(device=device), persistent=True)
        else:
            self.register_buffer("pos_mask", None, persistent=True)
            self.register_buffer("neg_mask", None, persistent=True)

    def dense_extra_mib(self) -> float:
        return float(self.out_features * self.in_features * self.a.element_size()) / 1024**2

    def auto_wants_dense_prefill(self) -> bool:
        global _AUTO_DENSE_USED_MIB
        if auto_prefill_mode() != "dense_if_possible":
            return False
        extra = self.dense_extra_mib()
        if _AUTO_DENSE_USED_MIB + extra > auto_max_extra_vram_mib():
            return False
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            if free / 1024**2 < extra + 64.0:
                return False
        _AUTO_DENSE_USED_MIB += extra
        return True

    def expanded_scales(self, dtype: torch.dtype) -> torch.Tensor:
        key = (self.scales.device, dtype)
        cached = self._expanded_scale_cache.get(key)
        if cached is None:
            cached = self.scales.to(dtype=dtype).repeat_interleave(self.group_size, dim=1)[:, : self.shape[1]].contiguous()
            self._expanded_scale_cache[key] = cached
        return cached

    def unpack_weight(self, dtype: torch.dtype, cache_scales: bool = True) -> torch.Tensor:
        codes = unpack_2bit_codes(self.packed_codes, self.code_numel)
        ternary = codes.to(torch.int8).sub_(1).view(self.shape).to(dtype=dtype)
        if cache_scales:
            scales = self.expanded_scales(dtype)
        else:
            scales = self.scales.to(dtype=dtype).repeat_interleave(self.group_size, dim=1)[:, : self.shape[1]]
        return ternary * scales

    def ternary_base(self, x: torch.Tensor) -> torch.Tensor:
        path = self.select_path(x)
        if path == "dense_cached":
            return F.linear(x, self.dense_weight, self.bias)
        if path == "dense_debug":
            return F.linear(x, self.dense_weight, self.bias)
        if path == "packed_fallback":
            return F.linear(x, self.unpack_weight(x.dtype), self.bias)
        if path == "temp_dense":
            return F.linear(x, self.unpack_weight(x.dtype, cache_scales=False), self.bias)
        if path == "triton_gemv":
            return self.triton_base(x)
        raise AssertionError(path)

    def select_path(self, x: torch.Tensor) -> str:
        if self.selective_dense:
            return "dense_cached"
        if self.runtime == "dense_debug":
            return "dense_debug"
        if self.runtime == "packed_fallback":
            return "packed_fallback"
        if self.runtime == "triton_gemv":
            return "triton_gemv"
        if self.runtime == "hybrid":
            active = self.decode_runtime if self.is_decode_input(x) else self.prefill_runtime
            if active in {"triton_gemv", "packed_fallback", "temp_dense", "dense_debug"}:
                return active
            raise AssertionError(active)
        if self.runtime == "auto":
            if self.is_decode_input(x):
                return "triton_gemv"
            mode = auto_prefill_mode()
            seq_len = self.sequence_len(x)
            if mode == "dense_if_possible" and seq_len <= auto_max_dense_prefill_tokens() and self.dense_weight is not None:
                return "dense_debug"
            if mode == "triton":
                return "triton_gemv"
            if mode == "packed":
                return "packed_fallback"
            return "temp_dense"
        raise AssertionError(self.runtime)

    @staticmethod
    def is_decode_input(x: torch.Tensor) -> bool:
        return PackedTernaryLoRALinear.sequence_len(x) == 1

    @staticmethod
    def sequence_len(x: torch.Tensor) -> int:
        if x.dim() >= 3:
            return int(x.shape[-2])
        return int(x.shape[0])

    def triton_base(self, x: torch.Tensor) -> torch.Tensor:
            original_shape = x.shape[:-1]
            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            y = _triton_ternary_linear(
                x_2d,
                self.pos_mask,
                self.neg_mask,
                self.scales,
                self.bias,
                self.out_features,
                self.in_features,
                self.group_size,
            )
            return y.reshape(*original_shape, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.profile:
            base = self.ternary_base(x)
            path = self.select_path(x)
            if self.dense_weight_has_lora and path in {"dense_cached", "dense_debug"}:
                return base
            residual = F.linear(F.linear(x, self.b), self.a)
            return base + residual * self.alpha

        path = self.select_path(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if path in {"dense_cached", "dense_debug"}:
            out = F.linear(x, self.dense_weight, self.bias)
        elif path == "packed_fallback":
            out = F.linear(x, self.unpack_weight(x.dtype), self.bias)
        elif path == "temp_dense":
            out = F.linear(x, self.unpack_weight(x.dtype, cache_scales=False), self.bias)
        elif path == "triton_gemv":
            out = self.triton_base(x)
        else:
            raise AssertionError(path)
        if not (self.dense_weight_has_lora and path in {"dense_cached", "dense_debug"}):
            out = out + F.linear(F.linear(x, self.b), self.a) * self.alpha
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        key = (self.module_name or "<unnamed>", path)
        stats = _PROFILE.setdefault(key, {"calls": 0, "total_ms": 0.0})
        stats["calls"] = int(stats["calls"]) + 1
        stats["total_ms"] = float(stats["total_ms"]) + elapsed_ms
        return out
