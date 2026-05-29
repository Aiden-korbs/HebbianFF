from __future__ import annotations

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    try:
        from liger_kernel.transformers import LigerRMSNorm
    except Exception:
        LigerRMSNorm = None
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False
    LigerFusedLinearCrossEntropyLoss = None
    LigerRMSNorm = None
