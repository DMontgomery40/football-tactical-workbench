from __future__ import annotations

import sys
from pathlib import Path

import huggingface_hub

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import soccermaster_adapter  # noqa: E402


def test_huggingface_hub_transformers_compat_shim_adds_is_offline_mode(monkeypatch) -> None:
    monkeypatch.delitem(huggingface_hub.__dict__, "is_offline_mode", raising=False)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    soccermaster_adapter._ensure_huggingface_hub_transformers_compat()

    assert "is_offline_mode" in huggingface_hub.__dict__
    assert huggingface_hub.is_offline_mode() is True

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert huggingface_hub.is_offline_mode() is False
