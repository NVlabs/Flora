"""Let vLLM load the Flora-Rewarder checkpoint, which carries empty ``_extra_state`` entries.

The checkpoint has 256 ``*._extra_state`` tensors on its language-model linear layers, left by
the training framework. Each holds a pickled ``None``: there was no FP8 state to save. vLLM 0.24
skipped them; newer releases fail with "There is no module or parameter named
'...._extra_state'". This module drops them before vLLM's weight loader sees them.

vLLM loads the model in separate processes, so a patch made in the calling process does not
reach it. ``load_llm`` therefore names ``DropEmptyExtraState`` as ``worker_extension_cls``: every
worker imports this module before loading weights, and the import applies the patch.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import torch

# pickle.dumps(None, protocol=4): the only _extra_state value this checkpoint holds.
EMPTY_EXTRA_STATE = b"\x80\x04N."
SUFFIX = "._extra_state"


def drop_empty_extra_state(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the weights without empty ``_extra_state`` entries; fail on any other one."""
    for name, tensor in weights:
        if name.endswith(SUFFIX):
            if tensor.dtype == torch.uint8 and bytes(tensor.flatten().tolist()) == EMPTY_EXTRA_STATE:
                continue
            raise ValueError(f"{name}: _extra_state is not empty; refusing to drop it")
        yield name, tensor


def _patch_model_class() -> None:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration as model_class

    original = model_class.load_weights
    if getattr(original, "_flora_drops_extra_state", False):
        return

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return original(self, drop_empty_extra_state(weights))

    load_weights._flora_drops_extra_state = True
    model_class.load_weights = load_weights


class DropEmptyExtraState:
    """vLLM worker extension; it adds no methods. Importing this module patches the loader."""


_patch_model_class()
