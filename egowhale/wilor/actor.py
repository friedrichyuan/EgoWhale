"""A standalone Ray-compatible WiLoR actor; importing it needs no other GPU models."""
from __future__ import annotations

import os

from egowhale.step import execute
from egowhale.wilor.step import WiLoR


class WiLoRActor:
    def __init__(self, options=None):
        self.step = WiLoR(**(options or {}))

    def ready(self):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        return 'cuda:' + visible.split(',')[0] if visible else 'cpu'

    def run(self, stage, src, dst, index=None):
        return execute(self.step, src, dst)

    def close(self):
        self.step.release()
