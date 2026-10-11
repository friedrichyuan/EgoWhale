"""Level 3: a VLM checks that the composite still shows the described task. Called only with a key.

EGOWHALE_VLM_API_KEY is the bearer token, EGOWHALE_VLM_BASE_URL an OpenAI-compatible endpoint, and
EGOWHALE_VLM_MODEL the model. A missing key, an unreachable endpoint, or a malformed reply is reported
in the result and never fails the episode by itself.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from pathlib import Path

import cv2
import h5py
import numpy as np

from egowhale.media import read_frames, video_shape

PROMPT = """You are evaluating whether a manipulation video matches its text description.
This is a ROBOT manipulation dataset. "Hand" refers to the robot's gripper/end-effector.
Many tasks use FAKE or SIMULATED objects as stand-ins for real objects. This is EXPECTED.
Task Description: {description}
Determine if the robot's actions match the described task.
Flag MAJOR MISMATCHES: wrong action type, wrong object category, wrong target location, or failed execution.
Be tolerant of: fake/toy objects, minor appearance variations, small spatial deviations, different grasping approaches.
Respond in JSON: {"is consistent": true/false, "confidence": 0.0-1.0, "reasoning": "..."}
"""


def audit(video: Path, start: int, frames: int, fps: float, description: str) -> dict:
    """Ask the VLM about `frames` frames from `start`, sampled at 4 fps. Only those frames are decoded."""
    key = os.environ.get("EGOWHALE_VLM_API_KEY", "").strip()
    if not key:
        return {"status": "skipped", "reason": "EGOWHALE_VLM_API_KEY is unset"}
    count = min(frames, video_shape(video)[0] - start)
    step = max(fps / 4.0, 1.0)
    indices = np.unique(np.round(np.arange(0, count, step)).astype(int))
    indices = indices[indices < count]
    content = []
    for frame in read_frames(video, start + indices) if len(indices) else []:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}})
    content.append({"type": "text", "text": PROMPT.replace("{description}", description)})
    body = json.dumps({
        "model": os.environ.get("EGOWHALE_VLM_MODEL", "qwen3.8-flash"),
        "messages": [{"role": "user", "content": content}],
    }).encode("utf-8")
    base = os.environ.get("EGOWHALE_VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    # Like a missing key, an unreachable or malformed audit is reported, not counted against the episode.
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            reply = json.loads(response.read().decode("utf-8"))
        text = reply["choices"][0]["message"]["content"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    if isinstance(text, list):
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    return _verdict(str(text))


def _verdict(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return {"status": "error", "raw": text}
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return {"status": "error", "raw": text}
    consistent = payload.get("is consistent", payload.get("is_consistent"))
    return {
        "status": "ok",
        "is_consistent": None if consistent is None else bool(consistent),
        "confidence": payload.get("confidence"),
        "reasoning": payload.get("reasoning"),
    }


def description(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        for key in ("llm_description", "task"):
            if key not in handle.attrs:
                continue
            value = handle.attrs[key]
            if isinstance(value, bytes):
                value = value.decode()
            text = str(value).strip()
            if text and text.lower() != "none":
                return text
    return "manipulation"
