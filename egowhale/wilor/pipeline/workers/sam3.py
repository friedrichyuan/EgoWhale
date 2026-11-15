"""SAM3 image hand masks streamed to compressed per-frame files."""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path

sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def author_masks(video, path, options):
    """Run the author's unchanged video method, then adapt its mask file."""
    import tempfile
    from types import SimpleNamespace

    import numpy as np

    from egowhale.media import read_frames, video_shape
    from egowhale.wilor.pipeline.author import REVISION, load_method
    from egowhale.wilor.pipeline.contracts import Artifact, write_artifact

    count, height, width, _ = video_shape(Path(video))
    estimated_gb = count * height * width * 3 / 1024**3
    if estimated_gb > options.get('max_video_gb', 4):
        raise ValueError('author video RGB input exceeds max_video_gb; split the video explicitly')
    directory = path.parent / 'hand_masks'
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='author_sam3_', dir=path.parent) as temporary:
        output = Path(temporary) / 'masks.npy'
        # The author method creates its own subprocess to isolate model dtypes.
        method = load_method('_generate_hand_masks')
        previous_tempdir = tempfile.tempdir
        try:
            # Author's delete=False RGB input must also be cleaned on timeout.
            # This worker is isolated; do not alter the parent or global pipeline.
            tempfile.tempdir = temporary
            method(SimpleNamespace(cfg={'sam3_checkpoint': options['checkpoint']}),
                   list(read_frames(Path(video), range(count))), str(output))
        finally:
            tempfile.tempdir = previous_tempdir
        if not output.is_file():
            raise RuntimeError('author SAM3 video subprocess failed; no masks produced')
        masks = np.load(output, mmap_mode='r', allow_pickle=False)
        if masks.shape != (count, height, width) or masks.dtype != np.bool_:
            raise ValueError('author mask timeline or dimensions differ from input')
        files, hashes = [], []
        for frame in range(count):
            file = directory / f'{frame:06d}.npz'
            np.savez_compressed(file, packed=np.packbits(masks[frame]), shape=np.array([height, width]))
            files.append(str(file.resolve()))
            hashes.append(hashlib.sha256(file.read_bytes()).hexdigest())
        del masks
    artifact = Artifact({'frame_index': np.arange(count)}, {'kind': 'masks', 'files': files, 'sha256': hashes,
                        'prompt': 'hand', 'backend': 'author-video', 'author_revision': REVISION})
    write_artifact(path, artifact)
    return {'ok': True, 'artifact': str(path)}


def infer(request):
    if request['options'].get('backend') == 'author-video':
        return author_masks(request['video'], Path(request['out']), request['options'])
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    from egowhale.wilor.pipeline.contracts import Artifact, write_artifact

    o = request['options']
    model = build_sam3_image_model(checkpoint_path=o['checkpoint'], load_from_HF=False, device='cuda', compile=False)
    processor = Sam3Processor(model, confidence_threshold=o.get('confidence', .5))
    path = Path(request['out'])
    directory = path.parent / 'hand_masks'
    directory.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(request['video'])
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    files, hashes = [], []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                state = processor.set_image(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                state = processor.set_text_prompt(state=state, prompt='hand')
            masks = state['masks'].detach().cpu().numpy().reshape((-1, *bgr.shape[:2]))
            union = masks.any(0) if len(masks) else np.zeros(bgr.shape[:2], bool)
            file = directory / f'{len(files):06d}.npz'
            np.savez_compressed(file, packed=np.packbits(union), shape=np.array(union.shape))
            files.append(str(file.resolve()))
            hashes.append(hashlib.sha256(file.read_bytes()).hexdigest())
    finally:
        cap.release()
    if not files or len(files) != expected:
        raise ValueError('incomplete mask video decoding')
    artifact = Artifact({'frame_index': np.arange(len(files))}, {'kind': 'masks', 'files': files, 'sha256': hashes, 'prompt': 'hand'})
    write_artifact(path, artifact)
    return {'ok': True, 'artifact': str(path)}


def main():
    protocol = sys.stdout
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                response = infer(json.loads(line))
        except Exception as error:
            import traceback
            traceback.print_exc(file=sys.stderr)
            response = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
        protocol.write(json.dumps(response) + '\n')
        protocol.flush()


if __name__ == '__main__':
    from egowhale.wilor.step import protect_worker_parent
    protect_worker_parent()
    main()
