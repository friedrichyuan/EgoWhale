"""Persistent single-GPU worker for the synchronized MP4 WiLoR node.

Requests are JSON lines on stdin. Model logs go to stdout; response lines have a
unique prefix so the parent can distinguish them from model output.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from egowhale.wilor.synced import CONFIG, RUNNER, SYNC_ROOT, identity, run
from egowhale.wilor.step import WiLoR

RESPONSE = '@@EGOWHALE_WILOR_RESPONSE@@'


def main():
    sys.path.insert(0, str(SYNC_ROOT))
    spec = importlib.util.spec_from_file_location('resident_hand_experiment', RUNNER)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    models = experiment.ResidentModels()
    print('常驻 worker 已启动，模型按需加载；后续视频复用同一模型', flush=True)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get('command') == 'stop':
                    break
                stage = request.get('stage')
                if stage not in ('camera', 'hand'):
                    raise ValueError('分卡 worker 只接受 camera 或 hand 阶段')
                src = Path(request['video']).resolve()
                dst = Path(request['output']).resolve()
                config = Path(request.get('config') or CONFIG).resolve()
                os.environ['EGOHAND_VIDEO_LABEL'] = request.get('label', '')
                os.environ['EGOHAND_PROGRESS_FILE'] = request.get('progress_file', '')
                if request.get('stage') == 'camera':
                    staging = dst / '.camera_stage'
                    if staging.exists():
                        raise ValueError('相机阶段目录已存在，请使用新的任务输出目录')
                    before = identity(src)
                    experiment.main(['--video', str(src), '--output', str(staging),
                                     '--config', str(config), '--stage', 'camera', '--prechecked'],
                                    resident=models)
                    if identity(src) != before:
                        raise ValueError('相机阶段运行时输入视频发生变化')
                    (staging / 'camera_input_identity.json').write_text(json.dumps(before))
                    print(RESPONSE + json.dumps({'ok': True}, ensure_ascii=False), flush=True)
                    continue
                node = WiLoR(camera_mode='da3', python=Path(sys.executable), pipeline_config=config)
                node.validate_outputs(src, dst)
                def execute(args):
                    if request.get('stage') == 'hand':
                        staging = dst / '.camera_stage'
                        if json.loads((staging / 'camera_input_identity.json').read_text()) != identity(src):
                            raise ValueError('DA3 处理后输入视频已变化')
                        work = dst / 'action/wilor_ego2robot'
                        required = ('video_L.mp4', 'input.json', 'cam_intrinsics.json',
                                    'cam_extrinsics.npz', 'da3_completed.json')
                        for name in required:
                            source, target = staging / name, work / name
                            if not source.is_file() or not source.stat().st_size or target.exists():
                                raise ValueError(f'DA3 阶段文件缺失或目标已存在：{name}')
                        for item in staging.iterdir():
                            if item.name != 'camera_input_identity.json':
                                item.replace(work / item.name)
                        (staging / 'camera_input_identity.json').unlink()
                        staging.rmdir()
                        print('已接收 DA3 结果，开始 WiLoR', flush=True)
                    return experiment.main([*args, '--prechecked'], resident=models)

                run(src, dst, config, Path(sys.executable), 'da3', None, .3, runner=execute)
                node.validate_outputs(src, dst)
                response = {'ok': True}
            except Exception as exc:
                traceback.print_exc()
                response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
            print(RESPONSE + json.dumps(response, ensure_ascii=False), flush=True)
            # A failed CUDA call can leave model state unsafe. Parent restarts for next job.
            if not response['ok']:
                return 1
    finally:
        models.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
