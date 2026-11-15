"""Worker-local compatibility adapters for the pinned Dyn-HaMR optimizer."""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def install_depth_bounds(z_min: float, z_max: float) -> None:
    from optim import losses
    original = getattr(losses.depth_constraint_loss, '_unbounded', losses.depth_constraint_loss)

    def bounded(joints3d, cam_R, cam_t, min_depth=0., max_depth=999.):
        return original(joints3d, cam_R, cam_t, min_depth=z_min, max_depth=z_max)

    bounded._unbounded = original
    losses.depth_constraint_loss = bounded


def configure_depth_loss(cfg, bounds):
    cfg.optim.loss_weights.depth_constraint = [100., 100., 100.] if bounds else [0., 0., 0.]
    if bounds:
        install_depth_bounds(*bounds)


def physical_run_mano(body_model, trans, root_orient, body_pose, is_right, betas=None, only_right=False):
    """Reflect local hand geometry, then rotate/translate in a physical world frame."""
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix

    b, t, _ = trans.shape
    betas = betas[:, None].expand(b, t, -1).reshape(b*t, -1)
    local = body_model(hand_pose=body_pose.reshape(b*t, -1), betas=betas,
                       global_orient=torch.zeros(b*t, 3, device=trans.device), transl=torch.zeros(b*t, 3, device=trans.device))
    mirror = torch.ones(b, t, 1, 3, device=trans.device)
    if not only_right:
        mirror[..., 0] = (2 * is_right - 1)[..., None]
    joints = local.joints.reshape(b, t, -1, 3) * mirror
    verts = local.vertices.reshape(b, t, -1, 3) * mirror
    root = joints[:, :, :1]
    rotation = axis_angle_to_matrix(root_orient)
    joints = torch.einsum('btij,btkj->btki', rotation, joints - root) + root + trans[:, :, None]
    verts = torch.einsum('btij,btkj->btki', rotation, verts - root) + root + trans[:, :, None]
    return {'joints': joints, 'vertices': verts, 'l_faces': body_model.faces_tensor[:, [0, 2, 1]],
            'r_faces': body_model.faces_tensor, 'is_right': is_right, 'body_pose': body_pose.clone()}


def install_camera_reader():
    import numpy as np
    import torch
    from data.dataset import CameraData

    def exact_camera(self):
        with np.load(Path(self.cam_dir) / 'cameras.npz') as data:
            interval = slice(self.sidx, self.eidx)
            self.cam_R = torch.tensor(data['w2c'][interval, :3, :3], dtype=torch.float32)
            self.cam_t = torch.tensor(data['w2c'][interval, :3, 3], dtype=torch.float32)
            self.intrins = torch.tensor(data['intrins'][interval], dtype=torch.float32)
    CameraData.load_data = exact_camera


def optimize(request):
    import os
    import runpy

    import cv2
    import numpy as np
    import torch

    from egowhale.wilor.compat import legacy_mano

    legacy_mano()
    from omegaconf import OmegaConf
    from pytorch3d.transforms import axis_angle_to_matrix

    from egowhale.wilor.camera import project
    from egowhale.wilor.pipeline.contracts import derive, read_artifact, write_artifact
    from egowhale.wilor.pipeline.hmp import run_prior, select_result

    o = request['options']
    repo = Path(o['repo']).resolve()
    code = repo / 'dyn-hamr'
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code / 'HMP'))
    sys.path.insert(0, str(code / 'src/human_body_prior'))
    upstream = runpy.run_path(str(code / 'run_opt.py'), run_name='wilor_optimizer')
    # All overrides are local to this short-lived process. Third-party source remains untouched.
    import body_model
    import optim.base_scene
    import optim.output
    body_model.run_mano = physical_run_mano
    optim.base_scene.run_mano = physical_run_mano
    optim.output.run_mano = physical_run_mano
    install_camera_reader()
    work = Path(request['workspace'])
    info = json.loads((work / 'input.json').read_text())
    images = work / 'images'
    images.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(request['video'])
    frame = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if not cv2.imwrite(str(images / f'{frame:06d}.jpg'), image):
                raise RuntimeError('cannot export optimizer image')
            frame += 1
    finally:
        cap.release()
    if frame != info['frames']:
        raise ValueError('optimizer video frame count mismatch')
    cfg = OmegaConf.load(code / 'confs/config.yaml')
    del cfg['defaults']
    del cfg['hydra']
    cfg.optim = OmegaConf.load(code / 'confs/optim.yaml').optim
    cfg.data = OmegaConf.load(code / 'confs/data/video_driod.yaml')
    cfg.data.update(root=str(work), seq='wilor', name='wilor', start_idx=0, end_idx=frame,
                    track_ids='-'.join(str(side) for side in info['sides']), sources={
                        'images': str(images), 'tracks': str(work / 'track_preds'),
                        'shots': str(work / 'shots.json'), 'cameras': str(work / 'cameras/shot-0')})
    cfg.paths.update(base_dir=str(repo), DATA_DIR=str(repo / '_DATA/data'), vposer=str(repo / '_DATA/unused_vposer'))
    cfg.MANO.MODEL_PATH = o['mano']
    cfg.MANO.MEAN_PARAMS = str(Path(o['mano']).parent / 'mano_mean_params.npz')
    cfg.model.opt_scale = info['opt_scale']
    cfg.optim.smooth.opt_scale = info['opt_scale']
    cfg.optim.root.num_iters = o.get('root_iters', 50)
    cfg.optim.smooth.num_iters = o.get('smooth_iters', 300)
    cfg.optim.options.lbfgs_max_iter = o.get('lbfgs_max_iter', 20)
    cfg.optim.options.vis_every = -1
    cfg.optim.options.save_every = 20
    cfg.optim.loss_weights.bio = [10., 10., 10.]
    configure_depth_loss(cfg, info['depth_bounds'])
    cfg.exp_name = 'wilor'
    cfg.data.frame_opts.fps = info['fps']
    cfg.fps = info['fps']
    cfg.run_opt = True
    cfg.run_prior = o.get('run_prior', True)
    cfg.is_static = info.get('is_static', False)
    cfg.run_vis = False
    torch.set_printoptions(threshold=1, edgeitems=1)
    upstream['set_seed'](42)
    output = work / 'result'
    output.mkdir(exist_ok=True)
    os.chdir(output)
    from data.dataset import MultiPeopleDataset
    from util.logger import Logger
    Logger.init(str(output / 'optimizer.log'))
    dataset = MultiPeopleDataset(cfg.data.sources, 'wilor', tid_spec=cfg.data.track_ids, end_idx=frame,
                                is_static=cfg.is_static)
    dataset.load_data()
    # Upstream fills missing confidence with one; retain zero data weight for unobserved samples.
    for i, visibility in enumerate(dataset.track_vis_masks):
        missing = ~visibility[dataset.start_idx:dataset.end_idx]
        dataset.data_dict['joints2d'][i][missing, :, 2] = 0
    OmegaConf.save(cfg, work / 'expanded-config.yaml', resolve=True)
    # Drive the same root -> smooth -> prior stages explicitly: upstream's HMP
    # assumes exactly 128 frames, so the prior adapter preserves our episode length.
    root_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    root_cfg.run_prior = False
    device = torch.device('cuda:0')
    upstream['run_opt'](root_cfg, dataset, str(output), device)
    if cfg.run_prior:
        run_prior(cfg, dataset, output, device, upstream, o, images)
    result_path = select_result(output, cfg.run_prior)
    tracks = read_artifact(work / 'tracks.npz', expected_kind=json.loads((work / 'tracks.json').read_text())['kind'])
    out = derive(tracks, 'refined', optimizer_executed=True, optimizer_result=str(result_path),
                 optimizer_stage='prior' if cfg.run_prior else 'smooth_fit', prior_executed=bool(cfg.run_prior),
                 expanded_config=str(work / 'expanded-config.yaml'), adapter='physical local mirror; exact camera reader')
    a = out.arrays
    a['valid'][:] = False
    with np.load(result_path) as z:
        t = z['trans'].shape[1]
        if t != dataset.seq_len:
            raise ValueError('optimizer output time mismatch')
        from body_model import MANO
        mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
        hand_model = MANO(batch_size=len(info['sides'])*t, pose2rot=True, **mano_cfg).cuda()
        trans, rot, pose, betas, right = [torch.tensor(z[key], device='cuda') for key in ('trans', 'root_orient', 'pose_body', 'betas', 'is_right')]
        with torch.no_grad():
            world = physical_run_mano(hand_model, trans, rot, pose, right, betas)
            cam_r = torch.tensor(z['cam_R'], device='cuda')
            cam_t = torch.tensor(z['cam_t'], device='cuda')
            points = torch.einsum('btij,btkj->btki', cam_r, world['joints']) + cam_t[:, :, None]
            physical_rot = cam_r @ axis_angle_to_matrix(rot)
            root_local = physical_run_mano(hand_model, torch.zeros_like(trans), torch.zeros_like(rot), pose, right, betas)['joints'][:, :, 0]
            camera_trans = torch.einsum('btij,btj->bti', cam_r, trans + root_local) + cam_t - root_local
        interval = slice(dataset.start_idx, dataset.end_idx)
        for b, side in enumerate(info['sides']):
            xyz = points[b].cpu().numpy()
            a['joints_camera'][interval, side] = xyz
            a['keypoints_2d'][interval, side] = project(xyz, a['intrinsic'])
            matrix = physical_rot[b].cpu().numpy()
            if side == 0:
                mirror = np.diag([-1., 1., 1.])
                matrix = mirror @ matrix @ mirror
            a['global_orient'][interval, side] = matrix
            a['hand_pose'][interval, side] = axis_angle_to_matrix(pose[b].reshape(t, 15, 3)).cpu().numpy()
            a['betas'][interval, side] = betas[b].cpu().numpy()
            a['translation'][interval, side] = camera_trans[b].cpu().numpy()
            eligible = tracks.arrays['valid'][interval, side]
            a['valid'][interval, side] = eligible & np.isfinite(xyz).all((1, 2)) & (xyz[..., 2] > 0).all(1)
        out.meta['optimized_camera_scale'] = float(z['world_scale'].reshape(-1)[0]) if 'world_scale' in z else 1.
        out.meta['camera_scale_optimized'] = bool(info['opt_scale'])
        out.meta['camera_assumption'] = info.get('camera_assumption', 'supplied motion')
        if not np.isfinite(out.meta['optimized_camera_scale']) or out.meta['optimized_camera_scale'] <= 0:
            raise ValueError('optimizer produced invalid camera scale')
    path = work / 'refined.npz'
    write_artifact(path, out)
    return {'ok': True, 'artifact': str(path)}


def main():
    protocol = sys.stdout
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                response = optimize(json.loads(line))
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
