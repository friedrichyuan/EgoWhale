"""Guard the quaternion gradient convention at the cuRobo/refinement boundary."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egowhale.action import base_ik as core


def test_refinement_fk_gradient_and_reused_buffers(monkeypatch):
    # Two nonparallel revolute axes make body/base gradient confusion observable.
    def poses(q):
        a, b = q.unbind(-1)
        zero, one = torch.zeros_like(a), torch.ones_like(a)
        qz = torch.stack((torch.cos(a / 2), zero, zero, torch.sin(a / 2)), -1)
        qy = torch.stack((torch.cos(b / 2), zero, torch.sin(b / 2), zero), -1)
        quat = core._quat_mul(qz, qy)
        pos = torch.stack((torch.cos(a) * torch.cos(b), torch.sin(a) * torch.cos(b), -torch.sin(b)), -1)
        axis_z = torch.stack((zero, zero, one), -1)
        axis_y = torch.stack((-torch.sin(a), torch.cos(a), zero), -1)
        axes = torch.stack((axis_z, axis_y), -1)
        linear = torch.linalg.cross(axes.transpose(-1, -2), pos.unsqueeze(-2).expand_as(axes.transpose(-1, -2)), dim=-1)
        jac = torch.cat((linear.transpose(-1, -2), axes), -2)
        return pos.unsqueeze(-2), quat.unsqueeze(-2), jac.unsqueeze(-3)

    class Kin:
        joint_names = ['first', 'second']
        compute_jacobian = False
        saved = None

        def compute_kinematics(self, state):
            pos, quat, jac = poses(state.position)
            if self.saved is None:
                self.saved = [pos.clone(), quat.clone(), jac.clone()]
            else:
                for dst, src in zip(self.saved, (pos, quat, jac), strict=True):
                    dst.copy_(src)
            return SimpleNamespace(tool_poses=SimpleNamespace(position=self.saved[0], quaternion=self.saved[1]), tool_jacobians=self.saved[2])

    js = SimpleNamespace(from_position=lambda position, joint_names: SimpleNamespace(position=position))
    monkeypatch.setitem(sys.modules, 'curobo.types', SimpleNamespace(JointState=js))
    q = torch.tensor([[[0.7, -0.4]], [[-1.2, 0.8]]], dtype=torch.float64, requires_grad=True)
    kin = Kin()
    assert torch.autograd.gradcheck(lambda x: core._RefinementFK.apply(x, kin), (q,), atol=1e-6)
    pos, quat = core._RefinementFK.apply(q, kin)
    weights = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=q.dtype)
    loss = pos.square().sum() + (quat * weights).sum()
    core._RefinementFK.apply(q + 0.3, kin)  # overwrite the library's scratch buffers
    actual = torch.autograd.grad(loss, q)[0]
    ref_p, ref_q, _ = poses(q)
    expected = torch.autograd.grad(ref_p.square().sum() + (ref_q * weights).sum(), q)[0]
    torch.testing.assert_close(actual, expected)
    assert kin.compute_jacobian is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires an explicitly allocated GPU')
@pytest.mark.parametrize('asset', ['piper', 'arx'])
def test_refinement_fk_real_asset_finite_difference(asset):
    from curobo.kinematics import Kinematics, KinematicsCfg

    from egowhale.robots import load_robot

    robot = load_robot(asset)
    group = robot.groups[0]
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(group.config, tool_frames=list(group.tools)))
    def tensor(value):
        return torch.tensor(value, device='cuda', dtype=torch.float32)

    q = tensor(np.asarray(group.home)[None] + 0.13).requires_grad_()
    base = tensor([0.02, -0.03, 0.01, 0.23]).requires_grad_()
    # A camera rotation which is not aligned with any joint axis.
    from scipy.spatial.transform import Rotation
    rotation = tensor(Rotation.from_euler('xyz', [0.4, -0.7, 0.3]).as_matrix())
    translation = tensor([0.2, 0.1, 0.3])
    wp, wq = tensor([0.3, -0.5, 0.8]), tensor([0.1, -0.6, 0.2, 0.7])

    def objective():
        p, r = core._fk_camera(kin, q, base, rotation, translation, group)
        return (p * wp).sum() + (r * wq).sum()

    dq, db = torch.autograd.grad(objective(), (q, base))
    for variable, gradient in ((q, dq), (base, db)):
        for i in range(variable.numel()):
            with torch.no_grad():
                original = variable.flatten()[i].item()
                variable.flatten()[i] = original + 1e-3
                plus = objective().item()
                variable.flatten()[i] = original - 1e-3
                minus = objective().item()
                variable.flatten()[i] = original
            assert gradient.flatten()[i].item() == pytest.approx((plus - minus) / 2e-3, abs=8e-4, rel=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires an explicitly allocated GPU')
@pytest.mark.parametrize('asset', ['piper', 'arx'])
@pytest.mark.parametrize('length', [1, 16])
def test_refinement_graph_and_angle_boundary(asset, length):
    from curobo.kinematics import Kinematics, KinematicsCfg
    from scipy.spatial.transform import Rotation

    from egowhale.robots import load_robot

    robot = load_robot(asset)
    group = robot.groups[0]
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(group.config, tool_frames=list(group.tools)))
    mask = core._periodic_joints(robot, group)
    arm = np.tile(group.home, (length, 1))
    if mask.any():
        angles = np.linspace(179., 181., length)
        angles = (angles + 180.) % 360. - 180.
        arm[:, mask] = np.deg2rad(angles[:, None])
    with torch.no_grad():
        q = torch.as_tensor(arm, device='cuda', dtype=torch.float32)
        p, r = core._fk_camera(kin, q, torch.zeros(4, device='cuda'), torch.eye(3, device='cuda'),
                               torch.zeros(3, device='cuda'), group)
        gp = p.cpu().numpy()
        quat = r.cpu().numpy()
    gr = Rotation.from_quat(quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(length, -1, 3, 3)
    _, output, losses = core._refine(kin, gp, gr, np.ones(gp.shape[:2], bool), np.eye(3), np.zeros(3),
                                      np.zeros(4), arm, group, fps=18., fixed_base=True, periodic=mask)
    assert np.isfinite(output).all() and all(np.isfinite(v) for v in losses.values())
    assert (output >= group.lower - 1e-6).all() and (output <= group.upper + 1e-6).all()
    if length > 1:
        assert abs(core._joint_delta(output[1:], output[:-1], mask)).max() < np.deg2rad(20)
    assert losses['position'] < .01
    assert losses['orientation'] < np.sin(np.deg2rad(10) / 2)**2
