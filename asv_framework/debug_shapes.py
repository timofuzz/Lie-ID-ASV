"""
Debug utility to print tensor shapes in SE2HamNODE forward pass.
"""
from asv_framework.paths import add_vendor_paths

add_vendor_paths()

from asv_framework.training.dataset import OtterStepDataset
from asv_framework.training.train_otter_se2 import yaw_to_R_flat
from asv_framework.models.se2_hamnode import SE2HamNODE
from torch.utils.data import DataLoader
import torch


def main():
    ds = OtterStepDataset("asv_framework/data/otter_rollouts.npz")
    loader = iter(DataLoader(ds, batch_size=4, shuffle=False))
    y0, y_target, dt = next(loader)
    q = y0[:, 0:3]
    q_dot = y0[:, 3:6]
    u = y0[:, 6:8]
    R_flat = yaw_to_R_flat(q[:, 2])
    q_full = torch.cat((q[:, 0:2], R_flat), dim=1)
    state = torch.cat((q_full, q_dot, u), dim=1)
    print("state", state.shape)

    model = SE2HamNODE(device="cpu")

    q, q_dot, u = torch.split(
        state, [model.posedim, model.twistdim, model.udim], dim=1
    )
    x, R_flat = torch.split(q, [model.xdim, model.Rdim], dim=1)
    q_dot_v, q_dot_w = torch.split(q_dot, [model.linveldim, model.angveldim], dim=1)
    M_q_inv1 = model.M_net1(x)
    M_q_inv2 = model.M_net2(R_flat)
    pv = torch.squeeze(
        torch.matmul(torch.inverse(M_q_inv1), torch.unsqueeze(q_dot_v, dim=2)), dim=2
    )
    pw = q_dot_w / M_q_inv2.view(-1, 1)
    q_p = torch.cat((q, pv, pw), dim=1)
    print("q", q.shape, "q_dot", q_dot.shape, "u", u.shape)
    print("x", x.shape, "R_flat", R_flat.shape)
    print("q_dot_v", q_dot_v.shape, "q_dot_w", q_dot_w.shape)
    print("M_q_inv1", M_q_inv1.shape, "M_q_inv2", M_q_inv2.shape)
    print("pv", pv.shape, "pw", pw.shape)
    print("q_p", q_p.shape)

    # Recompute forward pieces to inspect gradients
    M_q_inv2_scalar = M_q_inv2.view(-1, 1)
    V_q = model.V_net(q)
    g_q = model.g_net(q)
    p_aug_v = torch.unsqueeze(pv, dim=2)
    H = (
        torch.squeeze(
            torch.matmul(
                torch.transpose(p_aug_v, 1, 2), torch.matmul(M_q_inv1, p_aug_v)
            )
        )
        / 2.0
        + 0.5 * torch.squeeze(M_q_inv2_scalar) * torch.squeeze(pw) ** 2
        + torch.squeeze(V_q)
    )
    dH = torch.autograd.grad(H.sum(), q_p, create_graph=True, allow_unused=True)[0]
    if dH is None:
        dH = torch.zeros_like(q_p)
    dHdx, dHdR, dHdpv, dHdpw = torch.split(
        dH, [model.xdim, model.Rdim, model.linveldim, model.angveldim], dim=1
    )
    print("dHdpv", dHdpv.shape)
    Rmat = R_flat.view(-1, 2, 2)
    row0 = Rmat[:, 0, :]
    row1 = Rmat[:, 1, :]
    grad0 = dHdR[:, 0:2]
    grad1 = dHdR[:, 2:4]
    from asv_framework.models.se2_hamnode import cross2

    dpw = cross2(pv, dHdpv) + cross2(row0, grad0) + cross2(row1, grad1)
    print("dpw shape after cross2", dpw.shape)

    # Run a single forward step to trigger the full graph and catch shapes
    try:
        out = model(None, state)
        print("model output", out.shape)
    except Exception as e:
        print("forward error:", e)


if __name__ == "__main__":
    main()
