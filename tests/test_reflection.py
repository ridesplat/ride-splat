import math

import torch

from ridesplat.reflection import ReflectionModel

H, W = 8, 12
FRAMES = ["train_r00000", "train_r00001"]


def model(**kw):
    return ReflectionModel(H, W, FRAMES, device="cpu", **kw)


def test_starts_as_plain_3dgs_without_extras():
    m = model()
    T = torch.rand(3, H, W)
    pred, reg = m.predict(T, FRAMES[0])
    assert torch.equal(pred, T) and float(reg) == 0.0


def test_minimal_composite_is_scene_plus_weighted_reflection():
    weight = torch.rand(1, H, W)
    m = model(weight=weight)
    with torch.no_grad():
        m.R.copy_(torch.randn(3, H, W))
    T = torch.rand(3, H, W)
    pred, _ = m.predict(T, FRAMES[1])
    assert torch.allclose(pred, T + weight * m.R)


def test_paper_composite_matches_the_research_formula():
    weight = torch.rand(1, H, W)
    m = model(weight=weight, atten=True, atten_max=0.9, appearance="affine")
    with torch.no_grad():
        m.R.copy_(torch.randn(3, H, W))
        m.atten_logit.copy_(torch.randn(1, H, W))
        m.gain[FRAMES[0]].copy_(torch.tensor([1.1, 0.9, 1.0]).view(3, 1, 1))
        m.bias[FRAMES[0]].copy_(torch.tensor([0.02, -0.01, 0.0]).view(3, 1, 1))
    T = torch.rand(3, H, W)
    pred, reg = m.predict(T, FRAMES[0])
    g, b = m.gain[FRAMES[0]], m.bias[FRAMES[0]]
    A = 0.9 * torch.sigmoid(m.atten_logit)
    assert torch.allclose(pred, (1 - A) * (g * T + b) + m.R * weight)        # pred = T_eff + R_eff
    assert torch.allclose(reg, ((g - 1) ** 2).mean() + (b ** 2).mean())


def test_attenuation_starts_near_zero():
    A = model(atten=True, atten_max=0.9).attenuation()
    assert torch.allclose(A, torch.full((1, H, W), 0.9 / (1 + math.exp(3))))


def test_penalty_terms():
    m = model(l1=0.01, atten=True, atten_l1=0.01, appearance="affine", appearance_reg=10.0)
    with torch.no_grad():
        m.R.fill_(0.5)
    reg = torch.tensor(0.2)
    expected = 0.01 * 0.5 + 10.0 * 0.2 + 0.01 * float(m.attenuation().detach().mean())
    assert math.isclose(float(m.penalty(reg)), expected, rel_tol=1e-6)


def test_step_updates_only_the_frame_that_was_used():
    m = model(atten=True, appearance="affine")
    pred, reg = m.predict(torch.rand(3, H, W), FRAMES[0])
    ((pred - 0.3) ** 2).mean().add(m.penalty(reg)).backward()
    m.step()
    assert float(m.R.abs().sum()) > 0
    assert not torch.equal(m.gain[FRAMES[0]], torch.ones(3, 1, 1))
    assert torch.equal(m.gain[FRAMES[1]], torch.ones(3, 1, 1))


def test_state_round_trip():
    a = model(atten=True, appearance="affine")
    pred, reg = a.predict(torch.rand(3, H, W), FRAMES[1])
    pred.mean().add(a.penalty(reg)).backward()
    a.step()
    b = model(atten=True, appearance="affine")
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.R, b.R) and torch.equal(a.atten_logit, b.atten_logit)
    assert torch.equal(a.gain[FRAMES[1]], b.gain[FRAMES[1]])
