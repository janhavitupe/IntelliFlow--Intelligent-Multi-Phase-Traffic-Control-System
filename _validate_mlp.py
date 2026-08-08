"""
_validate_mlp.py

Gradient-check the pure-numpy MLP (rl/dqn.py) on a tiny toy problem BEFORE
it is wired into any RL machinery. This isolates backprop bugs from RL
non-convergence, per the Stage 2 guidance: both failure modes look identical
from the outside ("reward isn't going up"), so we must prove the network and
its gradients are correct independently.

Tests:
  TEST 1 - Numeric gradient check: analytic grads match finite differences
           (relative error < 1e-5) for a random small MLP + random batch.
  TEST 2 - XOR fit: a 2-layer MLP (2->8->8->1, ReLU) learns XOR to near-zero
           loss with SGD, proving forward/backward/step are individually
           correct end-to-end.
  TEST 3 - DQNAgent smoke: a tiny agent can do a forward + one replay update
           without error and its Q-shape is (batch, n_actions).
"""
import numpy as np

from rl.dqn import MLP, DQNAgent


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return passed


# ---------------------------------------------------------------------------
# TEST 1 : Numeric gradient check
# ---------------------------------------------------------------------------
def numeric_grad(model, x, dout, eps=1e-6):
    """Finite-difference estimate of dL/dW and dL/db for each layer.

    Approximates the exact gradient using central differences on the total
    loss L = sum(dout * output) (a linear functional of the output), which
    yields the same dW/db as backward(dout).
    """
    grads = {}
    for i in range(len(model.params)):
        W = model.params[i]["W"].copy()
        b = model.params[i]["b"].copy()
        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        # Bias grads.
        for r in range(b.shape[0]):
            bp = b.copy(); bp[r, 0] += eps
            model.params[i]["b"] = bp
            lp = _scalar_loss(model, x, dout)
            bm = b.copy(); bm[r, 0] -= eps
            model.params[i]["b"] = bm
            lm = _scalar_loss(model, x, dout)
            db[r, 0] = (lp - lm) / (2 * eps)
        model.params[i]["b"] = b
        # Weight grads.
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                wp = W.copy(); wp[r, c] += eps
                model.params[i]["W"] = wp
                lp = _scalar_loss(model, x, dout)
                wm = W.copy(); wm[r, c] -= eps
                model.params[i]["W"] = wm
                lm = _scalar_loss(model, x, dout)
                dW[r, c] = (lp - lm) / (2 * eps)
        model.params[i]["W"] = W
        grads[i] = {"dW": dW, "db": db}
    return grads


def _scalar_loss(model, x, dout):
    """L = sum(dout * output), a scalar whose grad w.r.t output is dout."""
    out = model.forward(x)
    return float(np.sum(dout * out))


# Small random MLP + batch.
_rng = np.random.default_rng(0)
X = _rng.normal(size=(5, 3))
DOUT = _rng.normal(size=(5, 2))
m = MLP(3, 2, hidden=(4, 4), seed=0, std=0.5)

m.forward(X)
m.zero_grad()
m.backward(DOUT)
analytic = {i: {"dW": m.grads[i]["dW"].copy(), "db": m.grads[i]["db"].copy()}
            for i in range(len(m.params))}
numeric = numeric_grad(m, X, DOUT)

max_err = 0.0
for i in range(len(m.params)):
    rel_w = np.max(np.abs(analytic[i]["dW"] - numeric[i]["dW"])) / (
        1e-8 + np.max(np.abs(numeric[i]["dW"]))
    )
    rel_b = np.max(np.abs(analytic[i]["db"] - numeric[i]["db"])) / (
        1e-8 + np.max(np.abs(numeric[i]["db"]))
    )
    max_err = max(max_err, rel_w, rel_b)

test1 = max_err < 1e-5
results = []
results.append(check(
    "TEST 1 Numeric gradient check (analytic vs finite-diff)",
    test1,
    f"max relative error = {max_err:.2e}",
))


# ---------------------------------------------------------------------------
# TEST 2 : XOR fit (backprop correctness end-to-end)
# ---------------------------------------------------------------------------
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
XOR_Y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Small init + modest lr keeps the toy stable. Gradients are already proven
# correct by TEST 1; this just confirms the optimizer end-to-end.
xor_net = MLP(2, 1, hidden=(8, 8), seed=1, std=0.3)
lr = 0.1
best = float("inf")
for it in range(40000):
    out = xor_net.forward(XOR_X)
    err = out - XOR_Y
    loss = 0.5 * np.sum(err ** 2)
    best = min(best, loss)
    xor_net.zero_grad()
    xor_net.backward(err)
    xor_net.step(lr)
    if loss < 1e-4:
        break

final_loss = 0.5 * np.sum((xor_net.forward(XOR_X) - XOR_Y) ** 2)
test2 = final_loss < 1e-3
results.append(check(
    "TEST 2 XOR fit (MLP learns non-linear XOR with SGD)",
    test2,
    f"final loss = {final_loss:.2e} (iters={it + 1})",
))


# ---------------------------------------------------------------------------
# TEST 3 : DQNAgent smoke test
# ---------------------------------------------------------------------------
agent = DQNAgent(obs_dim=4, n_actions=3, hidden=(4, 4),
                 batch_size=4, replay_size=100, seed=0)
# Push enough transitions to warm the buffer and trigger a training step.
for k in range(20):
    s = np.array([float(k)] * 4)
    s2 = np.array([float(k + 1)] * 4)
    agent.store(s, int(k % 3), -1.0, s2, k == 19)

# Shape check on a batch forward.
q = agent.policy_net.predict(np.array([[0.0, 1.0, 0.0, 1.0]]))
shape_ok = q.shape == (1, 3)
# Action selection returns a valid int.
act = agent.select_action(np.array([0.1, 0.2, 0.3, 0.4]))
act_ok = 0 <= act < 3
test3 = shape_ok and act_ok and len(agent.buffer) == 20
results.append(check(
    "TEST 3 DQNAgent smoke (forward + replay update + action selection)",
    test3,
    f"Qshape={q.shape} action={act} buffer={len(agent.buffer)}",
))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MLP GRADIENT / DQN VALIDATION SUMMARY")
print("=" * 60)
passed = sum(results)
total = len(results)
for i, r in enumerate(results, 1):
    print(f"  Test {i}: {'PASS' if r else 'FAIL'}")
print(f"{passed}/{total} checks passed")
raise SystemExit(0 if passed == total else 1)
