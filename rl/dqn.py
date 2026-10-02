"""
dqn.py

A hand-rolled, pure-numpy Deep Q-Network (Stage 2).

The function approximator is a small multilayer perceptron with ReLU hidden
units and a linear output head over the 10 phase-actions. The MLP and its
backprop are validated by a finite-difference gradient check
(tests/test_agents.py) INDEPENDENT of any RL machinery, so that a DQN
non-convergence can never be attributed to a backprop bug.

Why numpy:
    A 10-action / ~23-dim state problem is small enough that a hand-rolled
    MLP with manual forward/backward passes is simple to get right, and it
    removes an entire class of "why won't torch install/import" failure modes
    that have nothing to do with the project. It is also a better demo talking
    point ("we implemented the gradient updates ourselves").

Design:
    - MLP.forward(x)  : x (N x D) -> Q(s, a) (N x A), caching activations.
    - MLP.backward(d): accumulate grads for a batch of output gradients.
    - MLP.step(lr)    : one SGD update from accumulated grads.
    - Adam            : Adam optimizer over an MLP's parameters.
    - ReplayBuffer    : fixed-capacity ring buffer of transitions.
    - DQNAgent        : epsilon-greedy, replay, target network, Double-DQN
                        targets, Huber loss, gradient-norm clipping.
"""
import numpy as np

from config import rl as rl_config


class MLP:
    """
    Feed-forward MLP for Q-value approximation.

    Layers: input -> h1 -> ... -> output. ReLU on hidden, linear output.
    Weights use He initialization scaled by `std`. Supports batched
    forward/backward and a single SGD update step.

    Attributes:
        n_input (int): state dimension.
        n_output (int): number of actions.
        hidden  (tuple[int]): hidden widths.
        params  (dict layer -> dict W/b): trainable parameters.
        grads   (dict layer -> dict dW/db): accumulated gradients.
    """

    def __init__(self, n_input, n_output, hidden=(64, 64), seed=None, std=0.5):
        self.n_input = n_input
        self.n_output = n_output
        self.hidden = tuple(hidden)
        self._rng = np.random.default_rng(seed)
        self.grads = None            # populated by zero_grad()

        self.params = {}
        layer_sizes = [n_input] + list(self.hidden) + [n_output]
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W = self._rng.normal(0.0, std * np.sqrt(2.0 / fan_in), (fan_out, fan_in))
            b = np.zeros((fan_out, 1))
            self.params[i] = {"W": W, "b": b}

    # -------- forward --------

    def forward(self, x):
        """
        Forward pass, caching activations for backprop.

        Args:
            x (np.ndarray): (N, n_input).

        Returns:
            np.ndarray: (N, n_output) Q-values.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.ndim == 1:
            x = x[None, :]

        # a0 = the input acts as the 'activation' of layer 0, so backward can
        # uniformly read a{i} as the previous activations for layer i.
        self._cache = {"z0": x, "a0": x}
        current = x
        n_layers = len(self.params)
        for i in range(n_layers):
            W = self.params[i]["W"]
            b = self.params[i]["b"]
            z = current @ W.T + b.T          # (N, out_i)
            a = np.maximum(0, z) if i < n_layers - 1 else z
            self._cache[f"z{i+1}"] = z
            self._cache[f"a{i+1}"] = a
            current = a
        return current

    # -------- backward --------

    def backward(self, dout):
        """
        Backpropagate a (N, n_output) gradient of the loss w.r.t outputs.

        Accumulates into self.grads. Call zero_grad() first.

        Args:
            dout (np.ndarray): (N, n_output) upstream gradient.
        """
        n_layers = len(self.params)
        dl = dout
        for i in range(n_layers - 1, -1, -1):
            a_prev = self._cache[f"a{i}"]            # (N, in_i)
            dW = dl.T @ a_prev                       # (out_i, in_i)
            db = dl.sum(axis=0, keepdims=True).T     # (out_i, 1)
            self.grads[i]["dW"] = self.grads[i].get("dW", 0.0) + dW
            self.grads[i]["db"] = self.grads[i].get("db", 0.0) + db
            if i > 0:
                W = self.params[i]["W"]
                dz = dl @ W                          # (N, in_i)
                z_prev = self._cache[f"z{i}"]        # (N, in_i) pre-activation
                dl = dz * (z_prev > 0)               # ReLU mask

    # -------- optimizer --------

    def zero_grad(self):
        """Reset accumulated gradients."""
        self.grads = {i: {} for i in range(len(self.params))}

    def step(self, lr):
        """Apply one SGD update using accumulated gradients."""
        for i in range(len(self.params)):
            self.params[i]["W"] -= lr * self.grads[i]["dW"]
            self.params[i]["b"] -= lr * self.grads[i]["db"]

    # -------- predict --------

    def predict(self, x):
        """Inference: return Q-values (no mutation)."""
        return self.forward(x)

    def __repr__(self):
        return f"MLP({self.n_input}->{self.hidden}->{self.n_output})"


def value_transform(x, eps):
    """
    Invertible value rescaling h(x) = sign(x)(sqrt(|x| + 1) - 1) + eps*x
    (Pohlen et al. 2018, "Observe and Look Further"; used in R2D2).

    Compresses large magnitudes so one network can represent values that
    differ by orders of magnitude across traffic levels. Monotonic, so the
    greedy action (and hence the optimal policy) is unchanged.
    """
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1.0) + eps * x


def value_transform_inverse(x, eps):
    """Exact inverse of value_transform."""
    return np.sign(x) * (
        ((np.sqrt(1.0 + 4.0 * eps * (np.abs(x) + 1.0 + eps)) - 1.0) / (2.0 * eps)) ** 2 - 1.0
    )


class Adam:
    """
    Adam optimizer for an MLP's accumulated gradients.

    Per-parameter adaptive step sizes make training insensitive to the raw
    reward scale, which plain SGD is not (large TD errors -> huge steps).
    """

    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {i: {k: np.zeros_like(v) for k, v in p.items()}
                  for i, p in net.params.items()}
        self.v = {i: {k: np.zeros_like(v) for k, v in p.items()}
                  for i, p in net.params.items()}

    def step(self, max_grad_norm=None):
        """Apply one update; optionally clip the global gradient norm first."""
        grads = {i: {"W": g["dW"], "b": g["db"]} for i, g in self.net.grads.items()}
        if max_grad_norm is not None:
            norm = np.sqrt(sum(float((g ** 2).sum())
                               for layer in grads.values() for g in layer.values()))
            if norm > max_grad_norm:
                scale = max_grad_norm / norm
                grads = {i: {k: g * scale for k, g in layer.items()}
                         for i, layer in grads.items()}

        self.t += 1
        for i, layer in grads.items():
            for k, g in layer.items():
                self.m[i][k] = self.beta1 * self.m[i][k] + (1 - self.beta1) * g
                self.v[i][k] = self.beta2 * self.v[i][k] + (1 - self.beta2) * g * g
                m_hat = self.m[i][k] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][k] / (1 - self.beta2 ** self.t)
                self.net.params[i][k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ReplayBuffer:
    """Fixed-capacity ring buffer of (s, a, r, s', done, discount)."""

    def __init__(self, capacity=20000, seed=None):
        self.capacity = capacity
        self._data = []
        self._pos = 0
        self._rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done, discount):
        """Store one transition, evicting the oldest when full."""
        item = (np.asarray(s, dtype=np.float64), a, float(r),
                np.asarray(s2, dtype=np.float64), bool(done), float(discount))
        if len(self._data) < self.capacity:
            self._data.append(item)
        else:
            self._data[self._pos] = item
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size):
        """Random batch of transitions (list of tuples)."""
        indices = self._rng.integers(0, len(self._data), size=batch_size)
        return [self._data[i] for i in indices]

    def __len__(self):
        return len(self._data)

    def is_ready(self, batch_size):
        return len(self._data) >= batch_size


class DQNAgent:
    """
    Deep Q-Network agent over the raw 23-dim observation.

    Attributes:
        policy_net (MLP): online network.
        target_net (MLP): frozen copy for stable targets.
        buffer (ReplayBuffer): experience replay.
        gamma, lr: RL hyperparameters.
        epsilon: exploration schedule (decays per episode like tabular).
    """

    def __init__(
        self,
        obs_dim=None,
        n_actions=None,
        hidden=None,
        gamma=None,
        alpha=None,
        epsilon=None,
        epsilon_end=None,
        replay_size=None,
        batch_size=None,
        target_update=None,
        seed=None,
        value_rescaling=None,
    ):
        obs_dim = obs_dim if obs_dim is not None else rl_config.OBS_DIM
        n_actions = n_actions if n_actions is not None else rl_config.NUM_PHASES
        hidden = hidden if hidden is not None else rl_config.DQN_HIDDEN_LAYERS
        gamma = gamma if gamma is not None else rl_config.GAMMA
        alpha = alpha if alpha is not None else rl_config.DQN_LEARNING_RATE
        epsilon = epsilon if epsilon is not None else rl_config.EPSILON_START
        epsilon_end = epsilon_end if epsilon_end is not None else rl_config.EPSILON_END
        replay_size = (
            replay_size if replay_size is not None else rl_config.DQN_REPLAY_SIZE
        )
        batch_size = batch_size if batch_size is not None else rl_config.DQN_BATCH_SIZE
        target_update = (
            target_update if target_update is not None else rl_config.DQN_TARGET_UPDATE
        )

        np_seed = seed if seed is not None else rl_config.NUMPY_SEED
        self.policy_net = MLP(obs_dim, n_actions, hidden, seed=np_seed)
        self.target_net = MLP(obs_dim, n_actions, hidden, seed=np_seed + 1)
        self._sync_target()

        self.buffer = ReplayBuffer(replay_size, seed=np_seed)
        self.optimizer = Adam(self.policy_net, lr=alpha)
        self.huber_delta = rl_config.DQN_HUBER_DELTA
        # When on, the network outputs h(Q) instead of Q (see value_transform).
        self.value_rescaling = (
            value_rescaling if value_rescaling is not None
            else rl_config.DQN_VALUE_RESCALING
        )
        self.rescale_eps = rl_config.DQN_VALUE_RESCALING_EPS
        self.grad_clip = rl_config.DQN_GRAD_CLIP

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = alpha
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.target_update_steps = target_update
        self._train_steps = 0
        self._rng = np.random.default_rng(np_seed)

    # -------- target sync --------

    def _sync_target(self):
        """Copy online weights into the target network."""
        for i in range(len(self.policy_net.params)):
            self.target_net.params[i]["W"] = self.policy_net.params[i]["W"].copy()
            self.target_net.params[i]["b"] = self.policy_net.params[i]["b"].copy()

    # -------- action selection --------

    def epsilon_greedy(self, obs):
        """Epsilon-greedy action on a raw observation."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.policy_net.predict(obs)))

    def select_action(self, obs):
        """Greedy action (inference/demo time)."""
        return int(np.argmax(self.policy_net.predict(obs)))

    # -------- training step --------

    def store(self, s, a, r, s2, terminal, discount=None):
        """
        Store a transition and train when the buffer is warm.

        `terminal` means the task truly ended (no bootstrapping). A
        time-limit cut-off is not terminal; pass False for it.
        `discount` is this transition's discount factor (it depends on how
        long the step lasted); defaults to the per-step gamma.
        """
        discount = self.gamma if discount is None else discount
        self.buffer.push(s, a, r, s2, terminal, discount)
        if self.buffer.is_ready(self.batch_size):
            self._train_once()
            self._train_steps += 1
            if self._train_steps % self.target_update_steps == 0:
                self._sync_target()

    def _train_once(self):
        """Sample a batch and apply one Adam step on the Huber TD objective."""
        batch = self.buffer.sample(self.batch_size)
        states = np.stack([t[0] for t in batch])     # (B, D)
        actions = np.array([t[1] for t in batch])    # (B,)
        rewards = np.array([t[2] for t in batch])    # (B,)
        nexts = np.stack([t[3] for t in batch])      # (B, D)
        dones = np.array([t[4] for t in batch])      # (B,)
        discounts = np.array([t[5] for t in batch])  # (B,)

        # Double DQN: the online net picks a', the target net evaluates it.
        # Using max over the target net alone systematically overestimates.
        rows = np.arange(self.batch_size)
        best_next = self.policy_net.predict(nexts).argmax(axis=1)   # (B,)
        q_next = self.target_net.predict(nexts)[rows, best_next]    # (B,)
        if self.value_rescaling:
            # Network values live in h-space: map back, bootstrap, map again.
            q_next = value_transform_inverse(q_next, self.rescale_eps)
        targets = rewards + discounts * q_next * (1.0 - dones)
        if self.value_rescaling:
            targets = value_transform(targets, self.rescale_eps)

        # Forward on the policy net (must come after the predict above,
        # which overwrites the activation cache used by backward).
        q_all = self.policy_net.forward(states)      # (B, A)
        self.policy_net.zero_grad()
        dq = np.zeros_like(q_all)
        # Huber loss gradient: the TD error itself when small, clipped to
        # +/-delta when large, so outlier transitions cannot blow up updates.
        td = q_all[rows, actions] - targets
        dq[rows, actions] = np.clip(td, -self.huber_delta, self.huber_delta)
        self.policy_net.backward(dq / self.batch_size)
        self.optimizer.step(max_grad_norm=self.grad_clip)

    # -------- persistence --------

    def get_weights(self):
        """Snapshot of the policy weights (for best-checkpoint keeping)."""
        return {i: {k: v.copy() for k, v in p.items()}
                for i, p in self.policy_net.params.items()}

    def set_weights(self, weights):
        for i, p in weights.items():
            for k, v in p.items():
                self.policy_net.params[i][k] = v.copy()
        self._sync_target()

    def save(self, path):
        """Save the policy network weights to an .npz file."""
        arrays = {f"{k}{i}": v for i, p in self.policy_net.params.items()
                  for k, v in p.items()}
        np.savez(path, **arrays)

    def load(self, path):
        """Load policy weights saved by save() (and sync the target net)."""
        data = np.load(path)
        for i, p in self.policy_net.params.items():
            for k in p:
                p[k] = data[f"{k}{i}"].copy()
        self._sync_target()

    def __repr__(self):
        return f"DQNAgent({self.policy_net}, eps={self.epsilon:.3f})"

