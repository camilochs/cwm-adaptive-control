"""DQN baseline for adaptive parameter control in (1+1)-RLS_k.

Implements a Deep Q-Network that learns to select the flip count k
given the current optimizer state. Follows the standard DQN architecture
(Mnih et al., Nature 2015) with:
  - Experience replay buffer
  - Target network with periodic hard updates
  - Epsilon-greedy exploration with decay

State representation inspired by DE-DDQN (Sharma et al., 2019,
arXiv:1905.08006) which uses fitness landscape features as state:
  - normalized_fitness: fitness / max_fitness
  - progress: step / budget
  - region indicator: normal / valley_edge / valley (for Jump_k)
  - stagnation: consecutive non-improving steps / patience

Action space: discrete set of k values (not all 1..n, but a
representative subset matching the CWM candidate set for fair comparison).

Reward: -1 per step (goal: minimize steps to optimum).
Terminal: optimum reached OR budget exhausted.

Reference:
  Mnih, V. et al. "Human-level control through deep reinforcement
  learning." Nature 518.7540 (2015): 529-533.

  Sharma, M. et al. "Deep Reinforcement Learning Based Parameter
  Control in Differential Evolution." GECCO 2019. arXiv:1905.08006.
"""

import numpy as np
from collections import deque, namedtuple
from typing import List, Optional, Tuple
import random
import math

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Transition storage ─────────────────────────────────────────────────────

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ── Q-Network ──────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Simple MLP Q-network for discrete action selection.

    Architecture follows PyTorch DQN tutorial with 2 hidden layers.
    Input: state vector (4 features)
    Output: Q-value for each discrete action
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ── DQN Agent ──────────────────────────────────────────────────────────────

class DQNAgent:
    """DQN agent for selecting flip count k in (1+1)-RLS_k.

    Designed for both OneMax and Jump_k problems. The state representation
    includes problem-specific features that allow the agent to distinguish
    between the normal region and the valley in Jump_k.

    Args:
        candidate_ks: list of allowed k values (discrete action space)
        n: bitstring length
        jump_k: gap parameter (0 for OneMax)
        budget: step budget per episode
        lr: learning rate
        gamma: discount factor
        eps_start: initial epsilon for exploration
        eps_end: minimum epsilon
        eps_decay: epsilon decay per step
        batch_size: mini-batch size for replay
        target_update: steps between target network hard updates
        buffer_size: replay buffer capacity
        hidden: hidden layer size
    """

    # State features:
    #   0: normalized_fitness  (fitness / max_possible)
    #   1: progress            (step / budget)
    #   2: region_indicator    (1.0 = normal, 0.5 = valley_edge, 0.0 = valley)
    #   3: stagnation          (non-improving steps / 100, clipped to [0, 1])
    STATE_DIM = 4

    def __init__(
        self,
        candidate_ks: List[int],
        n: int = 50,
        jump_k: int = 2,
        budget: int = 10000,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 5000,
        batch_size: int = 64,
        target_update: int = 500,
        buffer_size: int = 50000,
        hidden: int = 128,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN baseline")

        self.candidate_ks = sorted(candidate_ks)
        self.n_actions = len(self.candidate_ks)
        self.k_to_idx = {k: i for i, k in enumerate(self.candidate_ks)}
        self.n = n
        self.jump_k = jump_k
        self.budget = budget
        self.max_fitness = n + jump_k if jump_k > 0 else n

        # Hyperparameters
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.lr = lr

        # Networks
        self.device = torch.device("cpu")
        self.policy_net = QNetwork(self.STATE_DIM, self.n_actions, hidden).to(self.device)
        self.target_net = QNetwork(self.STATE_DIM, self.n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        # Counters
        self.total_steps = 0
        self.train_losses = []

        # Per-episode state tracking
        self._stagnation = 0
        self._prev_fitness = None

    def _encode_state(self, fitness: int, step: int) -> np.ndarray:
        """Encode optimizer state as a feature vector."""
        norm_fitness = fitness / self.max_fitness

        progress = step / self.budget

        # Region indicator (Jump_k specific)
        if self.jump_k > 0:
            if fitness >= self.jump_k and fitness <= self.n:
                if fitness == self.n:
                    region = 0.5  # valley edge
                else:
                    region = 1.0  # normal
            elif fitness == self.max_fitness:
                region = 1.0  # optimum (treat as normal)
            else:
                region = 0.0  # valley
        else:
            region = 1.0  # OneMax: always "normal"

        stagnation = min(self._stagnation / 100.0, 1.0)

        return np.array([norm_fitness, progress, region, stagnation],
                        dtype=np.float32)

    def _get_epsilon(self) -> float:
        """Compute current epsilon with exponential decay."""
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-self.total_steps / self.eps_decay)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action index using epsilon-greedy policy.

        Returns:
            index into self.candidate_ks
        """
        if training:
            eps = self._get_epsilon()
            if random.random() < eps:
                return random.randrange(self.n_actions)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def select_k(self, fitness: int, step: int,
                 training: bool = True) -> int:
        """High-level interface: given optimizer state, return k value."""
        state = self._encode_state(fitness, step)
        action_idx = self.select_action(state, training=training)
        return self.candidate_ks[action_idx]

    def update_stagnation(self, fitness: int):
        """Update stagnation counter (call after each step)."""
        if self._prev_fitness is not None:
            if fitness > self._prev_fitness:  # strict: ties = stagnation
                self._stagnation = 0
            else:
                self._stagnation += 1
        self._prev_fitness = fitness

    def reset_episode(self):
        """Reset per-episode state."""
        self._stagnation = 0
        self._prev_fitness = None

    def store_transition(self, state: np.ndarray, action_idx: int,
                         reward: float, next_state: np.ndarray, done: bool):
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action_idx, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Perform one gradient step on a mini-batch from replay.

        Returns loss value or None if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*batch))

        state_batch = torch.tensor(np.array(batch.state),
                                   dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long,
                                    device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32,
                                    device=self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state),
                                        dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32,
                                  device=self.device)

        # Current Q values: Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # Target Q values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(dim=1)[0]
            target = reward_batch + self.gamma * next_q * (1.0 - done_batch)

        # Huber loss (smooth L1) — standard for DQN
        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (standard for DQN stability)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.total_steps += 1

        # Hard update target network
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.train_losses.append(loss_val)
        return loss_val

    def get_policy_fn(self):
        """Return a policy function compatible with run_episode().

        The returned function has signature (fitness, n, jump_k, step) -> k
        and runs the trained DQN in greedy mode (no exploration).
        Resets episode state on step=0 and updates stagnation each call.
        """
        self.policy_net.eval()

        def policy(fitness, n_, jump_k_, step):
            if step == 0:
                self.reset_episode()
            self.update_stagnation(fitness)
            return self.select_k(fitness, step, training=False)

        return policy

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "candidate_ks": self.candidate_ks,
            "n": self.n,
            "jump_k": self.jump_k,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]


# ── Training loop ──────────────────────────────────────────────────────────

def train_dqn_jump(
    n: int = 50,
    jump_k: int = 2,
    budget: int = 10000,
    n_episodes: int = 500,
    candidate_ks: List[int] = None,
    lr: float = 1e-3,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: int = 10000,
    batch_size: int = 64,
    target_update: int = 500,
    hidden: int = 128,
    verbose: bool = True,
    seed: int = 42,
) -> Tuple["DQNAgent", dict]:
    """Train a DQN agent on the Jump_k problem.

    The agent interacts with the (1+1)-RLS_k optimizer, choosing k at each
    step. Reward is -1 per step (minimise time to optimum). A bonus of +100
    is given when the optimum is reached, to create a clear signal.

    Args:
        n: bitstring length
        jump_k: gap parameter
        budget: step budget per episode
        n_episodes: training episodes
        candidate_ks: discrete action set
        lr, gamma, eps_start, eps_end, eps_decay: DQN hyperparameters
        batch_size: replay mini-batch size
        target_update: steps between target network sync
        hidden: hidden layer width
        verbose: print progress
        seed: random seed

    Returns:
        (agent, training_log)
    """
    from src.jump import JumpProblem, RLSOptimizer

    if candidate_ks is None:
        # Match the CWM candidate set for fair comparison
        candidate_ks = sorted(set(
            [1, 2, 3, 5, 8, 10, 15, 25, jump_k]
            + [max(1, n - i) for i in range(0, n, 5)]
            + [n]
        ))
        candidate_ks = [k for k in candidate_ks if 1 <= k <= n]

    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    agent = DQNAgent(
        candidate_ks=candidate_ks,
        n=n, jump_k=jump_k, budget=budget,
        lr=lr, gamma=gamma,
        eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
        batch_size=batch_size, target_update=target_update,
        hidden=hidden,
    )

    log = {
        "episode_steps": [],
        "episode_converged": [],
        "episode_reward": [],
        "mean_loss": [],
    }

    for ep in range(n_episodes):
        problem = JumpProblem(n, jump_k)
        rls = RLSOptimizer(problem, seed=seed + ep)
        rls.initialize()
        agent.reset_episode()

        total_reward = 0.0
        ep_losses = []

        while rls.step_count < budget and not rls.is_optimal():
            # Encode state
            state = agent._encode_state(rls.fitness_val, rls.step_count)

            # Select action
            action_idx = agent.select_action(state, training=True)
            k = agent.candidate_ks[action_idx]

            # Execute step
            old_fitness = rls.fitness_val
            new_fitness, improved = rls.step(k)

            # Update stagnation tracking
            agent.update_stagnation(new_fitness)

            # Compute reward
            done = rls.is_optimal() or rls.step_count >= budget

            if rls.is_optimal():
                reward = 100.0  # bonus for finding optimum
            else:
                reward = -1.0   # cost per step

            total_reward += reward

            # Encode next state
            next_state = agent._encode_state(rls.fitness_val, rls.step_count)

            # Store and learn
            agent.store_transition(state, action_idx, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

        log["episode_steps"].append(rls.step_count)
        log["episode_converged"].append(rls.is_optimal())
        log["episode_reward"].append(total_reward)
        log["mean_loss"].append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            recent_steps = log["episode_steps"][-50:]
            recent_conv = log["episode_converged"][-50:]
            eps = agent._get_epsilon()
            print(f"    Episode {ep+1}/{n_episodes}  "
                  f"mean_steps={np.mean(recent_steps):.0f}  "
                  f"conv={np.mean(recent_conv):.0%}  "
                  f"eps={eps:.3f}  "
                  f"loss={log['mean_loss'][-1]:.4f}")

    agent.policy_net.eval()
    return agent, log
