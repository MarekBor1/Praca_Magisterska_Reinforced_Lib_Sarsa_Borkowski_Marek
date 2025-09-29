from functools import partial
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass, Array, Scalar, PRNGKey, Shape

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class SarsaState(AgentState):
    """
    Container for the state of the SARSA agent.

    Attributes
    ----------
    Q : Array
        Q-values for each action.
    """
    Q: Array
    previous_action: int
    prev_env_state: int


class SARSA(BaseAgent):
    def __init__(self, action_space_size: int, obs_space_shape: int, alpha: float, gamma: float, epsilon: float, group_size: int) -> None:
        self.action_space_size = action_space_size
        self.obs_space_shape = obs_space_shape
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.group_size = group_size

        self.init = jax.jit(partial(self.init, obs_space_shape=obs_space_shape))
        self.update = jax.jit(partial(self.update, alpha=alpha, gamma=gamma, epsilon=epsilon ,group_size=group_size))
        self.sample = jax.jit(partial(self.sample, epsilon=epsilon, group_size=group_size))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'alpha': gym.spaces.Box(0.0, 1.0, (1,), float),
            'gamma': gym.spaces.Box(0.0, 1.0, (1,), float),
            'epsilon': gym.spaces.Box(0.0, 1.0, (1,), float)
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_state': gym.spaces.Box(-jnp.inf, jnp.inf, (6,), float),
            'action': self.action_space,
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float),
            'current_state': gym.spaces.Discrete(3)
            # 'next_env_state': gym.spaces.Box(-jnp.inf, jnp.inf, (6,), float)
        })
    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_state': gym.spaces.Box(-jnp.inf, jnp.inf, (6,), float),
            'current_state': gym.spaces.Discrete(3)
        })

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.action_space_size)  # Actions: 0, 1, ..., action_space_size - 1


    @staticmethod
    def init() -> SarsaState:
        """
        Initializes the SARSA agent's state with a Q-table for the given number of groups.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        obs_space_shape : Shape
            The shape of the observation space.
        num_groups : int
            Number of groups to initialize the Q-table with.
        action_space_size : int
            Number of actions available.

        Returns
        -------
        SarsaState
            The initial state of the SARSA agent.
        """
        Q = jnp.zeros((52, 12))
        return SarsaState(Q=Q, previous_action=0, prev_env_state=0)
    
    @staticmethod
    def update(
            state: SarsaState,
            action: int, 
            reward: Scalar, 
            alpha: float,
            gamma: float,
            current_state: int
    ) -> SarsaState:

        current_q = state.Q[state.prev_env_state, state.previous_action]
        next_q = state.Q[current_state, action]

        new_q = current_q + alpha * (reward + gamma * next_q - current_q)
        Q = state.Q.at[state.prev_env_state, action].set(new_q)

        return SarsaState(Q=Q, previous_action=action, prev_env_state=current_state)

    @staticmethod
    def sample(state: SarsaState,
               key: PRNGKey,
               epsilon: float,
               current_state: int) -> int:
        """
        Selects the next action using an ε-greedy policy.

        Parameters
        ----------
        state : SarsaState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        env_state: dict,
            status of the network eviroment
        epsilon : float
            Exploration rate.

        Returns
        -------
        int
            Selected action.
        """

        q_values = state.Q[current_state]


        max_q = (q_values == q_values.max()).astype(float)
        probs = (1 - epsilon) * max_q / jnp.sum(max_q) + epsilon / state.Q.shape[1]

        return jax.random.choice(key, state.Q.shape[1], p=probs)
