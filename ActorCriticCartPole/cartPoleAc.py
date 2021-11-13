#!/usr/bin/env python

"""Actor Critic on CartPole

Use Actor Critic methods to 'solve' the CartPole-v0 environment from Open AI.
This is basically a hello world of actor critic methods.
"""

import gym
import tqdm
import statistics
import collections
import numpy as np
import tensorflow as tf

from typing import Tuple, List
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Machine Epsilon - added to denomenators for stabilising arithemtic
_epsilon = np.finfo(np.float32).eps.item()
# The function used to calculate the loss of the critic.
_critic_loss_func = tf.keras.losses.Huber(
                        reduction = tf.keras.losses.Reduction.SUM)
_optimiser = tf.keras.optimizers.Adam(learning_rate = 0.001)
# Initialise the CartPole environment
_env = gym.make("CartPole-v0") 

class ActorCritic(tf.keras.Model):
    """Actor-Critic network with a two hidden layer.

    Actor and Critic have common two common hidden layes. Training the model
    takes the following form:
        - Run simulation and make agent collect data.
        - Compute expcteted return at each time step.
        - Compute the loss for the A2C model.
        - Compute gradient and update network params.
        - Repeat until the agent success criteria are reached or the episode
          has been cut short.
          
    Running the simulation and collecting data:
        1) Take the first state of the environment.
        2) Feed this state into the actor and critic. The actor proposes a
            policy function, whilst the critic proposes an estimated value
            function.
        3) An action is picked from the probability distribution provided by
            the actor.
        4) Perform the action on the environment to get the next state,
            reward for the current action.
        5) At this point the environment could terminate due to success, failure
            or max time steps being evaulated.
        6) Repeat steps 2 to 6 until environment is terminated.

    Computing the expected return compromises of:
        1) Take the rewards for each timestep from a single episode.
        2) Create a sequence where each term is equal to the sum of rewards from
            the previous time multiplied by a decaying factor plus the reward
            for the current time step.
            a(0) = reward(0)
            a(n+1) = d * a(n) + reward(n+1)
        3) Standardise for numerical stability.

    Computing Losses:
        Loss of the Actor:
            The actor returns a probability for each of the actions. This means
            we have to use a categorical loss. Hence, categorical crossentropy
            is used. The categorical cross-entropy value is multiplied by the
            advantage. The advantage is denoted as G - V: the difference between
            the expected return at a give state and the proposed value function.
            In other words we are seeing how much better the critic could have
            performed.
        Loss of the Critic:
            The critic returns a scalar value from its proposed value function.
            Hence, this is a regression problem. Huber loss is a good choice as
            it is not as heavily influence by outliers as the squared-error loss
            is, but is still differentiable everywhere.
        Combined Loss:
            This loss must compromise of the actor loss and the critic loss. A
            simple solution is to just sum the two together.
        
    Attributes:
        common_one: The first fully connected hidden layer, common to both
            actor and critic.
        common_two: The second fully connected hidden layer, common to both
            actor and critic.
        actor: The output layer representing the actor.
        critic: The output layer representing the critic.
    """

    def __init__(self,
                 num_actions: int):
        """Initialise."""
        super().__init__()

        self.common_one = layers.Dense(256, activation = "relu")
        self.common_two = layers.Dense(256, activation = "relu")

        # Actor provides us with the action to choose
        self.actor = layers.Dense(num_actions)
        # Critic provides us with an estimate for the value function 
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass of the network."""
        hidden_one = self.common_one(inputs)
        hidden_two = self.common_two(hidden_one)
        
        return self.actor(hidden_two), self.critic(hidden_two)
    
def env_step(
    action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the next state, reward, and done flag tuple, based on
    given action."""
    state, reward, done, _ = _env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.float32),
            np.array(done, np.float32))
    

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    """Wraps env_step to work on a tensor."""
    # Numpy function is used on tensors only.
    return tf.numpy_function(env_step,
                             [action],
                             [tf.float32, tf.float32, tf.float32])

def run_episode(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Executes one episode of the environment for training data."""

    # Initialise tensor arrays for probabilites, values and rewards
    action_probs = tf.TensorArray(dtype = tf.float32,
                                  size = 0,
                                  dynamic_size = True)
    critic_values = tf.TensorArray(dtype = tf.float32,
                                   size = 0,
                                   dynamic_size = True)
    rewards = tf.TensorArray(dtype = tf.float32,
                             size = 0,
                             dynamic_size = True)

    # Initial state is decided by environment, later states are influenced
    # by the model.
    initial_state_shape = initial_state.shape
    state = initial_state

    # Preform upto max_steps amount of steps in an episode. Each episode
    # collects a tuple of action, value and reward.
    for t in tf.range(max_steps):
        # Change the state to a batch with a batch size of 1.
        state = tf.expand_dims(state, 0)

        # Get actor probabilites and critic values.
        # Actor: policy function, Critic: value function
        action_logits_t, value = model(state)

        # Pick a next action from the action probability distibution.
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Get the next state decided from the environment.
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store critic values given.
        critic_values = critic_values.write(t, tf.squeeze(value))
        
        # Store log probs of chosen action.
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Store rewards given to the model.
        rewards = rewards.write(t, reward)

        # Exit loop if environment has finished. This can happend by
        # 'solving' the environment of perfoming terribly.
        if tf.cast(done, tf.bool):
            break
    # Stacking TensorArrays results in stacked Tensor
    action_probs = action_probs.stack()
    critic_values = critic_values.stack()
    rewards = rewards.stack()

    return action_probs, critic_values, rewards

def get_expected_return(rewards: tf.Tensor, gamma: float) -> tf.Tensor:
    """Calculate the expected return of a each timestep in an episode."""
    timesteps = tf.shape(rewards)[0]
    exp_returns = tf.TensorArray(dtype = tf.float32, size = timesteps)

    # Calculate discounted sum of rewards
    rewards = tf.cast(rewards[::-1], dtype = tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape

    for t in tf.range(timesteps):
        current_reward = rewards[t]
        discounted_sum = current_reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        exp_returns = exp_returns.write(t, discounted_sum)
    exp_returns = exp_returns.stack()[::-1]

    # Standardise the returns per episode
    mean = tf.math.reduce_mean(exp_returns)
    std = tf.math.reduce_std(exp_returns) + _epsilon
    exp_returns = (exp_returns - mean) / std

    return exp_returns

def get_loss(
             action_probs: tf.Tensor,
             values:  tf.Tensor,
             exp_returns: tf.Tensor) -> tf.Tensor:
    """Calculates the loss of the combined actor critic."""
    advantage = exp_returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = _critic_loss_func(values, exp_returns)
    
    return actor_loss + critic_loss            
    
@tf.function
def train(
          initial_state: tf.Tensor,
          model: tf.keras.Model,
          optimiser: tf.keras.optimizers.Optimizer,
          gamma: float,
          max_steps_per_episode: int) -> tf.Tensor:
    """Executes on training step on the model."""

    # Track variables for auto-differentiation.
    with tf.GradientTape() as gt:
        # Gather training data.
        action_probs, values, rewards = run_episode(
                                            initial_state,
                                            model,
                                            max_steps_per_episode)
        
        # Calculate expected returns given rewards and discount factor.
        exp_returns = get_expected_return(rewards, gamma)

        # Change format of training data.
        action_probs, values, exp_returns = [
            tf.expand_dims(x, 1) for x in [action_probs,
                                           values,
                                           exp_returns]]

        # Calculate loss of model given the outcome of the episode.
        loss = get_loss(action_probs, values, exp_returns)

    # Calculate gradient in the model using the gradient tape.
    gradients = gt.gradient(loss, model.trainable_variables)

    # Update parameters using optimiser.
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))

    # Sum the rewards.
    episode_rewards = tf.math.reduce_sum(rewards)

    return episode_rewards
    

def main():
    # Initialise the CartPole environment
    #_env = gym.make("CartPole-v0") 
    # Setting seed allows for reproducibility
    seed = 31419
    np.random.seed(seed)
    tf.random.set_seed(seed)
    _env.seed(seed)

    # Actor will be 2x32x32x2
    # Critic will be 2x32x32x1
    num_actions = _env.action_space.n     
    model = ActorCritic(num_actions)

    # Ensures the model has actually trained.
    min_episodes = 100
    
    # Ensures the model stops training at some point.
    max_episodes = 10000
    
    # This should be large enough for the model to explore properly, yet not
    # larger than necessary.
    max_steps_per_episode = 1000

    # Environment is 'solved' when avg reward is >= 195 for 100 episodes in
    # a row.
    reward_threshold = 195
    reward_avg = 0

    # Discount factor
    gamma = 0.9875

    # A queue of length up to min_episodes containing the rewards of most
    # recent episodes. 
    past_rewards: collections.deque = collections.deque(maxlen = min_episodes)

    with tqdm.trange(max_episodes) as training:
        for episodes in training:
            # Gets initial state from the environment
            initial_state = tf.constant(_env.reset(),
                                        dtype = tf.float32)

            # Get total reward from one episode during training
            episode_reward = int(train(initial_state,
                                       model,
                                       _optimiser,
                                       gamma,
                                       max_steps_per_episode))
            # Append new reward to the queue, if the queue is full, the
            # first item in the queue is removed prior to appending.
            past_rewards.append(episode_reward)

            # New mean is calculated
            reward_avg = statistics.mean(past_rewards)

            training.set_description(f'Episode {episodes}')
            training.set_postfix(episode_reward = episode_reward,
                                 reward_avg = reward_avg)

            # Stop training if the environment is 'solved'
            if reward_avg > reward_threshold and episodes >= min_episodes:
                break

    print(f'\nEnvironment solved in {episodes} episodes:')
    print(f'Average reward: {reward_avg:.2f}')
    # Train the model


if __name__ == "__main__":
    main()

    input("Exit...")
