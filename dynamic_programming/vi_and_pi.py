### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [0, nS-1] and actions in [0, nA-1],
        P[state][action] is a tuple of the form
        (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)
    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    i = 0
    while True:
        prev_value_function = value_function
        for s in range(nS):   
            state_val = 0.0
            for output in P[s][policy[s]]:
                probability, nextstate, reward, terminal = output
                cur_val = reward
                if not terminal:
                    cur_val += gamma*prev_value_function[nextstate]
                cur_val *= probability
                state_val += cur_val

            value_function[s] = state_val

        if np.max(np.absolute(np.subtract(value_function, prev_value_function))) < tol and i >= 500:
            break
        i += 1
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    for s in range(nS):
        action_vals = []

        for a in range(nA):
            
            action_val = 0.0
            for output in P[s][a]:
                probability, nextstate, reward, terminal = output
                cur_val = reward
                if not terminal:
                    cur_val += gamma * value_from_policy[nextstate]
                cur_val *= probability
                action_val += cur_val
            action_vals.append(action_val)
        
        optim_indices = np.argwhere(action_vals == np.max(action_vals)).flatten().tolist()
        optim_action_index = np.random.choice(optim_indices)
        new_policy[s] = optim_action_index
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    while True:
        
        # evaluation
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)

        # improvement
        prev_policy = policy
        policy = policy_improvement(P, nS, nA, value_function, policy, gamma)

        if np.max(np.absolute(np.subtract(policy, prev_policy))) < tol:
            break
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    i = 0
    while True:
        prev_value_function = value_function
        for s in range(nS):
            action_vals = []

            # for a single state, loop through possible actions
            for a in range(nA):
                sum_val = 0.0
                for output in P[s][a]:
                    probability, nextstate, reward, terminal = output
                    cur_val = reward
                    if not terminal:
                        cur_val += gamma*prev_value_function[nextstate]
                    cur_val *= probability
                    sum_val += cur_val
                action_vals.append(sum_val)
            # for a single state, fetch the best action
            optim_indices = np.argwhere(action_vals==np.max(action_vals)).flatten().tolist()
            optim_action_index = np.random.choice(optim_indices)
            optim_action_val = action_vals[optim_action_index]

            # update procedure for this single state
            value_function[s] = optim_action_val
            policy[s] = optim_action_index

        if np.max(np.absolute(np.subtract(value_function, prev_value_function))) < tol and i >= 1000:
            break
        i += 1
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function, policy

def render_single(env, policy, max_steps=100, show_rendering=True):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        if show_rendering:
            env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    if show_rendering:
        env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

def evaluate(env, policy, max_steps=100, max_episodes=32):
    """
    This function does not need to be modified,
    evaluates your policy over multiple episodes.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_rewards = []
    dones = []
    for _ in range(max_episodes):
        episode_reward = 0
        ob = env.reset()
        for t in range(max_steps):
            a = policy[ob]
            ob, rew, done, _ = env.step(a)
            episode_reward += rew
            if done:
                break

        episode_rewards.append(episode_reward)
        dones.append(done)

    episode_rewards = np.array(episode_rewards).mean()
    success = np.array(dones).mean()

    print(f"> Average reward over {max_episodes} episodes:\t\t\t {episode_rewards}")
    print(f"> Percentage of episodes goal reached:\t\t\t {success * 100:.0f}%")
