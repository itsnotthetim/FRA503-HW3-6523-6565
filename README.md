# FRA503-HW3: Cart Pole [ HW3 ]
Similar to the previous homework, this assignment focuses on the **Stabilizing Cart-Pole Task**, but using function approximation-based RL approaches instead of table-based RL approaches.

Additionally, as in the previous homework, the `CartPole` extension repository includes configurations for the **Swing-up Cart-Pole Task** as an optional resource for students seeking a more challenging task.

## Learning Objectives:
1. Understand how **function approximation** works and how to implement it.

2. Understand how **policy-based RL** works and how to implement it.

3. Understand how advanced RL algorithms balance exploration and exploitation.

4. Be able to differentiate RL algorithms based on stochastic or deterministic policies, as well as value-based, policy-based, or Actor-Critic approaches. 

5. Gain insight into different reinforcement learning algorithms, including Linear Q-Learning, Deep Q-Network (DQN), the REINFORCE algorithm, and the Actor-Critic algorithm. Analyze their strengths and weaknesses.


## Part 1: Understanding the Algorithm
 
 ### Linear Q-Learning
 Linear Q-Learning is a `value-based` method that approximates the action-value function $ Q(s, a)$ in every state and action, and store the Q-value in table as a linear combination of feature vector $ \phi(s, a) $. The policy is `deterministic` and select an optimal action by maximizing the action-value function illustrated as below 

 $$ a^* = \arg\max_a Q(s, a) $$

 and an $\epsilon$-greedy variant thereof the exploration because it's the linear approximation. The **observation space** is `continuous` (joint limit in radian) while the *action space* is `discrete` for maximizing step. The exploration and exploitation is balanced by adjusting an $\epsilon$ decayed in **$\epsilon$-greedy policy**.

 ### Deep Q-Network (DQN)
 DQN is the Q-Learning (`value-based`) that use neural network to approximates the action-value function $ Q(s, a)$ instead of storing all the Q-value in table. The policy is `deterministic` greedy for exploitation; $\epsilon$-greedy bahaviour during learning and at the test time the agent select an optimal action by maximizing the action-value function illustrated as below 
 $$ a^* = \arg\max_a Q(s, a) $$

while training injects stochastic exploration by taking a random action with probability $\epsilon$

The *observation space *is `continuous` which high/low-dimensional features are also work. The *action space* is only `discrete` which produces one scalar value per action. And the **balancing exploration and exploitation** with three components consist of **$\epsilon$-greedy decayed** from a large value to a small floor to focus on exploration and exploitation, **experience replay buffer** by storing past transition $(s, a, r, s')$ to re‑use rare but informative experiences, improving both exploration coverage and data efficiency. and **target network** to stabilize learning by copied set of parameter provides a slowly changing boostrap target  so that the greedy policy exploits estimates rather than oscillating ones.

### REINFORCE (Monte Carlo Policy Gradient)
REINFORCE is a policy‑based Monte Carlo algorithm that directly optimizes a stochastic policy $ \pi(a \mid s; \theta)$ by performing gradient ascent on the expected return:

$$\nabla_{\theta} J(\theta) = \mathbb{E} \left[ \nabla_{\theta} \log \pi(a \mid s; \theta) G_t \right]
$$

which 
- $\nabla_{\theta} J(\theta)$: radient of the expected return $J(\theta)$ with respect to policy parameters $\theta$

- $\mathbb{E}[]$: Expectation over the distribution induced by the policy

- $\log \pi(a \mid s; \theta)$: Log-probability of taking action $q$ under policy $\pi$

- $G_t$: Return (e.g., cumulative discounted reward) starting from time $t$

Learning policy is `stochastic` with supports both `discrete and continuous` in action spaces by choosing an appropriate distribution (softmax of gaussian) and can be `continuous and discrete` observation spaces. Exploration is sampling from the stochastic policy supplies intrinsic exploration; entropy bonus often added to delay premature exploitation  

### Advantage Actor‑Critic (A2C)
Actor-Critic method is the `combination between value-based and policy-based`. The actor learns a `stochastic` policy  $\log \pi(a \mid s; \theta)$ (softmax for discrete or Gaussian for continuous actions),  while the critic estimates a value function $V(s; w)$

$$A(s, a) = r + \gamma V(s'; w) - V(s; w) $$

It uses the advantage reduces variance in policy gradient update. This approach handles `continuous or discrete` observation spaces and can applies both `continuous or discrete` in action spaces. Exploration comes from the randomness in policy $\pi$ often boosted by `entropy`. Exploitation is guided by the `critic`, helping the actor choose high-reward actions.  