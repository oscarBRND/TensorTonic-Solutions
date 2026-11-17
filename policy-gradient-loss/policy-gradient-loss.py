import numpy as np 


def policy_gradient_loss(log_probs: list, rewards: list, gamma: float) -> float:
    """
    Returns the REINFORCE loss with a mean-return baseline.
    """
    T = len(log_probs)
    G = [0.0] * T
    G[T - 1] = float(rewards[T-1])
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t+1]
    mean_g = np.sum(G)/T
    A = np.asarray([g - mean_g for g in G])
    return -(1/T)*np.sum(log_probs*A)