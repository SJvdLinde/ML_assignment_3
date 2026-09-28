import numpy as np

def grow_to_shape(state, new_shape):

    if state.shape == new_shape:
        return state
    padding = [(0, new - old) for old, new in zip(state.shape, new_shape)]
    assert all(extra >= 0 for _, extra in padding), "optimiser state can only grow"
    return np.pad(state, padding)

class Adam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.first_moment = {}
        self.second_moment = {}
        self.step_count = 0

    def step(self, params, grads):
        self.step_count += 1
        correction1 = 1.0 - self.beta1 ** self.step_count
        correction2 = 1.0 - self.beta2 ** self.step_count

        for name, grad in grads.items():
            m = grow_to_shape(self.first_moment.get(name, np.zeros_like(grad)), grad.shape)
            v = grow_to_shape(self.second_moment.get(name, np.zeros_like(grad)), grad.shape)

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
            self.first_moment[name] = m
            self.second_moment[name] = v

            m_hat = m / correction1
            v_hat = v / correction2
            params[name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        for name, grad in grads.items():
            params[name] -= self.learning_rate * grad

def make_optimiser(name, learning_rate):
    if name == "adam":
        return Adam(learning_rate)
    if name == "sgd":
        return SGD(learning_rate)
    raise ValueError(f"don't know optimiser '{name}'")
