import numpy as np

def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))

ACTIVATIONS = {
    "tanh": (np.tanh, lambda z, a: 1.0 - a ** 2),
    "sigmoid": (sigmoid, lambda z, a: a * (1.0 - a)),
    "relu": (lambda z: np.maximum(z, 0.0), lambda z, a: (z > 0).astype(float)),
}

def softmax(output_in):
    shifted = output_in - output_in.max(axis=1, keepdims=True)
    exps = np.exp(shifted)

    return exps / exps.sum(axis=1, keepdims=True)

def one_hot(y, n_classes):
    targets = np.zeros((len(y), n_classes))
    targets[np.arange(len(y)), y] = 1.0

    return targets

class FeedForwardNet:

    def __init__(self, n_inputs, n_hidden, n_classes, rng, activation="tanh", direct=False):
        if n_hidden == 0 and not direct:
            raise ValueError("0 hidden neurons only works with direct connections")

        self.rng = rng
        self.activation = activation
        self.act_fn, self.act_derivative = ACTIVATIONS[activation]
        self.direct = direct

        self.params = {
            "W1": self.init_weights(n_inputs, n_hidden),
            "b1": np.zeros(n_hidden),
            "W2": self.init_weights(n_hidden, n_classes),
            "b2": np.zeros(n_classes),
        }
        if direct:
            self.params["W_direct"] = self.init_weights(n_inputs, n_classes)

    @property
    def n_inputs(self):
        return self.params["W1"].shape[0]

    @property
    def n_hidden(self):
        return self.params["W1"].shape[1]

    @property
    def n_classes(self):
        return self.params["b2"].shape[0]

    def n_weights(self):
        return sum(p.size for p in self.params.values())

    def init_weights(self, fan_in, fan_out, shape=None):

        if shape is None:
            shape = (fan_in, fan_out)
        if fan_in == 0:
            return np.zeros(shape)
        if self.activation == "relu":
            limit = np.sqrt(6.0 / fan_in)
        else:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, size=shape)

    def forward(self, X):
        hidden_in = X @ self.params["W1"] + self.params["b1"]
        hidden_out = self.act_fn(hidden_in)
        output_in = hidden_out @ self.params["W2"] + self.params["b2"]
        if self.direct:
            output_in = output_in + X @ self.params["W_direct"]
        probs = softmax(output_in)

        cache = (X, hidden_in, hidden_out)
        return probs, cache

    def predict_proba(self, X):
        probs, _ = self.forward(X)
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def cross_entropy(self, probs, y):
        prob_of_true_class = probs[np.arange(len(y)), y]
        return -np.mean(np.log(prob_of_true_class + 1e-12))

    def loss(self, X, y):
        probs, _ = self.forward(X)
        return self.cross_entropy(probs, y)

    def loss_and_gradients(self, X, y):
        probs, (X, hidden_in, hidden_out) = self.forward(X)
        batch_size = X.shape[0]
        loss = self.cross_entropy(probs, y)

        # softmax + cross-entropy together give this gradient
        d_output_in = (probs - one_hot(y, self.n_classes)) / batch_size

        grads = {}
        grads["W2"] = hidden_out.T @ d_output_in
        grads["b2"] = d_output_in.sum(axis=0)
        if self.direct:
            grads["W_direct"] = X.T @ d_output_in

        # push the error back though W2 and the activation
        d_hidden_out = d_output_in @ self.params["W2"].T
        d_hidden_in = d_hidden_out * self.act_derivative(hidden_in, hidden_out)

        grads["W1"] = X.T @ d_hidden_in
        grads["b1"] = d_hidden_in.sum(axis=0)
        return loss, grads

    # incrementing part

    def add_hidden_neuron(self):
        n_inputs, new_hidden, n_classes = self.n_inputs, self.n_hidden + 1, self.n_classes

        # random input weights
        new_in_weights = self.init_weights(n_inputs, new_hidden, shape=(n_inputs, 1))
        new_out_weights = np.zeros((1, n_classes))

        self.params["W1"] = np.hstack([self.params["W1"], new_in_weights])
        self.params["b1"] = np.append(self.params["b1"], 0.0)
        self.params["W2"] = np.vstack([self.params["W2"], new_out_weights])

    def add_output_neuron(self):

        self.params["W2"] = np.hstack([self.params["W2"], np.zeros((self.n_hidden, 1))])
        self.params["b2"] = np.append(self.params["b2"], 0.0)
        if self.direct:
            self.params["W_direct"] = np.hstack([self.params["W_direct"], np.zeros((self.n_inputs, 1))])

    def get_weights(self):
        return {name: values.copy() for name, values in self.params.items()}

    def set_weights(self, saved_weights):
        self.params = {name: values.copy() for name, values in saved_weights.items()}
