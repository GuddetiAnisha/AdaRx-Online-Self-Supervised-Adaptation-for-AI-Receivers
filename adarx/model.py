import numpy as np

def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class NeuralReceiver:
    """Two-layer MLP classifier with explicit NumPy backpropagation."""
    def __init__(self, seed=0, hidden=24):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.25, (4, hidden)); self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 0.25, (hidden, 4)); self.b2 = np.zeros(4)
        self.anchor = self.state()

    def state(self):
        return tuple(x.copy() for x in (self.w1, self.b1, self.w2, self.b2))

    def load(self, state):
        self.w1, self.b1, self.w2, self.b2 = (x.copy() for x in state)

    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        return h, softmax(h @ self.w2 + self.b2)

    def predict(self, x):
        return self.forward(x)[1].argmax(axis=1)

    def update(self, x, y, lr=0.02, steps=1, anchor_strength=0.0, weights=None):
        if len(x) == 0: return 0.0
        y = np.asarray(y, dtype=int)
        weights = np.ones(len(x)) if weights is None else np.asarray(weights)
        last = 0.0
        for _ in range(steps):
            h, p = self.forward(x)
            last = float(-np.mean(weights * np.log(p[np.arange(len(y)), y] + 1e-9)))
            grad = p
            grad[np.arange(len(y)), y] -= 1
            grad *= (weights / max(1, len(y)))[:, None]
            gw2 = h.T @ grad + anchor_strength * (self.w2 - self.anchor[2])
            gb2 = grad.sum(0) + anchor_strength * (self.b2 - self.anchor[3])
            gh = (grad @ self.w2.T) * (1 - h * h)
            gw1 = x.T @ gh + anchor_strength * (self.w1 - self.anchor[0])
            gb1 = gh.sum(0) + anchor_strength * (self.b1 - self.anchor[1])
            self.w1 -= lr * gw1; self.b1 -= lr * gb1
            self.w2 -= lr * gw2; self.b2 -= lr * gb2
        return last

    def drift(self):
        now = self.state()
        return float(np.sqrt(sum(np.sum((a-b)**2) for a,b in zip(now, self.anchor))))

