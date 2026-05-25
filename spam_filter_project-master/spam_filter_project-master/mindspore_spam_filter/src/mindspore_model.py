from __future__ import annotations

import numpy as np


def require_mindspore():
    try:
        import mindspore as ms
        from mindspore import Tensor, nn, ops
    except ImportError as exc:
        raise RuntimeError(
            "MindSpore is not installed in this Python environment. "
            "Install a MindSpore build matching your CPU/GPU/Ascend environment, "
            "then rerun this script."
        ) from exc
    return ms, Tensor, nn, ops


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def build_cells(input_dim: int, hidden_dim: int, dropout: float):
    ms, Tensor, nn, ops = require_mindspore()

    class SpamNet(nn.Cell):
        def __init__(self):
            super().__init__()
            if hidden_dim <= 0:
                self.net = nn.Dense(input_dim, 1)
            else:
                try:
                    dropout_layer = nn.Dropout(p=dropout)
                except TypeError:
                    dropout_layer = nn.Dropout(keep_prob=1.0 - dropout)
                self.net = nn.SequentialCell(
                    nn.Dense(input_dim, hidden_dim),
                    nn.ReLU(),
                    dropout_layer,
                    nn.Dense(hidden_dim, 1),
                )

        def construct(self, x):
            return self.net(x).squeeze(-1)

    class WeightedBCEWithLogitsLoss(nn.Cell):
        def __init__(self, pos_weight: float):
            super().__init__()
            self.pos_weight = Tensor(np.asarray(pos_weight, dtype=np.float32), ms.float32)

        def construct(self, logits, labels):
            zeros = ops.zeros_like(logits)
            base = ops.maximum(logits, zeros) - logits * labels + ops.log(1.0 + ops.exp(-ops.abs(logits)))
            weights = labels * self.pos_weight + (1.0 - labels)
            return ops.mean(base * weights)

    return SpamNet, WeightedBCEWithLogitsLoss


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    pos_weight: float | None = None,
    device_target: str = "CPU",
) -> tuple[object, dict[str, list[float]], np.ndarray]:
    ms, Tensor, nn, ops = require_mindspore()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=device_target)
    x_train = x_train.astype("float32")
    y_train = y_train.astype("float32")
    x_valid = x_valid.astype("float32")
    y_valid = y_valid.astype("float32")
    if pos_weight is None:
        positives = max(float(y_train.sum()), 1.0)
        negatives = max(float(len(y_train) - y_train.sum()), 1.0)
        pos_weight = negatives / positives

    SpamNet, WeightedLoss = build_cells(x_train.shape[1], hidden_dim, dropout)
    model = SpamNet()
    loss_fn = WeightedLoss(float(pos_weight))
    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

    def forward_fn(x, y):
        logits = model(x)
        return loss_fn(logits, y)

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
    history = {"train_loss": [], "valid_loss": []}
    rng = np.random.default_rng(42)

    for _ in range(epochs):
        model.set_train(True)
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = Tensor(x_train[idx], ms.float32)
            yb = Tensor(y_train[idx], ms.float32)
            loss, grads = grad_fn(xb, yb)
            optimizer(grads)
            losses.append(float(loss.asnumpy()))

        model.set_train(False)
        valid_logits = model(Tensor(x_valid, ms.float32))
        valid_loss = loss_fn(valid_logits, Tensor(y_valid, ms.float32))
        history["train_loss"].append(float(np.mean(losses)))
        history["valid_loss"].append(float(valid_loss.asnumpy()))

    model.set_train(False)
    valid_logits = model(Tensor(x_valid, ms.float32)).asnumpy()
    return model, history, valid_logits


def predict_logits(model: object, x: np.ndarray, batch_size: int = 512, device_target: str = "CPU") -> np.ndarray:
    ms, Tensor, _, _ = require_mindspore()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=device_target)
    model.set_train(False)
    logits = []
    for start in range(0, len(x), batch_size):
        batch = Tensor(x[start : start + batch_size].astype("float32"), ms.float32)
        logits.append(model(batch).asnumpy())
    return np.concatenate(logits)
