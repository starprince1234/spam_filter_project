from __future__ import annotations

import numpy as np


def require_mindspore():
    try:
        import mindspore as ms
        from mindspore import Tensor, nn, ops
    except ImportError as exc:
        raise RuntimeError("MindSpore is not installed. Install the CPU/GPU/Ascend build for your environment, then rerun.") from exc
    return ms, Tensor, nn, ops


def build_cells(input_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
    ms, Tensor, nn, ops = require_mindspore()

    class SpamClassifier(nn.Cell):
        def __init__(self):
            super().__init__()
            if hidden_dim <= 0:
                self.net = nn.Dense(input_dim, 1)
            else:
                try:
                    dropout_layer = nn.Dropout(p=dropout)
                except TypeError:
                    dropout_layer = nn.Dropout(keep_prob=1.0 - dropout)
                self.net = nn.SequentialCell(nn.Dense(input_dim, hidden_dim), nn.ReLU(), dropout_layer, nn.Dense(hidden_dim, 1))

        def construct(self, features):
            return self.net(features).squeeze(-1)

    class WeightedBCEWithLogitsLoss(nn.Cell):
        def __init__(self, pos_weight: float):
            super().__init__()
            self.pos_weight = Tensor(np.asarray(pos_weight, dtype=np.float32), ms.float32)

        def construct(self, logits, labels):
            zeros = ops.zeros_like(logits)
            base = ops.maximum(logits, zeros) - logits * labels + ops.log(1.0 + ops.exp(-ops.abs(logits)))
            weights = labels * self.pos_weight + (1.0 - labels)
            return ops.mean(base * weights)

    return SpamClassifier, WeightedBCEWithLogitsLoss


def predict_logits(network, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
    ms, Tensor, _, _ = require_mindspore()
    network.set_train(False)
    logits = []
    for start in range(0, len(features), batch_size):
        batch = Tensor(features[start : start + batch_size].astype(np.float32), ms.float32)
        logits.append(network(batch).asnumpy())
    return np.concatenate(logits)
