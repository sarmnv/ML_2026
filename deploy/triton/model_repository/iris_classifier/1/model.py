"""
Triton Python backend for Iris classifier.

Зачем Python backend, а не нативный ONNX:
  Три ONNX-модели (RF, CatBoost, NN) имеют разный формат выхода:
    - RF:        1D [class_id]                  — нет вероятностей
    - CatBoost:  2D [N, 3] probabilities        — OK
    - NN:        2D [N, 3] logits               — нет softmax
  Python backend нормализует их в единый формат: class_id + class_name + probabilities.

Альтернатива (без Python backend):
  Переконвертировать ONNX с правильным выходом:
    - RF:  skl2onnx с output={'label': ..., 'probabilities': ...}
    - NN:  torch.onnx.export с nn.Softmax в графе
  Тогда можно использовать 3 отдельных ONNX-модели без Python backend.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import triton_python_backend_utils as pb_utils

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class TritonPythonModel:
    def initialize(self, args):
        models_dir = Path(__file__).parent

        self.sessions: dict[str, ort.InferenceSession] = {}
        for name in ("iris_rf", "iris_catboost", "iris_nn"):
            path = models_dir / f"{name}.onnx"
            if path.exists():
                self.sessions[name] = ort.InferenceSession(str(path))
                print(f"[TritonPythonModel] Loaded {name}")
            else:
                print(f"[TritonPythonModel] WARNING: {path} not found")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input")
            model_name_tensor = pb_utils.get_input_tensor_by_name(request, "model_name")

            features = input_tensor.as_numpy()
            model_name = model_name_tensor.as_numpy()[0][0].decode()

            sess = self.sessions.get(model_name)
            if sess is None:
                raise pb_utils.TritonModelException(
                    f"Unknown model '{model_name}'. Available: {list(self.sessions.keys())}"
                )

            raw = sess.run(None, {sess.get_inputs()[0].name: features.astype(np.float32)})[0]

            # нормализация к probabilities [N, 3]
            if raw.ndim == 1:
                # RF: 1D class IDs
                probs = np.zeros((len(raw), 3), dtype=np.float32)
                probs[np.arange(len(raw)), raw.astype(int)] = 1.0
            elif raw.shape[1] == 3 and abs(raw.sum(axis=1).mean() - 1.0) > 0.01:
                # NN: logits → softmax
                probs = softmax(raw)
            else:
                # CatBoost: already probabilities
                probs = raw.astype(np.float32)

            class_ids = probs.argmax(axis=1).astype(np.int64)
            class_names = np.array(
                [[CLASS_NAMES[cid]] for cid in class_ids], dtype=object
            )

            responses.append(
                pb_utils.InferenceResponse(
                    [
                        pb_utils.Tensor("class_id", class_ids.reshape(-1, 1)),
                        pb_utils.Tensor("class_name", class_names),
                        pb_utils.Tensor("probabilities", probs),
                    ]
                )
            )

        return responses

    def finalize(self):
        pass
