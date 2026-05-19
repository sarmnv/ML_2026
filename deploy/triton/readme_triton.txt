=== Triton Inference Server ===

Зачем это нужно?
  Triton Inference Server (NVIDIA) — production-grade сервинг моделей
  с поддержкой множества фреймворков, dynamic batching, GPU, model versioning.

┌─────────────────────────────────────────┐
│         Triton Inference Server         │
├─────────────────────────────────────────┤
│ ✓ Dynamic batching (автоматическое)     │
│ ✓ Model versioning & A/B тестирование   │
│ ✓ Поддержка множества фреймворков       │
│ ✓ GPU/CPU автоматическое распределение  │
│ ✓ Prometheus-метрики «из коробки»       │
│ ✓ Health checks (ready/live)            │
│ ✓ Конкурентное выполнение запросов      │
└─────────────────────────────────────────┘


✅ Выбирайте Triton, если:
• У вас несколько моделей разных фреймворков в одном сервисе
• Нужен dynamic batching для повышения throughput
• Требуется быстрое переключение версий моделей (A/B тесты)
• Важен низкий latency при высокой нагрузке
• Используете GPU и хотите максимизировать утилизацию

❌ Рассмотрите альтернативы, если:
• У вас одна модель на чистом PyTorch → TorchServe проще
• Вы в экосистеме TensorFlow → TF Serving нативнее
• У вас уже есть Kubernetes-инфраструктура → Seldon может интегрироваться лучше


В нашем проекте Python backend используется, потому что три ONNX-модели
имеют разный формат выхода (RF → 1D class IDs, NN → logits без softmax).
Python backend нормализует их в единый формат.

Альтернатива (без Python backend):
  Переконвертировать ONNX с правильным выходом:
    - RF:  skl2onnx --output 'label probabilities'
    - NN:  torch.onnx.export с nn.Softmax в графе
  Тогда можно использовать 3 отдельных ONNX-модели напрямую.

--- Структура ---

triton/
├── docker-compose.yml                    # запуск Triton
├── model_repository/
│   └── iris_classifier/
│       ├── config.pbtxt                  # метаданные модели
│       └── 1/
│           ├── model.py                  # Python backend
│           ├── iris_rf.onnx
│           ├── iris_catboost.onnx
│           └── iris_nn.onnx
├── triton_client.py                      # тестовый клиент
└── readme_triton.txt                     # этот файл

--- Запуск ---

1. Установите Docker: https://docs.docker.com/engine/install/

2. Скопируйте ONNX модели (или создайте symlink):
     cp models/iris_*.onnx triton/model_repository/iris_classifier/1/

3. Запустите Triton:
     cd triton
     docker compose up

   Дождитесь в логах строки "iris_classifier: ... is ready".

4. В другом терминале запустите клиент:
     python triton/triton_client.py

   (Требуется tritonclient: pip install tritonclient[http])

--- Эндпоинты Triton ---

  GET  http://localhost:8000/v2/health/ready   — готовность сервера
  GET  http://localhost:8000/v2/models/        — список моделей
  POST http://localhost:8000/v2/models/iris_classifier/infer — инференс

Пример запроса (curl):

  curl -X POST http://localhost:8000/v2/models/iris_classifier/infer \
    -H "Content-Type: application/json" \
    -d '{
      "inputs": [
        {"name": "input", "shape": [1, 4], "datatype": "FP32", "data": [[5.1, 3.5, 1.4, 0.2]]},
        {"name": "model_name", "shape": [1, 1], "datatype": "BYTES", "data": [["iris_rf"]]}
      ]
    }'

--- Переменные окружения клиента ---

  TRITON_URL=localhost:8000          — адрес Triton сервера
  TRITON_MODEL=iris_classifier       — имя модели
  TRITON_TIMEOUT=5.0                 — таймаут запроса (сек)
  BATCH_SIZE=8                       — размер батча
  TRITON_MODEL_NAMES='["iris_rf"]'   — список моделей (JSON)

--- Проблемные места (для обсуждения) ---

  №  Проблема                Решение                          Строка
  1  Хардкод имени модели   os.getenv("TRITON_MODEL", ...)    12
  2  Нет обработки ошибок   try/except вокруг infer           82-91
  3  Нет таймаутов          request_timeout=TRITON_TIMEOUT    73
  4  Без батчинга           сравнение single vs batch         82-110
  5  Нет логгирования       logging.basicConfig + logger      30-34
  6  Модели в коде клиента  get_model_repository_index()      133-138

--- Bенчмарк (single-sample vs batch) ---

  Клиент автоматически сравнивает:
    - Single-sample:  avg latency + p99 latency
    - Batch (size=8): total time + time per sample

  Это иллюстрирует выгоду dynamic batching, который Triton делает
  на стороне сервера при высокой нагрузке.

--- Советы ---

- Для GPU: установите NVIDIA Container Toolkit и добавьте в docker-compose.yml:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

- Метрики Prometheus доступны на http://localhost:8002/metrics

- Для детального лога: tritonserver --log-verbose=1 (уже в docker-compose.yml)
