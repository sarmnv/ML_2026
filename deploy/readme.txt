=== Iris Classifier API ===

--- Запуск ноутбуков (по порядку) ---

0_train_models.ipynb    — обучение RF, CatBoost, IrisNet (native)
1_test_models.ipynb     — тестирование native моделей на тестовой выборке
2_convert_to_onnx.ipynb — конвертация всех трёх моделей в ONNX
3_test_onnx.ipynb       — тестирование ONNX моделей через onnxruntime
4_drift_monitor.ipynb   — генерация current.csv с искусственным сдвигом
                          + отчёт Evidently (Data + Target drift) → drift_report.html

--- Запуск FastAPI сервера ---

python3 app/main.py
Сервер доступен на http://0.0.0.0:8000

Эндпоинты:
  GET  /health               — статус + список загруженных моделей
  GET  /models               — список доступных моделей
  POST /predict              — предсказание (IrisFeatures в JSON body)
       ?model=iris_rf        — выбор модели (iris_rf / iris_catboost / iris_nn)
  POST /predict/batch        — пакетное предсказание (list of IrisFeatures)
  GET  /drift                — дрифт данных (по накопленным запросам к /predict)
                               признаки: K-S p_value, stattest_name, drift_detected
  GET  /metrics              — метрики Prometheus

Переменные окружения (опционально):
  DEFAULT_MODEL=iris_rf     — модель по умолчанию
  HOST=0.0.0.0              — хост
  PORT=8000                 — порт

Пример запроса:
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

  curl http://localhost:8000/drift

--- Smoke-тесты API ---

python3 app/test_api.py
(требуется запущенный сервер на localhost:8000)

--- Конвертация в ONNX (скрипты) ---

python3 convert_to_onnx.py
python3 test_onnx.py

--- Скрипты тестирования ---

python3 test_models.py   — тестирование native моделей
python3 test_onnx.py     — тестирование ONNX моделей

--- Мониторинг дрифта ---

# Offline (ноутбук): генерирует current.csv со сдвигом и HTML-отчёт
jupyter notebook 4_drift_monitor.ipynb

# Online (API): накапливает признаки через /predict, дрифт по запросу /drift
python3 app/main.py
curl http://localhost:8000/drift

--- Установка зависимостей ---

pip install -r requirements.txt
