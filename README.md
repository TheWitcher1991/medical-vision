# 🧠 Medical Vision AI

Система диагностики МРТ головного мозга с использованием глубокого обучения и LLM-анализа.

## Возможности

- **Классификация опухолей**: Норма, Глиома, Менингиома, Опухоль гипофиза
- **LLM-анализ**: Интерпретация результатов нейросети через Groq API (Llama 3.3)
- **Рекомендации**: Автоматические рекомендации по дальнейшим действиям
- **REST API**: FastAPI бэкенд для интеграции

## Архитектура

```
medical-vision/
├── backend/              # FastAPI сервер
│   ├── main.py           # API эндпоинты
│   ├── model.py         # MedClsNet архитектура
│   ├── inference.py     # Инференс модели
│   └── llm.py          # LLM клиент
├── frontend/             # Next.js приложение
│   ├── src/app/        # Страницы
│   └── components/     # UI компоненты
└── medclsnet.pth      # Обученная модель
```

## Технологии

- **ML**: PyTorch, ResNet18
- **Backend**: FastAPI, Uvicorn
- **LLM**: Groq API (Llama 3.3 70B)
- **Frontend**: Next.js 14, React, shadcn/ui, Tailwind CSS

## Установка

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

API доступно на `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Приложение доступно на `http://localhost:3000`

## API Endpoints

| Метод | Путь | Описание |
|-------|-----|-----------|
| GET | `/health` | Проверка здоровья |
| GET | `/classes` | Доступные классы |
| POST | `/diagnose` | Диагностика изображения |

## Пример запроса

```bash
curl -X POST http://localhost:8000/diagnose \
  -F "file=@brain.jpg"
```

## Ответ

```json
{
  "diagnosis": "tumor_meningioma",
  "diagnosis_ru": "Менингиома (опухоль)",
  "confidence": 0.786,
  "probabilities": {
    "normal": 0.212,
    "tumor_glioma": 0.002,
    "tumor_meningioma": 0.786,
    "tumor_pituitary": 0.0
  },
  "timestamp": "2026-04-29 00:00:00",
  "llm_analysis": "...",
  "recommendations": "Консультация нейрохирурга..."
}
```

## Классы диагностики

| Код | Название | Описание |
|-----|----------|----------|
| `normal` | Норма | Без патологий |
| `tumor_glioma` | Глиома | Первичная опухоль мозга |
| `tumor_meningioma` | Менингиома | Опухоль оболочек |
| `tumor_pituitary` | Опухоль гипофиза | Аденома гипофиза |

## Требования

- Python 3.10+
- Node.js 18+
- CUDA (опционально для GPU инференса)

## Лицензия

MIT