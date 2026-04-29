import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
from openai import OpenAI
from datetime import datetime

from model import MedClsNet

# Конфигурация
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LLM конфигурация
GROQ_API_KEY = "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"  # Вставьте ваш ключ
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
LLM_MODEL = "llama-3.3-70b-versatile"

CLASS_MAP = {
    "normal": "Норма",
    "tumor_glioma": "Глиома (опухоль)",
    "tumor_meningioma": "Менингиома (опухоль)",
    "tumor_pituitary": "Опухоль гипофиза",
}


CLASS_NAMES = ["normal", "tumor_glioma", "tumor_meningioma", "tumor_pituitary"]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model():
    """Загрузка обученной модели классификатора"""
    model = MedClsNet(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load("medclsnet.pth", map_location=DEVICE))
    model.eval()
    return model

def predict(image_path, model, class_names):
    """Классификация изображения"""
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        
        conf, idx = torch.max(probs, dim=1)
        all_probs = probs.cpu().numpy()[0]
        
        pred_class = class_names[idx.item()]
        confidence = conf.item()
        
        probabilities = {class_names[i]: float(all_probs[i]) for i in range(len(class_names))}

    return pred_class, confidence, probabilities

def get_llm_analysis(pred_class, confidence, probabilities, image_path):
    """Получение анализа от LLM"""
    
    prob_details = "\n".join([f"  - {cls}: {prob*100:.1f}%" for cls, prob in probabilities.items()])
    
    clinical_context = {
        "tumor_glioma": "Глиома - первичная опухоль головного мозга, возникающая из глиальных клеток. Требует нейрохирургического вмешательства и часто - адъювантной терапии.",
        "tumor_meningioma": "Менингиома - обычно доброкачественная опухоль мозговых оболочек. Растет медленно, прогноз часто благоприятный.",
        "tumor_pituitary": "Опухоль гипофиза - может вызывать эндокринные нарушения. Требует оценки гормонального статуса.",
        "normal": "Снимок без признаков патологии. Регулярное наблюдение рекомендуется."
    }
    
    system_prompt = """Вы - опытный нейрорадиолог и медицинский эксперт. 
    Анализируйте результаты компьютерной диагностики МРТ головного мозга.
    Давайте обоснованные, профессиональные заключения на русском языке.
    Будьте точны и информативны, но не пугайте пациента без необходимости."""
    
    user_prompt = f"""Проведите анализ результатов диагностики МРТ головного мозга.

    РЕЗУЛЬТАТЫ АВТОМАТИЧЕСКОГО АНАЛИЗА:
    - Диагноз: {pred_class} ({CLASS_MAP.get(pred_class, pred_class)})
    - Уверенность модели: {confidence*100:.1f}%
    
    Распределение вероятностей по классам:
    {prob_details}
    
    КЛИНИЧЕСКИЙ КОНТЕКСТ:
    {clinical_context.get(pred_class, "Требуется дополнительная клиническая оценка.")}
    
    ПОЖАЛУЙСТА, ПРЕДОСТАВЬТЕ:
    1. Интерпретацию результата (что означает этот диагноз)
    2. Рекомендации по дальнейшим действиям
    3. Какие дополнительные обследования могут потребоваться
    4. Прогноз и возможные варианты лечения (в общих чертах)
    
    Обратите внимание: результат основан на автоматическом анализе и требует подтверждения врачом.
    Ответ должен быть профессиональным, но понятным для пациента.
    Упомяните уровень уверенности модели в вашем анализе."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Ошибка при обращении к LLM: {str(e)}\n\nРекомендуется проконсультироваться с врачом."

def get_llm_comparison_analysis(probabilities, confidence, pred_class):
    """Альтернативный анализ: сравнение вариантов диагностики"""
    
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_probs[:3]
    
    comparison = "\n".join([f"{i+1}. {cls}: {prob*100:.1f}%" 
                           for i, (cls, prob) in enumerate(top_3)])
    
    system_prompt = "Вы - медицинский консультант, специализирующийся на дифференциальной диагностике."
    
    user_prompt = f"""Проведите дифференциальную диагностику на основе вероятностного распределения:

    Основной диагноз (уверенность {confidence*100:.1f}%): {pred_class}
    
    Альтернативные варианты:
    {comparison}
    
    Проанализируйте:
    1. Насколько значимо различие между этими диагнозами?
    2. Какие дополнительные признаки могли бы помочь уточнить диагноз?
    3. Рекомендации по дальнейшей диагностике для исключения ошибки.
    
    Ответ должен быть на русском языке, профессиональным и конструктивным."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Ошибка: {str(e)}"

def generate_markdown_report(image_path, pred_class, ru_label, confidence, probabilities, llm_analysis, comparison_analysis):
    """Генерация отчета в формате Markdown"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Создаем прогресс-бары для вероятностей
    prob_bars = ""
    for cls, prob in probabilities.items():
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        prob_bars += f"| {CLASS_MAP.get(cls, cls):20s} | {bar} | {prob*100:5.1f}% |\n"
    
    markdown_report = f"""# 🧠 Отчет по диагностике МРТ головного мозга

**Дата анализа:** {timestamp}  
**Файл изображения:** `{image_path}`

---

## 📊 Результаты автоматического анализа

| Параметр | Значение |
|----------|----------|
| **Диагноз** | **{ru_label}** |
| **Код** | `{pred_class}` |
| **Уверенность модели** | **{confidence*100:.2f}%** |

### Распределение вероятностей по классам

| Диагноз | Уверенность | Процент |
|---------|-------------|---------|
{prob_bars}

---

## 🤖 Анализ искусственного интеллекта

{llm_analysis}

---

## 🔄 Дифференциальная диагностика

{comparison_analysis}

---

## 📋 Рекомендации

### На основе анализа:

{generate_recommendations(pred_class, confidence)}

---

## ⚠️ Важное примечание

> **Данный анализ является вспомогательным и НЕ заменяет консультацию врача.**  
> Окончательный диагноз должен быть поставлен квалифицированным специалистом  
> на основе полного клинического обследования.

---

*Отчет сгенерирован автоматически системой диагностики Medical Vision*
"""
    
    return markdown_report

def generate_recommendations(pred_class, confidence):
    """Генерация рекомендаций на основе диагноза"""
    
    recommendations = {
        "normal": """
- **Плановое наблюдение:** Рекомендуется проходить профилактические осмотры 1 раз в год
- **Здоровый образ жизни:** Поддерживайте нормальное артериальное давление и уровень холестерина
- **При появлении симптомов:** Головные боли, головокружения, нарушения зрения - обратитесь к неврологу
""",
        "tumor_glioma": f"""
- **Срочная консультация:** Нейрохирурга и онколога (уверенность модели: {confidence*100:.1f}%)
- **Дополнительные исследования:** МРТ с контрастированием, возможно биопсия
- **Лечение:** Хирургическое удаление, лучевая терапия, химиотерапия по показаниям
- **Прогноз:** Зависит от степени злокачественности и своевременности лечения
""",
        "tumor_meningioma": f"""
- **Консультация:** Нейрохирурга (уверенность модели: {confidence*100:.1f}%)
- **Наблюдение:** При небольших размерах и отсутствии симптомов - динамическое наблюдение
- **Лечение:** Хирургическое удаление при симптомах или росте опухоли
- **Прогноз:** Обычно благоприятный при своевременном лечении
""",
        "tumor_pituitary": f"""
- **Консультации:** Нейрохирурга и эндокринолога (уверенность модели: {confidence*100:.1f}%)
- **Дополнительные исследования:** Гормональный профиль, МРТ гипофиза с контрастом
- **Лечение:** Медикаментозное, возможно хирургическое при аденомах
- **Прогноз:** Зависит от гормональной активности и размеров опухоли
"""
    }
    
    return recommendations.get(pred_class, """
- **Консультация специалиста:** Невролога или нейрохирурга
- **Дополнительная диагностика:** Уточняющие методы визуализации
- **Наблюдение:** Динамическое наблюдение для исключения прогрессирования
""")

def save_markdown_report(report, image_path):
    """Сохранение Markdown отчета в файл"""
    
    report_path = image_path.rsplit('.', 1)[0] + "_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path

def print_color_output(text, color='white'):
    """Цветной вывод в консоль (опционально)"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Диагностика МРТ головного мозга с LLM анализом"
    )

    parser.add_argument("--image", type=str, default="brain.jpg")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-comparison", action="store_true")

    # теперь md сохраняется ВСЕГДА
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    print_color_output("\n🔄 Загрузка модели...", "cyan")
    model = load_model()

    with open("medclsnet_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = config["class_names"]

    print_color_output("🔄 Выполняется анализ изображения...", "cyan")

    pred, conf, probs = predict(args.image, model, class_names)
    ru_label = CLASS_MAP.get(pred, pred)

    print("\n" + "=" * 70)
    print_color_output("# 🧠 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ", "bold")
    print("=" * 70)

    print(f"\nДиагноз: {ru_label}")
    print(f"Уверенность: {conf*100:.2f}%")

    llm_analysis = ""
    comparison = ""

    if not args.no_llm:
        llm_analysis = get_llm_analysis(pred, conf, probs, args.image)

        if not args.no_comparison:
            comparison = get_llm_comparison_analysis(probs, conf, pred)
    else:
        llm_analysis = "LLM анализ отключен."
        comparison = "Дифференциальный анализ отключен."

    # ---------------------------
    # СОХРАНЕНИЕ ВСЕГДА В MD
    # ---------------------------
    markdown_report = generate_markdown_report(
        args.image,
        pred,
        ru_label,
        conf,
        probs,
        llm_analysis,
        comparison
    )

    if args.output:
        report_path = args.output
    else:
        base = os.path.splitext(args.image)[0]
        report_path = f"{base}_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    print_color_output(f"\n✅ Markdown отчет сохранён: {report_path}", "green")

    print("\n" + "=" * 70)
    print_color_output("⚠️ Анализ не заменяет консультацию врача", "yellow")