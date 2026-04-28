from openai import OpenAI
from inference import CLASS_MAP


GROQ_API_KEY = "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"
llm_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
LLM_MODEL = "llama-3.3-70b-versatile"


def get_llm_analysis(pred_class: str, confidence: float, probabilities: dict) -> str:
    prob_details = "\n".join([f"  - {cls}: {prob*100:.1f}%" for cls, prob in probabilities.items()])
    
    clinical_context = {
        "tumor_glioma": "Глиома - первичная опухоль головного мозга. Требует нейрохирургического вмешательства.",
        "tumor_meningioma": "Менингиома - доброкачественная опухоль мозговых оболочек. Растет медленно, прогноз благоприятный.",
        "tumor_pituitary": "Опухоль гипофиза - может вызывать эндокринные нарушения.",
        "normal": "Снимок без признаков патологии. Регулярное наблюдение рекомендуется."
    }
    
    system_prompt = """Вы - опытный нейрорадиолог. Анализируйте результаты диагностики МРТ. Давайте профессиональные заключения на русском языке."""
    
    user_prompt = f"""Проведите анализ результатов диагностики МРТ:
    - Диагноз: {pred_class} ({CLASS_MAP.get(pred_class, pred_class)})
    - Уверенность: {confidence*100:.1f}%
    Вероятности: {prob_details}
    Клинический контекст: {clinical_context.get(pred_class, "Требуется клиническая оценка.")}
    Предоставьте: 1) Интерпретация 2) Рекомендации 3) Прогноз"""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.5,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка LLM: {str(e)}"


def get_recommendations(pred_class: str) -> str:
    recs = {
        "normal": "Плановое наблюдение 1 раз в год. Поддерживайте здоровый образ жизни.",
        "tumor_glioma": "Срочная консультация нейрохирурга. МРТ с контрастом. Возможна биопсия.",
        "tumor_meningioma": "Консультация нейрохирурга. При малых размерах - наблюдение.",
        "tumor_pituitary": "Консультации нейрохирурга и эндокринолога. Гормональный профиль."
    }
    return recs.get(pred_class, "Консультация специалиста.")