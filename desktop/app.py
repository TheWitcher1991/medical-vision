import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import torch

from infer import (
    load_model,
    predict,
    CLASS_MAP,
    get_llm_analysis,
    get_llm_comparison_analysis,
    generate_markdown_report,
    save_markdown_report,
)

# -----------------------------
# STYLE
# -----------------------------
BG = "#0f172a"
CARD = "#1e293b"
TEXT = "#f8fafc"
SUB = "#94a3b8"
BLUE = "#2563eb"
GREEN = "#16a34a"

# -----------------------------
# MODEL
# -----------------------------
model = load_model()

with open("medclsnet_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

class_names = config["class_names"]

selected_image = None

# -----------------------------
# ROOT
# -----------------------------
root = tk.Tk()
root.title("Medical Vision AI")
root.geometry("1100x760")
root.configure(bg=BG)

# -----------------------------
# MAIN LAYOUT
# -----------------------------
main = tk.Frame(root, bg=BG)
main.pack(fill="both", expand=True, padx=20, pady=20)

left = tk.Frame(main, bg=BG)
left.pack(side="left", fill="y", padx=(0, 15))

right = tk.Frame(main, bg=BG)
right.pack(side="right", fill="both", expand=True)

# -----------------------------
# LEFT PANEL
# -----------------------------
title = tk.Label(
    left,
    text="🧠 Medical Vision AI",
    font=("Segoe UI", 22, "bold"),
    bg=BG,
    fg=TEXT
)
title.pack(anchor="w", pady=(0, 5))

subtitle = tk.Label(
    left,
    text="Диагностика МРТ мозга",
    font=("Segoe UI", 11),
    bg=BG,
    fg=SUB
)
subtitle.pack(anchor="w", pady=(0, 20))

preview_card = tk.Frame(left, bg=CARD, width=420, height=320)
preview_card.pack_propagate(False)
preview_card.pack()

img_label = tk.Label(
    preview_card,
    text="Загрузите изображение",
    font=("Segoe UI", 12),
    bg=CARD,
    fg=SUB
)
img_label.pack(expand=True)

# -----------------------------
# BUTTONS UNDER IMAGE
# -----------------------------
btn_wrap = tk.Frame(left, bg=BG)
btn_wrap.pack(fill="x", pady=20)

# КНОПКА ЗАГРУЗКИ
# КНОПКА АНАЛИЗА ТЕПЕРЬ ВИДНА ВСЕГДА
# И НАХОДИТСЯ ПОД ФОТО

# -----------------------------
# RIGHT PANEL
# -----------------------------
result_card = tk.Frame(right, bg=CARD)
result_card.pack(fill="x")

result_title = tk.Label(
    result_card,
    text="Результат",
    font=("Segoe UI", 16, "bold"),
    bg=CARD,
    fg=TEXT
)
result_title.pack(anchor="w", padx=20, pady=(15, 5))

diagnosis_label = tk.Label(
    result_card,
    text="Ожидание анализа",
    font=("Segoe UI", 24, "bold"),
    bg=CARD,
    fg=BLUE
)
diagnosis_label.pack(anchor="w", padx=20)

confidence_label = tk.Label(
    result_card,
    text="",
    font=("Segoe UI", 11),
    bg=CARD,
    fg=SUB
)
confidence_label.pack(anchor="w", padx=20, pady=(5, 10))

progress = ttk.Progressbar(
    result_card,
    orient="horizontal",
    mode="determinate",
    length=500
)
progress.pack(anchor="w", padx=20, pady=(0, 20))

text_card = tk.Frame(right, bg=CARD)
text_card.pack(fill="both", expand=True, pady=(15, 0))

analysis_box = tk.Text(
    text_card,
    bg=CARD,
    fg=TEXT,
    insertbackground="white",
    relief="flat",
    wrap="word",
    font=("Segoe UI", 10)
)
analysis_box.pack(fill="both", expand=True, padx=15, pady=15)

# -----------------------------
# FUNCTIONS
# -----------------------------
def choose_image():
    global selected_image

    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg")]
    )

    if not path:
        return

    selected_image = path

    img = Image.open(path)
    img.thumbnail((400, 300))

    tk_img = ImageTk.PhotoImage(img)

    img_label.configure(image=tk_img, text="")
    img_label.image = tk_img


def run_ai():
    global selected_image

    if not selected_image:
        messagebox.showwarning("Ошибка", "Сначала выберите изображение")
        return

    diagnosis_label.config(text="Анализ...")
    root.update()

    pred, conf, probs = predict(selected_image, model, class_names)

    ru = CLASS_MAP.get(pred, pred)

    diagnosis_label.config(text=ru)
    confidence_label.config(
        text=f"Уверенность модели: {conf*100:.2f}%"
    )

    progress["value"] = conf * 100

    llm = get_llm_analysis(pred, conf, probs, selected_image)
    comp = get_llm_comparison_analysis(probs, conf, pred)

    analysis_box.delete("1.0", tk.END)
    analysis_box.insert(
        tk.END,
        llm + "\n\n--------------------\n\n" + comp
    )

    report = generate_markdown_report(
        selected_image,
        pred,
        ru,
        conf,
        probs,
        llm,
        comp
    )

    path = save_markdown_report(report, selected_image)

    analysis_box.insert(
        tk.END,
        f"\n\n✅ Отчёт сохранён:\n{path}"
    )


# -----------------------------
# BUTTONS CREATE LAST
# -----------------------------
btn1 = tk.Button(
    btn_wrap,
    text="📁 Загрузить снимок",
    command=choose_image,
    bg=BLUE,
    fg="white",
    font=("Segoe UI", 12, "bold"),
    relief="flat",
    padx=20,
    pady=12,
    width=18
)
btn1.pack(fill="x", pady=(0, 10))

btn2 = tk.Button(
    btn_wrap,
    text="🧠 АНАЛИЗИРОВАТЬ",
    command=run_ai,
    bg=GREEN,
    fg="white",
    font=("Segoe UI", 13, "bold"),
    relief="flat",
    padx=20,
    pady=14,
    width=18
)
btn2.pack(fill="x")

root.mainloop()