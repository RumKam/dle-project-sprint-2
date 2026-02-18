import torch
import evaluate
from tqdm import tqdm
import random

def compute_rouge(model, texts, tokenizer, device, seq_len, prompt_ratio=0.75):
    """Вычисляет ROUGE-метрику, сравнивая сгенерированные 
    моделью продолжения с эталонными текстами"""

    model.eval()
    predictions = []
    references = []

    for text in tqdm(texts, desc="ROUGE"):
        # Токенизация входного текста
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Пропускаем слишком короткие тексты
        if len(input_ids) < 2:
            continue
        
        # Разделение на промпт и эталонное продолжение
        split = int(len(input_ids) * prompt_ratio)
        prompt_ids = input_ids[:split]
        ref_ids = input_ids[split:]
        
        # Пропускаем тексты без продолжения
        if len(ref_ids) == 0:
            continue
        
        # Генерация текстов
        with torch.no_grad():
            gen_ids = model.generate(prompt_ids, max_length=split + len(ref_ids), device=device, seq_len=seq_len)
        
        # Извлечение сгенерированной части
        gen_part = gen_ids[split:]

        # Декодирование в текст для сравнения
        pred_text = tokenizer.decode(gen_part, skip_special_tokens=True)
        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

        predictions.append(pred_text)
        references.append(ref_text)

    if not predictions:
        return {}
    
    rouge = evaluate.load("rouge")

    return rouge.compute(predictions=predictions, references=references)

def show_examples(model, texts, tokenizer, device, seq_len, num_examples=3, prompt_ratio=0.75):
    """Демонстрирует примеры генерации текста моделью на основе заданных промптов"""

    model.eval()
    # Случайный выбор индексов для демонстрации
    indices = random.sample(range(len(texts)), min(num_examples, len(texts)))

    for idx in indices:
        text = texts[idx]
        # Токенизация выбранного текста
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Пропускаем слишком короткие
        if len(input_ids) < 2:
            continue

        # Разделение на промпт и эталонное продолжение
        split = int(len(input_ids) * prompt_ratio)
        prompt_ids = input_ids[:split]    
        ref_ids = input_ids[split:]    

        # Пропускаем тексты без продолжения
        if len(ref_ids) == 0:
            continue

        # Генерация продолжения текста
        with torch.no_grad():
            gen_ids = model.generate(prompt_ids, max_length=split + len(ref_ids), device=device, seq_len=seq_len)

        # Извлечение сгенерированной части
        gen_part = gen_ids[split:]
        
        # Декодирование всех частей для отображения
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
        pred_text = tokenizer.decode(gen_part, skip_special_tokens=True)
        
        print("\n" + "="*50)
        print(f"Промпт: {prompt_text}")
        print(f"Ожидалось: {ref_text}")
        print(f"Сгенерировано: {pred_text}")