from transformers import pipeline, AutoTokenizer
import evaluate
from tqdm import tqdm

def load_transformer_pipeline(model_name="distilgpt2", device=0):
    """
    Загружает пайплайн для генерации текста на основе трансформерной модели
    """
    # Загрузка токенизатора для указанной модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Установка pad_token равным eos_token для корректной обработки паддинга
    tokenizer.pad_token = tokenizer.eos_token
    
    # Создание пайплайна для генерации текста
    generator = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device=device,
        pad_token_id=tokenizer.eos_token_id
    )

    return generator

def evaluate_transformer_rouge(generator, texts, prompt_ratio=0.75, **gen_kwargs):
    """
    Оценивает модель на списке текстов с помощью ROUGE метрики
    """
    predictions = []
    references = []
    tok = generator.tokenizer

    # Проход по всем текстам
    for text in tqdm(texts, desc="Оценка ROUGE (transformer)"):
        # Токенизация текста без добавления специальных токенов
        input_ids = tok.encode(text, add_special_tokens=False)

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

        # Декодирование промпта в текст для передачи в пайплайн
        prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)
        
        # Генерация продолжения текста
        out = generator(
            prompt_text,
            max_new_tokens=len(ref_ids),  
            return_full_text=False,      
            **gen_kwargs
        )

        # Извлечение сгенерированного текста
        generated_part = out[0]['generated_text']
        predictions.append(generated_part)
        references.append(tok.decode(ref_ids, skip_special_tokens=True))

    # Возвращаем пустой словарь, если нет предсказаний
    if not predictions:
        return {}
    
    # Загрузка и вычисление метрики
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)

    return results

def show_transformer_examples(generator, texts, num_examples=5, prompt_ratio=0.75, **gen_kwargs):
    """
    Демонстрирует примеры генерации текста с помощью трансформерной модели
    """
    tok = generator.tokenizer

    for i, text in enumerate(texts[:num_examples]):
        # Токенизация текста
        input_ids = tok.encode(text, add_special_tokens=False)
        
        # Разделение на промпт и эталонное продолжение
        split = int(len(input_ids) * prompt_ratio)
        prompt_ids = input_ids[:split] 
        ref_ids = input_ids[split:]  
        
        # Декодирование в текст
        prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)
        ref_text = tok.decode(ref_ids, skip_special_tokens=True)

        # Генерация продолжения
        out = generator(
            prompt_text,
            max_new_tokens=len(ref_ids),
            return_full_text=False,
            **gen_kwargs
        )

        gen_text = out[0]['generated_text']

        print(f"\nПример {i+1}:")
        print(f"Промпт: {prompt_text}")
        print(f"Ожидалось: {ref_text}")
        print(f"Сгенерировано: {gen_text}")