from transformers import pipeline, AutoTokenizer
import evaluate
from tqdm import tqdm

def load_transformer_pipeline(model_name="distilgpt2", device=-1):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
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
    Оценивает модель на списке текстов
    """
    predictions = []
    references = []
    tok = generator.tokenizer

    for text in tqdm(texts, desc="Оценка ROUGE (transformer)"):

        input_ids = tok.encode(text, add_special_tokens=False)

        if len(input_ids) < 2:
            continue

        split = int(len(input_ids) * prompt_ratio)
        prompt_ids = input_ids[:split]
        ref_ids = input_ids[split:]

        if len(ref_ids) == 0:
            continue

        prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)
        out = generator(
            prompt_text,
            max_new_tokens=len(ref_ids),
            return_full_text=False,
            **gen_kwargs
        )

        generated_part = out[0]['generated_text']
        predictions.append(generated_part)
        references.append(tok.decode(ref_ids, skip_special_tokens=True))

    if not predictions:
        return {}
    
    rouge = evaluate.load("rouge")
    
    results = rouge.compute(predictions=predictions, references=references)

    return results

def show_transformer_examples(generator, texts, num_examples=5, prompt_ratio=0.75, **gen_kwargs):

    tok = generator.tokenizer

    for i, text in enumerate(texts[:num_examples]):
        input_ids = tok.encode(text, add_special_tokens=False)
        split = int(len(input_ids) * prompt_ratio)
        prompt_ids = input_ids[:split]
        ref_ids = input_ids[split:]
        prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)
        ref_text = tok.decode(ref_ids, skip_special_tokens=True)

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