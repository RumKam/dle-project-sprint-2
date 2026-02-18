import pandas as pd
import re
import html
from tqdm import tqdm

def clean_text(text):
    """Функция для очистки текста"""

    # Декодируем HTML-сущности
    text = html.unescape(text)
    # Удаляем ссылки
    text = re.sub(r'http\S+', '', text)
    # Удаляем всё, что в квадратных скобках
    text = re.sub(r'\[.*?\]', '', text)
    # Удаляем упоминания @username целиком
    text = re.sub(r'@\S+', '', text)
    # Удаляем хэштеги #hashtag целиком
    text = re.sub(r'#\S+', '', text)
    # Удаляем конструкции со звёздочками целиком
    text = re.sub(r'\*+.*?\*+', '', text)
    # Оставляем буквы, цифры, пробелы и основные знаки препинания
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s.,!?:;\"'()\-+$%&=]", '', text)
    # Нормализуем множественные знаки препинания
    text = re.sub(r'\.{4,}', '...', text)      # 4 и более точек -> ...
    text = re.sub(r'!{2,}', '!', text)          # !!!... -> !
    text = re.sub(r'\?{2,}', '?', text)         # ???... -> ?
    # Сжимаем множественные пробелы и убираем краевые
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def prepaire_text(df, column):
    """Функция для обработка текста"""

    # Приводим к нижнему регистру
    df['text_clean'] = df[column].str.lower()
    # Очищаем текст
    df['text_clean'] = df['text_clean'].apply(clean_text)
    # Удаляем пустые строки после очистки
    df = df[df['text_clean'].str.len() > 0].reset_index(drop=True)

    return df

def load_raw_data(filepath):
    """Функция для загрузки исходного файла"""

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    df = pd.DataFrame(lines, columns=['text'])
    
    return df

def tokenize_texts(texts, tokenizer):
    """Токенизация текстов"""
    
    tokenized = []
    for text in tqdm(texts, desc="Tokenizing"):
        if not isinstance(text, str) or pd.isna(text):
            continue
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=512)
        tokenized.append(ids)
    return tokenized