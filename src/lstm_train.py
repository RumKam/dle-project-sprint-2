import torch
import torch.nn as nn
from tqdm import tqdm
from .eval_lstm import compute_rouge

def train_epoch(model, loader, optimizer, criterion, device, return_grad_norm=True):

    model.train()
    total_loss = 0
    total_grad_norm = 0
    num_batches = 0

    for input_ids, targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, targets)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if return_grad_norm:
            total_grad_norm += grad_norm.item()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / len(loader)

    if return_grad_norm:
        avg_grad_norm = total_grad_norm / num_batches
        return avg_loss, avg_grad_norm
    else:
        return avg_loss

def train_model(model, train_loader, val_texts, tokenizer, device, seq_len,
                epochs, optimizer, criterion, save_path='best_lstm_model.pt',
                val_loader=None, prompt_ratio=0.75):
    """
    Полный цикл обучения с отслеживанием истории loss и нормы градиента.
    Параметры:
        val_loader: если передан, вычисляется loss на валидации (по батчам).
        prompt_ratio: доля промпта для вычисления ROUGE.
    Возвращает кортеж (model, train_losses, val_losses, grad_norms).
    """
    best_rouge1 = 0.0
    train_losses = []
    val_losses = []
    grad_norms = []

    for epoch in range(1, epochs + 1):
        # Обучаем эпоху с возвратом нормы градиента
        train_loss, avg_grad_norm = train_epoch(
            model, train_loader, optimizer, criterion, device, return_grad_norm=True
        )
        train_losses.append(train_loss)
        grad_norms.append(avg_grad_norm)

        # Вычисляем loss на валидации
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
        else:
            val_losses.append(None)

        # Вычисляем ROUGE на валидационных текстах
        val_rouge = compute_rouge(model, val_texts, tokenizer, device, seq_len, prompt_ratio)

        # Формируем строку вывода
        rouge_str = f" | Val ROUGE: {val_rouge}"
        grad_str = f" | Grad Norm: {avg_grad_norm:.4f}"
        val_loss_str = f" | Val Loss: {val_loss:.4f}" if val_loader else ""

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f}{val_loss_str}{grad_str}{rouge_str}")

        # Сохраняем лучшую модель по ROUGE-1
        if val_rouge and val_rouge.get('rouge1', 0) > best_rouge1:
            best_rouge1 = val_rouge['rouge1']

            torch.save(model.state_dict(), save_path)
            
            print(f"Saved best model with ROUGE-1 = {best_rouge1:.4f}")

    return model, train_losses, val_losses, grad_norms