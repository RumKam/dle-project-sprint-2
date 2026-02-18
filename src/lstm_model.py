import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """LSTM модель для генерации текста с embedding слоем и классификационным выходом"""

    def __init__(self, vocab_size, emb_dim=300, hidden_dim=256, num_layers=2, dropout=0.5, pad_idx=0):
        """Инициализация LSTM модели"""

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        """Инициализация весов модели с использованием xavier uniform"""

        # Инициализация embedding
        nn.init.xavier_uniform_(self.embedding.weight)

        # Инициализация весов LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Инициализация линейного слоя
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids):
        """Прямой проход через модель"""

        x = self.embedding(input_ids)     
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)            
        last_out = lstm_out[:, -1, :]           
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)      

        return logits

    def generate(self, prompt_ids, max_length, device='cpu', seq_len=None):
        """Генерация последовательности токенов на основе промпта """
        
        if seq_len is None:
            raise ValueError("seq_len must be provided for generation")
        
        self.eval()

        with torch.no_grad():
            generated = prompt_ids.copy()
            context_size = seq_len
            for _ in range(max_length - len(generated)):
                context = generated[-context_size:]
                if len(context) < context_size:
                    # Дополняем слева pad-токеном
                    context = [self.embedding.padding_idx] * (context_size - len(context)) + context

                input_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
                logits = self.forward(input_tensor)
                next_token = torch.argmax(logits, dim=-1).item()
                generated.append(next_token)

            return generated