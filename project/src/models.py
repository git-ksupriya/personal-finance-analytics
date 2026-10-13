import torch
import torch.nn as nn
import torch.nn.functional as F

class HurdleRNN(nn.Module):
    """
    Unified Recurrent Architecture supporting LSTM or Vanilla RNN with Hurdle Output Head.
    
    Dynamic input: (B, seq_len, dyn_dim=5)
    Static inputs:
      - Category embedding (14 categories -> cat_embed_dim=16)
      - User-level train stats (B, 2)
    Recurrent core:
      - rnn_type: 'lstm' or 'rnn'
      - 2 layers, hidden_dim (e.g. 64-128), dropout (between layers)
    Hurdle Heads:
      - Head A: P(spend > 0) -> Sigmoid, BCE loss
      - Head B: Predicted log1p(amt) if spend > 0 -> Linear/ReLU, Huber/MSE loss
    Final expected prediction = P(spend) * expm1(predicted_log_amt)
    """
    def __init__(
        self,
        rnn_type: str = "lstm",
        dyn_dim: int = 5,
        num_categories: int = 14,
        cat_embed_dim: int = 16,
        user_stat_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Category embedding
        self.cat_embedding = nn.Embedding(num_categories, cat_embed_dim)

        # Recurrent Core
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=dyn_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=dyn_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                nonlinearity='tanh',
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Must be 'lstm' or 'rnn'.")

        # Hurdle head fusion dimension
        fusion_dim = hidden_dim + cat_embed_dim + user_stat_dim

        # Common feature extraction layer
        self.fc_shared = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Head A: Probability of spending (binary classification)
        self.head_spend_prob = nn.Linear(64, 1)

        # Head B: Spending amount on positive spend (regression on log1p)
        self.head_spend_amt = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU() # Log1p amount is non-negative
        )

    def forward(self, x_dyn, cat_idx, user_stats):
        """
        x_dyn: (B, seq_len, 5)
        cat_idx: (B,)
        user_stats: (B, 2)
        """
        # Recurrent feature extraction
        out, _ = self.rnn(x_dyn)
        last_hidden = out[:, -1, :] # (B, hidden_dim)

        # Embed category
        cat_emb = self.cat_embedding(cat_idx) # (B, cat_embed_dim)

        # Fuse recurrent context with static user & category representations
        fused = torch.cat([last_hidden, cat_emb, user_stats], dim=-1)
        feat = self.fc_shared(fused)

        # Head A: Logits for spend probability
        spend_logits = self.head_spend_prob(feat).squeeze(-1)
        spend_prob = torch.sigmoid(spend_logits)

        # Head B: Predicted log1p(amt) given spend
        pred_log_amt = self.head_spend_amt(feat).squeeze(-1)

        return spend_logits, spend_prob, pred_log_amt
