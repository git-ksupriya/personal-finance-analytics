import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb

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


class HurdleCNN(nn.Module):
    """
    1D-CNN Sequence Architecture with Hurdle Dual-Output Heads.
    
    Dynamic input: (B, seq_len, dyn_dim=5)
    Static inputs:
      - Category embedding (14 categories -> cat_embed_dim=16)
      - User-level train stats (B, 2)
    CNN Feature Extractor:
      - Transpose to (B, dyn_dim, seq_len)
      - Conv1d(dyn_dim, 32) + BatchNorm + ReLU
      - Conv1d(32, 64) + BatchNorm + ReLU
      - AdaptiveAvgPool1d(1) -> (B, 64)
    Hurdle Heads:
      - Head A: P(spend > 0) -> BCEWithLogitsLoss
      - Head B: Predicted log1p(amt) given spend > 0 -> HuberLoss
    """
    def __init__(
        self,
        dyn_dim: int = 5,
        num_categories: int = 14,
        cat_embed_dim: int = 16,
        user_stat_dim: int = 2,
        conv_channels: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.cat_embedding = nn.Embedding(num_categories, cat_embed_dim)

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=dyn_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        fusion_dim = conv_channels + cat_embed_dim + user_stat_dim

        self.fc_shared = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head_spend_prob = nn.Linear(64, 1)
        self.head_spend_amt = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x_dyn, cat_idx, user_stats):
        # x_dyn: (B, seq_len, dyn_dim) -> (B, dyn_dim, seq_len)
        x_conv = x_dyn.permute(0, 2, 1)
        cnn_feat = self.conv_net(x_conv).squeeze(-1) # (B, conv_channels)

        cat_emb = self.cat_embedding(cat_idx)
        fused = torch.cat([cnn_feat, cat_emb, user_stats], dim=-1)
        feat = self.fc_shared(fused)

        spend_logits = self.head_spend_prob(feat).squeeze(-1)
        spend_prob = torch.sigmoid(spend_logits)
        pred_log_amt = self.head_spend_amt(feat).squeeze(-1)

        return spend_logits, spend_prob, pred_log_amt


class HurdleXGBoost:
    """
    Two-Stage Hurdle XGBoost Model:
    1. XGBClassifier predicts probability of spending P(spend > 0).
    2. XGBRegressor predicts log1p(amt) conditionally on non-zero spend instances.
    Final expected prediction = P(spend) * expm1(predicted_log_amt).
    """
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42):
        self.clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        self.reg = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1
        )

    @staticmethod
    def extract_tabular_features(X_dyn, X_stat):
        """
        Converts (B, seq_len, 5) dynamic sequence + (B, 3) static features into rich tabular feature matrix.
        """
        B = X_dyn.shape[0]
        # Summary statistics over sequence
        mean_log_amt = X_dyn[:, :, 0].mean(axis=1, keepdims=True)
        max_log_amt = X_dyn[:, :, 0].max(axis=1, keepdims=True)
        std_log_amt = X_dyn[:, :, 0].std(axis=1, keepdims=True)
        last_log_amt = X_dyn[:, -1, 0:1]
        spend_freq = (X_dyn[:, :, 2] > 0).mean(axis=1, keepdims=True)
        mean_txn_cnt = X_dyn[:, :, 1].mean(axis=1, keepdims=True)
        last_sin_woy = X_dyn[:, -1, 3:4]
        last_cos_woy = X_dyn[:, -1, 4:5]

        # Flattened full sequence
        flat_dyn = X_dyn.reshape(B, -1)

        # Concatenate tabular features
        features = np.hstack([
            flat_dyn,
            mean_log_amt, max_log_amt, std_log_amt, last_log_amt,
            spend_freq, mean_txn_cnt, last_sin_woy, last_cos_woy,
            X_stat # cat_idx, user_mean, user_std
        ])
        return features

    def fit(self, X_features, y_clf, y_reg):
        print(f"Fitting Hurdle XGBoost on {len(X_features):,} samples...")
        self.clf.fit(X_features, y_clf)
        
        # Train regression head strictly on positive spend samples
        mask_nonzero = (y_clf == 1)
        if mask_nonzero.sum() > 0:
            self.reg.fit(X_features[mask_nonzero], y_reg[mask_nonzero])
        else:
            self.reg.fit(X_features, y_reg)
        print("Hurdle XGBoost training complete.")

    def predict_proba(self, X_features):
        return self.clf.predict_proba(X_features)[:, 1]

    def predict_log_amount(self, X_features):
        return np.clip(self.reg.predict(X_features), 0, None)

    def predict(self, X_features):
        probs = self.predict_proba(X_features)
        log_amts = self.predict_log_amount(X_features)
        amts = np.expm1(log_amts)
        return probs * amts
