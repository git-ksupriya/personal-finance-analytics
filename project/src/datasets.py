import numpy as np
import torch
from torch.utils.data import Dataset
from config import MODELS_DIR

class SpendingSequenceDataset(Dataset):
    """
    PyTorch Dataset for weekly spending sequences.
    Returns:
        x_dyn: (seq_len, dynamic_feature_dim)
        cat_idx: int (category index for embedding)
        user_stats: (2,) (train mean spend, train std spend)
        y_clf: float (binary has_spend indicator)
        y_reg: float (log1p amount if spend > 0)
    """
    def __init__(self, x_dyn, x_stat, y_clf, y_reg):
        self.x_dyn = torch.from_numpy(x_dyn).float()
        self.cat_idx = torch.from_numpy(x_stat[:, 0]).long()
        self.user_stats = torch.from_numpy(x_stat[:, 1:]).float()
        self.y_clf = torch.from_numpy(y_clf).float()
        self.y_reg = torch.from_numpy(y_reg).float()

    def __len__(self):
        return len(self.y_clf)

    def __getitem__(self, idx):
        return (
            self.x_dyn[idx],
            self.cat_idx[idx],
            self.user_stats[idx],
            self.y_clf[idx],
            self.y_reg[idx]
        )

def load_data_split(seq_len=8):
    """Loads preprocessed tensors for a specific sequence window."""
    file_path = MODELS_DIR / f"processed_w{seq_len}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found: {file_path}. Run preprocess.py first.")

    data = np.load(file_path)
    train_ds = SpendingSequenceDataset(
        data['X_dyn_train'], data['X_stat_train'], data['y_clf_train'], data['y_reg_train']
    )
    val_ds = SpendingSequenceDataset(
        data['X_dyn_val'], data['X_stat_val'], data['y_clf_val'], data['y_reg_val']
    )
    test_ds = SpendingSequenceDataset(
        data['X_dyn_test'], data['X_stat_test'], data['y_clf_test'], data['y_reg_test']
    )
    return train_ds, val_ds, test_ds
