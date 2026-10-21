import os
import json
import time
import uuid
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import psycopg2

from config import (
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT,
    MODELS_DIR, REPORTS_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, PATIENCE, SEED
)
from datasets import load_data_split
from models import HurdleRNN, HurdleCNN, HurdleXGBoost

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_db_connection():
    return psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )

def train_epoch(model, dataloader, optimizer, criterion_clf, criterion_reg, device, clip_grad=None):
    model.train()
    total_loss, total_clf_loss, total_reg_loss = 0.0, 0.0, 0.0
    num_batches = 0

    for x_dyn, cat_idx, user_stats, y_clf, y_reg in dataloader:
        x_dyn = x_dyn.to(device)
        cat_idx = cat_idx.to(device)
        user_stats = user_stats.to(device)
        y_clf = y_clf.to(device)
        y_reg = y_reg.to(device)

        optimizer.zero_grad()
        spend_logits, spend_prob, pred_log_amt = model(x_dyn, cat_idx, user_stats)

        # Head A: Classification loss on all weeks
        loss_clf = criterion_clf(spend_logits, y_clf)

        # Head B: Regression loss ONLY on non-zero target weeks
        mask_nonzero = (y_clf > 0.5)
        if mask_nonzero.sum() > 0:
            loss_reg = criterion_reg(pred_log_amt[mask_nonzero], y_reg[mask_nonzero])
        else:
            loss_reg = torch.tensor(0.0, device=device)

        loss = loss_clf + loss_reg
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()

        total_loss += loss.item()
        total_clf_loss += loss_clf.item()
        total_reg_loss += loss_reg.item()
        num_batches += 1

    return (
        total_loss / max(1, num_batches),
        total_clf_loss / max(1, num_batches),
        total_reg_loss / max(1, num_batches)
    )

def eval_epoch(model, dataloader, criterion_clf, criterion_reg, device):
    model.eval()
    total_loss, total_clf_loss, total_reg_loss = 0.0, 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for x_dyn, cat_idx, user_stats, y_clf, y_reg in dataloader:
            x_dyn = x_dyn.to(device)
            cat_idx = cat_idx.to(device)
            user_stats = user_stats.to(device)
            y_clf = y_clf.to(device)
            y_reg = y_reg.to(device)

            spend_logits, spend_prob, pred_log_amt = model(x_dyn, cat_idx, user_stats)
            loss_clf = criterion_clf(spend_logits, y_clf)

            mask_nonzero = (y_clf > 0.5)
            if mask_nonzero.sum() > 0:
                loss_reg = criterion_reg(pred_log_amt[mask_nonzero], y_reg[mask_nonzero])
            else:
                loss_reg = torch.tensor(0.0, device=device)

            loss = loss_clf + loss_reg
            total_loss += loss.item()
            total_clf_loss += loss_clf.item()
            total_reg_loss += loss_reg.item()
            num_batches += 1

    return (
        total_loss / max(1, num_batches),
        total_clf_loss / max(1, num_batches),
        total_reg_loss / max(1, num_batches)
    )

def save_predictions_to_db(meta_df, p_spend, pred_amt, model_name, run_id):
    """Writes forecast predictions into PostgreSQL forecasts table."""
    try:
        conn = get_db_connection()
        records = []
        for i, row in meta_df.iterrows():
            records.append((
                int(row['user_id']),
                str(row['category']),
                str(row['target_week']),
                model_name,
                float(np.round(p_spend[i], 4)),
                float(np.round(pred_amt[i], 2)),
                run_id
            ))

        import io
        output = io.StringIO()
        pred_df = pd.DataFrame(records, columns=['user_id', 'category', 'target_week', 'model_name', 'p_spend', 'predicted_amt', 'run_id'])
        pred_df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)

        with conn.cursor() as cur:
            cur.copy_from(
                output,
                'forecasts',
                columns=['user_id', 'category', 'target_week', 'model_name', 'p_spend', 'predicted_amt', 'run_id']
            )
        conn.commit()
        conn.close()
        print(f"Logged {len(records):,} test predictions into 'forecasts' table for run {run_id}.")
    except Exception as e:
        print(f"Notice: database forecast logging skipped: {e}")

def log_model_run_to_db(run_id, model_name, hyperparams, metrics):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_runs (run_id, model_name, hyperparams, metrics)
                VALUES (%s, %s, %s, %s);
                """,
                (run_id, model_name, json.dumps(hyperparams), json.dumps(metrics))
            )
        conn.commit()
        conn.close()
        print(f"Logged run metadata and metrics into 'model_runs' for {run_id}.")
    except Exception as e:
        print(f"Notice: database run metadata logging skipped: {e}")

def train_neural_model(model_type="lstm", seq_len=8, hidden_dim=64, num_layers=2, max_epochs=NUM_EPOCHS, sample_frac=1.0):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Training {model_type.upper()} (window={seq_len}, hidden={hidden_dim}, device={device})")
    print(f"==========================================")

    train_ds, val_ds, test_ds = load_data_split(seq_len=seq_len)

    # Subsampling if fast mode requested
    if sample_frac < 1.0:
        n_train = int(len(train_ds) * sample_frac)
        n_val = int(len(val_ds) * sample_frac)
        train_ds = Subset(train_ds, indices=range(n_train))
        val_ds = Subset(val_ds, indices=range(n_val))
        print(f"Sampled subset: Train={len(train_ds):,}, Val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    if model_type in ["lstm", "rnn"]:
        model = HurdleRNN(
            rnn_type=model_type,
            dyn_dim=5,
            num_categories=14,
            cat_embed_dim=16,
            user_stat_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.2
        ).to(device)
    elif model_type == "cnn":
        model = HurdleCNN(
            dyn_dim=5,
            num_categories=14,
            cat_embed_dim=16,
            user_stat_dim=2,
            conv_channels=hidden_dim,
            dropout=0.2
        ).to(device)
    else:
        raise ValueError(f"Unknown neural model type: {model_type}")

    criterion_clf = nn.BCEWithLogitsLoss()
    criterion_reg = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    clip_grad = 1.0 if model_type == "rnn" else 5.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    run_id = f"{model_type}_w{seq_len}_{uuid.uuid4().hex[:8]}"

    history = []
    checkpoint_path = MODELS_DIR / f"{model_type}_w{seq_len}_best.pt"

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        t_loss, t_clf, t_reg = train_epoch(
            model, train_loader, optimizer, criterion_clf, criterion_reg, device, clip_grad=clip_grad
        )
        v_loss, v_clf, v_reg = eval_epoch(
            model, val_loader, criterion_clf, criterion_reg, device
        )
        scheduler.step(v_loss)

        history.append({
            'epoch': epoch, 'train_loss': t_loss, 'val_loss': v_loss,
            'train_clf_loss': t_clf, 'val_clf_loss': v_clf,
            'train_reg_loss': t_reg, 'val_reg_loss': v_reg
        })

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train: {t_loss:.4f} (clf={t_clf:.4f}, reg={t_reg:.4f}) | Val: {v_loss:.4f} (clf={v_clf:.4f}, reg={v_reg:.4f})")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': v_loss,
                'hyperparams': {
                    'model_type': model_type,
                    'seq_len': seq_len,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers
                }
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch} (Best epoch: {best_epoch}, Val loss: {best_val_loss:.4f})")
                break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s. Loading best weights...")

    # Load best model for evaluation on test set
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()

    all_p_spend, all_pred_amt = [], []
    with torch.no_grad():
        for x_dyn, cat_idx, user_stats, _, _ in test_loader:
            x_dyn = x_dyn.to(device)
            cat_idx = cat_idx.to(device)
            user_stats = user_stats.to(device)

            _, spend_prob, pred_log_amt = model(x_dyn, cat_idx, user_stats)
            pred_amt_val = torch.expm1(pred_log_amt)
            final_pred_amt = spend_prob * pred_amt_val

            all_p_spend.extend(spend_prob.cpu().numpy())
            all_pred_amt.extend(final_pred_amt.cpu().numpy())

    p_spend_arr = np.array(all_p_spend)
    pred_amt_arr = np.array(all_pred_amt)

    # Save predictions to file and DB
    np.save(MODELS_DIR / f"{model_type}_w{seq_len}_p_spend.npy", p_spend_arr)
    np.save(MODELS_DIR / f"{model_type}_w{seq_len}_pred_amt.npy", pred_amt_arr)

    meta_file = MODELS_DIR / f"processed_w{seq_len}_test_meta.csv"
    meta_df = pd.read_csv(meta_file)
    save_predictions_to_db(meta_df, p_spend_arr, pred_amt_arr, model_type.upper(), run_id)

    actual_amt = meta_df['actual_amt'].values
    test_mae = float(np.mean(np.abs(actual_amt - pred_amt_arr)))
    test_rmse = float(np.sqrt(np.mean((actual_amt - pred_amt_arr) ** 2)))

    metrics = {
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'test_overall_mae': test_mae,
        'test_overall_rmse': test_rmse,
        'training_time_sec': round(training_time, 2)
    }

    log_model_run_to_db(
        run_id=run_id,
        model_name=model_type.upper(),
        hyperparams={'seq_len': seq_len, 'hidden_dim': hidden_dim, 'lr': LEARNING_RATE},
        metrics=metrics
    )

    return model, history, metrics

def train_xgboost_model(seq_len=8, sample_frac=0.1):
    set_seed(SEED)
    print(f"\n==========================================")
    print(f"Training Hurdle XGBoost (window={seq_len})")
    print(f"==========================================")

    npz_path = MODELS_DIR / f"processed_w{seq_len}.npz"
    data = np.load(npz_path)

    X_dyn_train = data['X_dyn_train']
    X_stat_train = data['X_stat_train']
    y_clf_train = data['y_clf_train']
    y_reg_train = data['y_reg_train']

    X_dyn_test = data['X_dyn_test']
    X_stat_test = data['X_stat_test']

    # Subsample training data for fast, efficient tree training
    n_train = int(len(y_clf_train) * sample_frac)
    indices = np.random.choice(len(y_clf_train), size=n_train, replace=False)
    indices.sort()

    print(f"Extracting tabular features for train ({n_train:,} samples) and test ({len(X_dyn_test):,} samples)...")
    X_train_feat = HurdleXGBoost.extract_tabular_features(X_dyn_train[indices], X_stat_train[indices])
    y_clf_sample = y_clf_train[indices]
    y_reg_sample = y_reg_train[indices]

    start_time = time.time()
    xgb_hurdle = HurdleXGBoost(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=SEED)
    xgb_hurdle.fit(X_train_feat, y_clf_sample, y_reg_sample)
    training_time = time.time() - start_time

    # Save model
    joblib.dump(xgb_hurdle, MODELS_DIR / f"xgboost_w{seq_len}_model.pkl")

    # Run inference on test set in chunks to manage memory
    print(f"Generating test predictions for Hurdle XGBoost...")
    chunk_size = 50000
    all_p_spend = []
    all_pred_amt = []
    for i in range(0, len(X_dyn_test), chunk_size):
        end = min(i + chunk_size, len(X_dyn_test))
        x_chunk = HurdleXGBoost.extract_tabular_features(X_dyn_test[i:end], X_stat_test[i:end])
        probs = xgb_hurdle.predict_proba(x_chunk)
        preds = xgb_hurdle.predict(x_chunk)
        all_p_spend.extend(probs)
        all_pred_amt.extend(preds)

    p_spend_arr = np.array(all_p_spend)
    pred_amt_arr = np.array(all_pred_amt)

    np.save(MODELS_DIR / f"xgb_w{seq_len}_p_spend.npy", p_spend_arr)
    np.save(MODELS_DIR / f"xgb_w{seq_len}_pred_amt.npy", pred_amt_arr)

    meta_file = MODELS_DIR / f"processed_w{seq_len}_test_meta.csv"
    meta_df = pd.read_csv(meta_file)
    run_id = f"xgb_w{seq_len}_{uuid.uuid4().hex[:8]}"
    save_predictions_to_db(meta_df, p_spend_arr, pred_amt_arr, "XGBOOST", run_id)

    actual_amt = meta_df['actual_amt'].values
    test_mae = float(np.mean(np.abs(actual_amt - pred_amt_arr)))
    test_rmse = float(np.sqrt(np.mean((actual_amt - pred_amt_arr) ** 2)))

    metrics = {
        'test_overall_mae': test_mae,
        'test_overall_rmse': test_rmse,
        'training_time_sec': round(training_time, 2)
    }
    log_model_run_to_db(run_id, "XGBOOST", {'seq_len': seq_len, 'n_estimators': 100}, metrics)
    print(f"Hurdle XGBoost Evaluation: Overall MAE: ${test_mae:.2f} | RMSE: ${test_rmse:.2f}")
    return xgb_hurdle, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["lstm", "rnn", "cnn", "xgboost", "all"])
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    args = parser.parse_args()

    if args.model in ["lstm", "all"]:
        train_neural_model(model_type="lstm", seq_len=args.seq_len, hidden_dim=args.hidden_dim, max_epochs=args.epochs, sample_frac=args.sample_frac)
    if args.model in ["rnn", "all"]:
        train_neural_model(model_type="rnn", seq_len=args.seq_len, hidden_dim=args.hidden_dim, max_epochs=args.epochs, sample_frac=args.sample_frac)
    if args.model in ["cnn", "all"]:
        train_neural_model(model_type="cnn", seq_len=args.seq_len, hidden_dim=args.hidden_dim, max_epochs=args.epochs, sample_frac=args.sample_frac)
    if args.model in ["xgboost", "all"]:
        train_xgboost_model(seq_len=args.seq_len, sample_frac=0.1)
