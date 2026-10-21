import os
import sys
import json
import joblib
import numpy as np
import torch

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from lstm.train_lstm import LSTMModel
from rnn.train_rnn import RNNModel

class SpendingPredictor:
    def __init__(self, model_type="lstm"):
        self.model_type = model_type.lower()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.utils_dir = os.path.join(self.base_dir, "utils")
        
        scaler_path = os.path.join(self.utils_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            scaler_path = os.path.abspath(os.path.join(self.base_dir, "..", "preprocessing", "processed", "scaler.pkl"))
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Run preprocessing first.")
        self.scaler = joblib.load(scaler_path)

        if self.model_type == "lstm":
            self.model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=7, dropout=0.2)
            model_path = os.path.join(self.base_dir, "lstm", "lstm_model.pt")
        elif self.model_type == "rnn":
            self.model = RNNModel(input_size=1, hidden_size=64, num_layers=2, output_size=7, dropout=0.2)
            model_path = os.path.join(self.base_dir, "rnn", "rnn_model.pt")
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lstm' or 'rnn'.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}. Train the model first.")

        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict_next_7_days(self, sequence):
        """
        Given a list or array of past spending amounts (length >= 10),
        predict the likely next 7 days of spending amounts.
        Returns a dictionary mapping 'Day_1' through 'Day_7' to predicted amounts in INR.
        """
        if len(sequence) < 10:
            raise ValueError(f"Input sequence length must be at least 10 (got {len(sequence)}).")

        # Take last 10 timesteps
        input_seq = sequence[-10:]
        seq_array = np.array(input_seq, dtype=np.float32).reshape(-1, 1)
        scaled_seq = self.scaler.transform(seq_array).reshape(1, 10, 1)
        input_tensor = torch.tensor(scaled_seq, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = self.model(input_tensor).numpy().reshape(-1, 1)

        pred_raw = self.scaler.inverse_transform(pred_scaled).reshape(-1)
        
        predictions_7_days = {}
        for d in range(7):
            predictions_7_days[f"Day_{d+1}"] = round(float(pred_raw[d]), 2)

        return predictions_7_days

def main():
    print("--- 7-Day Multi-Step Spending Sequence Predictor Demo ---")
    sample_spending_sequence = [1200.0, 450.0, 3500.0, 890.0, 2100.0, 1500.0, 620.0, 4100.0, 950.0, 2800.0]
    print(f"Past Spending Sequence (10 timesteps): {sample_spending_sequence}")

    try:
        lstm_predictor = SpendingPredictor(model_type="lstm")
        lstm_preds = lstm_predictor.predict_next_7_days(sample_spending_sequence)
        print("\nPredicted Next 7 Days Spending (LSTM):")
        for day, val in lstm_preds.items():
            print(f"  {day}: INR {val:,.2f}")
    except Exception as e:
        print(f"LSTM Prediction Error: {e}")

    try:
        rnn_predictor = SpendingPredictor(model_type="rnn")
        rnn_preds = rnn_predictor.predict_next_7_days(sample_spending_sequence)
        print("\nPredicted Next 7 Days Spending (SimpleRNN):")
        for day, val in rnn_preds.items():
            print(f"  {day}: INR {val:,.2f}")
    except Exception as e:
        print(f"RNN Prediction Error: {e}")

if __name__ == "__main__":
    main()
