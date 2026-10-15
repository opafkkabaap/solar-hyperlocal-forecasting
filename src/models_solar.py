import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class SolarGenerationForecasterXGB:
    """
    Physics-informed XGBoost forecaster for solar active energy generation (kWh).
    Incorporates theoretical DC capacity, temperature derating, and inverter limits.
    """
    def __init__(self, rated_kwp=66.0, inverter_limit_kw=55.2, n_estimators=300, max_depth=6, learning_rate=0.05):
        self.rated_kwp = rated_kwp
        self.inverter_limit_kw = inverter_limit_kw
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = None

    def train(self, train_df, feature_cols):
        self.feature_cols = feature_cols
        X_train = train_df[feature_cols]
        y_train = train_df["energy_kwh"]
        
        print(f"Training Solar Generation XGBoost on {len(X_train)} samples with {len(feature_cols)} features...")
        self.model.fit(X_train, y_train)
        print("Solar generation model trained successfully.")

    def predict(self, df):
        X = df[self.feature_cols]
        pred_kwh = self.model.predict(X)
        
        # Physical constraints
        # 1. At night, solar generation is strictly 0.0
        night_mask = df["solar_elevation"] <= 0.0
        pred_kwh = np.where(night_mask, 0.0, pred_kwh)
        
        # 2. Clamped between 0 and physical inverter capacity
        max_hourly_kwh = self.inverter_limit_kw  # 1 hour max energy at full inverter AC rating
        pred_kwh = np.clip(pred_kwh, 0.0, max_hourly_kwh)
        return pred_kwh

    def get_feature_importances(self):
        return pd.Series(self.model.feature_importances_, index=self.feature_cols).sort_values(ascending=False)


class BiGRUPVModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


class SolarGenerationForecasterDeep:
    """
    Sequence-based Deep Learning Bi-GRU forecaster for comparison.
    """
    def __init__(self, seq_len=24, hidden_dim=64, num_layers=2, epochs=15, batch_size=64, lr=1e-3):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.feature_cols = None
        self.mean = None
        self.std = None
        self.target_max = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_sequences(self, df):
        features = df[self.feature_cols].values
        targets = df["energy_kwh"].values
        
        # Normalize
        norm_features = (features - self.mean) / self.std
        
        X_seq, y_seq, indices = [], [], []
        for i in range(self.seq_len, len(df)):
            X_seq.append(norm_features[i - self.seq_len:i])
            y_seq.append(targets[i])
            indices.append(i)
            
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), np.array(indices)

    def train(self, train_df, feature_cols):
        self.feature_cols = feature_cols
        raw_feat = train_df[feature_cols].values
        self.mean = np.mean(raw_feat, axis=0)
        self.std = np.std(raw_feat, axis=0) + 1e-6
        self.target_max = train_df["energy_kwh"].max() + 1e-3

        X_train, y_train, _ = self._prepare_sequences(train_df)
        
        dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = BiGRUPVModel(
            input_dim=len(feature_cols),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        print(f"Training Deep Bi-GRU on {len(X_train)} sequences of length {self.seq_len}...")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_y)
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch+1}/{self.epochs} - MSE Loss: {total_loss / len(X_train):.4f}")

    def predict(self, test_df):
        self.model.eval()
        X_test, _, valid_idx = self._prepare_sequences(test_df)
        
        dataset = TensorDataset(torch.tensor(X_test))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                preds.extend(pred.cpu().numpy())
                
        pred_arr = np.array(preds)
        
        # Apply physical constraints on valid test indices
        sub_test = test_df.iloc[valid_idx]
        night_mask = sub_test["solar_elevation"].values <= 0.0
        pred_arr = np.where(night_mask, 0.0, pred_arr)
        pred_arr = np.clip(pred_arr, 0.0, self.target_max)
        
        # Pad initial seq_len records with lag_24 or persistence
        full_preds = np.zeros(len(test_df), dtype=np.float32)
        full_preds[:self.seq_len] = test_df["energy_kwh_lag_24"].iloc[:self.seq_len].values
        full_preds[self.seq_len:] = pred_arr
        return full_preds
