"""
循环水系统 - 系统建模模块
基于 LSTM/GRU 的时间序列预测模型（黑盒/灰盒模型）
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from config import (LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
                    LSTM_LEARNING_RATE, LSTM_EPOCHS, MODEL_DIR, STATE_COLS,
                    CONTROL_COLS)


class LSTMPredictor(nn.Module):
    """基于 LSTM 的循环水系统预测模型（黑盒模型）
    输入：过去的系统状态和控制量序列
    输出：未来的系统状态预测
    """

    def __init__(self, input_dim, hidden_size=LSTM_HIDDEN_SIZE,
                 num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT,
                 output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

        self.ln = nn.LayerNorm(hidden_size * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_dim or input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, hidden=None):
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        out = self.ln(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

    def predict(self, x, device="cpu"):
        """预测接口"""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(device)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            pred = self.forward(x)
        return pred.cpu().numpy()


class GRUPredictor(nn.Module):
    """基于 GRU 的预测模型（备选方案）"""

    def __init__(self, input_dim, hidden_size=LSTM_HIDDEN_SIZE,
                 num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT,
                 output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

        self.ln = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, output_dim or input_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.ln(out[:, -1, :])
        return self.fc(out)


class PhysicsInformedLSTM(nn.Module):
    """物理信息融合 LSTM（灰盒模型）
    融合循环水系统的物理定律：
    - 质量守恒：入流量 = 出流量（管路节点处）
    - 能量守恒：热交换器换热量 = 流量 × 比热 × 温差
    - 泵相似定律：流量/转速 ∝ 转速的一次方，扬程/转速 ∝ 转速的平方
    - 阀门流量特性：Q ∝ Cv × sqrt(delta_P)，等百分比特性
    - 混合温度：T_mix = (Q1*T1 + Q2*T2) / (Q1 + Q2)

    特征索引（11维全特征）：
      状态(0-5): temp_pump_out, press_pump_out, flow_pump_DN300,
                 temp_tank_out, temp_he_sim, press_tank_out
      控制(6-10): valve_DN200, valve_DN300, valve_DN350, valve_DN400, pump_speed
    """

    def __init__(self, input_dim, state_dim=6, control_dim=5,
                 hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS,
                 dropout=LSTM_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.control_dim = control_dim

        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, input_dim),
        )

        # 物理参数（可学习）
        # 水的比热容缩放参数
        self.log_cp_water = nn.Parameter(torch.tensor(1.0))  # 4.18 kJ/(kg*K)
        # 泵的效率参数
        self.log_pump_eta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln(out[:, -1, :])
        return self.fc(out)

    def physics_loss(self, pred, target, lambda_physics=0.05):
        """物理约束损失
        pred, target: (batch, 11) 全特征张量
        特征顺序: [temp_pump(0), press_pump(1), flow_DN300(2),
                    temp_tank(3), temp_he(4), press_tank(5),
                    valve_DN200(6), valve_DN300(7), valve_DN350(8),
                    valve_DN400(9), pump_speed(10)]
        """
        mse_loss = nn.MSELoss()(pred, target)
        batch_size = pred.shape[0]
        physics_terms = {}

        # 1. 流量非负约束
        flow_cols = [2, -1]  # flow_DN300, 以及可能的其他流量
        flow_penalty = 0.0
        for idx in flow_cols:
            if idx < pred.shape[1]:
                flow_penalty += torch.relu(-pred[:, idx]).mean()
        physics_terms["flow_nonneg"] = flow_penalty.item()

        # 2. 压力非负且上限约束 (MPa)
        press_cols = [1, 5]  # press_pump, press_tank
        press_penalty = 0.0
        for idx in press_cols:
            if idx < pred.shape[1]:
                press_penalty += torch.relu(-pred[:, idx]).mean()
                press_penalty += torch.relu(pred[:, idx] - 3.0).mean()
        physics_terms["pressure"] = press_penalty.item()

        # 3. 质量守恒：系统流入流出平衡
        # 主管流量 ≈ M1支路流量 + M2泄压支路流量
        # flow_DN300 ≈ k1 * valve_DN200 + k2 * valve_DN300
        flow_total = pred[:, 2]  # flow_DN300
        valve_m1 = pred[:, 6]   # valve_DN200
        valve_m2 = pred[:, 7]   # valve_DN300
        flow_balance = torch.abs(
            flow_total - 3.0 * (valve_m1 + valve_m2)
        ).mean()
        physics_terms["mass_conservation"] = flow_balance.item()

        # 4. 能量守恒：热交换器换热量
        # Q = m_dot * cp * delta_T
        # 换热器入口温度T2 ≈ 混合温度
        # T2 ≈ alpha * T_tank + (1-alpha) * T_he_out
        t_tank = pred[:, 3]     # temp_tank_out
        t_he_out = pred[:, 4]   # temp_he_sim
        valve_m3 = pred[:, 8]   # valve_DN350 (recycle)
        # M3归一化比例
        m3_ratio = torch.sigmoid((valve_m3 - 99.6) * 10.0)  # 约0.5左右
        t2_predicted = m3_ratio * t_tank + (1 - m3_ratio) * t_he_out
        t_pump = pred[:, 0]     # temp_pump_out (close to T2 area)
        temp_mixing = torch.abs(t2_predicted - t_pump).mean()
        physics_terms["energy_mixing"] = temp_mixing.item()

        # 5. 泵相似定律：流量与转速近似正比
        # flow / pump_speed ≈ const (在固定系统阻力下)
        pump_speed = pred[:, 10] + 1e-6
        flow_pump = pred[:, 2] + 1e-6
        specific_flow = flow_pump / pump_speed
        specific_flow_target = target[:, 2] / (target[:, 10] + 1e-6)
        pump_law_penalty = torch.abs(
            specific_flow - specific_flow_target
        ).mean()
        physics_terms["pump_affinity"] = pump_law_penalty.item()

        # 6. 阀门特性：流量与阀位开度相关（等百分比特性）
        # Q/Q_max ≈ R^(x-1) 其中R是范围比(≈50)，x是相对开度
        m1_norm = valve_m1 / 100.0
        q_ratio = flow_total / (flow_total.detach().max() + 1e-6)
        valve_characteristic = torch.abs(
            q_ratio - m1_norm
        ).mean()
        physics_terms["valve_char"] = valve_characteristic.item()

        # 总物理损失
        physics_total = (
            flow_penalty * 0.5 +
            press_penalty * 1.0 +
            flow_balance * 0.1 +
            temp_mixing * 0.3 +
            pump_law_penalty * 0.2 +
            valve_characteristic * 0.05
        )

        total_loss = mse_loss + lambda_physics * physics_total
        physics_terms["physics_total"] = physics_total.item()
        physics_terms["mse"] = mse_loss.item()
        physics_terms["total"] = total_loss.item()

        return total_loss, physics_terms


class PhysicsInformedTrainer:
    """物理信息融合模型训练器"""

    def __init__(self, model, lambda_physics=0.05,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = model.to(device)
        self.lambda_physics = lambda_physics
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LSTM_LEARNING_RATE,
                                            weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2)
        self.best_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        self.physics_log = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss, physics_terms = self.model.physics_loss(
                pred, y, self.lambda_physics)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            self.physics_log.append(physics_terms)
        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        for X, y in val_loader:
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss, _ = self.model.physics_loss(pred, y, self.lambda_physics)
            total_loss += loss.item()
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs=LSTM_EPOCHS,
            patience=15, model_name="physics_lstm"):
        print(f"Physics-Informed LSTM training (device={self.device})")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                self._save(model_name)
                improved = "*"
            else:
                patience_counter += 1
                improved = ""

            if epoch % 10 == 0 or improved:
                # 显示物理约束项
                if self.physics_log:
                    last = self.physics_log[-1]
                    p_str = f"| phys={last.get('physics_total', 0):.4f}"
                else:
                    p_str = ""
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:4d} | train={train_loss:.4f} | "
                      f"val={val_loss:.4f} {p_str} | lr={lr:.2e} {improved}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self._load(model_name)
        print(f"Physics-LSTM complete, best val_loss={self.best_loss:.4f}")
        return self.train_losses, self.val_losses

    def _save(self, name):
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def _load(self, name):
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optim.AdamW(model.parameters(), lr=LSTM_LEARNING_RATE,
                                      weight_decay=1e-5)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2)
        self.criterion = nn.MSELoss()
        self.best_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        for X, y in val_loader:
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred, y)
            total_loss += loss.item()
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs=LSTM_EPOCHS,
            patience=15, model_name="lstm_model"):
        print(f"开始训练 (device={self.device}, epochs={epochs})")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                self._save(model_name)
                improved = "*"
            else:
                patience_counter += 1
                improved = ""

            if epoch % 10 == 0 or improved:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:4d} | train_loss={train_loss:.6f} | "
                      f"val_loss={val_loss:.6f} | lr={lr:.2e} {improved}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self._load(model_name)
        print(f"训练完成，最佳val_loss={self.best_loss:.6f}")
        return self.train_losses, self.val_losses

    def _save(self, name):
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def _load(self, name):
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])


def build_model(input_dim, model_type="lstm"):
    """构建模型工厂函数"""
    if model_type == "lstm":
        return LSTMPredictor(input_dim)
    elif model_type == "gru":
        return GRUPredictor(input_dim)
    elif model_type == "physics_lstm":
        state_dim = len(STATE_COLS)
        control_dim = len(CONTROL_COLS)
        return PhysicsInformedLSTM(input_dim, state_dim, control_dim)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


if __name__ == "__main__":
    from data_preprocessing import full_preprocessing_pipeline

    train_loader, val_loader, test_loader, _, _ = full_preprocessing_pipeline(
        max_files=3, window_size=30)

    sample_X, sample_y = next(iter(train_loader))
    input_dim = sample_X.shape[-1]
    print(f"Input dim: {input_dim}")

    model = build_model(input_dim, "lstm")
    trainer = ModelTrainer(model)
    train_losses, val_losses = trainer.fit(train_loader, val_loader, epochs=30)

    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
