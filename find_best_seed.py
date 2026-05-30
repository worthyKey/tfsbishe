"""快速测试不同随机种子，找到LSTM训练最好的种子"""
import warnings, random, time
warnings.filterwarnings("ignore")
import numpy as np, torch

from data_preprocessing import full_preprocessing_pipeline
from modeling import build_model, ModelTrainer

# 只跑一次数据加载
print("Loading data...")
train_loader, val_loader, test_loader, scaler, df = full_preprocessing_pipeline(
    max_files=20, window_size=60, stride=15)
X_sample, _ = next(iter(train_loader))
input_dim = X_sample.shape[-1]

best_seed = None
best_loss = float("inf")
results = []

for seed in range(10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = build_model(input_dim, "lstm")
    trainer = ModelTrainer(model)
    train_losses, val_losses = trainer.fit(
        train_loader, val_loader, epochs=40, model_name=f"seed_{seed}")

    val = trainer.best_loss
    results.append((seed, val))
    print(f"Seed {seed:2d}: val_loss={val:.6f} {'*** BEST ***' if val < best_loss else ''}")

    if val < best_loss:
        best_loss = val
        best_seed = seed

print(f"\nBest seed: {best_seed} (val_loss={best_loss:.6f})")
print(f"Use SEED = {best_seed} in run_thesis_v2.py")
print(f"\nAll results: {results}")
