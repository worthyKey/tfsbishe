"""诊断SVM预测问题"""
import warnings, numpy as np
warnings.filterwarnings("ignore")

from data_preprocessing import full_preprocessing_pipeline
from config import STATE_COLS, CONTROL_COLS

print("Loading data...")
train_loader, val_loader, test_loader, scaler, df = full_preprocessing_pipeline(
    max_files=20, window_size=30, stride=20)

state_cols = [c for c in STATE_COLS if c in df.columns]
ctrl_cols = [c for c in CONTROL_COLS if c in df.columns]

states = df[state_cols].values
controls = df[ctrl_cols].values

print(f"\n数据统计 (原始值):")
for i, col in enumerate(state_cols):
    print(f"  {col}: mean={states[:,i].mean():.2f}, std={states[:,i].std():.2f}, "
          f"min={states[:,i].min():.2f}, max={states[:,i].max():.2f}")
for i, col in enumerate(ctrl_cols):
    print(f"  {col}: mean={controls[:,i].mean():.2f}, std={controls[:,i].std():.2f}, "
          f"min={controls[:,i].min():.2f}, max={controls[:,i].max():.2f}")

# 训练SVM
from optimization import SVMSurrogateModel
print("\n训练SVM...")
sample_size = min(5000, len(states))
indices = np.random.choice(len(states), sample_size, replace=False)
X_sample = controls[indices]
y_sample = states[indices]

svm = SVMSurrogateModel(kernel="rbf", C=10.0, epsilon=0.01)
svm.fit(X_sample, y_sample)

# 测试SVM对随机输入的预测
print("\nSVM预测测试 (10个随机控制输入):")
for i in range(10):
    if i == 0:
        # 使用一个真实数据点
        ctrl = controls[100]
    else:
        ctrl = np.random.uniform(controls.min(axis=0)*0.8, controls.max(axis=0)*1.2, len(ctrl_cols))
    pred = svm.predict(ctrl.reshape(1, -1))[0]
    print(f"  Input  {i}: {[f'{c:.0f}' for c in ctrl]}")
    print(f"  Output {i}: {[f'{p:.2f}' for p in pred]}")
    print()

# 检查：用不同的控制参数，SVM预测是否变化？
print("\n变量敏感性测试:")
base_ctrl = controls.mean(axis=0)
for ci in range(len(ctrl_cols)):
    low = base_ctrl.copy()
    low[ci] = controls[:, ci].min()
    high = base_ctrl.copy()
    high[ci] = controls[:, ci].max()
    pred_low = svm.predict(low.reshape(1, -1))[0]
    pred_high = svm.predict(high.reshape(1, -1))[0]
    diff = np.abs(pred_low - pred_high)
    print(f"  {ctrl_cols[ci]}: low→high diff = {[f'{d:.3f}' for d in diff[:3]]}...")
