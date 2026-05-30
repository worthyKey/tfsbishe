"""Streamlined thesis pipeline v2 - with temperature fixes"""
import warnings, os, time, random, numpy as np, torch
warnings.filterwarnings("ignore")

# 固定随机种子
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from config import RESULT_DIR, STATE_COLS, CONTROL_COLS, TARGET_FLOW, TARGET_TEMP, MAX_PRESSURE
os.makedirs(RESULT_DIR, exist_ok=True)
t0 = time.time()

# ===== Stage 1: Data =====
print("\n" + "=" * 60)
print("Stage 1/5: Data Loading (20 files)")
print("=" * 60)
from data_preprocessing import full_preprocessing_pipeline
train_loader, val_loader, test_loader, scaler, df = full_preprocessing_pipeline(
    max_files=20, window_size=60, stride=15)
X_sample, y_sample = next(iter(train_loader))
input_dim = X_sample.shape[-1]
print(f"Data: {df.shape[0]} rows, input_dim={input_dim}")

# ===== Stage 2: Physical Mapping =====
print("\n" + "=" * 60)
print("Stage 2/5: Physical Mapping")
print("=" * 60)
from physical_mapping import PhysicalMapping, TargetCalculator
pm = PhysicalMapping()
pm.fit_all(df)
df_t = TargetCalculator.add_target_columns(df)
print("Physical mapping completed")

# ===== Stage 3: LSTM =====
print("\n" + "=" * 60)
print("Stage 3/5: LSTM Training (50 epochs)")
print("=" * 60)
from modeling import build_model, ModelTrainer
lstm = build_model(input_dim, "lstm")
trainer = ModelTrainer(lstm)
train_losses, val_losses = trainer.fit(train_loader, val_loader, epochs=50, model_name="thesis_lstm")

from evaluation import (plot_training_history, plot_predictions,
                        plot_error_distribution, plot_system_overview)
plot_training_history(train_losses, val_losses,
                      save_path=os.path.join(RESULT_DIR, "training_history.png"))
plot_system_overview(df, save_path=os.path.join(RESULT_DIR, "system_overview.png"))

lstm.eval()
all_preds, all_targets = [], []
for X, y in test_loader:
    pred = lstm(X).detach().numpy()
    all_preds.append(pred)
    all_targets.append(y.numpy())
preds = np.vstack(all_preds)
targets = np.vstack(all_targets)
plot_predictions(preds, targets, save_path=os.path.join(RESULT_DIR, "predictions.png"))
plot_error_distribution(preds, targets, save_path=os.path.join(RESULT_DIR, "error_distribution.png"))

# ===== Stage 4: Optimization + DRL =====
print("\n" + "=" * 60)
print("Stage 4/5: SVM+GA Optimization + DRL Training")
print("=" * 60)

from optimization import build_optimization_pipeline
svm_m, ga_o = build_optimization_pipeline(df)
best_c, best_s, ga_hist = ga_o.optimize(pop_size=50, generations=80, verbose=False)
print(f"GA best fitness: {ga_hist[-1]['best_fitness']:.2f}")

from evaluation import plot_ga_convergence
plot_ga_convergence(ga_hist, save_path=os.path.join(RESULT_DIR, "ga_convergence.png"))

from drl_controller import build_drl_pipeline
drl_ctrl = build_drl_pipeline(lstm, scaler, df)
drl_rewards = drl_ctrl.train(episodes=150, max_steps=80, verbose=False,
                             pretrain=True, ddpg_finetune=False)
print(f"DRL avg reward (last 20): {np.mean(drl_rewards[-20:]):.1f}")

from evaluation import plot_drl_rewards
plot_drl_rewards(drl_rewards, save_path=os.path.join(RESULT_DIR, "drl_rewards.png"))

# ===== Stage 5: Closed-Loop =====
print("\n" + "=" * 60)
print("Stage 5/5: Closed-Loop Validation")
print("=" * 60)
from closed_loop import run_full_closed_loop_validation
ga_h, drl_h, ga_r, drl_r = run_full_closed_loop_validation(lstm, scaler, df, ga_o, drl_ctrl)

# ===== Report =====
print("\n" + "=" * 60)
print("Generating Thesis Report")
print("=" * 60)

from evaluation import compute_metrics
test_metrics = {}
sc_used = [c for c in STATE_COLS if c in df.columns]
for i, col in enumerate(sc_used[:preds.shape[1]]):
    m = compute_metrics(preds[:, i], targets[:, i])
    test_metrics[col] = m

overall_rmse = np.sqrt(np.mean((preds - targets) ** 2))

report_path = os.path.join(RESULT_DIR, "thesis_report.txt")
lines = []
lines.append("=" * 70)
lines.append("  循环水系统数据驱动控制 - 论文实验结果")
lines.append(f"  生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 70)
lines.append("")
lines.append("-" * 70)
lines.append("1. 数据概况")
lines.append("-" * 70)
lines.append(f"  数据文件: 20对FB/TP (69对可用)")
lines.append(f"  总数据量: {df.shape[0]:,}行 x {df.shape[1]}列, 滑动窗口60步")
lines.append(f"  训练/验证/测试: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
lines.append("")
lines.append("-" * 70)
lines.append("2. 物理映射与等效目标")
lines.append("-" * 70)
lines.append(f"  流量映射: Q_he = Q_total * M1/(M1+M2)")
lines.append(f"    换热器入口估算均值: {df_t['target_flow_he'].mean():.0f} m3/h")
lines.append(f"  温度映射: T2 = alpha*T_cold + (1-alpha)*T_hot")
lines.append(f"    T2估算均值: {df_t['target_temp_t2'].mean():.1f} C")
lines.append(f"    设计目标: 流量{TARGET_FLOW} m3/h, 温度{TARGET_TEMP} C")
lines.append(f"  注: 等效目标为传感器值经物理映射后的值,与设计目标通过P&ID关联")
lines.append("")
lines.append("-" * 70)
lines.append("3. LSTM模型性能")
lines.append("-" * 70)
lines.append(f"  参数量: 977,291 | 最佳验证损失: {trainer.best_loss:.6f} | 整体RMSE: {overall_rmse:.6f}")
for col, m in test_metrics.items():
    lines.append(f"  {col}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
lines.append("")
lines.append("-" * 70)
lines.append("4. SVM+GA参数优化")
lines.append("-" * 70)
lines.append(f"  GA最佳适应度: {ga_hist[-1]['best_fitness']:.2f}")
ctrl_cols_used = [c for c in CONTROL_COLS if c in df.columns]
lines.append(f"  最佳控制: " + ", ".join([f"{c}={best_c[i]:.2f}" for i, c in enumerate(ctrl_cols_used)]))
lines.append("")
lines.append("-" * 70)
lines.append("5. DRL智能控制")
lines.append("-" * 70)
lines.append(f"  训练策略: GA行为克隆(BC)预训练, 150条专家轨迹")
lines.append(f"  Actor网络通过监督学习模仿GA优化器输出的最优控制动作")
lines.append("")
lines.append("-" * 70)
lines.append("6. 闭环仿真验证（核心结果）")
lines.append("-" * 70)
lines.append(f"  GA闭环: 流量达标率={ga_r['flow_rate']:.1f}%  温度达标率={ga_r['temp_rate']:.1f}%  压力达标率={ga_r['pressure_rate']:.1f}%")
lines.append(f"    平均流量={ga_r['mean_flow']:.1f} m3/h  平均温度={ga_r['mean_temp']:.2f} C  平均能耗={ga_r['mean_energy']:.1f}")
lines.append(f"  DRL闭环: 流量达标率={drl_r['flow_rate']:.1f}%  温度达标率={drl_r['temp_rate']:.1f}%  压力达标率={drl_r['pressure_rate']:.1f}%")
lines.append(f"    平均流量={drl_r['mean_flow']:.1f} m3/h  平均温度={drl_r['mean_temp']:.2f} C  平均能耗={drl_r['mean_energy']:.1f}")
lines.append("")
lines.append("=" * 70)

report = "\n".join(lines)
print("\n" + report)
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

elapsed = time.time() - t0
print(f"\nPipeline complete! Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
for f in sorted(os.listdir(RESULT_DIR)):
    if f.endswith(('.png', '.txt', '.csv')):
        size = os.path.getsize(os.path.join(RESULT_DIR, f))
        print(f"  {f}: {size:,} bytes")
