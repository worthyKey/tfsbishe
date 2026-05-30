"""
循环水系统 - 评估与可视化模块
计算 RMSE、MAE、R2 等指标，生成可视化图表
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import RESULT_DIR, TARGET_FLOW, TARGET_TEMP, MAX_PRESSURE

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def compute_metrics(y_true, y_pred):
    """计算回归评估指标"""
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # MAPE
    mask = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0.0

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE(%)": mape, "MSE": mse}


def evaluate_model(model, test_loader, device="cpu", scaler=None):
    """全面评估模型性能"""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y.numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # 如果提供了scaler，反标准化
    if scaler is not None:
        try:
            preds = scaler.inverse_transform(preds)
            targets = scaler.inverse_transform(targets)
        except Exception:
            pass

    # 逐特征计算指标
    results = {}
    n_features = preds.shape[1]
    for i in range(n_features):
        results[f"feature_{i}"] = compute_metrics(targets[:, i], preds[:, i])

    # 总体指标
    results["overall"] = compute_metrics(targets.flatten(), preds.flatten())

    return results, preds, targets


def evaluate_control_performance(predictions, targets, state_cols=None):
    """评估控制性能：检查是否满足技术指标"""
    report = {}

    if state_cols is None:
        from config import STATE_COLS as state_cols

    n_states = min(len(state_cols), predictions.shape[1])

    for i in range(n_states):
        col = state_cols[i] if i < len(state_cols) else f"var_{i}"
        pred = predictions[:, i]
        true = targets[:, i]

        if "flow" in col.lower():
            report[col] = {
                "target": TARGET_FLOW,
                "tolerance": 1.0,
                "mean_pred": float(np.mean(pred)),
                "in_range": float(np.mean(np.abs(pred - TARGET_FLOW) <= 1.0) * 100),
            }
        elif "temp" in col.lower():
            report[col] = {
                "target": TARGET_TEMP,
                "tolerance": 2.0,
                "mean_pred": float(np.mean(pred)),
                "in_range": float(np.mean(np.abs(pred - TARGET_TEMP) <= 2.0) * 100),
            }
        elif "press" in col.lower():
            report[col] = {
                "target": f"<={MAX_PRESSURE}",
                "mean_pred": float(np.mean(pred)),
                "in_range": float(np.mean(pred <= MAX_PRESSURE) * 100),
            }

    return report


def plot_training_history(train_losses, val_losses, save_path=None):
    """绘制训练损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="训练损失", alpha=0.7)
    ax.plot(val_losses, label="验证损失", alpha=0.7)
    ax.set_xlabel("轮次")
    ax.set_ylabel("损失")
    ax.set_title("模型训练损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(preds, targets, feature_names=None, n_features=4,
                     save_path=None):
    """绘制预测值与真实值对比"""
    n = min(n_features, preds.shape[1])
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t = np.arange(min(500, len(preds)))

    for i in range(n):
        ax = axes[i]
        ax.plot(t, targets[t, i], label="真实值", alpha=0.7, linewidth=1)
        ax.plot(t, preds[t, i], label="预测值", alpha=0.7, linewidth=1,
                linestyle="--")
        name = feature_names[i] if feature_names and i < len(feature_names) else f"变量 {i+1}"
        ax.set_ylabel(name)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("时间步")
    fig.suptitle("模型预测 vs 真实值", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_distribution(preds, targets, save_path=None):
    """绘制误差分布直方图"""
    errors = (preds - targets).flatten()
    errors = errors[np.isfinite(errors)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(errors, bins=50, density=True, alpha=0.7, color="steelblue",
            edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("预测误差")
    ax.set_ylabel("概率密度")
    ax.set_title("误差分布直方图")

    ax = axes[1]
    ax.scatter(targets.flatten()[:2000], preds.flatten()[:2000],
               alpha=0.3, s=2)
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
            "r--", linewidth=1)
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.set_title("预测值 vs 真实值")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ga_convergence(history, save_path=None):
    """绘制遗传算法收敛曲线"""
    gens = [h["gen"] for h in history]
    fitnesses = [h["best_fitness"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, fitnesses, "b-", linewidth=1.5)
    ax.set_xlabel("代数")
    ax.set_ylabel("最佳适应度")
    ax.set_title("遗传算法收敛曲线")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drl_rewards(episode_rewards, save_path=None):
    """绘制DRL训练曲线（BC Actor Loss 或 DDPG奖励）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 判断数据类型：BC loss（递减）还是DDPG reward（递增）
    values = np.array(episode_rewards, dtype=np.float64)
    is_bc = len(values) > 0 and np.all(values >= 0) and np.median(values) < 1.0

    ax = axes[0]
    if is_bc and len(values) > 5:
        # BC训练：Actor Loss曲线
        ax.plot(values, "b-", linewidth=1.2, label="BC Actor Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("行为克隆(BC)预训练 - Actor网络损失")
        # 对数坐标使下降趋势更明显
        if np.min(values) > 0:
            ax.set_yscale("log")
        ax.legend()
    else:
        ax.plot(values, alpha=0.5, linewidth=0.8, color="steelblue")
        if len(values) >= 10:
            ma = np.convolve(values, np.ones(10) / 10, mode="valid")
            ax.plot(range(9, len(values)), ma, "r-", linewidth=1.5,
                    label="10回合滑动平均")
        ax.set_xlabel("回合")
        ax.set_ylabel("总奖励")
        ax.set_title("DRL训练奖励曲线")
        ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if is_bc and len(values) > 5:
        # BC训练：显示最终收敛状态
        ax.axhline(y=values[-1], color="r", linestyle="--", linewidth=1.2,
                   label=f"最终Loss={values[-1]:.6f}")
        ax.axhline(y=np.min(values), color="g", linestyle="--", linewidth=1.2,
                   label=f"最低Loss={np.min(values):.6f}")
        # 最后10个epoch的均值
        last10 = values[-min(10, len(values)):]
        ax.bar(0, np.mean(last10), color="steelblue", label=f"最后10 epoch均值={np.mean(last10):.6f}")
        ax.set_xlabel("")
        ax.set_ylabel("MSE Loss")
        ax.set_title("BC训练收敛指标")
        ax.legend()
    elif len(values) >= 20:
        window = min(50, len(values))
        ma = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(ma, "g-", linewidth=1.5)
        ax.set_xlabel("回合")
        ax.set_ylabel("滑动平均奖励")
        ax.set_title("平滑奖励趋势")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_system_overview(df, save_path=None):
    """绘制系统运行数据概览"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:8]
    n = len(numeric_cols)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    sample = df.iloc[:3000] if len(df) > 3000 else df
    for i, col in enumerate(numeric_cols):
        axes[i].plot(sample[col].values, linewidth=0.5, color=f"C{i}")
        axes[i].set_ylabel(col[:30])
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("采样序号")
    fig.suptitle("系统运行数据概览", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_report(metrics, control_perf, save_path=None):
    """生成评估报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("循环水系统模型评估报告")
    lines.append("=" * 60)

    lines.append("\n[模型预测精度]")
    for k, v in metrics.items():
        if isinstance(v, dict):
            lines.append(f"  {k}: RMSE={v['RMSE']:.4f}, MAE={v['MAE']:.4f}, "
                         f"R2={v['R2']:.4f}")

    if control_perf:
        lines.append("\n[控制性能评估]")
        for col, perf in control_perf.items():
            lines.append(f"  {col}:")
            lines.append(f"    目标: {perf['target']}, 均值: {perf['mean_pred']:.3f}")
            if "in_range" in perf:
                lines.append(f"    达标率: {perf['in_range']:.1f}%")

    lines.append("\n[技术指标验证]")
    lines.append(f"  流量控制精度: {TARGET_FLOW}+/-1 m3/h")
    lines.append(f"  温度控制精度: {TARGET_TEMP}+/-2 C")
    lines.append(f"  压力控制: ≤{MAX_PRESSURE} MPa")

    report = "\n".join(lines)
    print(report)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def full_evaluation(model, test_loader, df, scaler=None, state_cols=None):
    """完整评估流程"""
    print("=" * 60)
    print("开始完整评估...")

    # 1. 模型预测精度评估
    print("\n[1/4] 计算预测指标...")
    metrics, preds, targets = evaluate_model(model, test_loader, scaler=scaler)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}: RMSE={v['RMSE']:.4f}, MAE={v['MAE']:.4f}, R2={v['R2']:.4f}")

    # 2. 控制性能评估
    print("\n[2/4] 评估控制性能...")
    control_perf = evaluate_control_performance(preds, targets, state_cols)
    for col, perf in control_perf.items():
        if "in_range" in perf:
            print(f"  {col}: 达标率={perf['in_range']:.1f}%")

    # 3. 生成图表
    print("\n[3/4] 生成可视化图表...")
    plot_predictions(
        preds, targets,
        feature_names=state_cols,
        save_path=os.path.join(RESULT_DIR, "predictions.png"),
    )
    plot_error_distribution(
        preds, targets,
        save_path=os.path.join(RESULT_DIR, "error_distribution.png"),
    )
    plot_system_overview(
        df,
        save_path=os.path.join(RESULT_DIR, "system_overview.png"),
    )

    # 4. 生成报告
    print("\n[4/4] 生成评估报告...")
    report = generate_report(
        metrics, control_perf,
        save_path=os.path.join(RESULT_DIR, "evaluation_report.txt"),
    )

    print(f"\n评估完成！结果保存在 {RESULT_DIR}")
    return metrics, control_perf, report


if __name__ == "__main__":
    import torch
    from data_preprocessing import full_preprocessing_pipeline
    from modeling import build_model

    # 加载数据并测试评估
    train_loader, val_loader, test_loader, scaler, df = full_preprocessing_pipeline(
        max_files=3)

    sample_X, _ = next(iter(train_loader))
    model = build_model(sample_X.shape[-1], "lstm")

    with torch.no_grad():
        preds = model(sample_X).numpy()
        targets = sample_X[:, -1, :].numpy()

    metrics = compute_metrics(targets.flatten(), preds.flatten())
    print(f"快速评估: RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
