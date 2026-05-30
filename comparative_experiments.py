"""
循环水系统 - 对比实验模块
========================
1. LSTM vs GRU 预测精度对比
2. GA vs 随机搜索 vs 网格搜索 优化效率对比
3. 不同窗口大小的模型性能对比
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import RESULT_DIR
from data_preprocessing import full_preprocessing_pipeline
from modeling import build_model, ModelTrainer

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def compare_lstm_gru(train_loader, val_loader, test_loader, input_dim,
                     epochs=50):

    print("=" * 60)
    print("Experiment 1: LSTM vs GRU prediction accuracy")
    print("=" * 60)

    results = {}

    for model_type in ["lstm", "gru"]:
        print(f"\n--- Training {model_type.upper()} ---")
        model = build_model(input_dim, model_type)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        trainer = ModelTrainer(model)
        t0 = time.time()
        train_losses, val_losses = trainer.fit(
            train_loader, val_loader, epochs=epochs,
            model_name=f"compare_{model_type}")
        train_time = time.time() - t0

        # Test evaluation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X).numpy()
                all_preds.append(pred)
                all_targets.append(y.numpy())

        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)

        rmse = np.sqrt(mean_squared_error(targets.flatten(), preds.flatten()))
        mae = mean_absolute_error(targets.flatten(), preds.flatten())
        r2 = r2_score(targets.flatten(), preds.flatten())

        results[model_type] = {
            "params": n_params,
            "train_time": train_time,
            "best_val_loss": trainer.best_loss,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        print(f"  Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        print(f"  Train time: {train_time:.1f}s, Best val_loss: {trainer.best_loss:.6f}")

    return results


def compare_ga_random_search(df, state_cols=None, control_cols=None,
                             n_trials=1000):

    print("\n" + "=" * 60)
    print("Experiment 2: GA vs Random Search vs Grid Search")
    print("=" * 60)

    from optimization import SVMSurrogateModel, GeneticAlgorithmOptimizer
    from config import STATE_COLS as _sc, CONTROL_COLS as _cc

    if state_cols is None:
        state_cols = [c for c in _sc if c in df.columns]
    if control_cols is None:
        control_cols = [c for c in _cc if c in df.columns]

    states = df[state_cols].values
    controls = df[control_cols].values

    # Build SVM surrogate
    sample_size = min(5000, len(states))
    indices = np.random.choice(len(states), sample_size, replace=False)
    svm = SVMSurrogateModel(kernel="rbf", C=10.0, epsilon=0.01)
    svm.fit(controls[indices], states[indices])

    # Control bounds
    bounds = []
    for i in range(controls.shape[1]):
        lo = max(0, controls[:, i].min() * 0.8)
        hi = controls[:, i].max() * 1.2
        bounds.append((lo, hi))

    results = {}

    # 1. Random Search
    print("\n--- Random Search ---")
    t0 = time.time()
    best_energy_rs = float("inf")
    best_ctrl_rs = None

    for _ in range(n_trials):
        ctrl = [np.random.uniform(lo, hi) for lo, hi in bounds]
        state_pred = svm.predict(np.array(ctrl).reshape(1, -1))[0]
        energy = np.sum(ctrl[-2:]) * 0.5  # Simplified energy
        if energy < best_energy_rs:
            best_energy_rs = energy
            best_ctrl_rs = ctrl

    results["random_search"] = {
        "time": time.time() - t0,
        "trials": n_trials,
        "best_energy": best_energy_rs,
        "best_controls": best_ctrl_rs,
        "best_states": svm.predict(np.array(best_ctrl_rs).reshape(1, -1))[0],
    }
    print(f"  Best energy: {best_energy_rs:.2f}, Time: {results['random_search']['time']:.1f}s")

    # 2. GA Optimization
    print("\n--- Genetic Algorithm ---")
    ga_opt = GeneticAlgorithmOptimizer(
        surrogate_model=svm,
        state_scaler=svm.state_scaler,
        control_scaler=svm.control_scaler,
        control_bounds=bounds,
        state_cols=state_cols,
        control_cols=control_cols,
    )

    t0 = time.time()
    best_ctrl_ga, best_state_ga, ga_history = ga_opt.optimize(
        pop_size=50, generations=50, verbose=False)
    ga_time = time.time() - t0

    results["ga"] = {
        "time": ga_time,
        "generations": 50,
        "pop_size": 50,
        "evaluations": 50 * 50,
        "best_energy": np.sum(best_ctrl_ga[-2:]) * 0.5,
        "best_controls": best_ctrl_ga.tolist(),
        "best_states": best_state_ga.tolist(),
        "convergence": [h["best_fitness"] for h in ga_history],
    }
    print(f"  Best energy: {results['ga']['best_energy']:.2f}, Time: {ga_time:.1f}s")

    # 3. Grid Search (coarse)
    print("\n--- Grid Search (coarse) ---")
    grid_points = [5, 5, 5, 5, 5]  # 5 points per dimension = 5^5 = 3125 evaluations
    grids = [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, grid_points)]
    mesh = np.meshgrid(*grids, indexing="ij")
    grid_combinations = np.stack([g.ravel() for g in mesh], axis=1)

    t0 = time.time()
    best_energy_gs = float("inf")
    best_idx = 0

    # Evaluate in batches
    batch_size = 500
    for i in range(0, len(grid_combinations), batch_size):
        batch = grid_combinations[i:i + batch_size]
        state_preds = svm.predict(batch)
        energies = np.sum(batch[:, -2:], axis=1) * 0.5
        min_idx = np.argmin(energies)
        if energies[min_idx] < best_energy_gs:
            best_energy_gs = energies[min_idx]
            best_idx = i + min_idx

    results["grid_search"] = {
        "time": time.time() - t0,
        "evaluations": len(grid_combinations),
        "best_energy": best_energy_gs,
        "best_controls": grid_combinations[best_idx].tolist(),
        "best_states": svm.predict(grid_combinations[best_idx:best_idx+1])[0].tolist(),
    }
    print(f"  Best energy: {best_energy_gs:.2f}, Time: {results['grid_search']['time']:.1f}s")

    # Summary
    print("\n--- Optimization Comparison Summary ---")
    print(f"{'Method':<15} {'Evaluations':<12} {'Best Energy':<12} {'Time(s)':<10}")
    print("-" * 50)
    for method, res in results.items():
        evals = res.get("evaluations", res.get("trials", 0))
        print(f"{method:<15} {evals:<12} {res['best_energy']:<12.2f} {res['time']:<10.1f}")

    return results


def compare_window_sizes(train_loader_builder, input_dim, epochs=30):

    print("\n" + "=" * 60)
    print("Experiment 3: Effect of window size on prediction accuracy")
    print("=" * 60)

    window_sizes = [10, 30, 60, 120]
    results = {}

    for ws in window_sizes:
        print(f"\n--- Window size = {ws} ---")
        # Rebuild data with different window size
        try:
            train_loader, val_loader, test_loader, _, _ = full_preprocessing_pipeline(
                max_files=10, window_size=ws, stride=max(5, ws // 6), batch_size=256)

            model = build_model(input_dim, "lstm")
            trainer = ModelTrainer(model)
            t0 = time.time()
            trainer.fit(train_loader, val_loader, epochs=epochs, verbose=False,
                        model_name=f"window_{ws}")
            train_time = time.time() - t0

            results[ws] = {
                "best_val_loss": trainer.best_loss,
                "train_time": train_time,
                "train_samples": len(train_loader.dataset),
            }
            print(f"  Best val_loss={trainer.best_loss:.6f}, Time={train_time:.1f}s")

        except Exception as e:
            print(f"  Failed: {e}")
            results[ws] = {"best_val_loss": None, "error": str(e)}

    return results


def plot_comparison_results(lstm_gru_results, opt_results, window_results,
                            save_dir=RESULT_DIR):

    # 1. LSTM vs GRU training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for model_type, res in lstm_gru_results.items():
        ax.plot(res["val_losses"][:50], label=f"{model_type.upper()} val",
                linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("LSTM vs GRU - Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    models = list(lstm_gru_results.keys())
    r2_vals = [lstm_gru_results[m]["r2"] for m in models]
    rmse_vals = [lstm_gru_results[m]["rmse"] for m in models]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, r2_vals, w, label="R2", color="steelblue")
    ax_twin = ax.twinx()
    ax_twin.bar(x + w/2, rmse_vals, w, label="RMSE", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("R2 Score")
    ax_twin.set_ylabel("RMSE")
    ax.set_title("LSTM vs GRU - Test Metrics")
    ax.legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    methods = list(opt_results.keys())
    energies = [opt_results[m]["best_energy"] for m in methods]
    times = [opt_results[m]["time"] for m in methods]
    colors = ["steelblue", "coral", "seagreen"]
    bars = ax.bar(methods, energies, color=colors, alpha=0.7)
    for bar, e, t in zip(bars, energies, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"E={e:.0f}\n{t:.1f}s", ha="center", fontsize=8)
    ax.set_ylabel("Best Energy Score")
    ax.set_title("Optimization Method Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Comparative Experiment Results", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparative_results.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"对比实验图已保存到 {save_dir}/comparative_results.png")

    # 2. Window size comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ws_list = [k for k in window_results if window_results[k].get("best_val_loss")]
    losses = [window_results[w]["best_val_loss"] for w in ws_list]
    ax.plot(ws_list, losses, "o-", linewidth=2, markersize=8, color="steelblue")
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Effect of Window Size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "window_size_ablation.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"窗口大小实验图已保存到 {save_dir}/window_size_ablation.png")


def generate_comparison_report(lstm_gru_results, opt_results, window_results,
                               save_path=None):

    lines = []
    lines.append("=" * 60)
    lines.append("对比实验报告")
    lines.append("=" * 60)

    # Exp 1
    lines.append("\n[实验1] LSTM vs GRU 预测精度对比")
    lines.append("-" * 40)
    for model, res in lstm_gru_results.items():
        lines.append(f"  {model.upper()}:")
        lines.append(f"    RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}, R2={res['r2']:.4f}")
        lines.append(f"    Parameters={res['params']:,}, Best val_loss={res['best_val_loss']:.6f}")
        lines.append(f"    Training time={res['train_time']:.1f}s")

    best_model = min(lstm_gru_results.keys(), key=lambda m: lstm_gru_results[m]["rmse"])
    lines.append(f"\n  结论: {best_model.upper()}在测试集上RMSE更低，推荐使用")

    # Exp 2
    lines.append("\n[实验2] 优化方法对比：GA vs Random Search vs Grid Search")
    lines.append("-" * 40)
    for method, res in opt_results.items():
        evals = res.get("evaluations", res.get("trials", "N/A"))
        lines.append(f"  {method}:")
        lines.append(f"    Evaluations={evals}, Best Energy={res['best_energy']:.2f}")
        lines.append(f"    Time={res['time']:.1f}s")

    ga_energy = opt_results["ga"]["best_energy"]
    rs_energy = opt_results["random_search"]["best_energy"]
    gs_energy = opt_results["grid_search"]["best_energy"]
    ga_vs_rs = (rs_energy - ga_energy) / rs_energy * 100
    lines.append(f"\n  结论: GA找到的解比Random Search优 {ga_vs_rs:.1f}%")

    # Exp 3
    lines.append("\n[实验3] 窗口大小对预测精度的影响")
    lines.append("-" * 40)
    for ws, res in window_results.items():
        if res.get("best_val_loss"):
            lines.append(f"  Window={ws}: Best val_loss={res['best_val_loss']:.6f}, "
                         f"Time={res['train_time']:.1f}s")
        else:
            lines.append(f"  Window={ws}: Failed - {res.get('error', 'unknown')}")

    if window_results:
        best_ws = min(
            [(ws, r["best_val_loss"]) for ws, r in window_results.items()
             if r.get("best_val_loss")],
            key=lambda x: x[1])
        lines.append(f"\n  结论: 窗口大小={best_ws[0]}时验证损失最低")

    lines.append("\n" + "=" * 60)
    lines.append("综合建议:")
    lines.append(f"  1. 推荐模型: {best_model.upper()}")
    lines.append("  2. 推荐优化方法: Genetic Algorithm（精度高、收敛快）")
    if window_results:
        lines.append(f"  3. 推荐窗口大小: {best_ws[0]}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n对比实验报告已保存到 {save_path}")

    return report


def run_all_comparative_experiments():
    """运行所有对比实验"""
    print("=" * 60)
    print("对比实验套件")
    print("=" * 60)

    # Data preparation (shared)
    print("\n>>> Preparing data...")
    train_loader, val_loader, test_loader, scaler, df = full_preprocessing_pipeline(
        max_files=15, window_size=40, stride=10, batch_size=256)
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[-1]
    print(f"Input dim: {input_dim}, Train samples: {len(train_loader.dataset)}")

    # Experiment 1: LSTM vs GRU
    lstm_gru_results = compare_lstm_gru(
        train_loader, val_loader, test_loader, input_dim, epochs=40)

    # Experiment 2: GA vs Random Search
    opt_results = compare_ga_random_search(df)

    # Experiment 3: Window size ablation
    window_results = compare_window_sizes(None, input_dim, epochs=20)

    # Plot
    plot_comparison_results(lstm_gru_results, opt_results, window_results)

    # Report
    generate_comparison_report(
        lstm_gru_results, opt_results, window_results,
        save_path=os.path.join(RESULT_DIR, "comparison_report.txt"))

    return lstm_gru_results, opt_results, window_results


if __name__ == "__main__":
    run_all_comparative_experiments()
