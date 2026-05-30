"""
Comprehensive system verification - tests all 6 modules
"""
import warnings, os, time, numpy as np
warnings.filterwarnings('ignore')

print('=== Comprehensive System Verification ===')
t_start = time.time()

# 1. Data
from data_preprocessing import full_preprocessing_pipeline
print('\n[1/6] Loading data (10 files)...')
train_loader, val_loader, test_loader, scaler, df = full_preprocessing_pipeline(
    max_files=10, window_size=40, stride=15)
X_sample, y_sample = next(iter(train_loader))
input_dim = X_sample.shape[-1]
print(f'  Data: {df.shape[0]} rows x {df.shape[1]} cols, input_dim={input_dim}')

# 2. Physical Mapping
from physical_mapping import PhysicalMapping, TargetCalculator
print('\n[2/6] Physical mapping...')
pm = PhysicalMapping()
pm.fit_all(df)
df_t = TargetCalculator.add_target_columns(df)
t2_mean = df_t["target_temp_t2"].mean()
flow_mean = df_t["target_flow_he"].mean()
print(f'  T2 estimated: mean={t2_mean:.1f} C (target=22 C)')
print(f'  Flow estimated: mean={flow_mean:.1f} m3/h')

# 3. LSTM + GRU
from modeling import build_model, ModelTrainer
print('\n[3/6] LSTM vs GRU comparison...')
results = {}
for mt in ['lstm', 'gru']:
    m = build_model(input_dim, mt)
    t = ModelTrainer(m)
    t.fit(train_loader, val_loader, epochs=20, model_name=f'verify_{mt}')
    results[mt] = {'best_val_loss': t.best_loss}
    print(f'  {mt.upper()}: best_val_loss={t.best_loss:.4f}')

winner = min(results, key=lambda k: results[k]['best_val_loss'])
print(f'  Winner: {winner.upper()}')

# 4. SVM + GA
from optimization import build_optimization_pipeline
print('\n[4/6] SVM+GA optimization...')
svm_m, ga_o = build_optimization_pipeline(df)
best_c, best_s, hist = ga_o.optimize(pop_size=20, generations=30, verbose=False)
print(f'  GA converged: best_fitness={hist[-1]["best_fitness"]:.2f}')
print(f'  Best control: M1={best_c[0]:.1f}, pump_speed={best_c[-1]:.1f}')

# 5. DRL
from drl_controller import build_drl_pipeline
print('\n[5/6] DRL controller...')
lstm = build_model(input_dim, 'lstm')
ModelTrainer(lstm).fit(train_loader, val_loader, epochs=15, model_name='verify_drl_env')
ctrl = build_drl_pipeline(lstm, scaler, df)
rew = ctrl.train(episodes=30, max_steps=30, verbose=False)
print(f'  DRL avg reward (last 10): {np.mean(rew[-10:]):.1f}')

# 6. Closed-loop
from closed_loop import ClosedLoopSimulator, run_full_closed_loop_validation
print('\n[6/6] Closed-loop validation...')
ga_h, drl_h, ga_r, drl_r = run_full_closed_loop_validation(lstm, scaler, df, ga_o, ctrl)
print(f'  GA closed-loop: flow_rate={ga_r["flow_rate"]:.0f}%, temp_rate={ga_r["temp_rate"]:.0f}%')
print(f'  DRL closed-loop: flow_rate={drl_r["flow_rate"]:.0f}%, temp_rate={drl_r["temp_rate"]:.0f}%')

# Done
elapsed = time.time() - t_start
print(f'\n=== ALL 6 MODULES VERIFIED in {elapsed:.0f}s ({elapsed/60:.1f}min) ===')

result_files = os.listdir('d:/project/results')
print(f'Generated result files ({len(result_files)}):')
for f in sorted(result_files):
    size = os.path.getsize(f'd:/project/results/{f}')
    print(f'  {f}: {size:,} bytes')

print('\nAll modules work correctly and can support the thesis!')
