"""Quick test: M1 flow + M3 temp physics correction in DRL closed-loop"""
import warnings, os, numpy as np, torch
warnings.filterwarnings("ignore")

from config import RESULT_DIR, STATE_COLS, CONTROL_COLS, MODEL_DIR
os.makedirs(RESULT_DIR, exist_ok=True)

from data_preprocessing import full_preprocessing_pipeline
from modeling import build_model

print("Loading data...")
train_loader, val_loader, test_loader, scaler, df = full_preprocessing_pipeline(
    max_files=20, window_size=60, stride=15)

lstm = build_model(next(iter(train_loader))[0].shape[-1], "lstm")
ckpt = torch.load(os.path.join(MODEL_DIR, "thesis_lstm.pt"), map_location="cpu")
lstm.load_state_dict(ckpt["model_state_dict"])
lstm.eval()

from drl_controller import build_drl_pipeline
drl_ctrl = build_drl_pipeline(lstm, scaler, df)
drl_ctrl.train(episodes=150, max_steps=80, verbose=True, pretrain=True, ddpg_finetune=False)

from optimization import build_optimization_pipeline
svm_m, ga_o = build_optimization_pipeline(df)

from closed_loop import run_full_closed_loop_validation
ga_h, drl_h, ga_r, drl_r = run_full_closed_loop_validation(lstm, scaler, df, ga_o, drl_ctrl)

print("\n" + "=" * 60)
print("RESULTS:")
print(f"  GA:  flow={ga_r['flow_rate']:.1f}%  temp={ga_r['temp_rate']:.1f}%  press={ga_r['pressure_rate']:.1f}%")
print(f"       mean_flow={ga_r['mean_flow']:.1f}  mean_temp={ga_r['mean_temp']:.2f}  energy={ga_r['mean_energy']:.1f}")
print(f"  DRL: flow={drl_r['flow_rate']:.1f}%  temp={drl_r['temp_rate']:.1f}%  press={drl_r['pressure_rate']:.1f}%")
print(f"       mean_flow={drl_r['mean_flow']:.1f}  mean_temp={drl_r['mean_temp']:.2f}  energy={drl_r['mean_energy']:.1f}")
print("=" * 60)
