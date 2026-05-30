"""
循环水系统数据驱动控制 - 配置文件
"""
import os

# 数据路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "shuju")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

for d in [MODEL_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# CSV 读取配置
CSV_ENCODING = "utf-16-le"
CSV_DELIMITER = ";"

# 数据预处理参数
TARGET_SAMPLING_RATE = "1s"       # 目标采样率
WINDOW_SIZE = 60                   # 滑动窗口大小（秒）
STRIDE = 10                        # 滑动步长
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 滤波器参数
SMOOTHING_METHOD = "savgol"        # savgol / moving_avg / lowpass
SAVGOL_WINDOW = 11
SAVGOL_ORDER = 3

# 特征列定义
FB_COLUMNS = [
    "timestamp",
    "valve_DN200_fb",      # DN200电动调节阀反馈
    "valve_DN300_fb",      # DN300电动调节阀反馈
    "valve_DN350_fb",      # DN350电动调节阀反馈
    "valve_DN400_fb",      # DN400电动调节阀反馈
    "pump_speed_fb",       # 增压泵转速反馈
    "pump_current_fb",     # 增压泵电流反馈
]

TP_COLUMNS = [
    "timestamp",
    "temp_pump_out",       # 增压泵后温度
    "temp_tank_out",       # 冷水罐出口温度
    "temp_he_sim",         # 换热器模拟出口温度
    "press_pump_out",      # 增压泵后压力
    "press_valve_DN400",   # DN400电动调节阀后压力
    "press_check_valve",   # 止回阀后压力
    "press_tank_out",      # 冷水罐出口压力
    "flow_pump_DN300",     # DN300增压泵后流量
    "flow_pump_DN400",     # DN400增压泵后流量
]

# 控制变量与状态变量映射
# 对应循环水系统控制回路：
# 温度回路：通过循环水流量u3控制换热器入水温度T2
# 压力回路：通过电动阀M2输出u2控制入水水压P1
# 流量回路：通过电动阀M1输出u1控制换热器入水流量
STATE_COLS = [
    "temp_pump_out",       # T2 - 换热器入水温度
    "press_pump_out",      # P1 - 入水水压
    "flow_pump_DN300",     # 换热器入水流量
    "temp_tank_out",       # 冷水罐出口温度
    "temp_he_sim",         # 换热器出口温度
    "press_tank_out",      # 冷水罐出口压力
]

CONTROL_COLS = [
    "valve_DN200_fb",      # M1 - 主线路电动阀
    "valve_DN300_fb",      # M2 - 泄压电动阀
    "valve_DN350_fb",      # M3 - 混合电动阀
    "valve_DN400_fb",      # M4 - 排水电动阀
    "pump_speed_fb",       # 增压泵转速
]

# 技术指标
TARGET_FLOW = 27.0          # m³/h
TARGET_TEMP = 22.0          # ℃
MAX_PRESSURE = 3.0          # MPa
FLOW_TOLERANCE = 1.0        # m³/h
TEMP_TOLERANCE = 2.0        # ℃

# LSTM 模型参数
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 3
LSTM_DROPOUT = 0.2
LSTM_BATCH_SIZE = 256
LSTM_EPOCHS = 100
LSTM_LEARNING_RATE = 1e-3

# SVM + GA 优化参数
GA_POPULATION_SIZE = 50
GA_GENERATIONS = 100
GA_MUTATION_RATE = 0.1
GA_CROSSOVER_RATE = 0.8

# DRL (DDPG) 参数
DRL_STATE_DIM = len(STATE_COLS)
DRL_ACTION_DIM = len(CONTROL_COLS)
DRL_HIDDEN_DIM = 256
DRL_ACTOR_LR = 1e-4
DRL_CRITIC_LR = 3e-4
DRL_GAMMA = 0.99
DRL_TAU = 0.005
DRL_MEMORY_SIZE = 100000
DRL_BATCH_SIZE = 128
DRL_MAX_EPISODES = 500
DRL_MAX_STEPS = 200
