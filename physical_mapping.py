"""
循环水系统 - 物理映射模块
建立传感器测量值到换热器入口控制目标的物理关系

系统管路结构：
  冷水罐 → 增压泵 → [T1, P1] → ┬─ M1 → 流量计1 → [T2] → 热交换器 → 热水排出(M4)
                              │
                              └─ M2 → 单向阀 → 泄压排水

  热交换后 → ┬─ M4 → 单向阀 → 热水排出
            │
            └─ M3 → 与冷水混合(回收) → 回到换热器入口

控制目标（换热器入口处）：
  - 流量: 27 m³/h (换热器入口)
  - 温度 T2: 22°C (换热器入口)
  - 压力 P1: ≤3 MPa (系统主管)

传感器测量点：
  - temp_pump_out: 增压泵出口温度（T1附近）
  - temp_tank_out: 冷水罐出口温度
  - temp_he_sim: 换热器模拟出口温度
  - press_pump_out: 增压泵出口压力（P1）
  - flow_pump_DN300: DN300管路流量（M1+M2分流前总流量）
  - flow_pump_DN400: DN400管路流量
  - valve_DN200_fb: M1阀位反馈
  - valve_DN300_fb: M2阀位反馈
  - valve_DN350_fb: M3阀位反馈
  - valve_DN400_fb: M4阀位反馈
  - pump_speed_fb: 增压泵转速
  - pump_current_fb: 增压泵电流

物理映射关系：
  1. 流量映射：换热器入口流量 ≈ M1阀位比例 × 主管流量
     由于M1是全量程电动阀，流量与阀位开度近似线性（在固定压差下）
     Q_he = k_flow * valve_M1_position

  2. 温度映射：换热器入口温度T2 = 混合温度
     T2 = (Q_cold * T_cold + Q_recycle * T_hot) / (Q_cold + Q_recycle)
     其中 Q_cold = M1阀位 * 主管流量
          Q_recycle = M3阀位 * 循环流量
     简化：T2 = alpha * T_tank + (1-alpha) * T_he_outlet
     其中 alpha = f(M1, M3) 为混合比例

  3. 压力映射：P1 ≈ 泵出口压力 - 管路沿程损失
     P1 = press_pump_out（直接测量）
     压力控制通过M2泄压阀实现

  4. 能耗映射：总能耗 ≈ 泵功率 = f(泵转速, 泵电流)
     P_total = pump_current * voltage（电压恒定）
     简化：E ≈ k * pump_current
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


class PhysicalMapping:
    """传感器数据到控制目标的物理映射模型"""

    def __init__(self):
        self.flow_model = None      # 换热器入口流量映射
        self.temp_model = None      # 换热器入口温度T2映射
        self.pressure_identity = True  # P1直接测量
        self.energy_coefficient = None  # 能耗系数

    def fit_flow_mapping(self, df):
        """建立换热器入口流量映射
        Q_he = f(valve_M1, flow_total, press_diff)
        由于没有直接测量换热器入口流量，基于物理原理推断：
        - 换热器入口流量与M1阀位开度成正比
        - 与M2阀位开度成反比（M2分流越多，M1得到的越少）
        - 与泵转速正相关
        """
        X = df[["valve_DN200_fb", "valve_DN300_fb", "pump_speed_fb",
                "flow_pump_DN300"]].values
        # 目标换热器入口流量：通过M1的流量 ≈ 总流量 * M1开度比例
        # 估算：Q_he_estimated = flow_pump_DN300 * valve_DN200 / (valve_DN200 + valve_DN300)
        # 这是基于并联管路分流原理的近似
        m1 = df["valve_DN200_fb"].values
        m2 = df["valve_DN300_fb"].values
        total_flow = df["flow_pump_DN300"].values

        # 避免除零
        m_sum = np.maximum(m1 + m2, 0.1)
        q_he_estimated = total_flow * m1 / m_sum

        # 拟合模型
        self.flow_model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=1.0)
        )
        self.flow_model.fit(X, q_he_estimated)

        # 计算拟合精度
        y_pred = self.flow_model.predict(X)
        r2 = 1 - np.sum((q_he_estimated - y_pred) ** 2) / np.sum(
            (q_he_estimated - q_he_estimated.mean()) ** 2)
        print(f"流量映射模型 R2={r2:.4f}")
        print(f"  换热器入口估算流量范围: {q_he_estimated.min():.1f} ~ {q_he_estimated.max():.1f} m3/h")
        print(f"  换热器入口估算流量均值: {q_he_estimated.mean():.1f} m3/h")

        return self

    def fit_temp_mapping(self, df):
        """建立换热器入口温度T2映射
        T2 = 冷水与回收热水的混合温度
        T2 = f(T_tank, T_he_out, valve_M3)
        M3开度决定回收热水比例
        """
        X = df[["temp_tank_out", "temp_he_sim", "valve_DN350_fb",
                "valve_DN200_fb"]].values
        # T2估算：冷水罐温度和换热器出口温度的加权平均
        # 权重由M3阀位（回收比例）决定
        m3 = df["valve_DN350_fb"].values
        # M3归一化到[0,1]：99.6~99.7基本恒定，实际变化范围很小
        m3_norm = (m3 - m3.min()) / max(m3.max() - m3.min(), 1e-6)

        t_cold = df["temp_tank_out"].values
        t_hot = df["temp_he_sim"].values

        # 混合温度：M3越大 → 回收越多 → T2越高
        alpha = 0.7 - 0.3 * m3_norm  # 冷水比例在40%~70%之间
        t2_estimated = alpha * t_cold + (1 - alpha) * t_hot

        # 拟合模型
        self.temp_model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=1.0)
        )
        self.temp_model.fit(X, t2_estimated)

        y_pred = self.temp_model.predict(X)
        r2 = 1 - np.sum((t2_estimated - y_pred) ** 2) / np.sum(
            (t2_estimated - t2_estimated.mean()) ** 2)
        print(f"温度映射模型 R2={r2:.4f}")
        print(f"  换热器入口估算温度范围: {t2_estimated.min():.1f} ~ {t2_estimated.max():.1f} C")
        print(f"  换热器入口估算温度均值: {t2_estimated.mean():.1f} C")

        return self

    def fit_energy_model(self, df):
        """建立能耗模型
        E = f(pump_speed, pump_current, valve_positions)
        简化：E ∝ pump_current（电流直接反映功率）
        """
        self.energy_coefficient = {
            "voltage": 380.0,  # 标准工业电压(V)
            "power_factor": 0.85,  # 功率因数
        }
        # 三相电机功率: P = sqrt(3) * U * I * cos_phi
        current = df["pump_current_fb"].values
        power = np.sqrt(3) * 380 * current * 0.85 / 1000  # kW
        print(f"能耗模型:")
        print(f"  平均功率: {power.mean():.1f} kW")
        print(f"  功率范围: {power.min():.1f} ~ {power.max():.1f} kW")
        print(f"  估算日能耗: {power.mean() * 24:.1f} kWh")

        return self

    def map_to_targets(self, sensor_values):
        """将传感器值映射到控制目标值"""
        n = len(sensor_values)
        targets = np.zeros((n, 3))  # [flow, temp, pressure]

        # 流量映射
        if self.flow_model is not None:
            # 需要 M1, M2, pump_speed, flow_total 四个特征
            x_flow = sensor_values[:, [0, 1, 4, 6]] if sensor_values.shape[1] >= 7 else sensor_values
            targets[:, 0] = self.flow_model.predict(x_flow)

        # 温度映射
        if self.temp_model is not None:
            x_temp = sensor_values[:, [2, 4, 2, 0]]  # 近似
            targets[:, 1] = self.temp_model.predict(x_temp)

        # 压力：直接测量
        if sensor_values.shape[1] >= 4:
            targets[:, 2] = sensor_values[:, 3]
        else:
            targets[:, 2] = sensor_values[:, 0]

        return targets

    def fit_all(self, df):
        """训练所有物理映射模型"""
        print("=" * 50)
        print("建立传感器→控制目标物理映射")
        print("=" * 50)
        self.fit_flow_mapping(df)
        self.fit_temp_mapping(df)
        self.fit_energy_model(df)
        print("=" * 50)
        return self


class TargetCalculator:
    """计算传感器测量值对应的控制目标值（用于奖励函数和约束检查）"""

    @staticmethod
    def compute_flow_target(df):
        """估算换热器入口流量（基于管路分流原理）"""
        m1 = df["valve_DN200_fb"].values
        m2 = df["valve_DN300_fb"].values
        flow_total = df["flow_pump_DN300"].values
        m_sum = np.maximum(m1 + m2, 0.1)
        return flow_total * m1 / m_sum

    @staticmethod
    def compute_temp_target(df):
        """估算换热器入口温度（基于混合原理）"""
        m3 = df["valve_DN350_fb"].values
        m3_norm = (m3 - m3.min()) / max(m3.max() - m3.min(), 1e-6)

        t_cold = df["temp_tank_out"].values
        t_hot = df["temp_he_sim"].values
        alpha = 0.7 - 0.3 * m3_norm
        return alpha * t_cold + (1 - alpha) * t_hot

    @staticmethod
    def compute_energy(df):
        """计算瞬时功率 (kW)"""
        current = df["pump_current_fb"].values
        return np.sqrt(3) * 380 * current * 0.85 / 1000

    @staticmethod
    def add_target_columns(df):
        """为DataFrame添加控制目标列"""
        df = df.copy()
        df["target_flow_he"] = TargetCalculator.compute_flow_target(df)
        df["target_temp_t2"] = TargetCalculator.compute_temp_target(df)
        df["target_energy_kw"] = TargetCalculator.compute_energy(df)
        return df


if __name__ == "__main__":
    from data_preprocessing import DataLoader_CWS
    import warnings
    warnings.filterwarnings("ignore")

    loader = DataLoader_CWS()
    df = loader.load_and_merge(max_files=30)

    # 建立物理映射
    pm = PhysicalMapping()
    pm.fit_all(df)

    # 计算控制目标值
    df_targets = TargetCalculator.add_target_columns(df)
    print("\n控制目标统计:")
    for col in ["target_flow_he", "target_temp_t2", "target_energy_kw"]:
        if col in df_targets.columns:
            s = df_targets[col]
            print(f"  {col}: mean={s.mean():.2f}, std={s.std():.2f}, "
                  f"min={s.min():.2f}, max={s.max():.2f}")
