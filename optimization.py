"""
循环水系统 - 参数优化模块
采用 SVM + 遗传算法(GA) 进行离线/在线运行参数优化
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import random
import warnings
warnings.filterwarnings("ignore")

from config import (GA_POPULATION_SIZE, GA_GENERATIONS, GA_MUTATION_RATE,
                    GA_CROSSOVER_RATE, TARGET_FLOW, TARGET_TEMP, MAX_PRESSURE,
                    FLOW_TOLERANCE, TEMP_TOLERANCE, STATE_COLS, CONTROL_COLS)


class SVMSurrogateModel:
    """基于 SVR 的代理模型，用于拟合控制变量到系统状态的映射"""

    def __init__(self, kernel="rbf", C=10.0, epsilon=0.01):
        self.state_scaler = StandardScaler()
        self.control_scaler = StandardScaler()
        self.model = MultiOutputRegressor(
            SVR(kernel=kernel, C=C, epsilon=epsilon, gamma="scale")
        )

    def fit(self, X_control, y_state):
        """训练 SVM 代理模型
        Args:
            X_control: (n_samples, n_controls) 控制变量
            y_state:   (n_samples, n_states)   系统状态
        """
        X_scaled = self.control_scaler.fit_transform(X_control)
        y_scaled = self.state_scaler.fit_transform(y_state)
        self.model.fit(X_scaled, y_scaled)
        # 计算训练精度
        y_pred = self.model.predict(X_scaled)
        mse = np.mean((y_scaled - y_pred) ** 2)
        print(f"SVM代理模型训练完成, MSE={mse:.6f}")
        return self

    def predict(self, X_control):
        """预测给定控制量对应的系统状态"""
        X_scaled = self.control_scaler.transform(X_control)
        y_scaled = self.model.predict(X_scaled)
        return self.state_scaler.inverse_transform(y_scaled)


class GeneticAlgorithmOptimizer:
    """遗传算法优化器
    目标：在满足约束条件下最小化能耗（增压泵电流最小）
    """

    def __init__(self, surrogate_model, state_scaler, control_scaler,
                 control_bounds, state_cols, control_cols):
        self.surrogate = surrogate_model
        self.state_scaler = state_scaler
        self.control_scaler = control_scaler
        self.bounds = np.array(control_bounds)  # (n_controls, 2)
        self.n_controls = len(control_bounds)
        self.state_cols = state_cols
        self.control_cols = control_cols

        # 确定各状态变量和目标的索引
        self.flow_idx = None
        self.temp_idx = None
        self.press_idx = None
        for i, col in enumerate(state_cols):
            if "flow" in col.lower():
                self.flow_idx = i
            if "temp" in col.lower() and self.temp_idx is None:
                self.temp_idx = i
            if "press" in col.lower() and self.press_idx is None:
                self.press_idx = i

        # 默认使用设计目标（会被closed_loop覆盖）
        self.equiv_flow_lo = TARGET_FLOW - FLOW_TOLERANCE
        self.equiv_flow_hi = TARGET_FLOW + FLOW_TOLERANCE
        self.equiv_temp_lo = TARGET_TEMP - TEMP_TOLERANCE
        self.equiv_temp_hi = TARGET_TEMP + TEMP_TOLERANCE

    def fitness(self, individual):
        """适应度函数：最小化能耗 + 约束惩罚（使用等效目标）"""
        controls = np.array(individual).reshape(1, -1)
        states = self.surrogate.predict(controls)[0]

        # 通过物理映射计算等效控制目标
        targets = self._compute_equivalent_from_svm(states, individual)
        penalty = 0.0

        # 流量约束：等效目标范围
        if targets["flow"] is not None:
            flow = targets["flow"]
            flow_center = (self.equiv_flow_lo + self.equiv_flow_hi) / 2
            flow_half = max(self.equiv_flow_hi - self.equiv_flow_lo, 1) / 2
            flow_dist = abs(flow - flow_center) / flow_half
            if flow_dist > 1.0:
                penalty += (flow_dist - 1.0) ** 2 * 500

        # 温度约束：等效目标范围
        if targets["temp"] is not None:
            temp = targets["temp"]
            temp_center = (self.equiv_temp_lo + self.equiv_temp_hi) / 2
            temp_half = max(self.equiv_temp_hi - self.equiv_temp_lo, 0.5) / 2
            temp_dist = abs(temp - temp_center) / temp_half
            if temp_dist > 1.0:
                penalty += (temp_dist - 1.0) ** 2 * 200

        # 压力约束：≤ 3 MPa
        if self.press_idx is not None and self.press_idx < len(states):
            press = states[self.press_idx]
            if press > MAX_PRESSURE:
                penalty += (press - MAX_PRESSURE) ** 2 * 500

        # 控制量范围约束
        for i, (lo, hi) in enumerate(self.bounds):
            if individual[i] < lo:
                penalty += (lo - individual[i]) ** 2 * 1000
            elif individual[i] > hi:
                penalty += (individual[i] - hi) ** 2 * 1000

        # 能耗
        ctrl_arr = np.array(individual)
        energy = np.mean(np.abs(ctrl_arr)) * 0.005

        return -(energy + penalty)  # 最大化适应度

    def _set_equivalent_targets(self, flow_lo, flow_hi, temp_lo, temp_hi):
        """设置等效控制目标（由ClosedLoopSimulator提供）"""
        self.equiv_flow_lo = flow_lo
        self.equiv_flow_hi = flow_hi
        self.equiv_temp_lo = temp_lo
        self.equiv_temp_hi = temp_hi

    def _compute_equivalent_from_svm(self, states, individual):
        """将SVM预测的传感器值映射为等效控制目标"""
        result = {"flow": None, "temp": None}
        ctrl = individual

        # 流量映射：Q_he = Q_total * M1/(M1+M2)
        if self.flow_idx is not None and self.flow_idx < len(states):
            total_flow = max(states[self.flow_idx], 0.01)
            # M1 = first control (valve_DN200), M2 = second (valve_DN300)
            m1 = max(ctrl[0], 0.1) if len(ctrl) > 0 else 42.0
            m2 = max(ctrl[1], 0.1) if len(ctrl) > 1 else 99.0
            result["flow"] = total_flow * m1 / (m1 + m2)

        # 温度映射：T2 = alpha*T_cold + (1-alpha)*T_he
        if self.temp_idx is not None:
            t_cold_idx = None
            t_hot_idx = None
            for i, col in enumerate(self.state_cols):
                low = col.lower()
                if "tank" in low:
                    t_cold_idx = i
                elif "he" in low:
                    t_hot_idx = i
            if t_cold_idx is not None and t_hot_idx is not None:
                if t_cold_idx < len(states) and t_hot_idx < len(states):
                    t_cold = states[t_cold_idx]
                    t_hot = states[t_hot_idx]
                    m3 = ctrl[2] if len(ctrl) > 2 else 99.7
                    m3_norm = max(0, min(1, (m3 - 99.6) / 0.15))
                    alpha = 0.7 - 0.3 * m3_norm
                    result["temp"] = alpha * t_cold + (1 - alpha) * t_hot

        return result

    def _create_individual(self):
        return [random.uniform(lo, hi) for lo, hi in self.bounds]

    def _crossover(self, p1, p2):
        if random.random() < GA_CROSSOVER_RATE:
            alpha = random.random()
            c1 = [alpha * a + (1 - alpha) * b for a, b in zip(p1, p2)]
            c2 = [(1 - alpha) * a + alpha * b for a, b in zip(p1, p2)]
            return c1, c2
        return p1[:], p2[:]

    def _mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < GA_MUTATION_RATE:
                lo, hi = self.bounds[i]
                sigma = (hi - lo) * 0.1
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(lo, min(hi, individual[i]))
        return individual

    def _tournament_select(self, population, fitnesses, k=3):
        idx = random.sample(range(len(population)), k)
        best = max(idx, key=lambda i: fitnesses[i])
        return population[best][:]

    def optimize(self, pop_size=GA_POPULATION_SIZE,
                 generations=GA_GENERATIONS, verbose=True):
        """运行遗传算法优化"""
        population = [self._create_individual() for _ in range(pop_size)]
        best_history = []

        for gen in range(generations):
            fitnesses = [self.fitness(ind) for ind in population]

            best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            best_history.append({
                "gen": gen,
                "best_fitness": fitnesses[best_idx],
                "best_individual": population[best_idx][:],
            })

            # 精英保留
            new_population = [population[best_idx][:]]

            # 生成新一代
            while len(new_population) < pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.extend([c1, c2])

            population = new_population[:pop_size]

            if verbose and gen % 20 == 0:
                best_ctrl = best_history[-1]["best_individual"]
                states = self.surrogate.predict(np.array(best_ctrl).reshape(1, -1))[0]
                print(f"Gen {gen:4d} | Fitness={fitnesses[best_idx]:.4f} | "
                      f"Energy={np.sum(best_ctrl[-2:])*.5:.2f}")

        best = best_history[-1]
        best_controls = np.array(best["best_individual"])
        best_states = self.surrogate.predict(best_controls.reshape(1, -1))[0]

        print(f"\n优化完成!")
        print(f"最优控制参数: {dict(zip(self.control_cols, best_controls))}")
        print(f"预测状态: {dict(zip(self.state_cols, best_states))}")

        return best_controls, best_states, best_history


class OnlineOptimizer:
    """在线优化器：结合实时数据持续调整控制参数"""

    def __init__(self, surrogate_model, ga_optimizer, update_interval=60):
        self.surrogate = surrogate_model
        self.ga = ga_optimizer
        self.update_interval = update_interval
        self.current_best = None
        self.history = []

    def step(self, current_state, time_step):
        """根据当前状态返回优化后的控制参数"""
        if time_step % self.update_interval == 0 or self.current_best is None:
            best_ctrl, best_state, _ = self.ga.optimize(
                pop_size=30, generations=30, verbose=False
            )
            self.current_best = best_ctrl
            self.history.append({
                "time_step": time_step,
                "control": best_ctrl,
                "state": best_state,
            })

        return self.current_best


def build_optimization_pipeline(df, state_cols=None, control_cols=None):
    """构建完整的优化流程"""
    if state_cols is None:
        state_cols = STATE_COLS
    if control_cols is None:
        control_cols = CONTROL_COLS

    # 提取数据
    states = df[[c for c in state_cols if c in df.columns]].values
    controls = df[[c for c in control_cols if c in df.columns]].values

    # 剔除异常值
    mask = np.ones(len(states), dtype=bool)
    for i in range(states.shape[1]):
        mean, std = states[:, i].mean(), states[:, i].std()
        if std > 0:
            mask &= (np.abs(states[:, i] - mean) < 3 * std)
    for i in range(controls.shape[1]):
        mean, std = controls[:, i].mean(), controls[:, i].std()
        if std > 0:
            mask &= (np.abs(controls[:, i] - mean) < 3 * std)

    states = states[mask]
    controls = controls[mask]

    # 为GA优化采样（SVR复杂度O(n²)，样本不宜过大）
    sample_size = min(5000, len(states))
    indices = np.random.choice(len(states), sample_size, replace=False)
    X_sample = controls[indices]
    y_sample = states[indices]
    print(f"SVM训练样本: {sample_size}")

    # 控制量边界：影响流量的变量(M1,泵速)用全范围，其他用均值±3.5σ
    control_bounds = []
    flow_ctrl_names = ["dn200", "speed"]  # 能降低Q_he的关键控制量
    for i in range(controls.shape[1]):
        mean_i = controls[:, i].mean()
        std_i = controls[:, i].std()
        if std_i < 1e-6:
            std_i = 0.1
        col_lower = control_cols[i].lower() if i < len(control_cols) else ""
        # 流量控制变量：使用min~max全范围（允许降流量到设计目标）
        is_flow_ctrl = any(kw in col_lower for kw in flow_ctrl_names)
        if is_flow_ctrl:
            lo = max(0, controls[:, i].min() * 0.5)
            hi = controls[:, i].max() * 1.1
        else:
            lo = max(0, mean_i - 3.5 * std_i)
            hi = mean_i + 3.5 * std_i
        control_bounds.append((float(lo), float(hi)))

    # 训练SVM代理模型
    svm_model = SVMSurrogateModel(kernel="rbf", C=20.0, epsilon=0.01)
    svm_model.fit(X_sample, y_sample)

    # 创建GA优化器
    ga_opt = GeneticAlgorithmOptimizer(
        surrogate_model=svm_model,
        state_scaler=svm_model.state_scaler,
        control_scaler=svm_model.control_scaler,
        control_bounds=control_bounds,
        state_cols=state_cols,
        control_cols=control_cols,
    )

    return svm_model, ga_opt


if __name__ == "__main__":
    from data_preprocessing import full_preprocessing_pipeline

    _, _, _, _, df = full_preprocessing_pipeline(max_files=3, window_size=10, stride=5)
    svm_model, ga_opt = build_optimization_pipeline(df)
    best_ctrl, best_state, history = ga_opt.optimize(
        pop_size=30, generations=50, verbose=True
    )
