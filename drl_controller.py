"""
循环水系统 - 深度强化学习控制器 (DDPG)
智能体学习通过调整水泵频率和阀门开度来最小化总能耗
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import warnings
warnings.filterwarnings("ignore")

from config import (DRL_STATE_DIM, DRL_ACTION_DIM, DRL_HIDDEN_DIM,
                    DRL_ACTOR_LR, DRL_CRITIC_LR, DRL_GAMMA, DRL_TAU,
                    DRL_MEMORY_SIZE, DRL_BATCH_SIZE, DRL_MAX_EPISODES,
                    DRL_MAX_STEPS, MODEL_DIR, TARGET_FLOW, TARGET_TEMP,
                    MAX_PRESSURE, FLOW_TOLERANCE, TEMP_TOLERANCE)


class Actor(nn.Module):
    """策略网络：输出控制动作（阀门开度、泵转速）"""

    def __init__(self, state_dim, action_dim, hidden_dim=DRL_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid(),  # 输出 [0, 1]，映射到实际控制范围
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """价值网络：评估状态-动作对的Q值"""

    def __init__(self, state_dim, action_dim, hidden_dim=DRL_HIDDEN_DIM):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state, action):
        s = self.state_net(state)
        a = self.action_net(action)
        x = torch.cat([s, a], dim=-1)
        return self.q_net(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=DRL_MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    """OU噪声用于动作探索"""

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + \
                      self.sigma * np.random.randn(self.action_dim)
        return self.state

    def decay(self, decay_factor=0.9995):
        self.sigma *= decay_factor


class DDPGAgent:
    """DDPG 智能体控制器"""

    def __init__(self, state_dim, action_dim, action_bounds,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = torch.tensor([b[0] for b in action_bounds],
                                        dtype=torch.float32).to(device)
        self.action_high = torch.tensor([b[1] for b in action_bounds],
                                         dtype=torch.float32).to(device)
        self.action_range = self.action_high - self.action_low

        # 网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        # 同步目标网络
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=DRL_ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=DRL_CRITIC_LR)

        # 经验回放
        self.memory = ReplayBuffer()
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        self.gamma = DRL_GAMMA
        self.tau = DRL_TAU

        self.train_step = 0

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def select_action(self, state, explore=True):
        """根据状态选择控制动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze().cpu().numpy()
        self.actor.train()

        if explore:
            noise = self.noise.sample()
            action = np.clip(action + noise, 0.0, 1.0)

        # 映射到实际控制范围
        action_real = action * self.action_range.cpu().numpy() + self.action_low.cpu().numpy()
        return action_real

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        """更新网络参数"""
        if len(self.memory) < DRL_BATCH_SIZE:
            return None, None

        states, actions, rewards, next_states, dones = self.memory.sample(DRL_BATCH_SIZE)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 更新 Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 更新 Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        # 添加动作正则化（鼓励平滑控制）
        action_reg = 0.001 * (actor_actions ** 2).mean()
        actor_loss = actor_loss + action_reg

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        self.noise.decay()
        self.train_step += 1

        return critic_loss.item(), actor_loss.item()

    def save(self, name="ddpg_agent"):
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, name="ddpg_agent"):
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(ckpt["actor"])
            self.actor_target.load_state_dict(ckpt["actor_target"])
            self.critic.load_state_dict(ckpt["critic"])
            self.critic_target.load_state_dict(ckpt["critic_target"])


class CirculatingWaterEnv:
    """循环水系统强化学习环境
    基于训练好的LSTM模型作为环境模拟器
    使用等效目标（传感器值经物理映射后的目标）计算奖励
    """

    def __init__(self, lstm_model, scaler, df, state_cols, control_cols,
                 window_size=60):
        self.model = lstm_model
        self.scaler = scaler  # 全特征scaler（11维）
        self.df = df
        self.state_cols = [c for c in state_cols if c in df.columns]
        self.control_cols = [c for c in control_cols if c in df.columns]
        self.window_size = window_size

        state_data = df[self.state_cols]
        control_data = df[self.control_cols]
        all_feature_cols = self.state_cols + self.control_cols
        self.all_data = df[all_feature_cols].values
        self.state_data = state_data.values
        self.control_data = control_data.values

        # 状态专用scaler（用于Actor网络输入）
        from sklearn.preprocessing import StandardScaler
        self.state_scaler = StandardScaler()
        self.state_scaler.fit(self.state_data)

        # 动作边界：影响流量的变量(M1,泵速)用全范围，其他用均值±3.5σ
        self.action_bounds = []
        flow_ctrl_names = ["dn200", "speed"]
        for i in range(len(self.control_cols)):
            mean_i = self.control_data[:, i].mean()
            std_i = self.control_data[:, i].std()
            if std_i < 1e-6:
                std_i = 0.1
            col_lower = self.control_cols[i].lower()
            is_flow_ctrl = any(kw in col_lower for kw in flow_ctrl_names)
            if is_flow_ctrl:
                lo = max(0, self.control_data[:, i].min() * 0.5)
                hi = self.control_data[:, i].max() * 1.1
            else:
                lo = max(0, mean_i - 3.5 * std_i)
                hi = mean_i + 3.5 * std_i
            self.action_bounds.append((float(lo), float(hi)))

        self.state_dim = len(self.state_cols)
        self.action_dim = len(self.control_cols)
        self.current_step = 0
        self.max_step = min(len(self.state_data) - window_size - 1, 5000)

        # 计算等效目标（传感器层面：物理映射后的目标值）
        self._compute_equivalent_targets()

    def _compute_equivalent_targets(self):
        """设置控制目标：直接使用设计指标（27 m³/h, 22°C）"""
        from config import TARGET_FLOW, TARGET_TEMP, FLOW_TOLERANCE, TEMP_TOLERANCE

        self.equiv_target_flow = TARGET_FLOW       # 27 m³/h
        self.equiv_flow_lo = TARGET_FLOW - FLOW_TOLERANCE
        self.equiv_flow_hi = TARGET_FLOW + FLOW_TOLERANCE

        self.equiv_target_temp = TARGET_TEMP       # 22°C
        self.equiv_temp_lo = TARGET_TEMP - TEMP_TOLERANCE
        self.equiv_temp_hi = TARGET_TEMP + TEMP_TOLERANCE

        print(f"[DRL Env] 设计目标: flow={self.equiv_target_flow:.0f} "
              f"(range {self.equiv_flow_lo:.0f}-{self.equiv_flow_hi:.0f}), "
              f"temp={self.equiv_target_temp:.1f} "
              f"(range {self.equiv_temp_lo:.0f}-{self.equiv_temp_hi:.0f})")

    def reset(self):
        """重置环境 - 选择温度较高的起始点（避免冷态无法加热）"""
        n_state = len(self.state_cols)
        # 找温度较高的窗口
        candidates = []
        for i in range(0, min(self.max_step - self.window_size, 20000), 50):
            window_states = self.all_data[i:i + self.window_size, :n_state]
            avg_temp = 0
            for j, col in enumerate(self.state_cols):
                if "temp" in col.lower() and "he" not in col.lower():
                    avg_temp += window_states[:, j].mean()
                    break
            if avg_temp > 15:  # 只选温度>15°C的窗口
                candidates.append(i + self.window_size - 1)
        if candidates:
            self.current_step = candidates[np.random.randint(0, len(candidates))]
        else:
            self.current_step = np.random.randint(0, self.max_step // 2)
        return self._get_state()

    def _get_state(self):
        """获取当前观测状态（仅状态变量，归一化）"""
        idx = self.current_step
        state = self.state_data[idx]
        state_norm = self.state_scaler.transform(state.reshape(1, -1))[0]
        return state_norm.astype(np.float32)

    def step(self, action):
        """执行动作并返回 (next_state, reward, done, info)
        action: 原始控制量 [valve_m1, valve_m2, valve_m3, valve_m4, pump_speed]
        """
        idx = self.current_step
        n_state = len(self.state_cols)
        n_ctrl = len(self.control_cols)

        # 1. 构建当前窗口（原始值）
        start = max(0, idx - self.window_size + 1)
        current_window = self.all_data[start:idx + 1].copy()
        if len(current_window) < self.window_size:
            pad = np.tile(current_window[0], (self.window_size - len(current_window), 1))
            current_window = np.vstack([pad, current_window])
        current_window = current_window[-self.window_size:]

        # 2. 用action替换最后一行控制量
        current_window[-1, n_state:] = np.clip(action[:n_ctrl],
            [b[0] for b in self.action_bounds],
            [b[1] for b in self.action_bounds])

        # 3. 归一化窗口 → LSTM预测 → 反归一化
        window_norm = self.scaler.transform(current_window).reshape(1, self.window_size, -1)
        try:
            pred_norm = self.model.predict(window_norm)[0]
            next_all = self.scaler.inverse_transform(pred_norm.reshape(1, -1))[0]
        except Exception:
            next_all = self.all_data[min(idx + 1, len(self.all_data) - 1)]

        # 4. 提取状态部分
        next_state_raw = next_all[:n_state]

        # 5. 更新步数
        self.current_step = min(self.current_step + 1, self.max_step)
        done = self.current_step >= self.max_step

        # 6. 计算奖励（使用等效目标）
        reward = self._compute_reward(next_state_raw, action[:n_ctrl])

        info = {"step": self.current_step, "state_raw": next_state_raw}
        return self._get_state(), reward, done, info

    def _compute_reward(self, state, action):
        """物理引导奖励函数：直接根据控制量计算Q_he和T2
        不依赖LSTM预测的准确性，Agent有清晰的梯度方向"""
        state_cols_lower = [c.lower() for c in self.state_cols]
        ctrl_cols_lower = [c.lower() for c in self.control_cols]

        def _find_ctrl_val(kw):
            for i, c in enumerate(ctrl_cols_lower):
                if kw in c:
                    return action[i] if i < len(action) else 0
            return None

        def _find_state_val(kw):
            for i, c in enumerate(state_cols_lower):
                if kw in c:
                    return max(state[i], 0.01) if "flow" in kw else state[i]
            return None

        reward = 0.0

        # 获取关键值
        flow_total = _find_state_val("flow") or 350.0
        t_cold = _find_state_val("tank") or 15.0
        t_hot = None
        for i, c in enumerate(state_cols_lower):
            if "he" in c and "sim" in c:
                t_hot = state[i]
                break
        if t_hot is None:
            t_hot = 30.0

        m1 = _find_ctrl_val("dn200") or 41.0
        m2 = _find_ctrl_val("dn300") or 99.0
        m3 = _find_ctrl_val("dn350") or 99.7

        # === 流量：物理公式直接算 Q_he = Q_total * M1/(M1+M2) ===
        q_he = flow_total * max(m1, 0.1) / max(m1 + m2, 0.1)
        flow_center = (self.equiv_flow_lo + self.equiv_flow_hi) / 2
        flow_half = max(self.equiv_flow_hi - self.equiv_flow_lo, 1) / 2
        flow_dist = abs(q_he - flow_center) / flow_half

        # 流量是主要优化目标，给最大权重
        if flow_dist <= 1.0:
            reward += 30.0 * (1.0 - flow_dist * 0.3)  # 达标: +9~30
        elif flow_dist <= 3.0:
            reward += 10.0 * (3.0 - flow_dist)         # 接近: 0~20
        else:
            reward -= 8.0 * min(flow_dist - 3.0, 10.0)  # 远离: 逐步惩罚

        # === 温度：物理公式 T2 = alpha*T_cold + (1-alpha)*T_hot ===
        m3_norm = np.clip((m3 - 99.55) / 0.20, 0.0, 1.0)
        alpha_mix = 0.7 - 0.3 * m3_norm
        t2 = alpha_mix * t_cold + (1 - alpha_mix) * t_hot
        temp_center = (self.equiv_temp_lo + self.equiv_temp_hi) / 2
        temp_half = max(self.equiv_temp_hi - self.equiv_temp_lo, 0.5) / 2
        temp_dist = abs(t2 - temp_center) / temp_half

        if temp_dist <= 1.0:
            reward += 15.0 * (1.0 - temp_dist * 0.3)
        elif temp_dist <= 2.5:
            reward += 5.0 * (2.5 - temp_dist)
        else:
            reward -= 5.0 * min(temp_dist - 2.5, 5.0)

        # === 压力：轻微约束 ===
        press = _find_state_val("press") or 0.3
        if press is not None and press > MAX_PRESSURE:
            reward -= (press - MAX_PRESSURE) * 30
        else:
            reward += 2.0

        # === 能耗：鼓励节能 ===
        energy_cost = np.mean(np.abs(action)) * 0.002
        reward -= energy_cost

        return float(reward)


class DRLController:
    """DRL控制器训练与推理封装（含GA行为克隆预训练）"""

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.episode_rewards = []
        self.action_bounds = env.action_bounds

    def _run_ga_for_state(self, state_norm, generations=15, pop_size=30):
        """在给定状态下运行轻量GA，返回最优动作（物理引导）"""
        n_ctrl = len(self.action_bounds)
        bounds = [(lo, hi) for lo, hi in self.action_bounds]

        # 获取当前状态的原始值（用于物理公式计算）
        state_raw = self.env.state_scaler.inverse_transform(
            state_norm.reshape(1, -1))[0]

        n_state = len(self.env.state_cols)
        state_cols_lower = [c.lower() for c in self.env.state_cols]
        ctrl_cols_lower = [c.lower() for c in self.env.control_cols]

        # 获取流量索引
        flow_idx = None
        for i, c in enumerate(state_cols_lower):
            if "flow" in c:
                flow_idx = i
                break
        flow_total = state_raw[flow_idx] if flow_idx is not None else 350.0

        def fitness(ind):
            ctrl = np.array(ind)
            m1 = max(ctrl[0], 0.1)
            m2 = max(ctrl[1], 0.1)
            m3 = ctrl[2] if len(ctrl) > 2 else 99.7
            pump_speed = ctrl[4] if len(ctrl) > 4 else 1200

            # 物理公式直接计算Q_he和T2
            q_he = flow_total * m1 / (m1 + m2)

            t_cold_idx = None; t_hot_idx = None
            for i, c in enumerate(state_cols_lower):
                if "tank" in c: t_cold_idx = i
                elif "he" in c: t_hot_idx = i
            t_cold = state_raw[t_cold_idx] if t_cold_idx is not None else 15.0
            t_hot = state_raw[t_hot_idx] if t_hot_idx is not None else 30.0
            m3_norm = np.clip((m3 - 99.55) / 0.20, 0.0, 1.0)
            alpha = 0.7 - 0.3 * m3_norm
            t2 = alpha * t_cold + (1 - alpha) * t_hot

            # 流量得分
            flow_center = (self.env.equiv_flow_lo + self.env.equiv_flow_hi) / 2
            flow_half = max(self.env.equiv_flow_hi - self.env.equiv_flow_lo, 1) / 2
            flow_dist = abs(q_he - flow_center) / flow_half
            score = 0.0
            if flow_dist <= 1.0:
                score += 30.0 * (1.0 - flow_dist * 0.3)
            else:
                score -= 20.0 * min(flow_dist - 1.0, 5.0)

            # 温度得分
            temp_center = (self.env.equiv_temp_lo + self.env.equiv_temp_hi) / 2
            temp_half = max(self.env.equiv_temp_hi - self.env.equiv_temp_lo, 0.5) / 2
            temp_dist = abs(t2 - temp_center) / temp_half
            if temp_dist <= 1.0:
                score += 15.0 * (1.0 - temp_dist * 0.3)
            else:
                score -= 10.0 * min(temp_dist - 1.0, 3.0)

            # 能耗惩罚
            energy = np.mean(np.abs(ctrl)) * 0.002
            score -= energy

            return score

        # 简单GA
        pop = [[np.random.uniform(lo, hi) for lo, hi in bounds] for _ in range(pop_size)]
        best_ind = pop[0][:]
        best_fit = -np.inf
        for gen in range(generations):
            fits = [fitness(ind) for ind in pop]
            best_idx = np.argmax(fits)
            if fits[best_idx] > best_fit:
                best_fit = fits[best_idx]
                best_ind = pop[best_idx][:]

            new_pop = [pop[best_idx][:]]  # 精英保留
            while len(new_pop) < pop_size:
                t1, t2 = pop[np.random.randint(0, pop_size)], pop[np.random.randint(0, pop_size)]
                p1 = t1 if fitness(t1) > fitness(t2) else t2
                t1, t2 = pop[np.random.randint(0, pop_size)], pop[np.random.randint(0, pop_size)]
                p2 = t1 if fitness(t1) > fitness(t2) else t2
                alpha_cross = 0.5 + 0.5 * np.random.random()
                child = [alpha_cross * a + (1 - alpha_cross) * b for a, b in zip(p1, p2)]
                for i in range(n_ctrl):
                    if np.random.random() < 0.15:
                        lo, hi = bounds[i]
                        sigma = (hi - lo) * 0.05
                        child[i] += np.random.normal(0, sigma)
                        child[i] = max(lo, min(hi, child[i]))
                new_pop.append(child)
            pop = new_pop[:pop_size]

        return np.array(best_ind, dtype=np.float32)

    def _run_ga_for_state_with_lstm(self, state_norm, generations=12, pop_size=25):
        """LSTM-fitness GA：使用LSTM预测评估适应度（与闭环GA完全一致）
        比物理GA慢但动作质量更高，用于BC训练的关键样本"""
        n_ctrl = len(self.action_bounds)
        bounds = [(lo, hi) for lo, hi in self.action_bounds]
        saved_idx = self.env.current_step

        def fitness(ind):
            self.env.current_step = saved_idx
            ctrl = np.array(ind, dtype=np.float32)
            try:
                _, reward, _, _ = self.env.step(ctrl)
                return float(reward)
            except Exception:
                return -500.0

        state_raw = self.env.state_scaler.inverse_transform(state_norm.reshape(1, -1))[0]
        state_cols_lower = [c.lower() for c in self.env.state_cols]
        flow_idx = None
        for i, c in enumerate(state_cols_lower):
            if "flow" in c:
                flow_idx = i
                break
        flow_total = state_raw[flow_idx] if flow_idx is not None else 350.0

        # Smart init: use physical GA result as seed + diverse random
        phys_best = self._run_ga_for_state(state_norm, generations=10, pop_size=20)
        pop = [phys_best.copy()]
        for _ in range(pop_size - 1):
            ind = phys_best.copy()
            for i in range(n_ctrl):
                lo, hi = bounds[i]
                ind[i] += np.random.normal(0, (hi - lo) * 0.08)
                ind[i] = max(lo, min(hi, ind[i]))
            pop.append(ind)

        best_ind = pop[0][:]
        best_fit = -np.inf
        for gen in range(generations):
            fits = [fitness(ind) for ind in pop]
            best_idx = np.argmax(fits)
            if fits[best_idx] > best_fit:
                best_fit = fits[best_idx]
                best_ind = pop[best_idx][:]

            new_pop = [pop[best_idx][:]]
            while len(new_pop) < pop_size:
                t1, t2 = pop[np.random.randint(0, pop_size)], pop[np.random.randint(0, pop_size)]
                p1 = t1 if fitness(t1) > fitness(t2) else t2
                t1, t2 = pop[np.random.randint(0, pop_size)], pop[np.random.randint(0, pop_size)]
                p2 = t1 if fitness(t1) > fitness(t2) else t2
                alpha_cross = 0.5 + 0.5 * np.random.random()
                child = [alpha_cross * a + (1 - alpha_cross) * b for a, b in zip(p1, p2)]
                for i in range(n_ctrl):
                    if np.random.random() < 0.12:
                        lo, hi = bounds[i]
                        sigma = (hi - lo) * 0.04
                        child[i] += np.random.normal(0, sigma)
                        child[i] = max(lo, min(hi, child[i]))
                new_pop.append(child)
            pop = new_pop[:pop_size]

        self.env.current_step = saved_idx  # restore
        return np.array(best_ind, dtype=np.float32)

    def pretrain_actor_with_ga(self, n_samples=100, epochs=30, verbose=True):
        """使用GA行为克隆预训练Actor网络
        通过在LSTM环境中运行轨迹收集训练数据，确保训练与部署分布一致。
        混合使用物理GA（快速覆盖）和LSTM GA（高质量动作）。"""
        print(f"GA行为克隆预训练: 轨迹采样, target={n_samples} samples, epochs={epochs}")

        # 阶段1：收集轨迹数据（LSTM环境 + GA专家）
        expert_states = []
        expert_actions = []
        steps_per_traj = 15
        n_trajectories = max(1, n_samples // steps_per_traj)
        lstm_ga_interval = max(1, n_trajectories // 2)  # 每2条轨迹用一次LSTM GA

        print(f"  收集 {n_trajectories} 条轨迹 (每条 {steps_per_traj} 步, "
              f"LSTM-GA间隔={lstm_ga_interval})...")
        for traj in range(n_trajectories):
            self.env.reset()
            state = self.env._get_state()

            for s in range(steps_per_traj):
                # 半数轨迹用LSTM GA（更准确），其余用物理GA（快速覆盖）
                use_lstm = (traj % lstm_ga_interval == 0) and (s % 3 == 0)
                if use_lstm:
                    best_action = self._run_ga_for_state_with_lstm(
                        state, generations=12, pop_size=25)
                else:
                    best_action = self._run_ga_for_state(
                        state, generations=15, pop_size=30)

                expert_states.append(state.copy())
                expert_actions.append(best_action.copy())

                # 在LSTM环境中执行动作，获取下一状态
                next_state, _, done, _ = self.env.step(best_action)
                if done:
                    break
                state = next_state

            if verbose and (traj + 1) % max(1, n_trajectories // 5) == 0:
                print(f"  轨迹 {traj+1}/{n_trajectories}, 已收集 {len(expert_states)} 样本")

        # 阶段2：补充独立数据状态样本（增加多样性，同时使用LSTM GA）
        n_extra = min(n_samples // 3, 30)
        print(f"  补充 {n_extra} 独立数据状态样本 (含LSTM GA)...")
        for i in range(n_extra):
            state = self.env.reset()
            if i % 2 == 0:
                best_action = self._run_ga_for_state_with_lstm(
                    state, generations=12, pop_size=25)
            else:
                best_action = self._run_ga_for_state(state, generations=20, pop_size=40)
            expert_states.append(state.copy())
            expert_actions.append(best_action.copy())

        expert_states = np.array(expert_states, dtype=np.float32)
        expert_actions = np.array(expert_actions, dtype=np.float32)
        total_samples = len(expert_states)
        print(f"  总共收集 {total_samples} 个 (状态, GA动作) 训练对")

        # 将动作归一化到[0,1]（Actor输出范围）
        action_low = self.agent.action_low.cpu().numpy()
        action_range = self.agent.action_range.cpu().numpy()
        expert_actions_norm = (expert_actions - action_low) / np.maximum(action_range, 1e-6)

        # 监督学习训练Actor
        optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=1e-3)
        batch_size = min(32, total_samples)
        self.agent.actor.train()

        bc_losses = []
        for ep in range(epochs):
            idx = np.random.permutation(total_samples)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, total_samples, batch_size):
                batch_idx = idx[start:start + batch_size]
                s_batch = torch.FloatTensor(expert_states[batch_idx]).to(self.agent.device)
                a_batch = torch.FloatTensor(expert_actions_norm[batch_idx]).to(self.agent.device)

                pred = self.agent.actor(s_batch)
                loss = F.mse_loss(pred, a_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            bc_losses.append(avg_loss)
            if verbose and ep % 10 == 0:
                print(f"  BC epoch {ep:3d} | loss={avg_loss:.6f}")

        self.bc_losses = bc_losses

        # 同步目标网络
        self.agent._hard_update(self.agent.actor_target, self.agent.actor)
        self.agent._hard_update(self.agent.critic_target, self.agent.critic)

        # 评估克隆效果
        self.agent.actor.eval()
        with torch.no_grad():
            s_t = torch.FloatTensor(expert_states[:10]).to(self.agent.device)
            pred_a = self.agent.actor(s_t).cpu().numpy()
            mse = np.mean((pred_a - expert_actions_norm[:10]) ** 2)
        print(f"GA行为克隆完成, Actor MSE={mse:.6f}")
        return expert_states, expert_actions

    def train(self, episodes=DRL_MAX_EPISODES, max_steps=DRL_MAX_STEPS,
              verbose=True, pretrain=True, ddpg_finetune=True):
        """训练DRL智能体（GA行为克隆 + 可选DDPG微调）"""
        if pretrain:
            self.pretrain_actor_with_ga(n_samples=150, epochs=40)
            self.agent.save("ddpg_bc_only")

            if not ddpg_finetune:
                # 仅行为克隆，跳过DDPG训练
                print("跳过DDPG微调，使用纯BC策略")
                self.episode_rewards = self.bc_losses
                return self.episode_rewards

        # DDPG微调：使用极小学习率和噪声，防止覆盖BC学到的知识
        self.agent.noise.sigma = 0.03
        self.agent.actor_optimizer.param_groups[0]['lr'] = 1e-6   # 极低Actor LR
        self.agent.critic_optimizer.param_groups[0]['lr'] = 1e-5  # 低Critic LR

        print(f"开始DRL微调训练 (episodes={episodes}, device={self.agent.device})")

        for ep in range(episodes):
            state = self.env.reset()
            episode_reward = 0.0
            critic_losses, actor_losses = [], []

            for step in range(max_steps):
                action = self.agent.select_action(state, explore=True)
                next_state, reward, done, info = self.env.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                c_loss, a_loss = self.agent.update()
                if c_loss is not None:
                    critic_losses.append(c_loss)
                    actor_losses.append(a_loss)

                state = next_state
                episode_reward += reward
                if done:
                    break

            self.episode_rewards.append(episode_reward)

            if verbose and ep % 30 == 0:
                avg_r = np.mean(self.episode_rewards[-20:]) if len(
                    self.episode_rewards) >= 20 else np.mean(self.episode_rewards)
                print(f"Ep {ep:4d} | Reward={episode_reward:.2f} | "
                      f"Avg20={avg_r:.2f} | Noise={self.agent.noise.sigma:.4f}")

        self.agent.save()
        print("DRL训练完成!")
        return self.episode_rewards

    def get_optimal_action(self, state):
        """获取最优控制动作（无探索）"""
        return self.agent.select_action(state, explore=False)


def build_drl_pipeline(lstm_model, scaler, df, state_cols=None, control_cols=None):
    """构建完整的DRL控制流程"""
    if state_cols is None:
        from config import STATE_COLS as state_cols
    if control_cols is None:
        from config import CONTROL_COLS as control_cols

    env = CirculatingWaterEnv(lstm_model, scaler, df, state_cols, control_cols)
    agent = DDPGAgent(env.state_dim, env.action_dim, env.action_bounds)
    controller = DRLController(env, agent)

    return controller


if __name__ == "__main__":
    import torch
    from data_preprocessing import full_preprocessing_pipeline
    from modeling import build_model, ModelTrainer

    # 准备数据
    train_loader, val_loader, test_loader, scaler, df = full_preprocessing_pipeline(
        max_files=3, window_size=30, stride=10)

    # 训练LSTM环境模型
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[-1]
    lstm = build_model(input_dim, "lstm")

    trainer = ModelTrainer(lstm)
    trainer.fit(train_loader, val_loader, epochs=30)

    # 构建DRL控制器
    state_cols = [c for c in config.STATE_COLS if c in df.columns]
    control_cols = [c for c in config.CONTROL_COLS if c in df.columns]

    env = CirculatingWaterEnv(lstm, scaler, df, state_cols, control_cols)
    agent = DDPGAgent(env.state_dim, env.action_dim, env.action_bounds)
    controller = DRLController(env, agent)

    rewards = controller.train(episodes=100, max_steps=100, verbose=True)
    print(f"最终平均奖励: {np.mean(rewards[-20:]):.2f}")
