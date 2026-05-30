"""
循环水系统 - 数据预处理模块
功能：数据加载、清洗、去噪、标准化、序列构建
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
from glob import glob
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

from config import (DATA_DIR, CSV_ENCODING, CSV_DELIMITER, WINDOW_SIZE,
                    STRIDE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
                    SAVGOL_WINDOW, SAVGOL_ORDER, STATE_COLS, CONTROL_COLS,
                    FB_COLUMNS, TP_COLUMNS)


class DataLoader_CWS:
    """循环水系统数据加载器"""

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.fb_files = []
        self.tp_files = []
        self._scan_files()

    def _scan_files(self):
        """扫描所有数据文件，按FB/TP配对
        文件名规则：以 -FB.csv / -fb.csv 结尾的为阀门反馈数据；
                   以 -TP.csv / -tp.csv 结尾的为温度压力数据；
                   以 -BF.csv 结尾的也视为FB类；
                   以 -TB.csv 结尾的也视为TP类。
        """
        all_files = glob(os.path.join(self.data_dir, "*.csv"))
        for f in all_files:
            name = os.path.basename(f)
            upper = name.upper()
            # 匹配文件名末尾的 FB/TP/BF/TB（在 .csv 之前）
            if re.search(r"-[FB][BF]\.csv$", upper, re.IGNORECASE):
                self.fb_files.append(f)
            elif re.search(r"-[TP][PB]\.csv$", upper, re.IGNORECASE):
                self.tp_files.append(f)

        print(f"找到 {len(self.fb_files)} 个FB文件, {len(self.tp_files)} 个TP文件")

    @staticmethod
    def _read_csv(filepath):
        """读取单个UTF-16 LE CSV文件"""
        try:
            df = pd.read_csv(filepath, sep=CSV_DELIMITER, encoding=CSV_ENCODING,
                             low_memory=False)
            # 标准化列名
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            df = pd.read_csv(filepath, sep=CSV_DELIMITER, encoding=CSV_ENCODING,
                             engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            return df

    def load_and_merge(self, max_files=None):
        """加载并配对FB和TP数据，合并为统一数据集"""
        fb_dfs, tp_dfs = [], []

        fb_to_load = self.fb_files[:max_files] if max_files else self.fb_files
        tp_to_load = self.tp_files[:max_files] if max_files else self.tp_files

        # FB文件固定7列: timestamp + 6个变量
        FB_STD_COLS = FB_COLUMNS[:7] if len(FB_COLUMNS) >= 7 else FB_COLUMNS
        # TP文件固定10列: timestamp + 9个变量
        TP_STD_COLS = TP_COLUMNS[:10] if len(TP_COLUMNS) >= 10 else TP_COLUMNS

        for f in fb_to_load:
            try:
                df = self._read_csv(f)
                # 重命名列为标准名（按位置，跳过时间戳列）
                if len(df.columns) >= len(FB_STD_COLS):
                    rename_map = {df.columns[i]: FB_STD_COLS[i]
                                  for i in range(len(FB_STD_COLS))}
                else:
                    rename_map = {df.columns[0]: "timestamp"}
                    for i in range(1, len(df.columns)):
                        if i < len(FB_STD_COLS):
                            rename_map[df.columns[i]] = FB_STD_COLS[i]
                df = df.rename(columns=rename_map)
                fb_dfs.append(df)
            except Exception as e:
                print(f"跳过 {os.path.basename(f)}: {e}")

        for f in tp_to_load:
            try:
                df = self._read_csv(f)
                if len(df.columns) >= len(TP_STD_COLS):
                    rename_map = {df.columns[i]: TP_STD_COLS[i]
                                  for i in range(len(TP_STD_COLS))}
                else:
                    rename_map = {df.columns[0]: "timestamp"}
                    for i in range(1, len(df.columns)):
                        if i < len(TP_STD_COLS):
                            rename_map[df.columns[i]] = TP_STD_COLS[i]
                df = df.rename(columns=rename_map)
                tp_dfs.append(df)
            except Exception as e:
                print(f"跳过 {os.path.basename(f)}: {e}")

        if not fb_dfs or not tp_dfs:
            raise ValueError("未能加载任何数据文件")

        fb_all = pd.concat(fb_dfs, ignore_index=True)
        tp_all = pd.concat(tp_dfs, ignore_index=True)

        # 解析时间戳
        fb_all["timestamp"] = pd.to_datetime(fb_all["timestamp"])
        tp_all["timestamp"] = pd.to_datetime(tp_all["timestamp"])

        # 删除重复时间戳
        fb_all = fb_all.drop_duplicates(subset="timestamp").sort_values("timestamp")
        tp_all = tp_all.drop_duplicates(subset="timestamp").sort_values("timestamp")

        print(f"FB数据: {fb_all.shape[0]} 行, 时间 {fb_all['timestamp'].min()} ~ {fb_all['timestamp'].max()}")
        print(f"TP数据: {tp_all.shape[0]} 行, 时间 {tp_all['timestamp'].min()} ~ {tp_all['timestamp'].max()}")

        # 按最近时间戳合并FB和TP
        merged = pd.merge_asof(fb_all, tp_all, on="timestamp",
                               direction="nearest", tolerance=pd.Timedelta("10s"))
        merged = merged.dropna()

        # 确保所有目标列存在
        target_cols = ["timestamp"] + FB_STD_COLS[1:] + TP_STD_COLS[1:]
        available = [c for c in target_cols if c in merged.columns]
        merged = merged[available]

        print(f"合并后数据: {merged.shape[0]} 行, {merged.shape[1]} 列")
        if merged.shape[0] > 0:
            print(f"时间范围: {merged['timestamp'].min()} ~ {merged['timestamp'].max()}")
        return merged

    @staticmethod
    def clean_data(df):
        """数据清洗：去除异常值和缺失值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # 缺失值：前向填充
        df[numeric_cols] = df[numeric_cols].fillna(method="ffill").fillna(method="bfill")

        # 异常值：3倍标准差截断
        for col in numeric_cols:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                lower = mean - 4 * std
                upper = mean + 4 * std
                df[col] = df[col].clip(lower, upper)

        return df

    @staticmethod
    def denoise(df, method="savgol"):
        """数据去噪"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_denoised = df.copy()

        for col in numeric_cols:
            if method == "savgol":
                w = min(SAVGOL_WINDOW, len(df) - 2)
                if w % 2 == 0:
                    w += 1
                if w >= 5:
                    try:
                        df_denoised[col] = savgol_filter(df[col].values, w, SAVGOL_ORDER)
                    except Exception:
                        pass
            elif method == "moving_avg":
                df_denoised[col] = df[col].rolling(window=5, center=True).mean()
                df_denoised[col] = df_denoised[col].fillna(df[col])

        return df_denoised

    @staticmethod
    def normalize(train_df, val_df=None, test_df=None):
        """标准化：对特征和目标分别标准化"""
        feature_cols = [c for c in (STATE_COLS + CONTROL_COLS)
                        if c in train_df.columns]

        scaler_X = StandardScaler()
        train_X = scaler_X.fit_transform(train_df[feature_cols].values)

        result = {"scaler_X": scaler_X, "train_X": train_X}

        for name, df in [("val_X", val_df), ("test_X", test_df)]:
            if df is not None:
                result[name] = scaler_X.transform(df[feature_cols].values)

        return result

    @staticmethod
    def build_sequences(data, window_size=WINDOW_SIZE, stride=STRIDE):
        """构建滑动窗口序列用于时间序列预测
        Args:
            data: (N, features) array
        Returns:
            X: (num_samples, window_size, features) - 历史窗口
            y: (num_samples, features) - 下一步预测值
        """
        X, y = [], []
        for i in range(0, len(data) - window_size, stride):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class CWSDataset(Dataset):
    """循环水系统 PyTorch 数据集"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders(X, y, batch_size=256):
    """构建训练/验证/测试 DataLoader"""
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    train_loader = DataLoader(CWSDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CWSDataset(X_val, y_val),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CWSDataset(X_test, y_test),
                             batch_size=batch_size, shuffle=False)

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    return train_loader, val_loader, test_loader


def full_preprocessing_pipeline(max_files=10, window_size=WINDOW_SIZE,
                                stride=STRIDE, batch_size=256):
    """完整的预处理流程"""
    print("=" * 60)
    print("步骤1: 加载数据...")
    loader = DataLoader_CWS()
    df = loader.load_and_merge(max_files=max_files)

    print("\n步骤2: 数据清洗...")
    df = DataLoader_CWS.clean_data(df)

    print("\n步骤3: 数据去噪...")
    df = DataLoader_CWS.denoise(df)

    print("\n步骤4: 数据标准化...")
    norm_result = DataLoader_CWS.normalize(df)
    data = norm_result["train_X"]

    print("\n步骤5: 构建序列...")
    X, y = DataLoader_CWS.build_sequences(data, window_size, stride)

    print("\n步骤6: 构建DataLoader...")
    train_loader, val_loader, test_loader = build_dataloaders(X, y, batch_size)

    print(f"\n输入形状 X: {X.shape}, 输出形状 y: {y.shape}")
    return train_loader, val_loader, test_loader, norm_result["scaler_X"], df


if __name__ == "__main__":
    train_loader, val_loader, test_loader, scaler, df = full_preprocessing_pipeline(
        max_files=5)
    print("\n预处理完成!")
    print(f"Scaler input_dim: {scaler.n_features_in_}")
