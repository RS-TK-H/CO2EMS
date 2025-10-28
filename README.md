# CO2EMS

这是一个示例项目，展示如何利用车辆的发动机转速、车速、电池荷电状态等固有数据来训练不同类型的模型预测 CO₂ 尾气排放量。

## 功能概览

* 支持决策树（DT）、随机森林（RF）、XGBoost、支持向量机（SVM）、多层感知机（ANN）、SimpleRNN 与 LSTM 等多种回归模型。
* 自动发现目标列（列名包含 `co2`），也可以通过命令行指定特征列和目标列。
* 同时兼容 CSV 与 Excel（含多表）输入，支持使用通配符一次性加载多份训练数据。
* 可选保存测试集/验证集的时间序列与散点图，并将评估指标导出为 JSON。
* 序列模型基于滑动窗口构造样本，便于处理时间序列或归一化后的数据片段。

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **依赖说明**：
> * 使用 Excel 文件时需要 `openpyxl`。
> * 训练 XGBoost、RNN、LSTM 模型分别需要额外安装 `xgboost` 与 `tensorflow`，本仓库的 `requirements.txt` 已包含这些依赖。

## 快速上手

项目提供了一个示例数据集 `data/sample_vehicle_data.csv`，以及统一的训练脚本 `src/train_co2_model.py`。以下命令将训练一个随机森林模型并保存结果：

```bash
python src/train_co2_model.py data/sample_vehicle_data.csv --model rf \
    --model-output runs/rf_model.joblib --plots-dir runs/plots
```

脚本会输出 MAE、RMSE、R² 等指标，并将模型保存到 `runs/rf_model.joblib`。如果提供 `--plots-dir`，脚本还会生成测试集的时间序列图和散点图。

### 选择不同模型

| 模型 | 命令示例 |
| ---- | -------- |
| 决策树 | `--model dt` |
| 随机森林 | `--model rf` |
| XGBoost | `--model xgb` |
| 支持向量机 | `--model svm` |
| 多层感知机（ANN） | `--model ann` |
| SimpleRNN | `--model rnn --sequence-length 32 --epochs 80` |
| LSTM | `--model lstm --sequence-length 32 --epochs 80` |

对序列模型需要提供合理的 `--sequence-length`（滑动窗口长度）、`--epochs` 和 `--batch-size`。训练完成后请将 `--model-output` 指向一个目录，例如 `runs/lstm_model/`，目录下会保存 `model.keras` 与 `scaler.joblib`。

### 合并多份训练数据

```bash
python src/train_co2_model.py "CO2_Direct_2025_t1/normalized_part_*.xlsx" \
    --exclude-sheet OBD_WLTC_Cold_day1 --model xgb
```

* 支持使用通配符一次性载入多份数据，脚本会自动拼接。
* `--sheet` 参数可指定 Excel 中的单个工作表；若不指定则读取全部工作表。
* 可通过 `--exclude-sheet` 重复指定需要排除的表，例如验证集。

### 使用验证集与指标导出

```bash
python src/train_co2_model.py data/train.csv --validation data/val.xlsx \
    --validation-sheet Sheet1 --metrics-output runs/metrics.json --plots-dir runs/plots
```

当提供 `--validation` 时，脚本会在训练完成后对验证集再次评估，并将结果追加到 `--metrics-output` 指定的 JSON 文件中。

## 自定义特征与目标列

* 默认情况下会自动选择包含 `co2` 的第一列作为目标列，其他所有数值列作为特征。
* 可以通过 `--target-column` 显式指定目标列，通过 `--feature-columns` 指定特征列（空格分隔）。

## 数据准备建议

1. 确保所有输入特征为数值型（缺失值会被自动填充为 0）。
2. 如果使用归一化后的数据，请保证同一文件中包含所需的特征列与目标列。
3. 序列模型会根据滑动窗口创建样本，因此数据的行顺序应当保持时间顺序。
