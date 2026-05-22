---
name: "math-modeling-organizer"
description: "Organizes mathematical modeling project files by question subproblems (e.g., 1.1, 1.2). Invoke when creating code for math modeling competition questions that require structured folder organization."
---

# 数学建模项目文件组织器

## 功能说明

本技能用于规范数学建模竞赛项目的文件组织结构，按照问题编号（如第一问的1.1、1.2、1.3）自动创建文件夹体系，确保代码、结果、数据分离存放。

## 何时触发

- 用户开始解答数学建模竞赛的新问题
- 用户提到"创建文件夹结构"、"组织代码"、"按问题编号存放"
- 用户需要为子问题（1.1、1.2等）创建独立工作空间

## 文件夹结构规则

### 基本结构

```
项目根目录/
├── 问题1/
│   ├── 1.1/
│   │   ├── code/          # 代码文件
│   │   ├── results/       # 生成的结果（图表、输出文件）
│   │   ├── data/          # 生成的中间数据
│   │   └── conclusion.txt # 结论文件
│   ├── 1.2/
│   │   ├── code/
│   │   ├── results/
│   │   ├── data/
│   │   └── conclusion.txt
│   └── 1.3/
│       ├── code/
│       ├── results/
│       ├── data/
│       └── conclusion.txt
├── 问题2/
│   ├── 2.1/
│   │   ├── code/
│   │   ├── results/
│   │   ├── data/
│   │   └── conclusion.txt
│   └── 2.2/
│       ├── code/
│       ├── results/
│       ├── data/
│       └── conclusion.txt
└── ...
```

### 文件夹命名规范

- **父文件夹**：使用中文"问题+数字"，如`问题1`、`问题2`
- **子文件夹**：使用问题编号，如`1.1`、`1.2`、`2.1`
- **功能子文件夹**：
  - `code/`：存放所有.py代码文件
  - `results/`：存放生成的图表、分析报告等
  - `data/`：存放处理后的中间数据、CSV文件
  - `conclusion.txt`：存放该子问题的结论

## 使用流程

### 1. 创建文件夹结构

当用户开始解答新问题时，按照以下格式创建：

```bash
mkdir -p 问题1/1.1/{code,results,data}
mkdir -p 问题1/1.2/{code,results,data}
mkdir -p 问题1/1.3/{code,results,data}
```

### 2. 代码文件存放

- 所有.py文件放在对应子问题的`code/`文件夹中
- 文件名建议：`main.py`、`model.py`、`analyze.py`等

### 3. 结果输出

- 代码生成的图表、报告放在`results/`文件夹
- 文件名建议包含时间戳或版本号

### 4. 数据管理

- 中间处理结果、预测数据放在`data/`文件夹
- 文件名建议：`processed_data.csv`、`predictions.csv`

### 5. 结论记录

- 每个子问题完成后，在`conclusion.txt`中记录：
  - 使用的模型/方法
  - 关键参数
  - 主要结果
  - 结论总结

## 代码模板示例

### 代码文件开头规范

```python
# 问题1.1 主代码
import pandas as pd
import numpy as np

# 路径配置
BASE_DIR = "问题1/1.1"
DATA_DIR = f"{BASE_DIR}/data"
RESULTS_DIR = f"{BASE_DIR}/results"

# 数据加载
df = pd.read_excel("附件1：小区基础数据.xlsx")

# 处理逻辑
# ...

# 结果保存
df_result.to_csv(f"{DATA_DIR}/result.csv", index=False)
```

### 结论文件格式

```
问题：1.1 XXXX问题
方法：XXX模型/算法
关键参数：param1=value1, param2=value2
主要结果：
- 结果1
- 结果2
结论：XXX
```

## 注意事项

1. 每个子问题完全独立，不依赖其他子问题的代码
2. 路径使用相对路径，确保可移植性
3. 文件名使用英文，避免编码问题
4. 定期清理临时文件，保持目录整洁
5. 重要结果及时备份到`conclusion.txt`
