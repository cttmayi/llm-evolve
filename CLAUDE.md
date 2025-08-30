# CLAUDE.md

该文件为Claude Code（claude.ai/code）在处理此代码库中的代码时提供指导。

## 概述

OpenEvolve是Google DeepMind的AlphaEvolve系统的开源实现 - 一个使用LLM通过迭代进化优化代码的进化代码智能体。该框架可以进化多种语言（Python、R、Rust等）的代码，用于科学计算、优化和算法发现等任务。

## 基本命令

### 开发环境设置
```bash
# 以开发模式安装所有依赖
uv pip install -e ".[dev]"
```

### 运行测试
```bash
# 运行所有测试
python -m unittest discover tests
```

### 代码格式化
```bash
# 使用Black格式化
python -m black openevolve examples tests scripts

# 或使用Makefile
make lint
```

### 运行OpenEvolve
```bash
# 基本进化运行
python openevolve-run.py path/to/problem --iterations 1000

# 从检查点恢复
python openevolve-run.py path/to/problem --checkpoint path/to/checkpoint_directory --iterations 50
```

### 可视化
```bash
# 查看进化树
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

## 高层架构

### 核心组件

1. **控制器（`openevolve/controller.py`）**：使用ProcessPoolExecutor进行并行迭代执行的主要协调器。

2. **数据库（`openevolve/database.py`）**：实现具有基于岛屿进化的MAP-Elites算法：
   - 程序映射到多维特征网格
   - 多个隔离的种群（岛屿）独立进化
   - 岛屿之间的周期性迁移防止收敛
   - 单独跟踪绝对最佳程序

3. **评估器（`openevolve/evaluator.py`）**：级联评估模式：
   - 阶段1：快速验证
   - 阶段2：基本性能测试  
   - 阶段3：全面评估
   - 程序必须在每个阶段通过阈值

4. **LLM集成（`openevolve/llm/`）**：具有多个模型、可配置权重和重试逻辑的异步生成的集成方法。

5. **迭代（`openevolve/iteration.py`）**：从岛屿采样的工作进程，通过LLM生成突变，评估程序并存储工件。

### 关键架构模式

- **基于岛屿的进化**：多个种群分别进化并进行周期性迁移
- **MAP-Elites**：通过将程序映射到特征网格单元来保持多样性
- **工件系统**：程序返回调试数据的旁路通道，存储为JSON或文件
- **进程工作器模式**：每个迭代在具有数据库快照的新进程中运行
- **双重选择**：用于灵感的程序与显示给LLM的程序不同
- **延迟迁移**：岛屿基于世代计数而非迭代进行迁移

### 代码进化标记

使用以下方式标记要进化的代码部分：
```python
# EVOLVE-BLOCK-START
# 要进化的代码放在这里
# EVOLVE-BLOCK-END
```

### 配置

具有层次结构的基于YAML的配置：
- LLM模型和参数
- 进化策略（基于差异与完全重写）
- 数据库和岛屿设置
- 评估参数

### 重要模式

1. **检查点/恢复**：自动保存整个系统状态并能够无缝恢复
2. **并行评估**：通过TaskPool并发评估多个程序
3. **错误恢复**：单个故障不会使系统崩溃 - 广泛的重试逻辑和超时保护
4. **提示工程**：基于模板的系统，具有上下文感知构建和进化历史

### 开发说明

- 需要Python >=3.9
- 使用OpenAI兼容API进行LLM集成
- 测试使用unittest框架
- Black进行代码格式化
- 工件阈值：小文件（<10KB）存储在数据库中，大文件保存到磁盘
- 进程工作器加载数据库快照以实现真正的并行性