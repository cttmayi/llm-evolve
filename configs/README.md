# llmEvolve 配置文件

此目录包含 llmEvolve 的配置文件，针对不同用例提供了示例。

## 配置文件

### `default_config.yaml`
主配置文件，包含所有可用选项和合理的默认值。此文件包括：
- 所有配置参数的完整文档
- 所有设置的默认值
- **基于岛屿进化的参数** 以实现适当的进化多样性

将此文件作为您自己配置的模板。

### `island_config_example.yaml`
一个实用的示例配置，演示了正确的基于岛屿进化设置。展示了：
- 大多数用例的推荐岛屿设置
- 平衡的迁移参数
- 完整的工作配置

### `island_examples.yaml`
针对不同场景的多种示例配置：
- **最大多样性**：多个岛屿，频繁迁移
- **专注探索**：少量岛屿，罕见迁移
- **平衡方法**：默认推荐设置
- **快速探索**：小规模快速测试
- **大规模进化**：复杂优化运行

包含根据问题特征选择参数的指南。

## 基于岛屿进化参数

实现适当进化多样性的关键新参数是：

```yaml
database:
  num_islands: 5                      # 独立种群的数量
  migration_interval: 50              # 每N代迁移一次
  migration_rate: 0.1                 # 迁移顶级程序的比例
```

### 参数指南

- **num_islands**：大多数问题使用3-10个（越多=越多样）
- **migration_interval**：25-100代（越高=越独立）
- **migration_rate**：0.05-0.2（5%-20%，越高=知识共享越快）

### 何时使用何种配置

- **复杂问题** → 更多岛屿，较不频繁的迁移
- **简单问题** → 较少岛屿，更频繁的迁移
- **长时间运行** → 更多岛屿以保持多样性
- **短时间运行** → 较少岛屿以更快收敛

## 使用方法

复制任何这些文件作为您配置的起点：

```bash
cp configs/default_config.yaml my_config.yaml
# 编辑 my_config.yaml 以满足您的特定需求
```

然后与 llmEvolve 一起使用：

```python
from llm_evolve import llmEvolve
evolve = llmEvolve(
    initial_program_path="program.py",
    evaluation_file="evaluator.py", 
    config_path="my_config.yaml"
)
```