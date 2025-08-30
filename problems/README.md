# OpenEvolve 示例

本目录包含一系列示例，演示如何使用 OpenEvolve 进行各种任务，包括优化、算法发现和代码进化。每个示例展示了 OpenEvolve 功能的不同方面，并为创建您自己的进化编码项目提供模板。

## 快速开始模板

要创建您自己的 OpenEvolve 示例，需要三个基本组件：

### 1. 初始程序 (`initial_program.py`)

您的初始程序必须包含**一个** `EVOLVE-BLOCK`：

```python
# EVOLVE-BLOCK-START
def your_function():
    # 您的初始实现放在这里
    # 这是 OpenEvolve 将修改的唯一部分
    pass
# EVOLVE-BLOCK-END

# 帮助函数和其他在进化块外的代码
def helper_function():
    # 这部分代码不会被 OpenEvolve 修改
    pass
```

**关键要求：**
- ✅ **恰好一个 EVOLVE-BLOCK**（不是多个块）
- ✅ 使用 `# EVOLVE-BLOCK-START` 和 `# EVOLVE-BLOCK-END` 标记
- ✅ 只将您想要进化的代码放在块内
- ✅ 帮助函数和导入放在块外

### 2. 评估器 (`evaluator.py`)

您的评估器必须返回一个包含特定指标名称的**字典**：

```python
def evaluate(program_path: str) -> Dict:
    """
    评估程序并返回指标字典。
    
    关键：必须返回字典，而不是 EvaluationResult 对象。
    """
    try:
        # 导入并运行您的程序
        # 计算指标
        
        return {
            'combined_score': 0.8,  # 进化的主要指标（必需）
            'accuracy': 0.9,        # 您的自定义指标
            'speed': 0.7,
            'robustness': 0.6,
            # 添加您想要跟踪的其他指标
        }
    except Exception as e:
        return {
            'combined_score': 0.0,  # 即使出错也要返回 combined_score
            'error': str(e)
        }
```

**关键要求：**
- ✅ **返回字典**，而不是 `EvaluationResult` 对象
- ✅ **必须包含 `'combined_score'`** - 这是 OpenEvolve 使用的主要指标
- ✅ 更高的 `combined_score` 值应该表示更好的程序
- ✅ 优雅地处理异常并在失败时返回 `combined_score: 0.0`

### 3. 配置 (`config.yaml`)

基本配置结构：

```yaml
# 进化设置
max_iterations: 100
checkpoint_interval: 10
parallel_evaluations: 1

# LLM 配置
llm:
  api_base: "https://api.openai.com/v1"  # 或您的 LLM 提供商
  models:
    - name: "gpt-4"
      weight: 1.0
  temperature: 0.7
  max_tokens: 4000
  timeout: 120

# 数据库配置（MAP-Elites 算法）
database:
  population_size: 50
  num_islands: 3
  migration_interval: 10
  feature_dimensions:  # 必须是列表，不是整数
    - "score"
    - "complexity"

# 评估设置
evaluator:
  timeout: 60
  max_retries: 3

# 提示配置
prompt:
  system_message: |
    您是一位专业程序员。您的目标是改进
    EVOLVE-BLOCK 中的代码以获得更好的任务性能。
    
    专注于算法改进和代码优化。
  num_top_programs: 3
  num_diverse_programs: 2

# 日志
log_level: "INFO"
```

**关键要求：**
- ✅ **`feature_dimensions` 必须是列表**（例如 `["score", "complexity"]`），不是整数
- ✅ 根据您的用例设置适当的超时时间
- ✅ 为您的提供商配置 LLM 设置
- ✅ 使用有意义的 `system_message` 来指导进化

## 常见配置错误

❌ **错误：** `feature_dimensions: 2`  
✅ **正确：** `feature_dimensions: ["score", "complexity"]`

❌ **错误：** 返回 `EvaluationResult` 对象  
✅ **正确：** 返回 `{'combined_score': 0.8, ...}` 字典

❌ **错误：** 使用 `'total_score'` 指标名称  
✅ **正确：** 使用 `'combined_score'` 指标名称

❌ **错误：** 多个 EVOLVE-BLOCK 部分  
✅ **正确：** 恰好一个 EVOLVE-BLOCK 部分

## MAP-Elites 特征维度最佳实践

使用自定义特征维度时，您的评估器必须返回**原始连续值**，而不是预计算的分箱索引：

### ✅ 正确：返回原始值
```python
def evaluate(program_path: str) -> Dict:
    # 计算实际测量值
    prompt_length = len(generated_prompt)  # 实际字符计数
    execution_time = measure_runtime()     # 秒为单位的时间
    memory_usage = get_peak_memory()       # 使用的字节数
    
    return {
        "combined_score": accuracy_score,
        "prompt_length": prompt_length,    # 原始计数，不是分箱索引
        "execution_time": execution_time,  # 原始秒数，不是分箱索引  
        "memory_usage": memory_usage       # 原始字节数，不是分箱索引
    }
```

### ❌ 错误：返回分箱索引
```python
def evaluate(program_path: str) -> Dict:
    prompt_length = len(generated_prompt)
    
    # 不要这样做 - 预先计算分箱
    if prompt_length < 100:
        length_bin = 0
    elif prompt_length < 500:
        length_bin = 1
    # ... 更多分箱逻辑
    
    return {
        "combined_score": accuracy_score,
        "prompt_length": length_bin,  # ❌ 这是分箱索引，不是原始值
    }
```

### 为什么这很重要
- OpenEvolve 内部使用 min-max 缩放
- 分箱索引会被错误地缩放，就好像它们是原始值一样
- 随着新程序改变最小/最大范围，网格位置变得不稳定
- 这违反了 MAP-Elites 原则并导致进化效果不佳

### 好的特征维度示例
- **计数**：令牌计数、行数、字符计数
- **性能**：执行时间、内存使用、吞吐量
- **质量**：准确率、精确率、召回率、F1 分数  
- **复杂度**：圈复杂度、嵌套深度、函数计数

## 运行您的示例

```bash
# 基本运行
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 100

# 从检查点恢复
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50

# 查看结果
python scripts/visualizer.py --path path/to/openevolve_output/checkpoints/checkpoint_100/
```

## 高级配置选项

### LLM 集成（多个模型）
```yaml
llm:
  models:
    - name: "gpt-4"
      weight: 0.7
    - name: "claude-3-sonnet"
      weight: 0.3
```

### 岛屿进化（种群多样性）
```yaml
database:
  num_islands: 5        # 更多岛屿 = 更多多样性
  migration_interval: 15  # 岛屿交换程序的频率
  population_size: 100   # 更大种群 = 更多探索
```

### 级联评估（多阶段测试）
```yaml
evaluator:
  cascade_stages:
    - stage1_timeout: 30   # 快速验证
    - stage2_timeout: 120  # 全面评估
```

## 示例目录

### 🧮 数学优化

#### [函数最小化](function_minimization/)
**任务：** 寻找复杂非凸函数的全局最小值  
**成就：** 从随机搜索演化为复杂的模拟退火算法  
**关键教训：** 展示优化算法的自动发现  
```bash
cd examples/function_minimization
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

#### [圆形填充](circle_packing/)
**任务：** 在单位正方形中填充26个圆以最大化半径之和  
**成就：** 匹配 AlphaEvolve 论文结果 (2.634/2.635)  
**关键教训：** 演示从几何启发式到数学优化的进化  
```bash
cd examples/circle_packing
python ../../openevolve-run.py initial_program.py evaluator.py --config config_phase_1.yaml
```

### 🔧 算法发现

#### [信号处理](signal_processing/)
**任务：** 为音频处理设计数字滤波器  
**成就：** 发现了具有优越特性的新颖滤波器设计  
**关键教训：** 展示特定领域算法的进化  
```bash
cd examples/signal_processing
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

#### [Rust 自适应排序](rust_adaptive_sort/)
**任务：** 创建适应数据模式的排序算法  
**成就：** 演化的排序策略超越了传统算法  
**关键教训：** 多语言支持（Rust）和算法适应  
```bash
cd examples/rust_adaptive_sort
python ../../openevolve-run.py initial_program.rs evaluator.py --config config.yaml
```

### 🚀 性能优化

#### [MLX Metal 内核优化](mlx_metal_kernel_opt/)
**任务：** 为 Apple Silicon 优化注意力机制  
**成就：** 比基线实现快 2-3 倍  
**关键教训：** 硬件特定优化和性能调优  
```bash
cd examples/mlx_metal_kernel_opt
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### 🌐 Web 和数据处理

#### [使用 optillm 的 Web 爬虫](web_scraper_optillm/)
**任务：** 从 HTML 页面提取 API 文档  
**成就：** 演示了具有 readurls 和 MoA 的 optillm 集成  
**关键教训：** 展示与 LLM 代理系统和测试时计算的集成  
```bash
cd examples/web_scraper_optillm
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### 💻 编程挑战

#### [在线编程评判](online_judge_programming/)
**任务：** 解决竞争性编程问题  
**成就：** 自动化解决方案生成和提交  
**关键教训：** 与外部评估系统的集成  
```bash
cd examples/online_judge_programming
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### 📊 机器学习和 AI

#### [LLM 提示优化](llm_prompt_optimazation/)
**任务：** 演化提示以获得更好的 LLM 性能  
**成就：** 发现了有效的提示工程技术  
**关键教训：** 自我改进的 AI 系统和提示进化  
```bash
cd examples/llm_prompt_optimazation
python ../../openevolve-run.py initial_prompt.txt evaluator.py --config config.yaml
```

#### [LM-Eval 集成](lm_eval/)
**任务：** 与语言模型评估工具集成  
**成就：** 自动化基准改进  
**关键教训：** 与标准 ML 评估框架的集成  

#### [符号回归](symbolic_regression/)
**任务：** 从数据中发现数学表达式  
**成就：** 科学方程的自动发现  
**关键教训：** 科学发现和数学建模  

### 🔬 科学计算

#### [R 鲁棒回归](r_robust_regression/)
**任务：** 开发鲁棒的统计回归方法  
**成就：** 对异常值具有抵抗性的新颖统计算法  
**关键教训：** 多语言支持（R）和统计算法进化  
```bash
cd examples/r_robust_regression
python ../../openevolve-run.py initial_program.r evaluator.py --config config.yaml
```

### 🎯 高级功能

#### [带工件的圆形填充](circle_packing_with_artifacts/)
**任务：** 具有详细执行反馈的圆形填充  
**成就：** 高级调试和工件收集  
**关键教训：** 使用 OpenEvolve 的工件系统进行详细分析  
```bash
cd examples/circle_packing_with_artifacts
python ../../openevolve-run.py initial_program.py evaluator.py --config config_phase_1.yaml
```

## 最佳实践

### 🎯 设计有效的评估器
- 使用反映您目标的有意义指标
- 包括质量和效率措施
- 优雅地处理边缘情况和错误
- 为调试提供信息性反馈

### 🔧 配置调优
- 从较小种群和较少迭代开始测试
- 增加 `num_islands` 以获得更多样化的探索
- 根据您希望 LLM 的创造程度调整 `temperature`
- 为您的计算环境设置适当的超时时间

### 📈 进化策略
- 使用不同配置的多个阶段
- 从探索开始，然后专注于利用
- 考虑对昂贵的测试使用级联评估
- 监控进度并根据需要调整配置

### 🐛 调试
- 在 `openevolve_output/logs/` 中检查日志
- 在检查点目录中检查失败的程序
- 使用工件了解程序行为
- 在进化前独立测试您的评估器

## 获取帮助

- 📖 查看各个示例 README 以获取详细演练
- 🔍 查看主 [OpenEvolve 文档](../README.md)
- 💬 在 [GitHub 仓库](https://github.com/codelion/openevolve) 上打开问题

每个示例都是自包含的，包含入门所需的所有必要文件。选择一个与您的用例相似的示例，并使其适应您的特定问题！