# 实验工具

所有运行环境共用这套入口。模型路径、设备和实验参数决定实际运行方式。

| 文件或目录 | 用途 |
| --- | --- |
| `launch.py` | 统一入口，准备、预检、突发请求和在线负载实验 |
| `prepare.py` | 从模型 tokenizer、数据集和样本索引生成冻结计划 |
| `run_matrix.py`、`block_design.py` | 配置矩阵、策略执行顺序和完整调用复用 |
| `audit.py`、`analyze.py` | 完整性检查与配对分析 |
| `common/`、`online/` | 计时、请求回放、在线指标与报告 |
| `benchmark.py` | 基础测量与指标工具 |
| `simulation/` | 不执行模型前向的 CPU 调度模拟 |
| `math500/` | 在线输入编码和相关测试所依赖的辅助模块 |
| `metadata/` | 探索与排除样本的索引，帮助重建输入集合 |

命令见 [复现指南](../docs/REPRODUCE.md)，结果见 [报告索引](../reports/README.md)。
