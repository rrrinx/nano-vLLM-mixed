# 调度性能评测

| 场景 | 验证问题 | 报告 |
| --- | --- | --- |
| 突发请求 | 已有 Decode 遇到新到达的长 Prefill，混合批如何影响两类请求？ | [突发请求调度](burst_requests/README.md) |
| 长输入在线服务 | 请求持续到达时，吞吐、完成延迟、生成停顿和首 token 等待如何变化？ | [性能报告](long_input_serving/README.md) · [Notebook](long_input_serving/mixed_scheduling_report.ipynb) |

两种场景均保留全部正式配置，不筛选获益记录。所有数值可以用公开测量重算，数据导出方式及限制见 [公开数据说明](data/README.md)。
