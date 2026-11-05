# 子任务方案（2026-11-05）

问题：把每个任务拆成子任务，LiLa-WAM 只作为"按子任务 VTT 执行"的执行器，由 Sub Task Chooser 决定当前子任务何时结束、
下一个做什么；这样的分工在 RMBench 全部 10 个任务上能否超过同样微调的原模型？

- 子任务：语言标注中连续相同的描述合并为一个子任务（swap_blocks 用事件边界）；子任务类型 = (任务, 描述)，
  子任务 VTT = 该类型所有片段 CLS(末) − CLS(首) 的平均（`scripts/step36_subtask_library.py`，`artifacts/subtasks/`）。
- 执行器输入（接在 64 个视觉 token 之后）：子任务 token = Linear(LN(子任务 VTT)) + 类型嵌入；选中的记忆 token =
  子任务 token 对记忆条目的注意力（条目 0 = 第 0 帧，之后每个完成的子任务一条：首帧 CLS + 末帧 CLS，最多 12 条）。
  类型嵌入是对用户设计的补充：swap_blocks 中三次移动的子任务 VTT 范数只有 0.6–1.5、一致性 0.15，单靠 VTT 难以区分。
- Chooser：2 层 transformer，输入 DiT 最后一层的整条输出序列 + 当前帧 CLS + 当前子任务类型 + 查询 token；
  输出"当前子任务是否结束"和"下一个子任务类型"（只在本任务的类型中选）。不输入已进行的块数（避免学成计时器）。
- 时序约定：每个动作块先用"决定前"的状态在流时间 0 前向一次，Chooser 决定；若结束，用当前帧 CLS 写入记忆条目、
  换子任务类型，再用"决定后"的状态预测本块动作。训练与推理一致，无标签泄漏。
- 方案一（P1，联合）：执行器与 Chooser 一起训练，总损失 = L_fm + 0.5·L_ff + 0.1·L_chooser，Chooser 损失也训练 DiT。
- 方案二（P2，分开）：先只训练执行器（L_fm + 0.5·L_ff），冻结后单独训练 Chooser（`scripts/step38_train_chooser.py`）。
- 从头训练（用户决定）：DINOv3 冻结，训练设置与第一轮基线 baseline-seed0 完全相同（85,928 步、batch 64、学习率 2e-4 → 5e-5，约 17.4 个 epoch，不热启动），基线即为匹配的对照。
- 本轮不含失败标签（演示中没有失败；见讨论记录），失败记忆与重做留到之后。
- 评测：每组 10 个任务 × 50 次，种子与第一轮相同，配对比较（`analysis/subtask_summary.py`）。
- 局限：子任务库用了全部 50 条演示（含每任务 2 条留出演示，只影响类型均值）；每组一个训练种子。

运行：`bash scripts/step39_subtask_smoke.sh`（先检查）→ `bash scripts/step40_subtask_all.sh`（全部，后台，约 3 天）。
