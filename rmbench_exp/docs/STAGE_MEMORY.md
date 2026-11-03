# 阶段记忆实验（第 1 期）

执行手册：Claude Doc "LiLa-WAM 阶段记忆实验执行手册"。本文件记录在代码中的实现、与手册不同的地方，以及运行步骤。

## 问题

给 LiLa-WAM 增加按阶段写入的记忆（每个阶段结束时，把该阶段首帧和末帧的 DINOv3 CLS 压成一个 token），
能否解决"画面相同但该做的事不同"的问题？第 1 期用 oracle 边界，测的是这种记忆机制的上限。

## 与手册不同的地方（已和用户确认）

1. **任务改为 swap_blocks**（手册为 put_back_block）。原因：
   - put_back_block 需要记住的信息（原来在哪块垫子）在第 0 帧就可见，F 组（只记第 0 帧）与 M 组拿到的关键信息相同，
     只能检验"有无记忆"，不能检验"按阶段写入"；
   - put_back_block 在第一轮中 0/50 成功，主要卡在按钮按得太浅（操作），成功率可能测不出记忆的作用；
   - swap_blocks 第一轮 43/43 条失败都是"不知道进行到哪一步"（换完重来 22 条、提前按按钮 16 条），
     且第 3 次移动完成后的画面与初始画面相同，第 0 帧不包含进度信息。
2. **阶段边界来自机器人状态，不来自合并标注。** swap_blocks 的标注文字会在一次移动的中途改变
   （方块被搬动后"左边的方块"变成"右边的方块"），按相同描述合并得不到 4 个阶段。定义：
   第 k 次松开夹爪后，两只机械臂都回到起始位置（2 cm 内）的那一帧 = 第 k 次移动结束
   （`stages.swap_blocks_event_ends`）。与标注片段结束帧相差 5–13 帧（50 条全部 < 1 个动作块）。
3. **oracle 与训练边界同一定义**（`rmbench_lila/oracle.py`）：方块已在目标托盘（容差与任务的成功判定相同）、
   两夹爪张开、两臂回到起始位置。只读取仿真器状态，不调用、不改变任务的成功判定。
4. 4 个阶段（移动 1、2、3、按按钮），槽位数 N = 3。
5. 单任务训练时 VTT 是常数向量，不提供任务信息；保留它只是为了热启动权重能严格加载。

## 实现位置（全部在子项目内，主项目文件不改）

| 内容 | 文件 |
|---|---|
| 边界计算、取整到 16 帧网格、扰动、"t 时刻哪些槽位已写入" | `rmbench_lila/stages.py` |
| 每帧 CLS 缓存 | `scripts/step21_cache_cls.py` |
| 训练样本的记忆（不使用 t 之后结束的阶段） | `rmbench_lila/data/stage_features.py` |
| 记忆 token、模型与 wrapper 扩展、推理时的记忆维护 | `rmbench_lila/stage_memory.py` |
| 模型变体 `lila_wam_stage_memory`、热启动加载 | `rmbench_lila/models.py` |
| 评测 oracle（客户端钩子，不改 RMBench） | `rmbench_lila/oracle.py`, `eval_hooks/sitecustomize.py`, `scripts/step24_eval_with_oracle.sh` |
| 配置 A / F / M | `configs/stage_memory/` |
| 检查：等价性、索引、无未来泄露、训练/推理记忆一致 | `tests/test_stage_memory.py`, `scripts/step23_stage_checks.py` |

序列：[动作 32 | 状态 1 | 寄存器 2 | VTT 1 | 视觉 64 | 记忆 N]，即 0–31、32、33–34、35、36–99、100–(99+N)。
前瞻解码器的 "visual" 切片改为显式的 `x[:, 36:100]`（主项目写法 `x[:, dino_start:]` 在追加记忆后会把记忆 token 也包括进去）。

旧实验不受影响：配置里没有 `model.stage_memory` 或 `enabled: false` 时，所有新代码都不会被调用。

## 分组

| 组 | 配置 | 评测 |
|---|---|---|
| A | `configs/stage_memory/A.yaml`（原模型） | `step24 ... sm_A-seed0 100 0` |
| F | `F.yaml`（1 个槽位，始终是第 0 帧 CLS） | `step24 ... sm_F-seed0 100 0` |
| M | `M.yaml`（3 个槽位，阶段记忆） | `step24 ... sm_M-seed0 100 0` |
| M-empty | M 的 checkpoint，评测时从不写入 | `step24 ... sm_M-seed0 100 0 empty` |

## 步骤

1. `python scripts/step20_make_boundaries.py --task swap_blocks` → 边界（已完成）
2. `python scripts/step21_cache_cls.py --task swap_blocks` → CLS 缓存
3. 用 45 条训练演示重算统计量和 VTT：
   `python scripts/step04_compute_stats.py --config configs/stage_memory/A.yaml` 与
   `python scripts/step05_compute_vtt.py --config configs/stage_memory/A.yaml`
4. 检查：`python -m pytest tests -q`；`python scripts/step23_stage_checks.py consistency --config configs/stage_memory/M.yaml`；
   过拟合（2 条演示）；显存与速度
5. 三组训练，热启动 `checkpoints_vla/9task_visual/checkpoint_epoch_8.pt`（RoboTwin visual，单阶段训练，没有第二阶段微调）；
   超参数：1e-4 → 1e-5 余弦、200 步预热、8,000 步、batch 64（理由见 `configs/stage_memory/A.yaml`），三组相同
6. 留出集按阶段 L_fm：`python scripts/step23_stage_checks.py heldout --run runs/sm_M-seed0`
7. 闭环评测（每组 100 次，同一种子），失败分类：`python analysis/state_diagnostics.py <ckpt 目录>`

## 第 1 期结果（2026-11-03，swap_blocks，每组 50 次闭环评测，种子参数 0）

| 组 | 成功 | 95% CI | 换完又重来 | 移动 1 次就按 | 移动 2 次就按 | 其他顺序错误 | 放置失败/反复重做 |
|---|---|---|---|---|---|---|---|
| A 原模型 | 3/50 (6%) | 2–16% | 32 | 6 | 6 | 2 | 1 |
| F 只记第 0 帧 | 2/50 (4%) | 1–14% | 34 | 4 | 4 | 1 | 5 |
| **M 阶段记忆** | **26/50 (52%)** | 39–65% | 1 | 0 | 17 | 1 | 5 |
| M-empty（M 的权重，不写记忆） | 0/50 (0%) | 0–7% | 41 | 0 | 0 | 0 | 9 |

配对检验（McNemar，相同种子）：M vs A：M 独成 24、A 独成 1，p = 1.5×10⁻⁶；M vs F：p = 2.4×10⁻⁷；
M vs M-empty：p = 1.2×10⁻⁷；A vs F：p = 1；A vs M-empty：p = 0.25。分类：`analysis/swap_blocks_progress.py`。

手册第 6 节的三条标准：(1) M 显著高于 A——满足；(2) M 减少的主要是记忆类失败（"换完又重来"32→1、"移动 1 次就按"6→0）——满足；
(3) M-empty 回落到 A 的水平（0% vs 6%，p = 0.25），且失败方式与 A 相同（换完又重来 41 次）——满足。
F 与 A 无差别：第 0 帧不含进度信息，按阶段写入是必要的。

局限：oracle 边界（上限）；单一训练种子；M 中 3 条失败与 oracle 写入推迟有关（机械臂未回到起始位置）；
M 余下的主要失败是"移动 2 次就按按钮"（17 条，此时第 1、2 槽位已正确写入）。

## 第 1 期补充检查（进行中）

1. **oracle 专家回放检查**（手册检查清单，之前漏做）：`scripts/step26_oracle_replay_check.py`，在 RMBench 环境中、
   RMBench 根目录下运行。用演示的种子和保存的关节路径重放 50 条演示，在每个保存帧上记录 oracle 的 stages_done，
   与训练边界（`boundaries_gt.json`）和标注片段结束帧比较。通过标准：每条演示的每次移动，差值 ≤ 1 个动作块（16 帧）。
2. **M0（不加边界扰动的 M）**：`configs/stage_memory/M_nojitter.yaml`，只改 `jitter_p: 0`。
   预先定下的判断：若"移动 2 次就按"（M 为 17 条）明显减少，且留出集探针中"移动 2 次后、槽位 3 为空"的按按钮比例
   低于 0.32，则扰动是原因之一；否则更可能是图像记忆捷径，下一步单独检验 `dataset.image_aug`。
