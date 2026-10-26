# 用 RMBench 检验 LiLa-WAM 在记忆任务上的局限：实验计划

依据：LiLa-WAM 仓库 `AyanamiiiiRei/LiLa-WAM`（commit 339860f），RMBench 仓库 `robotwin-Platform/rmbench`（commit 7d1cdfc）。本计划只根据两者的代码整理，RMBench 的实际数据尚未检查，第 2 步必须先做。

---

## 0. 要证明的是什么

**假设：** LiLa-WAM 在 RMBench 上表现差，主要是因为它**没有记忆**：每次决策只看当前一帧图像和当前状态（`camera_indices: [0]`、`state_indices: [0]`），无法利用几百步之前的信息。

**必须排除的其他解释：** 成功率低还可能来自以下原因，实验设计要能把它们区分开：

| 其他可能的解释 | 用哪个实验排除 |
|---|---|
| 数据处理有错误（颜色通道、动作对齐、字段拼错） | 第 7 步的检查 |
| 每个任务只有 50 条演示，数据太少 | 对照实验 A |
| 没有域随机化、相机视场角不同等环境差异 | 对照实验 A |
| 任务本身的操作难度高（抓取、放置失败） | 失败原因分析（第 11 步） |

**最有力的证据**是对照实验 B：给 LiLa-WAM 加上历史信息后成功率明显提升，说明缺的确实是记忆。

---

## 1. 准备环境

1. 按 RMBench 的 README 安装：`conda create -n RMBench python=3.10`，`git clone --recurse-submodules`，运行 `scripts/_install.sh`（包括 CuRobo 和 XPolicyLab 子模块）。
2. 下载资源和数据：`bash scripts/_download_assets.sh`，`bash scripts/_download_data.sh`。数据会解压到 `data/demo_clean/<任务名>/aloha_agilex/`。
3. 准备 LiLa-WAM 的环境和 DINOv3 权重（`dinov3-vitl16-pretrain-lvd1689m`）。两个项目最好用两个独立的 conda 环境，评测时通过策略服务器连接。

## 2. 检查实际数据（先做这一步，再改代码）

对每个任务抽 1–2 条演示，确认以下内容：

- [ ] 字段结构：`h5py.File(path).visit(print)`，与第 3 步的字段对照表核对。
- [ ] 各字段的形状、数据类型和取值范围：关节角是否在 ±π 内，夹爪是否在 0–1 之间。
- [ ] `action[t]` 是否等于 `state[t+1]`（代码如此写入，用数据验证一下）。
- [ ] `additional_info/frequency` 的值（预计为 15）。
- [ ] 图像格式：用 RMBench 的 `data/decode_image_bit.py` 中的 `_has_rgb_marker` 检查是新格式还是旧格式；解码后保存几张图片，**目视确认颜色正常**。
- [ ] 每个任务的演示条数和每条演示的帧数，算出帧数的分布。RMBench 任务的时间跨度很长，这会影响训练样本数和 epoch 的设置。
- [ ] 每个任务"需要记住什么"：阅读 `envs/<任务名>.py` 和 `description/task_instruction/<任务名>.json`，为每个任务写一句话说明它依赖的记忆内容。第 11 步的失败分析要用到。

## 3. 修改数据加载代码（`dataloader/dataset.py`）

| 位置 | 原代码 | 改为 |
|---|---|---|
| `_load_episodes` | 扫描 `{任务}/demo_clean/data/*.hdf5` | 扫描 `demo_clean/{任务}/aloha_agilex/data/*.hdf5`；任务名取自中间那一层目录 |
| `_build_index_map` | 用 `joint_action/vector` 读帧数 | 改用 `action/left_arm_joint_states` 的长度 |
| `_load_hdf5_data`：动作 | `joint_action/vector` | `action/` 中 `left_arm_joint_states`、`left_ee_joint_states`、`right_arm_joint_states`、`right_ee_joint_states` 依次拼接，共 14 维 |
| `_load_hdf5_data`：状态 | `endpose/left_endpose` 等 | `state/` 中 `left_ee_poses`、`left_ee_joint_states`、`right_ee_poses`、`right_ee_joint_states` 依次拼接，共 16 维 |
| `_load_hdf5_data`：图像 | `observation/head_camera/rgb` | `vision/cam_head/colors` |
| `_decode` | `cv2.imdecode`，不转换颜色 | **改用 RMBench 的 `decode_image_bit`**，之后不要再做任何颜色转换 |

需要做出并记录的决定：

- **动作对齐方式。** RMBench 的 `action[t] = state[t+1]`，而 LiLa-WAM 在 RoboTwin 上的动作块第一步是当前状态。建议直接使用 RMBench 的 `action` 组（即"预测下一步"），并在训练和推理中保持一致。不要和 RoboTwin 上训练的模型混用。
- **状态中的夹爪。** RoboTwin 用的是 `endpose/left_gripper`，RMBench 只有 `left_ee_joint_states`。确认两者的含义和取值范围，评测时的状态必须用同样的定义。
- **未来帧偏移。** 保持 `future_frame_offset = 32`（等于动作块长度），越界时取最后一帧。

## 4. 重新计算归一化统计量

1. 按第 3 步的字段修改 `utils/calc_stat_remove_outlier.py`。
2. **先关闭删除功能**（`--delete_outliers` 默认是开启的，会直接删除文件），只统计、不删除，看一下有多少条超出 ±π。每个任务只有 50 条演示，不要轻易删除。
3. 生成新的统计文件（如 `utils/stat-rmbench.json`），训练和推理都通过 `--norm_stats_path` 指向它。注意 `VLAWrapper.load_norm_stats` 中读取的键名是 `'robotwin2'`，新文件要用同样的结构，或者修改这一行。

## 5. 重新计算 VTT

1. 修改 `utils/precompute_task_cond.py`：目录扫描、图像字段名、解码函数（同样改用 `decode_image_bit`）。
2. 对 RMBench 的每个任务运行一次，输出到新目录（如 `data-rmbench-taskcond/`）。
3. 计算任务之间 VTT 的余弦相似度。如果某些任务的 VTT 非常接近，模型就难以区分它们，这本身也是一个值得记录的发现。

## 6. 新建训练配置（复制 `configs/robotwin_all.yaml`）

| 配置项 | 设置 |
|---|---|
| `dataset.dataset_dir` | RMBench 数据的 `demo_clean` 目录 |
| `dataset.data_mode` | `"clean"` |
| `dataset.camera_names` | 按第 3 步的修改方式对应头部相机 |
| `dataset.image_size` | `[320, 240]`（RMBench 头部相机就是这个分辨率，视场角为 50°） |
| `dataset.task_cond_dir` | 第 5 步的输出目录 |
| `model.vision_encoder.feat_layers` | **与你在 RoboTwin 上的实验保持一致**（仓库默认 `[-12, -8, -4]`，论文主设置为第 17、21 层），并在报告中写明 |
| `training.epochs` | 数据量约为 RoboTwin 的十分之一，按**总训练步数**而不是 epoch 数来对齐，并保存多个 checkpoint 以便比较 |
| `training.batch_size` / `grad_accum_steps` | 按 16 GB 显存调整。注意：`grad_accum_steps > 1` 时，屏幕日志中的 `Loss` 会显示为真实值的若干倍（`train.py` 第 297 行），CSV 中的值才是对的 |

训练方式：建议和 RMBench 的 `m1_mix` 一样，**用所有任务训练一个多任务模型**；如果算力允许，再对几个 M(n) 任务单独训练做对比。

## 7. 正式训练前的检查（不要跳过）

1. **颜色检查**：从 dataloader 中取一个 batch，把 `pixel_values` 反归一化后保存成图片，和 RMBench 自带的视频（`video/episode_xxx.mp4`）对比。
2. **动作检查**：用 `utils/plot_action_curves.py`（需要改字段名）画出几条演示的 14 维动作曲线，确认左右臂、夹爪的顺序正确，曲线平滑、没有跳变。
3. **形状检查**：在 `VLAModel.forward` 中打印各组 token 的形状，确认序列仍是 `(B, 100, 768)`。
4. **过拟合测试**：只用一个小 batch 反复训练几百步，损失应该能降到很低。降不下去说明数据或代码有问题。
5. **开环测试**：在留出的几条演示上，比较模型预测的动作块和真实动作的误差。这一步不需要仿真环境。

## 8. 训练

- 主实验：完整版 LiLa-WAM（$\mathcal{L}_{fm} + 0.5\,\mathcal{L}_{ff}$）。
- 可选：去掉 $\mathcal{L}_{ff}$ 的消融版，看"预测未来"对记忆任务是否有帮助（预期帮助不大，因为它预测的是未来而不是回顾过去）。
- 记录：配置文件、git commit、随机种子、损失曲线（`log/loss/` 下的 CSV）。

## 9. 接入 RMBench 的评测

1. 为 LiLa-WAM 写一个策略适配器。RMBench 支持两种方式：`policy/` 目录下的原始适配器，或者 XPolicyLab 的适配器（README 推荐后者）。可以参考 `XPolicyLab/policy/Mem_0/` 的写法。
2. 适配器中复用 `robotwin_infer.py` 的推理逻辑，但需要检查：
   - **观测字段**：评测时环境传来的观测格式可能和 RoboTwin 不同。图像应该已经是解码好的 RGB（README 中说明了），不要再做颜色转换；状态要按第 3 步的定义拼成 16 维。
   - **任务向量**：每个任务开始前调用 `set_task(task_name)`，加载第 5 步的 VTT。
   - **动作**：输出 14 维绝对关节角，评测时使用 `--action-type joint`；每次推理执行前 16 步。
   - **机器人类型**：数据是 `aloha_agilex` 采集的，评测也要用同样的机器人。README 示例中的 `--env-cfg-type arx_x5` 是另一种机器人，不要照抄。
   - **B 样条平滑**：保持和 RoboTwin 实验相同的设置（默认开启），并在报告中写明。
3. 评测参数：最大步数使用 `env_cfg/task_config/_eval_step_limit.yml` 中的值；每个任务的评测次数**与 RMBench 论文的协议保持一致**（查阅论文或 Hugging Face 上的评测日志确认），至少不少于 50 次；固定随机种子。

## 10. 对照实验

**对照实验 A：数据量和环境的影响**

在原 RoboTwin 数据上，选几个**不需要记忆**的任务，每个任务只用 **50 条干净演示**训练 LiLa-WAM，并在干净场景中评测。

- 如果成功率仍然较高，就说明"每个任务 50 条、没有随机化"本身不是 LiLa-WAM 在 RMBench 上失败的主要原因。

**对照实验 B：加入历史信息（最关键）**

给 LiLa-WAM 一定的历史信息，看成功率是否提升。按改动从小到大排列：

1. **状态历史**（几乎不用改模型）：设置 `state_indices: [-k, …, -1, 0]` 和 `proprio_len: k+1`。模型本来就支持多个状态 token，起点之前的帧会被截断到第 0 帧。
2. **首帧图像**：把这条演示的第一帧也作为输入，即额外的一组视觉 token。需要修改数据加载、adapter 调用和序列拼接。
3. **多帧图像历史**：每隔若干步取一帧历史图像。改动最大，计算量也会增加。

如果加入历史后，成功率在依赖记忆的任务上明显提升，而在不依赖记忆的任务上变化不大，就是"局限来自缺乏记忆"的有力证据。

**对照实验 C：与已有方法比较**

把结果和 RMBench 论文中报告的方法（Mem-0、DP、ACT、π0.5、X-VLA 等）放在一起。注意核对评测次数、机器人类型、训练数据是否一致；不一致的地方要在报告中说明。

## 11. 结果分析

1. **成功率及其置信区间**：按二项分布计算每个任务的置信区间（50 次评测时，成功率约 50% 的任务误差在 ±14 个百分点左右）。比较两个版本时，在同一组随机种子上做**配对比较**。
2. **失败原因分类**（直接支持"缺乏记忆"这一结论）：用评测视频（`eval_video_log: true`），把每次失败归入以下类别：
   - **记忆错误**：操作动作本身正确，但做错了对象或顺序，比如重复做已经做过的步骤、忘记被遮住的物体在哪里、按错按钮次数；
   - **操作错误**：抓取失败、放置位置不准、碰倒物体；
   - **超时或卡住**。
   如果 LiLa-WAM 的失败**主要是记忆错误**，而操作本身大多成功，结论就很清楚。
3. **按任务类型比较**：分开报告 M(1) 和 M(n) 任务的结果，看需要记忆的信息越多，成功率是否越低。

## 12. 常见陷阱清单

- [ ] 图像颜色：必须用 `decode_image_bit`，训练和评测都不要额外转换颜色。
- [ ] 动作对齐：`action[t] = state[t+1]`，训练和推理保持同一种约定。
- [ ] 归一化统计量：必须在 RMBench 数据上重新计算，不要沿用 `stat-500-all.json`。
- [ ] VTT：必须在 RMBench 任务上重新计算，配置中的路径要和实际目录一致。
- [ ] 异常值删除脚本默认会**直接删除文件**，运行前先备份数据或关闭删除功能。
- [ ] 去除静止片段的脚本会**覆盖原文件且默认不备份**；对记忆任务，"静止等待"本身可能有意义，不建议使用。
- [ ] 评测时的机器人类型必须是 `aloha_agilex`。
- [ ] 评测次数要足够，`eval.sh` 中的默认值 `TEST_NUM=5` 远远不够。
- [ ] 所有实验固定随机种子，并记录配置和 commit。
