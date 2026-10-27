# 设计决定与代码检查中的发现

依据：LiLa-WAM fork `2bba854`（计划写于 `339860f`，两者间只有注释修改），RMBench `7d1cdfc`，XPolicyLab `fa431ec`。
标 ⚠ 的条目需要你确认或在报告中写明。

## 与计划不同或计划中需要更正的地方

1. **主项目的异常值删除无法关闭。** `utils/calc_stat_remove_outlier.py` 中
   `--delete_outliers` 是 `action='store_true', default=True`，命令行无法关掉，运行即删除文件。
   计划第 4 步"先关闭删除功能"做不到，因此子项目重写了统计脚本（`stats.py`），**只读、不删除**，可疑演示只列出。
2. **主项目学习率调度在梯度累积时不完整。** `train.py` 中 `T_max = epochs × len(dataloader)`，但调度器每个*优化器步*才走一步，
   当 `grad_accum_steps > 1` 时余弦曲线只走完 1/grad_accum，学习率降不到 `lr_min`。子项目按优化器步数设定调度。
3. **主项目没有设置随机种子**，且 dataloader 出错时会**静默换成随机样本**。子项目固定所有种子，读取出错直接报错。
4. ⚠ **评测次数**：RMBench 的 `eval_policy_xpolicylab.py` 默认 `test_num=100`；适配器默认用 100（`TEST_NUM` 可改）。请与 RMBench 论文核对。
5. **评测机器人**：下载的数据是 `aloha_agilex`（`data/_download.py` 写死）；Mem-0 适配器是为 `arx_x5`（dual_x5）写的，不要照抄。适配器会检查维度。

## 数据约定

6. **动作对齐**：默认 `rmbench.action_alignment: next`，即动作块 = `action/` 组（`action[t] = state[t+1]`）。
   可选 `current`（从 `state/` 组读，块首 = 当前姿态，同 RoboTwin 上的习惯）。训练和推理由同一个配置决定，不会不一致。
7. **状态**：`ee16` = 左末端位姿(7) + 左夹爪(1) + 右末端位姿(7) + 右夹爪(1)，与 RoboTwin 上的 16 维语义相同。
   夹爪来自 `state/left_ee_joint_states`；评测时 RMBench 的 `convert_state` 也优先取 `joint_action/left_gripper`，二者同源。
   第 2 步脚本会检查位姿后 4 维是否为单位四元数。
7b. **四元数符号统一（`rmbench.state_transforms: [quat_continuity]`，默认开启）**。仿真器有时会在相邻帧间翻转末端
    四元数的符号（q 与 −q 表示同一姿态），在状态输入中造成虚假的大幅跳变。第 7 步检查各任务前 5 条演示：
    battery_try 左 6 / 右 20 次，observe_and_pickup 右 3 次，其余 8 个任务为 0；抽查的 RoboTwin 数据为 0
    （只抽到 adjust_bottle 一个任务）。规则：每条演示首帧取 w ≥ 0，之后每帧取与上一帧点积非负的符号。
    训练按整条演示处理，推理按步在线处理，结果完全相同（`tests/test_transforms.py`）。统计量在处理后的状态上计算。
8. **图像**：一律用 `decode_image_bit`；之后不做任何颜色转换。训练、VTT、推理共用 `image.to_model_input`（INTER_LINEAR 缩放 + ImageNet 归一化）。
9. **留出集**：每个任务最后 2 条演示（按编号）不参加训练、统计量和 VTT，用于开环测试。
9b. **样本按帧均匀抽取（已确认，与 LiLa-WAM 原做法一致）**。各任务的训练量与其总帧数成正比：
    `blocks_ranking_try` 约占 22%，`observe_and_pickup` 约占 2%（第 2 步报告）。报告中需写明这一占比，
    分析短任务的成功率时要考虑训练量少这一因素。
9c. **超出 ±π 的 9 条演示全部保留**（battery_try 6 条、swap_blocks 3 条）。
    逐帧检查：被标记维度（5、8、9）的最大单步变化都 ≤ 0.30 rad，没有 ±π 处的跳变，是真实的大角度旋转。
    代价：第 5 维（左腕旋转）归一化范围变为 [-5.34, 4.01]，比其他任务宽。
9d. ⚠ **部分任务的 VTT 很弱**（第 5 步）：`place_block_mat`（范数 1.20，演示间一致性 cos 0.47）、`swap_T`（1.64，0.48）、
    `cover_blocks`（2.09，0.73）；其余任务范数 5.8–12.3、一致性 ≥ 0.91。这三个任务结束时场景接近初始状态，
    "末帧 − 首帧"几乎不含任务信息。最相似的任务对：put_back_block–rearrange_blocks（0.89）。
    这是"缺乏记忆"之外的另一种可能解释（任务指定不清），分析这三个任务时必须考虑，必要时加一组用任务编号代替 VTT 的对照。
10. **统计文件格式**：仍用主项目要求的顶层键 `robotwin2`（`VLAWrapper.load_norm_stats` 写死了这个键），`meta.dataset = "rmbench"` 注明来源。

## 模型与训练

11. **`feat_layers`**：`[-12, -8, -4]`（仓库默认，即第 13、17、21 个 block），与你在 RoboTwin 上的实验一致（已确认）。
    注意论文主设置是第 17、21 个 block（`[-8, -4]`）；报告中写明使用的是仓库默认的三层。
11b. **前瞻解码器的条件 token：`visual`**（已确认：只与 RoboTwin 上的 visual 版本对照）。
    这是你在主项目中加的选项：`VLAModel.future_cond_tokens`，由你修改的 `ModelFactory` 从
    `model.future_feat.cond_tokens` 读取（`"all"` = 原行为，`"visual"` = 只用视觉 token）。
    子项目在 `lila_api.py` 中再设置一次作为保护；配置检查只接受 `all`/`visual`（`null` 会让 `ModelFactory` 传入 None 而报错）。
    这两处主项目修改需要提交到 git，`run_info.json` 才能记录到准确的模型版本。
12. 训练按**优化器步数**与 RoboTwin visual 实验对齐：**85,928 步** = 8 epoch × 10,741 步/epoch（687,482 样本，
    `configs/mini9_visual.yaml`，`data_mode: both`）。在 RMBench（316,786 样本）上相当于约 17.4 个 epoch，
    学习率余弦曲线（2e-4 → 5e-5）的长度与 RoboTwin 相同。RoboTwin 实测约 0.79 秒/步，预计约 19 小时。
    **batch 64 × 梯度累积 1 = 等效 64**（已用 RoboTwin 配置核实），与你在 RoboTwin 上的 visual 实验一致
    （论文为 128）。实测 batch 64 显存峰值 8.3 GiB（+约 0.8 GiB 优化器状态），16 GB 显卡足够。训练为 bf16（与主项目一致）。
12b. **断点续训与后台运行，机制同你在 RoboTwin 上用的 `train_resumable.py`**：每 1000 步原子保存 `latest.pt`
    （权重、优化器、学习率调度、数据位置、随机数状态）；每个 epoch 用（种子 + epoch）固定数据顺序，续训时精确跳过已训练的 batch；
    同一命令再次运行即自动续训；`STOP` 文件、Ctrl+C、`kill` 都会先保存再退出；`scripts/train_bg.sh` 用 `nohup` 后台启动。
    相比原脚本额外保存随机数状态，并在续训时删除 `loss.csv` 中会被重新训练的行。
13. checkpoint 保持主项目格式（`model_state_dict` 等），主项目的 `robotwin_infer.py` 也能加载。
14. 推理：10 步 Euler、执行前 16 步、B 样条平滑（8 个控制点，与主项目相同），`noise_seed` 固定后评测可复现。

## 结果记录

R1. **baseline-seed0 训练（2026-10-26 19:48 → 10-27 14:14，约 18.4 小时，0.77 秒/步）**：85,928 步无中断、无 NaN；
    流匹配损失 0.17 → 0.0023，前瞻损失 0.31 → 0.091。
R2. **开环测试（20 条留出演示，每 16 步预测一次）**：手臂关节误差（执行部分）随训练单调下降，M(1) 平均
    0.040 → 0.012 rad、M(n) 0.035 → 0.0097 rad（10k → 85,928 步），没有过拟合迹象；夹爪误差 < 0.011。
    **闭环评测使用最终 checkpoint（step 85,928）**，也与 RoboTwin 使用最后一个 epoch 的做法一致。
    M(n) 的开环误差并不比 M(1) 大：开环时模型每步都拿到真实观测，不需要回忆过去，因此开环测试测不出记忆缺陷。
    个别时刻误差为中位数的 10–20 倍（place_block_mat t≈992、swap_blocks t≈336、cover_blocks t≈656），
    可能是"选择下一个目标"的决策点，闭环失败分析时值得对照。

R3. **闭环试跑（2026-10-27，最终 checkpoint，每个任务 2 次，种子 100000/100001，均失败）**：
    - 流程正确：`put_back_block` 第一阶段（不需要记忆：抓方块 → 放到中间 → 左臂按按钮）两次都完整完成；
      之后把方块放回到距原位置约 2 mm 处并松开夹爪。失败原因：按按钮太浅（演示中按到 z≈0.926，评测中只到
      0.948 / 0.933），按键很可能没被记录，任务停留在第 0 阶段。这是**操作精度**问题，不是记忆问题；
      B 样条平滑已排除（对演示轨迹分块平滑只损失 ≤0.3 mm）。两次方块原本都在右侧垫子，所以"放回原处"
      是记住了还是默认猜右边，需要更多评测才能区分。
    - `observe_and_pickup`：种子 100001 挡板落下后机器人 250 步一动不动（演示中挡板落下后先静止约 30 帧再开始抓取，
      无记忆时"该继续等还是该动手"的画面完全相同）；种子 100000 去抓了左侧的物体而不是目标（蓝色铃铛在右侧）。
      两者都符合"缺乏记忆"（时间记忆 / 目标记忆）。
    这说明失败原因必须逐条分类（第 11 步），不能把所有失败都算作记忆失败。

## 评测

15. RMBench 评测循环对每个环境步恰好调用一次 `update_obs`，所以适配器内的状态历史与训练时的 15 Hz 帧一一对应。
    ⚠ 这假设一次 `take_action` 对应数据中的一帧；第 7 步开环测试和评测视频可以帮助核实。
16. RMBench 只在 `_result.txt` 写总成功率；适配器通过 `on_trial_end` 把每条评测（任务、种子、是否成功、步数）写入 `episodes.jsonl`，用于置信区间和配对检验。

17b. **RMBench 的评测视频是第三人称相机**（`_base_task.py` 第 1539 行写入 `third_view_rgb`），不是策略看到的头部相机；
    训练数据的 `video/*.mp4` 则是头部相机，两者不能直接对比。适配器默认（`record_policy_view: true`）把策略实际收到的
    头部相机画面、状态和执行的动作逐条保存到 `runs/<run>/rmbench_eval/ckpt_<step>/policy_view/<任务>_seed<种子>.mp4/.npz`，
    用于失败原因分析。

## 尚未完成

17. ⚠ `rmbench_lila/tasks.py` 中各任务"需要记住什么"是根据任务说明写的初稿，需要读 `envs/<任务>.py` 核实。
18. 对照实验 B2/B3（图像历史）的模型改动尚未实现，见 `EXTENDING.md`。
