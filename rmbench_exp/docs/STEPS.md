# 逐步操作手册

步骤编号与 [`PLAN.md`](PLAN.md) 一致。所有命令都在 `LiLa-WAM/rmbench_exp/` 下运行；
`--config` 默认是 `configs/exp/baseline.yaml`，任何配置项都可以在命令末尾用 `键=值` 覆盖。

## 进度

- [ ] 1 准备环境
- [ ] 2 检查实际数据，核对 `rmbench_lila/tasks.py` 中的记忆说明
- [ ] 3 （已由代码完成）数据加载：`schema.py` + `episodes.py` + `data/`
- [ ] 4 归一化统计量
- [ ] 5 VTT 及任务相似度
- [ ] 6 确认训练配置（尤其 `feat_layers`、batch size）
- [ ] 7 训练前检查：color / actions / shapes / overfit + 开环测试
- [ ] 8 训练：baseline，可选 no_foresight
- [ ] 9 安装适配器并在 RMBench 中评测
- [ ] 10 对照实验 A / B / C
- [ ] 11 结果分析：置信区间、失败原因、M1 vs Mn

---

## 1. 准备环境（两个独立的 conda 环境）

⚠ RMBench 的 `scripts/_install.sh` **不会创建环境**，它把包装进当前激活的环境，并且固定了 `torch==2.4.1`、
`huggingface_hub==0.25.0` 等版本。必须先创建并激活 `RMBench` 环境再运行它，否则会破坏 LiLaWAM 环境。

```bash
# A. LiLaWAM：训练与策略服务器（按 LiLa-WAM README）
conda create -n LiLaWAM python=3.10 -y && conda activate LiLaWAM
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.0.0rc0 omegaconf accelerate h5py
pip install scipy opencv-python tqdm matplotlib pytest          # rmbench_exp 额外需要
bash xpolicylab_adapter/LiLa_WAM/install.sh                     # 第 9 步评测时需要（XPolicyLab，不改动已有包）

# B. RMBench：仿真评测客户端（按 RMBench README）
conda create -n RMBench python=3.10 -y && conda activate RMBench
cd ~/lilawam/RMBench && git submodule update --init --recursive XPolicyLab
# CuRobo 需要 CUDA 编译器；系统没有时装进环境（与 RoboTwin 环境相同的组合；系统 gcc 过新时必须用 conda 的 gcc 11）
conda install -y -c conda-forge cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105 gcc_linux-64=11.4.0 gxx_linux-64=11.4.0
conda deactivate && conda activate RMBench && export CUDA_HOME=$CONDA_PREFIX
bash scripts/_install.sh && bash scripts/_download_assets.sh && bash scripts/_download_data.sh
# _install.sh 之后修正版本冲突（RMBench 固定 scipy 1.10.1，需要 numpy 1.x）
pip install "numpy==1.26.4" "opencv-python-headless==4.10.0.84" && pip check
python -c "import sapien, mplib, curobo; print('RMBench OK')"
export RMBENCH_ROOT=$PWD RMBENCH_DATA=$PWD/data/demo_clean
```
DINOv3 权重放在 `LiLa-WAM/dinov3_pretrain/dinov3-vitl16-pretrain-lvd1689m`（与主项目相同）。
自检：`conda activate LiLaWAM && cd LiLa-WAM/rmbench_exp && python -m pytest tests -q`。

## 2. 检查实际数据

```bash
python scripts/step02_inspect_data.py --data_root $RMBENCH_DATA --per_task 2
```
阅读 `artifacts/inspect/report.md`，确认：
- 每条抽检演示的 `problems: none`；
- `action[t]==state[t+1]` 的最大差值约为 0；`frequency` 为 15；
- `images/*_hdf5_vs_video.png` 左右两半颜色一致（报告中 `|hdf5 - video|` 应明显小于 `|swapped - video|`）；
- 长度分布表：用它估算每个 epoch 的样本数（第 6 步）。

然后逐个阅读 `$RMBENCH_ROOT/envs/<任务>.py`，修改 `rmbench_lila/tasks.py` 中的 `memory` 说明并设 `verified=True`。

## 4. 归一化统计量

```bash
python scripts/step04_compute_stats.py           # 输出 artifacts/stats/stats-rmbench.json
```
此脚本**只读**。被标记（flagged）的演示会列出来但默认仍参与统计；看过之后若确认是坏数据，再加 `--exclude_flagged`。
默认只用训练集（不含每个任务最后 2 条留出演示）。

## 5. VTT

```bash
python scripts/step05_compute_vtt.py             # 输出 artifacts/vtt/rmbench/
```
查看 `meta.json` 中的 `most_similar_pairs` 和 `cosine_similarity.csv`：相似度很高的任务对值得在报告中提到。

## 6. 训练配置

在 `configs/base.yaml` 中确认：
- `model.vision_encoder.feat_layers`：**必须与你在 RoboTwin 上的实验一致**（默认按论文写成 `[-8, -4]`）；
- `training.batch_size × grad_accum_steps`：等效 batch（论文 128）。16 GB 显存先试 `batch_size=32`，用 shapes 检查看显存；
- `training.total_steps`：按步数而非 epoch。启动时日志会打印对应的 epoch 数。

## 7. 训练前检查

```bash
python scripts/step07_sanity_checks.py color      # artifacts/sanity/color：左=dataloader，右=mp4
python scripts/step07_sanity_checks.py actions    # artifacts/sanity/actions：动作/状态曲线与跳变
python scripts/step07_sanity_checks.py shapes     # 打印 token 序列，应为 (B, 100, 768)
python scripts/step07_sanity_checks.py overfit --config configs/exp/debug_overfit.yaml
```
开环测试在第 8 步得到 checkpoint 后运行：
```bash
python scripts/step07_open_loop_eval.py --run runs/baseline-seed0
```

## 8. 训练（后台运行、可断点续训）

```bash
conda activate LiLaWAM
cd ~/lilawam/LiLa-WAM/rmbench_exp
export RMBENCH_DATA=~/lilawam/RMBench/data/demo_clean
bash scripts/train_bg.sh configs/exp/baseline.yaml          # 后台启动；关掉终端也不会停
tail -f runs/baseline-seed0/train.log                       # 查看进度（Ctrl+C 只退出查看，不影响训练）
```
- **暂停**：`touch runs/baseline-seed0/STOP`，训练在当前一步结束后保存并退出。
- **继续**：再次运行同一条 `train_bg.sh` 命令，自动从 `runs/baseline-seed0/latest.pt` 继续（使用运行目录中保存的配置）。
- 每 1000 步覆盖保存一次 `latest.pt`（断点，约 1.3 GB）；每 10000 步另存只含权重的 `checkpoints/step_*.pt`（评测用，约 0.4 GB）。
- 断电或死机：最多重复训练最近不到 1000 步；同一 epoch 内已训练的 batch 会被精确跳过（固定的每 epoch 数据顺序）。
- 换随机种子：`bash scripts/train_bg.sh configs/exp/baseline.yaml "" experiment.seed=1`（运行目录为 `runs/baseline-seed1`）。
- 前台运行也可以：`python scripts/step08_train.py --config configs/exp/baseline.yaml`，Ctrl+C 会保存断点后退出。

运行目录 `runs/<实验名>-seed<种子>/`：`config.yaml`、`norm_stats.json`、`vtt/`、`train_episodes.json`、
`run_info.json`（两边的 git commit、命令行、每次续训的记录）、`loss.csv`、`train.log`、`latest.pt`、`checkpoints/`。

## 9. RMBench 评测

```bash
# 策略环境中安装 XPolicyLab（一次）
pip install -e $RMBENCH_ROOT/XPolicyLab
# 把适配器链接到 RMBench（只创建一个符号链接）
RMBENCH_ROOT=$RMBENCH_ROOT bash scripts/step09_install_adapter.sh
# 在 RMBench 目录下评测一个任务
cd $RMBENCH_ROOT
TEST_NUM=100 bash XPolicyLab/policy/LiLa_WAM/eval.sh RMBench cover_blocks baseline-seed0 \
    aloha_agilex joint 0 0 0 <策略conda环境> RMBench
```
- 第 3 个参数是运行目录（相对 `rmbench_exp/runs/`）；`checkpoint`、`noise_seed` 在 `deploy.yml` 中设置。
- 机器人必须是 `aloha_agilex`，动作类型必须是 `joint`，适配器会检查。
- 每条评测记录写入 `runs/<run>/rmbench_eval/ckpt_<步数>/episodes.jsonl`；视频在 RMBench 的 `eval_result/` 下。
- `TEST_NUM` 默认 100（RMBench 评测脚本的默认值）；请与 RMBench 论文的协议核对。
- 不同模型使用**相同的种子参数**（第 6 个参数），RMBench 会生成相同的测试场景序列，从而可以做配对比较。

## 10. 对照实验

**A：数据量和环境**（使用主项目原样代码）
```bash
python scripts/step10a_make_robotwin_subset.py --src <RoboTwin 数据根目录> --dst <输出目录> --n 50
# 修改 configs/control_a/robotwin_50clean.yaml 中的 dataset_dir，然后在 LiLa-WAM 根目录：
python utils/precompute_task_cond.py --config rmbench_exp/configs/control_a/robotwin_50clean.yaml --output_dir ./data-control_a-taskcond
python train.py --config rmbench_exp/configs/control_a/robotwin_50clean.yaml --norm_stats_path ./utils/stat-500-all.json
```
训练步数尽量与 RMBench 主实验对齐（主项目按 epoch 计；日志里有每个 epoch 的步数）。

**B：加入历史信息**
- B1 状态历史：`configs/exp/state_hist8.yaml`（连续 8 步）和 `state_hist_sparse.yaml`（同样 8 个 token，回看约 16 秒）。
- B2/B3 图像历史：数据侧已有 `first_image` feature；模型侧需要新增一个模型变体，做法见 [`EXTENDING.md`](EXTENDING.md)。

**C：与已有方法比较**：把第 11 步的表与 RMBench 论文 / Hugging Face 评测日志放在一起，注明评测次数、机器人、训练数据是否一致。

## 11. 结果分析

```bash
python analysis/results.py table   runs/baseline-seed0                         # 每任务成功率 + 95% CI，M1/Mn 汇总
python analysis/results.py compare runs/baseline-seed0 runs/state_hist8-seed0   # 同种子配对比较（McNemar）
python analysis/failure_labels.py init      runs/baseline-seed0/rmbench_eval/ckpt_<步数>
# 看视频，在 failure_labels.csv 中填 label：memory / manipulation / stuck / other
python analysis/failure_labels.py summarize runs/baseline-seed0/rmbench_eval/ckpt_<步数>
```
