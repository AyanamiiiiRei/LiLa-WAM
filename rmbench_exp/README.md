# rmbench_exp：用 RMBench 检验 LiLa-WAM 在记忆任务上的局限

这是 LiLa-WAM 仓库中的一个**独立子项目**，实现 [`docs/PLAN.md`](docs/PLAN.md) 中的实验计划。
逐步操作见 [`docs/STEPS.md`](docs/STEPS.md)，所有设计决定和代码检查中发现的问题记录在
[`docs/DECISIONS.md`](docs/DECISIONS.md)，如何添加新功能见 [`docs/EXTENDING.md`](docs/EXTENDING.md)。

## 独立性约定

1. **不修改主项目的任何文件。** 主项目（`models/`、`dataloader/`、`train.py` 等）只被**导入**，不被编辑。
   `git diff main -- . ':!rmbench_exp'` 应当始终为空。
2. **只有一个文件接触主项目：** [`rmbench_lila/lila_api.py`](rmbench_lila/lila_api.py)。
   它列出了用到的全部主项目接口（`ModelFactory`、`VLAWrapper`、`VLAModel.forward`）。主项目接口变了，只改这一个文件。
3. **对 RMBench 也只增加一个符号链接**（评测适配器，第 9 步），删掉链接即完全卸载。
4. **每次训练的运行目录自成一体**（配置、归一化统计量、VTT、训练集清单、git commit、损失曲线、checkpoint），
   评测只需要这个目录。

## 目录结构

```
rmbench_exp/
├── configs/
│   ├── base.yaml              完整配置（字段与主项目配置兼容 + rmbench/experiment/paths 三个新段）
│   ├── exp/*.yaml             实验配置，只写与 base 不同的部分（_base_ 继承）
│   └── control_a/             对照实验 A：给主项目原样使用的配置
├── rmbench_lila/              Python 包（分层见 __init__.py）
│   ├── schema.py              RMBench 字段 ↔ 向量的唯一定义（HDF5 键、评测观测键、维度）
│   ├── codec.py               图像解码（decode_image_bit，不做任何颜色转换）
│   ├── tasks.py               RMBench 任务表：M1/Mn、步数上限、需要记住什么
│   ├── config.py              配置继承、命令行覆盖、一致性检查
│   ├── episodes.py            扫描 / 划分 / 读取演示（不依赖 torch）
│   ├── data/                  Dataset：由注册的 feature 拼装样本
│   ├── lila_api.py            ★ 唯一导入主项目的模块
│   ├── models.py              模型变体注册表（默认 lila_wam = 主项目原模型）
│   ├── stats.py / vtt.py      归一化统计量 / VTT
│   ├── trainer.py             按步数训练、固定随机种子、自成一体的运行目录
│   ├── policy.py              推理（开环评测与 RMBench 适配器共用）
│   └── evalstats.py           Wilson 置信区间、McNemar 配对检验
├── scripts/stepXX_*.py        与计划步骤一一对应的脚本
├── xpolicylab_adapter/LiLa_WAM/   RMBench(XPolicyLab) 评测适配器
├── analysis/                  结果表、配对比较、失败原因标注
└── tests/                     用合成数据测试数据通路（无需真实数据和 GPU）
```

## 快速开始

```bash
cd LiLa-WAM/rmbench_exp
export RMBENCH_DATA=/path/to/RMBench/data/demo_clean
python scripts/step02_inspect_data.py --data_root $RMBENCH_DATA     # 先检查数据
python scripts/step04_compute_stats.py                              # 统计量（不删除任何文件）
python scripts/step05_compute_vtt.py                                # VTT
python scripts/step07_sanity_checks.py color                        # 训练前检查（共 4 项）
python scripts/step08_train.py --config configs/exp/baseline.yaml   # 训练
```

依赖：LiLa-WAM 的训练环境，另加 `h5py omegaconf scipy opencv-python tqdm matplotlib`
（评测时策略环境还需要 XPolicyLab，见第 9 步）。测试：`python -m pytest tests -q`。
