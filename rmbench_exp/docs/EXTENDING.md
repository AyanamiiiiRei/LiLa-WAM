# 如何扩展

原则：**新功能 = 新文件 + 新配置**，尽量不改已有代码；任何情况下都不改主项目。

| 想做的事 | 改哪里 |
|---|---|
| 新实验（改超参数、任务子集、种子） | 只新建 `configs/exp/<名字>.yaml`，写 `_base_: ../base.yaml` 和不同的项 |
| 新的状态 / 动作表示 | `schema.py` 中注册新的 `VectorLayout`，配置里 `rmbench.state_layout: <名字>` |
| 新的输入（如首帧图像） | `data/features.py`（或新模块）中写一个类并 `@FEATURES.register("名字")`，加入 `dataset.features` |
| 新的模型变体 | 新建模块，`@MODELS.register("名字")`，配置里 `model.variant: <名字>` |
| 训练中的附加操作（如定期开环评测） | 写回调函数 `cb(trainer, metrics)`，加入 `Trainer.callbacks` |
| 主项目接口变化 | 只改 `lila_api.py` |
| 新的分析 | `analysis/` 下新脚本，读 `episodes.jsonl` |

## 示例：对照实验 B2（加入首帧图像）

1. **数据**：已经有 `first_image` feature，在配置中加入即可：
   ```yaml
   dataset:
     features: [state, action_chunk, image, future_image, task_cond, first_image]
   ```
2. **模型**：新建 `rmbench_lila/variants/first_frame.py`：
   - 定义 `FirstFrameVLAModel(VLAModel)`（从 `lila_api` 导入主项目的 `VLAModel`，**继承**而不是修改），
     在 `forward` 中多接收一组视觉 token（首帧经过同一个 adapter，加一个新的 type embedding），拼到序列里；
   - 定义一个 `VLAWrapper` 的子类，`forward(batch)` 中额外取 `batch["first_pixel_values"]` 的 DINOv3 特征；
   - `@MODELS.register("lila_wam_first_frame")` 注册构建函数。
   需要导入的主项目类先在 `lila_api.py` 中导出（这是唯一允许导入主项目的地方）。
3. **推理**：`policy.py` 已保存 `first_rgb`；给新变体的 `predict_chunk` 传入首帧（在 `policy.py` 中把条件构造提取为按变体分派的函数）。
4. **配置**：`configs/exp/first_frame.yaml` 设 `model.variant: lila_wam_first_frame`。
5. **测试**：在 `tests/` 中为新 feature 和新变体加测试。

## 约定

- 包内分层：`schema/codec/tasks/config` → `episodes` → `data` → `lila_api/models` → `stats/vtt/trainer/policy`，低层不导入高层。
- 只读原则：任何脚本都不修改、删除数据文件；输出都写到 `artifacts/` 或 `runs/`（二者已在 `.gitignore` 中）。
- 每次运行需要的信息都写进运行目录，保证事后能复现。
