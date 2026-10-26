"""RMBench task registry: type (M1 / Mn), eval step limit, and what must be remembered.

Source: rmbench@7d1cdfc README (M(n) = battery_try, blocks_ranking_try, cover_blocks,
press_button; all others M(1)), env_cfg/task_config/_eval_step_limit.yml, and
description/task_instruction/<task>.json.

The ``memory`` notes are a FIRST DRAFT from the instructions only. Step 02 asks you to
confirm each one by reading envs/<task>.py; edit them here (one place) when you do.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskInfo:
    name: str
    kind: str              # "M1" or "Mn"
    step_limit: int | None
    memory: str            # what information from the past the policy needs
    verified: bool = False  # set True after reading envs/<task>.py


TASKS: dict[str, TaskInfo] = {t.name: t for t in [
    TaskInfo("battery_try", "Mn", 1000,
             "Which battery orientation combinations were already tried (needle feedback) "
             "so as not to repeat a failed one."),
    TaskInfo("blocks_ranking_try", "Mn", 3500,
             "Which block arrangements were already tried before pressing the button again."),
    TaskInfo("cover_blocks", "Mn", 1500,
             "Which colour is under which lid once all blocks are covered (needed to uncover "
             "in red-green-blue order)."),
    TaskInfo("press_button", "Mn", 1500,
             "The two numbers seen at the start, and how many times each button has been "
             "pressed so far."),
    TaskInfo("observe_and_pickup", "M1", 250,
             "Identity of the target object on the shelf before the screen hides it."),
    TaskInfo("place_block_mat", "M1", 1500,
             "Original positions of the blocks on the blue mat, to put them back."),
    TaskInfo("put_back_block", "M1", 500,
             "Which of the four mats the block started on."),
    TaskInfo("rearrange_blocks", "M1", 700,
             "Which block started on a mat vs between the mats (phase after the button press)."),
    TaskInfo("swap_T", "M1", 600,
             "Initial poses of both T-blocks (one is overwritten while swapping)."),
    TaskInfo("swap_blocks", "M1", 1000,
             "Original tray of each block during the 3-move swap, and whether the swap is done."),
]}

M1_TASKS = sorted(n for n, t in TASKS.items() if t.kind == "M1")
MN_TASKS = sorted(n for n, t in TASKS.items() if t.kind == "Mn")
ALL_TASKS = sorted(TASKS)


def task_kind(name: str) -> str:
    return TASKS[name].kind if name in TASKS else "unknown"
