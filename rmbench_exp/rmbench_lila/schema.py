"""Single source of truth for how RMBench data maps onto LiLa-WAM vectors.

Everything that knows an HDF5 key or an eval-observation key lives here, so the
dataset, the stats script, the VTT script and the eval adapter cannot disagree.

RMBench HDF5 layout (written by rmbench/envs/utils/pkl2hdf5.py):

    state/<field>    = recorded[t]     for t = 0 .. N-2
    action/<field>   = recorded[t+1]   for t = 0 .. N-2      (action[t] == state[t+1])
    vision/cam_head/colors             JPEG bits, one per frame (N-1 frames)
    additional_info/frequency          collection frequency (expected 15)

Fields: left/right_arm_joint_states (6), left/right_ee_joint_states (1, gripper),
left/right_ee_poses (7). Arm dims are verified against real files by step 02.

To add a new layout (e.g. joint-space proprioception), register another
``VectorLayout`` in ``STATE_LAYOUTS`` / ``ACTION_LAYOUTS`` and select it in the
config — no other file has to change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    h5_key: str      # dataset name inside the HDF5 group
    obs_key: str     # key inside obs["state"] at eval time (XPolicyLab format)
    dim: int


@dataclass(frozen=True)
class VectorLayout:
    name: str
    fields: tuple[Field, ...]
    gripper_dims: tuple[int, ...]   # indices (in the concatenated vector) that are grippers
    angle_dims: tuple[int, ...]     # indices that are joint angles (radians)

    @property
    def dim(self) -> int:
        return sum(f.dim for f in self.fields)

    def slices(self) -> list[tuple[Field, slice]]:
        out, off = [], 0
        for f in self.fields:
            out.append((f, slice(off, off + f.dim)))
            off += f.dim
        return out


def _dims(start: int, n: int) -> tuple[int, ...]:
    return tuple(range(start, start + n))


# ---------------------------------------------------------------- state layouts
# "ee16": same semantics as LiLa-WAM on RoboTwin (endpose 7 + gripper 1, per arm).
STATE_EE16 = VectorLayout(
    name="ee16",
    fields=(
        Field("left_ee_poses", "left_ee_pose", 7),
        Field("left_ee_joint_states", "left_ee_joint_state", 1),
        Field("right_ee_poses", "right_ee_pose", 7),
        Field("right_ee_joint_states", "right_ee_joint_state", 1),
    ),
    gripper_dims=(7, 15),
    angle_dims=(),
)

# "joint14": alternative proprioception in joint space (same layout as the action).
STATE_JOINT14 = VectorLayout(
    name="joint14",
    fields=(
        Field("left_arm_joint_states", "left_arm_joint_state", 6),
        Field("left_ee_joint_states", "left_ee_joint_state", 1),
        Field("right_arm_joint_states", "right_arm_joint_state", 6),
        Field("right_ee_joint_states", "right_ee_joint_state", 1),
    ),
    gripper_dims=(6, 13),
    angle_dims=_dims(0, 6) + _dims(7, 6),
)

# --------------------------------------------------------------- action layouts
# Absolute joint targets, XPolicyLab packing order [LA(6), LG, RA(6), RG].
ACTION_JOINT14 = VectorLayout(
    name="joint14",
    fields=STATE_JOINT14.fields,
    gripper_dims=(6, 13),
    angle_dims=_dims(0, 6) + _dims(7, 6),
)

STATE_LAYOUTS = {l.name: l for l in (STATE_EE16, STATE_JOINT14)}
ACTION_LAYOUTS = {l.name: l for l in (ACTION_JOINT14,)}

# ------------------------------------------------------------------- HDF5 groups
STATE_GROUP = "state"
ACTION_GROUP = "action"
VISION_GROUP = "vision"
IMAGE_KEY = "colors"
FREQUENCY_KEY = "additional_info/frequency"

# Container key the main project's VLAWrapper.load_norm_stats reads from the stats JSON.
NORM_STATS_CONTAINER_KEY = "robotwin2"

# LiLa-WAM camera name -> RMBench / XPolicyLab camera name
CAMERAS = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
    "front_camera": "cam_third_view",
}

# Where the "action" chunk is read from:
#   next    : action/<field>[t + k]   -> chunk[0] is the NEXT target (RMBench native, default)
#   current : state/<field>[t + k]    -> chunk[0] is the CURRENT pose (LiLa-WAM/RoboTwin habit)
ACTION_ALIGNMENTS = {"next": ACTION_GROUP, "current": STATE_GROUP}


def get_state_layout(name: str) -> VectorLayout:
    if name not in STATE_LAYOUTS:
        raise KeyError(f"Unknown state layout {name!r}; known: {sorted(STATE_LAYOUTS)}")
    return STATE_LAYOUTS[name]


def get_action_layout(name: str) -> VectorLayout:
    if name not in ACTION_LAYOUTS:
        raise KeyError(f"Unknown action layout {name!r}; known: {sorted(ACTION_LAYOUTS)}")
    return ACTION_LAYOUTS[name]


def camera_h5_name(camera: str) -> str:
    """Accept either LiLa-WAM ('head_camera') or RMBench ('cam_head') camera names."""
    return CAMERAS.get(camera, camera)
