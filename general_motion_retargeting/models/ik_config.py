from typing import Annotated, Dict, List, Tuple

from pydantic import BaseModel, model_validator

Vec3 = Tuple[float, float, float]
Quat = Annotated[Tuple[float, float, float, float], "unit quaternion (w, x, y, z)"]
IKParam = Annotated[Tuple[float, float], "IKParam: (pos_weight, rot_weight)"]


class RetargetConfig(BaseModel):
    # ---------- Root & 基本假设 ----------
    robot_root_name: str
    human_root_name: str
    human_upper_root_name: str

    ground_height: float
    human_height_assumption: float

    # ---------- 整体缩放 ----------
    upper_body_scale: float
    lower_body_scale: float

    # ---------- 上下半身划分 ----------
    upper_body_name_list: List[str]
    lower_body_name_list: List[str]

    # ---------- 人体关节缩放表 ----------
    human_scale_table: Dict[str, float]  # 所有需要缩放的关节，上下半身不需要微调时赋1.0

    # ---------- IK 匹配 ----------
    ik_match_table: Dict[str, str]  # 机器人关节名 -> 人体关节名
    robot_joint_offset: Dict[str, Tuple[Vec3, Quat]]  # 机器人关节名 -> (位置偏移, 旋转偏移)

    # ---------- IK 开关 ----------
    use_ik_match_table1: bool
    use_ik_match_table2: bool

    # ---------- IK 参数表 ----------
    ik_param1: Dict[str, IKParam]
    ik_param2: Dict[str, IKParam]

    @model_validator(mode="after")
    def check_cross_fields(self):
        ik_match_keys = set(self.ik_match_table.keys())
        if self.use_ik_match_table1:
            ik_param1_keys = set(self.ik_param1.keys())
            missing_in_match = ik_param1_keys - ik_match_keys
            if missing_in_match:
                raise ValueError(
                    f"ik_param1 contains keys not present in ik_match_table: " f"{sorted(missing_in_match)}"
                )

        if self.use_ik_match_table2:
            ik_param2_keys = set(self.ik_param2.keys())
            missing_in_match = ik_param2_keys - ik_match_keys
            if missing_in_match:
                raise ValueError(
                    f"ik_param2 contains keys not present in ik_match_table: " f"{sorted(missing_in_match)}"
                )

        robot_joint_offset_keys = set(self.robot_joint_offset.keys())
        missing_in_match = robot_joint_offset_keys - ik_match_keys
        if missing_in_match:
            raise ValueError(
                f"robot_joint_offset contains keys not present in ik_match_table: " f"{sorted(missing_in_match)}"
            )

        human_scale_keys = set(self.human_scale_table.keys())
        ik_match_values = set(self.ik_match_table.values())
        missing_in_scale = ik_match_values - human_scale_keys
        if missing_in_scale:
            raise ValueError(
                f"ik_match_table refers to unknown human joints "
                f"(not in human_scale_table): {sorted(missing_in_scale)}"
            )

        if self.human_upper_root_name not in self.human_scale_table:
            raise ValueError(
                f"human_upper_root_name '{self.human_upper_root_name}' " f"not found in human_scale_table keys"
            )

        if self.human_root_name not in self.human_scale_table:
            raise ValueError(f"human_root_name '{self.human_root_name}' " f"not found in human_scale_table keys")

        if set(self.upper_body_name_list).intersection(set(self.lower_body_name_list)):
            raise ValueError("upper_body_name_list and lower_body_name_list should not have overlapping joint names")

        return self


if __name__ == "__main__":
    import json
    from pathlib import Path

    with open(Path(__file__).parents[1] / "ik_configs/ik_config.schema.json", "w", encoding="utf-8") as f:
        json.dump(RetargetConfig.model_json_schema(), f, indent=2, ensure_ascii=False)
