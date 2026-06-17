#!/usr/bin/env python3
"""Capture YOLO dataset images from a RealSense camera without ROS.

The script mirrors the current ROS path:
RealSense RGB8 -> BGR OpenCV image -> ROI -> img_NNNN.jpg.
"""

from __future__ import annotations

import argparse
import ast
import glob
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_LAUNCH_PATH = (
    WORKSPACE_ROOT
    / "src"
    / "realsense-ros"
    / "realsense2_camera"
    / "launch"
    / "rs_launch.py"
)
DEFAULT_CONFIG_PATH = (
    WORKSPACE_ROOT
    / "src"
    / "my_cv"
    / "config"
    / "realsense_dataset_params.yaml"
)

ROI_X = 0
ROI_Y = 0
ROI_W = 640
ROI_H = 480

DEFAULT_PARAMS: dict[str, Any] = {
    "enable_color": True,
    "rgb_camera.color_profile": "640,480,15",
    "rgb_camera.color_format": "RGB8",
    "rgb_camera.enable_auto_exposure": False,
    "rgb_camera.exposure": 145,
    "rgb_camera.enable_auto_white_balance": False,
    "rgb_camera.white_balance": 4200.0,
    "rgb_camera.saturation": 55,
    "rgb_camera.contrast": 55,
    "rgb_camera.gamma": 300,
    "rgb_camera.sharpness": 30,
    "rgb_camera.gain": 64,
}

COLOR_PARAM_NAMES = {
    "enable_color",
    "rgb_camera.color_profile",
    "rgb_camera.color_format",
    "rgb_camera.enable_auto_exposure",
    "rgb_camera.exposure",
    "rgb_camera.enable_auto_white_balance",
    "rgb_camera.white_balance",
    "rgb_camera.saturation",
    "rgb_camera.contrast",
    "rgb_camera.gamma",
    "rgb_camera.sharpness",
    "rgb_camera.gain",
}


def desktop_dataset_dir() -> Path:
    return Path.home() / "Desktop" / "dataset" / "images" / "train"


def normalize_ros_value(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if value == "''":
            return ""
        try:
            if "." in lowered:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def load_yaml_params(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a flat parameter map: {path}")
    return {str(k): normalize_ros_value(v) for k, v in data.items()}


def literal_eval_node(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def load_launch_params(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    params: dict[str, Any] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "configurable_parameters" for t in node.targets):
            continue
        items = literal_eval_node(node.value)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str) and name in COLOR_PARAM_NAMES and "default" in item:
                params[name] = normalize_ros_value(item["default"])

    # Match rs_launch.py behavior: later parameter dictionaries override earlier ones.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        override = literal_eval_node(node)
        if not isinstance(override, dict):
            continue
        if any(isinstance(k, str) and k in COLOR_PARAM_NAMES for k in override):
            for key, value in override.items():
                if isinstance(key, str) and key in COLOR_PARAM_NAMES:
                    params[key] = normalize_ros_value(value)

    return params


def parse_profile(value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    return fallback


def bool_param(params: dict[str, Any], name: str, default: bool = False) -> bool:
    value = params.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def float_param(params: dict[str, Any], name: str, default: float) -> float:
    try:
        return float(params.get(name, default))
    except (TypeError, ValueError):
        return default


def find_sensor(device: rs.device, stream: rs.stream) -> rs.sensor | None:
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == stream:
                return sensor
    return None


def set_option_if_supported(sensor: rs.sensor | None, option: rs.option, value: float) -> None:
    if sensor is not None and sensor.supports(option):
        sensor.set_option(option, value)


def apply_sensor_options(profile: rs.pipeline_profile, params: dict[str, Any]) -> None:
    device = profile.get_device()
    color_sensor = find_sensor(device, rs.stream.color)

    auto_exposure = 1.0 if bool_param(params, "rgb_camera.enable_auto_exposure") else 0.0
    set_option_if_supported(color_sensor, rs.option.enable_auto_exposure, auto_exposure)
    if not auto_exposure:
        set_option_if_supported(color_sensor, rs.option.exposure, float_param(params, "rgb_camera.exposure", 145.0))

    auto_wb = 1.0 if bool_param(params, "rgb_camera.enable_auto_white_balance") else 0.0
    set_option_if_supported(color_sensor, rs.option.enable_auto_white_balance, auto_wb)
    if not auto_wb:
        set_option_if_supported(color_sensor, rs.option.white_balance, float_param(params, "rgb_camera.white_balance", 4200.0))

    set_option_if_supported(color_sensor, rs.option.saturation, float_param(params, "rgb_camera.saturation", 55.0))
    set_option_if_supported(color_sensor, rs.option.contrast, float_param(params, "rgb_camera.contrast", 55.0))
    set_option_if_supported(color_sensor, rs.option.gamma, float_param(params, "rgb_camera.gamma", 300.0))
    set_option_if_supported(color_sensor, rs.option.sharpness, float_param(params, "rgb_camera.sharpness", 30.0))
    set_option_if_supported(color_sensor, rs.option.gain, float_param(params, "rgb_camera.gain", 64.0))


def next_image_index(save_dir: Path) -> int:
    existing = sorted(glob.glob(str(save_dir / "img_*.jpg")))
    if not existing:
        return 0
    last_name = Path(existing[-1]).name
    return int(last_name.split("_")[1].split(".")[0]) + 1


def build_pipeline(params: dict[str, Any]) -> tuple[rs.pipeline, rs.pipeline_profile]:
    pipeline = rs.pipeline()
    config = rs.config()

    width, height, fps = parse_profile(params.get("rgb_camera.color_profile"), (640, 480, 15))
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    profile = pipeline.start(config)
    apply_sensor_options(profile, params)
    return pipeline, profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RealSense YOLO images on Windows without ROS.")
    parser.add_argument("--launch", type=Path, default=DEFAULT_LAUNCH_PATH, help="rs_launch.py to mirror.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Flat YAML params overriding launch defaults. Example: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--save-dir", type=Path, default=desktop_dataset_dir(), help="Output image directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = dict(DEFAULT_PARAMS)
    params.update(load_launch_params(args.launch))
    params.update(load_yaml_params(args.config))

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    image_count = next_image_index(save_dir)

    pipeline, _profile = build_pipeline(params)
    print(f"Save directory: {save_dir}")
    print("Press SPACE to save image, ESC to quit.")
    cv2.namedWindow("ROI Viewer", cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            roi = bgr[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

            cv2.imshow("ROI Viewer", roi)
            key = cv2.waitKey(10)
            if key == 27:
                break
            if key == 32:
                filename = save_dir / f"img_{image_count:04}.jpg"
                cv2.imwrite(str(filename), roi)
                print(f"Saved: {filename}")
                image_count += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
