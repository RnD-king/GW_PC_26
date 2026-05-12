from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import rclpy
import torch
import torch.nn as nn
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def _runtime_config_path() -> Path:
    env_path = os.environ.get("YOLO26_RUNTIME_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    return Path(get_package_share_directory("vision")) / "config" / "yolo26_runtime.yaml"


def _runtime_params() -> Dict:
    with _runtime_config_path().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("yolo26_runtime", {}).get("ros__parameters", {})


def _is_jetson_target() -> bool:
    return platform.machine() in ("aarch64", "arm64")


def _runtime_path_from_keys(
    env_name: str,
    pc_keys: Sequence[str],
    jetson_keys: Sequence[str],
) -> str:
    env_path = os.environ.get(env_name)
    if env_path:
        return env_path
    params = _runtime_params()
    if _is_jetson_target():
        keys = tuple(jetson_keys) + tuple(pc_keys)
    else:
        keys = tuple(pc_keys) + tuple(jetson_keys)
    for key in keys:
        value = params.get(key)
        if value and Path(value).expanduser().is_file():
            return str(Path(value).expanduser())
    for key in keys:
        value = params.get(key)
        if value:
            return str(Path(value).expanduser())
    raise FileNotFoundError(f"Missing one of {keys} in {_runtime_config_path()}")


def _runtime_path(env_name: str, pc_key: str, jetson_key: str) -> str:
    return _runtime_path_from_keys(env_name, (pc_key,), (jetson_key,))


DEFAULT_PPO_POLICY_CKPT = _runtime_path_from_keys(
    "G1_PPO_POLICY_CKPT",
    ("pc_ppo_policy_checkpoint", "pc_policy_checkpoint"),
    ("jetson_ppo_policy_checkpoint", "jetson_policy_checkpoint"),
)
DEFAULT_BC_POLICY_CKPT = _runtime_path_from_keys(
    "G1_BC_POLICY_CKPT",
    ("pc_bc_checkpoint", "pc_bc_policy_checkpoint", "pc_obs_norm_ckpt"),
    ("jetson_bc_checkpoint", "jetson_bc_policy_checkpoint", "jetson_obs_norm_ckpt"),
)
DEFAULT_OBS_NORM_CKPT = _runtime_path_from_keys(
    "G1_OBS_NORM_CKPT",
    ("pc_obs_norm_ckpt", "pc_bc_checkpoint", "pc_bc_policy_checkpoint"),
    ("jetson_obs_norm_ckpt", "jetson_bc_checkpoint", "jetson_bc_policy_checkpoint"),
)


def _default_policy_checkpoint(policy_mode: str) -> str:
    generic_env = os.environ.get("G1_POLICY_CKPT")
    if generic_env:
        return generic_env
    if policy_mode == "bc":
        return DEFAULT_BC_POLICY_CKPT
    if policy_mode == "ppo":
        return DEFAULT_PPO_POLICY_CKPT
    raise ValueError("policy_mode must be one of: ppo, bc")

BASE_FEATURE_NAMES = [
    "u_err_near",
    "u_err_lookahead",
    "u_err_ctrl",
    "slope",
    "n_visible",
    "in_recovery",
    "vx_prev",
    "wz_prev",
]


class HighLevelMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int = 2):
        super().__init__()
        dims = [int(input_dim)] + [int(v) for v in hidden_dims]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _torch_load(path: str) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_dims(spec: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in str(spec).split(",")]
    dims = tuple(int(p) for p in parts if p)
    return dims if dims else (128, 128)


def _build_feature_names(history_state: int) -> List[str]:
    if history_state <= 1:
        return list(BASE_FEATURE_NAMES)
    out: List[str] = []
    for k in range(history_state - 1, -1, -1):
        out.extend([f"{name}_t-{k}" for name in BASE_FEATURE_NAMES])
    return out


def _copy_actor_weights_from_ppo(model: HighLevelMLP, state: Dict[str, torch.Tensor]) -> None:
    mapped = {}
    for key, value in state.items():
        if key.startswith("actor."):
            mapped["net." + key[len("actor."):]] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load PPO actor cleanly: missing={missing}, unexpected={unexpected}"
        )


def load_policy_model(
    checkpoint_path: str,
    checkpoint_kind: str,
    input_dim: int,
    hidden_dims: Sequence[int],
    device: torch.device,
) -> HighLevelMLP:
    ckpt = _torch_load(checkpoint_path)
    state = ckpt.get("model_state_dict", ckpt)
    keys = set(state.keys())

    kind = checkpoint_kind.lower()
    if kind == "auto":
        if any(k.startswith("actor.") for k in keys):
            kind = "ppo"
        elif any(k.startswith("net.") for k in keys):
            kind = "bc"
        else:
            raise RuntimeError(f"Cannot infer checkpoint kind from keys in {checkpoint_path}")

    if kind == "bc":
        input_dim = int(ckpt.get("input_dim", input_dim))
        hidden_dims = tuple(int(v) for v in ckpt.get("hidden_dims", hidden_dims))
        output_dim = int(ckpt.get("output_dim", 2))
        model = HighLevelMLP(input_dim, hidden_dims, output_dim=output_dim)
        model.load_state_dict(state)
    elif kind == "ppo":
        model = HighLevelMLP(input_dim, hidden_dims, output_dim=2)
        _copy_actor_weights_from_ppo(model, state)
    else:
        raise ValueError("checkpoint_kind must be one of: auto, ppo, bc")

    model.to(device)
    model.eval()
    return model


def load_obs_norm(
    obs_norm_ckpt: str,
    expected_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    ckpt = _torch_load(obs_norm_ckpt)
    mean = torch.as_tensor(ckpt["feature_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(ckpt["feature_std"], dtype=torch.float32, device=device)
    names = list(ckpt.get("feature_names", []))
    if int(mean.numel()) != expected_dim or int(std.numel()) != expected_dim:
        raise RuntimeError(
            f"Obs norm dim mismatch: expected={expected_dim}, "
            f"mean={mean.numel()}, std={std.numel()}"
        )
    return mean, std, names


class FeatureHistory:
    def __init__(self, history_state: int, feature_dim: int):
        self.history_state = max(1, int(history_state))
        self.feature_dim = int(feature_dim)
        self.buffer = np.zeros((self.history_state, self.feature_dim), dtype=np.float32)
        self.initialized = False

    def update(self, feature: np.ndarray) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape != (self.feature_dim,):
            raise ValueError(f"Expected feature shape {(self.feature_dim,)}, got {feature.shape}")
        if not self.initialized:
            self.buffer[:] = feature[None, :]
            self.initialized = True
        else:
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = feature
        return self.buffer.reshape(-1).copy()


class HighLevelPolicyNode(Node):
    def __init__(self):
        super().__init__("g1_vision_highlevel_policy_node")

        self.declare_parameter("features_topic", "/g1_vision/features")
        self.declare_parameter("cmd_topic", "/g1_vision/cmd_vel")
        self.declare_parameter("policy_mode", "ppo")
        self.declare_parameter("policy_checkpoint", "")
        self.declare_parameter("checkpoint_kind", "auto")
        self.declare_parameter("obs_norm_ckpt", DEFAULT_OBS_NORM_CKPT)
        self.declare_parameter("obs_norm_mode", "bc")
        self.declare_parameter("history_state", 2)
        self.declare_parameter("hidden_dims", "128,128")
        self.declare_parameter("publish_hz", 5.0)
        self.declare_parameter("feature_stale_timeout_sec", 0.5)
        self.declare_parameter("publish_zero_before_first_feature", True)
        self.declare_parameter("vx_min", 0.10)
        self.declare_parameter("vx_max", 0.85)
        self.declare_parameter("wz_min", -1.20)
        self.declare_parameter("wz_max", 1.20)
        self.declare_parameter("stale_vx", 0.0)
        self.declare_parameter("stale_wz", 0.0)
        self.declare_parameter("obs_norm_clip", 8.0)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("log_cycle_timing", True)
        self.declare_parameter("cycle_timing_log_period_sec", 1.0)

        self.features_topic = str(self.get_parameter("features_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.history_state = int(self.get_parameter("history_state").value)
        self.feature_dim = len(BASE_FEATURE_NAMES)
        self.obs_dim = self.feature_dim * self.history_state
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.stale_timeout = float(self.get_parameter("feature_stale_timeout_sec").value)
        self.publish_zero_before_first = bool(
            self.get_parameter("publish_zero_before_first_feature").value
        )
        self.vx_min = float(self.get_parameter("vx_min").value)
        self.vx_max = float(self.get_parameter("vx_max").value)
        self.wz_min = float(self.get_parameter("wz_min").value)
        self.wz_max = float(self.get_parameter("wz_max").value)
        self.stale_vx = float(self.get_parameter("stale_vx").value)
        self.stale_wz = float(self.get_parameter("stale_wz").value)
        self.obs_norm_clip = float(self.get_parameter("obs_norm_clip").value)
        self.log_cycle_timing = bool(self.get_parameter("log_cycle_timing").value)
        self.cycle_timing_log_period_sec = float(
            self.get_parameter("cycle_timing_log_period_sec").value
        )

        device_name = str(self.get_parameter("device").value)
        self.device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

        self.policy_mode = str(self.get_parameter("policy_mode").value).lower()
        if self.policy_mode not in ("ppo", "bc"):
            raise ValueError("policy_mode must be one of: ppo, bc")
        policy_checkpoint_param = str(self.get_parameter("policy_checkpoint").value).strip()
        policy_checkpoint = policy_checkpoint_param or _default_policy_checkpoint(self.policy_mode)
        checkpoint_kind = str(self.get_parameter("checkpoint_kind").value)
        hidden_dims = _parse_dims(str(self.get_parameter("hidden_dims").value))
        self.model = load_policy_model(
            policy_checkpoint,
            checkpoint_kind,
            input_dim=self.obs_dim,
            hidden_dims=hidden_dims,
            device=self.device,
        )

        self.obs_norm_mode = str(self.get_parameter("obs_norm_mode").value).lower()
        self.obs_mean = torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)
        self.obs_std = torch.ones(self.obs_dim, dtype=torch.float32, device=self.device)
        if self.obs_norm_mode == "bc":
            obs_norm_ckpt = str(self.get_parameter("obs_norm_ckpt").value)
            self.obs_mean, self.obs_std, ckpt_names = load_obs_norm(
                obs_norm_ckpt, self.obs_dim, self.device
            )
            expected_names = _build_feature_names(self.history_state)
            if ckpt_names and ckpt_names != expected_names:
                self.get_logger().warn(
                    "Obs norm feature_names differ from expected g1_vision order. "
                    "Continue only if this checkpoint was trained with the same feature contract."
                )
        elif self.obs_norm_mode != "none":
            raise ValueError("obs_norm_mode must be one of: bc, none")

        self.history = FeatureHistory(self.history_state, self.feature_dim)
        self.latest_feature: np.ndarray | None = None
        self.latest_feature_time = None
        self.feature_count = 0
        self.pub_count = 0
        self._last_stale_warn_ns = 0
        self._last_cmd_log_ns = 0
        self._last_timing_log_ns = 0
        self._last_timer_perf = None

        self.feature_sub = self.create_subscription(
            Float32MultiArray,
            self.features_topic,
            self._on_features,
            10,
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        period = 1.0 / max(self.publish_hz, 1e-6)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info("g1_vision policy node started.")
        self.get_logger().info(f"  features_topic : {self.features_topic}")
        self.get_logger().info(f"  cmd_topic      : {self.cmd_topic}")
        self.get_logger().info(f"  policy_mode    : {self.policy_mode}")
        self.get_logger().info(f"  policy_ckpt    : {policy_checkpoint}")
        self.get_logger().info(f"  checkpoint_kind: {checkpoint_kind}")
        self.get_logger().info(f"  obs_norm_mode  : {self.obs_norm_mode}")
        self.get_logger().info(f"  obs_dim        : {self.obs_dim}")
        self.get_logger().info(f"  publish_hz     : {self.publish_hz:.2f}")

    def _on_features(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != self.feature_dim:
            self.get_logger().warn(
                f"Drop feature with wrong dim: got={len(msg.data)} expected={self.feature_dim}"
            )
            return
        self.latest_feature = np.asarray(msg.data, dtype=np.float32)
        self.latest_feature_time = self.get_clock().now()
        self.feature_count += 1

    def _publish_cmd(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)
        self.pub_count += 1

    def _publish_stale_cmd(self) -> None:
        self._publish_cmd(self.stale_vx, self.stale_wz)

    def _feature_is_stale(self) -> bool:
        if self.latest_feature is None or self.latest_feature_time is None:
            return True
        age = (self.get_clock().now() - self.latest_feature_time).nanoseconds * 1e-9
        return age > self.stale_timeout

    def _should_log(self, attr: str, period_sec: float) -> bool:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = int(getattr(self, attr))
        if now_ns - last_ns < int(period_sec * 1e9):
            return False
        setattr(self, attr, now_ns)
        return True

    def _on_timer(self) -> None:
        t_start = time.perf_counter()
        last_timer_perf = self._last_timer_perf
        self._last_timer_perf = t_start
        cycle_interval_sec = 0.0 if last_timer_perf is None else t_start - last_timer_perf
        is_stale = self._feature_is_stale()
        t_stale_check = time.perf_counter()

        if is_stale:
            if self.publish_zero_before_first or self.latest_feature is not None:
                self._publish_stale_cmd()
            t_publish = time.perf_counter()
            if self._should_log("_last_stale_warn_ns", 1.0):
                self.get_logger().warn(
                    "No fresh /g1_vision/features. Publishing stale fallback command."
                )
            self._log_cycle_timing(
                stale_check_sec=t_stale_check - t_start,
                history_sec=0.0,
                tensor_sec=0.0,
                norm_sec=0.0,
                inference_sec=0.0,
                publish_sec=t_publish - t_stale_check,
                total_sec=t_publish - t_start,
                cycle_interval_sec=cycle_interval_sec,
                stale=True,
            )
            return

        obs_np = self.history.update(self.latest_feature)
        t_history = time.perf_counter()
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_tensor = time.perf_counter()
        if self.obs_norm_mode == "bc":
            obs = (obs - self.obs_mean) / torch.clamp(self.obs_std, min=1e-6)
            obs = torch.clamp(obs, -self.obs_norm_clip, self.obs_norm_clip)
        t_norm = time.perf_counter()

        with torch.no_grad():
            action = self.model(obs).squeeze(0).detach().cpu().numpy()
        t_inference = time.perf_counter()

        vx = float(np.clip(action[0], self.vx_min, self.vx_max))
        wz = float(np.clip(action[1], self.wz_min, self.wz_max))
        self._publish_cmd(vx, wz)
        t_publish = time.perf_counter()

        if self._should_log("_last_cmd_log_ns", 1.0):
            self.get_logger().info(f"cmd vx={vx:.3f} wz={wz:.3f}")
        self._log_cycle_timing(
            stale_check_sec=t_stale_check - t_start,
            history_sec=t_history - t_stale_check,
            tensor_sec=t_tensor - t_history,
            norm_sec=t_norm - t_tensor,
            inference_sec=t_inference - t_norm,
            publish_sec=t_publish - t_inference,
            total_sec=t_publish - t_start,
            cycle_interval_sec=cycle_interval_sec,
            stale=False,
        )

    def _log_cycle_timing(
        self,
        *,
        stale_check_sec: float,
        history_sec: float,
        tensor_sec: float,
        norm_sec: float,
        inference_sec: float,
        publish_sec: float,
        total_sec: float,
        cycle_interval_sec: float,
        stale: bool,
    ) -> None:
        if not self.log_cycle_timing:
            return
        if self.cycle_timing_log_period_sec > 0.0 and not self._should_log(
            "_last_timing_log_ns", self.cycle_timing_log_period_sec
        ):
            return
        self.get_logger().info(
            "cycle timing "
            f"stale={stale} "
            f"interval={cycle_interval_sec * 1000.0:.3f}ms "
            f"total={total_sec * 1000.0:.3f}ms "
            f"stale_check={stale_check_sec * 1000.0:.3f}ms "
            f"history={history_sec * 1000.0:.3f}ms "
            f"tensor={tensor_sec * 1000.0:.3f}ms "
            f"norm={norm_sec * 1000.0:.3f}ms "
            f"infer={inference_sec * 1000.0:.3f}ms "
            f"publish={publish_sec * 1000.0:.3f}ms"
        )


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HighLevelPolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
