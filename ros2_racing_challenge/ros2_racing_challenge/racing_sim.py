import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ros2_racing_challenge_msgs.msg import CarCommand, CarState
import rclpy
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


Point2D = Tuple[float, float]


@dataclass
class Obstacle:
    center_x: float
    center_y: float
    size_x: float
    size_y: float


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float
    steering: float
    lateral_speed: float = 0.0
    yaw_rate: float = 0.0


class RacingSimulator(Node):
    def __init__(self) -> None:
        super().__init__('racing_sim')

        self.dt = 1.0 / 30.0
        self.wheelbase = 1.0
        self.front_overhang = 0.35
        self.rear_overhang = 0.25
        self.vehicle_width = 0.65
        self.axle_track = 0.48
        self.wheel_length = 0.32
        self.wheel_width = 0.10
        self.body_height = 0.18
        self.wheel_height = 0.24
        self.max_speed = 9.0
        self.max_steering = 0.60
        self.command_timeout = 0.5
        self.visualization_period = 1.0 / 20.0
        self.max_trail_points = 400
        self.goal_laps = 10
        self.max_accel = 2.8
        self.max_decel = 4.5
        self.max_steering_rate = 6.0
        self.gravity = 9.81
        self.vehicle_mass = 25.0
        self.yaw_inertia = 4.6
        self.cg_to_front_axle = 0.40
        self.cg_to_rear_axle = self.wheelbase - self.cg_to_front_axle
        self.base_front_cornering_stiffness = 260.0
        self.base_rear_cornering_stiffness = 520.0
        self.base_static_friction_coefficient = 0.85
        self.base_kinetic_friction_coefficient = 0.62
        self.front_cornering_stiffness = self.base_front_cornering_stiffness
        self.rear_cornering_stiffness = self.base_rear_cornering_stiffness
        self.front_static_friction_coefficient = (
            self.base_static_friction_coefficient
        )
        self.rear_static_friction_coefficient = (
            self.base_static_friction_coefficient
        )
        self.front_kinetic_friction_coefficient = (
            self.base_kinetic_friction_coefficient
        )
        self.rear_kinetic_friction_coefficient = (
            self.base_kinetic_friction_coefficient
        )
        self.lateral_speed_damping = 1.2
        self.yaw_rate_damping = 2.0
        self.max_lateral_speed = 3.0
        self.collision_penalty = 5.0
        self.off_track_penalty_rate = 5.0
        self.surface_friction_multiplier = 1.0
        self.surface_friction_loss = 0.0
        self.friction_loss_per_lap = 0.0
        self.friction_loss_per_lap_range = (0.02, 0.03)
        self.max_surface_friction_loss = 0.15

        self.random_generator = random.Random()

        self.straight_half_length = 10.0
        self.centerline_radius = 5.0
        self.track_half_width = 2.0
        self.outer_radius = self.centerline_radius + self.track_half_width
        self.inner_radius = self.centerline_radius - self.track_half_width
        self.track_center_y = self.centerline_radius
        self.start_line_x = 0.0
        self.track_length = (
            4.0 * self.straight_half_length +
            2.0 * math.pi * self.centerline_radius
        )
        self.centerline = self.make_stadium_path(self.centerline_radius)
        self.outer_boundary = self.make_stadium_path(self.outer_radius)
        self.inner_boundary = self.make_stadium_path(self.inner_radius)
        self.waypoint_progresses = [
            0.0,
            0.25 * self.track_length,
            0.50 * self.track_length,
            0.75 * self.track_length,
        ]
        self.next_waypoint_index = 1
        self.obstacles = [
            Obstacle(
                center_x=4.0,
                center_y=self.centerline_radius + 1.00,
                size_x=0.5,
                size_y=2.0,
            ),
            Obstacle(
                center_x=-4.0,
                center_y=self.centerline_radius - 1.00,
                size_x=0.5,
                size_y=2.0,
            ),
        ]
        self.state = VehicleState(
            x=0.25,
            y=-self.track_center_y,
            yaw=0.0,
            speed=0.0,
            steering=0.0,
        )
        self.command_accel = 0.0
        self.command_steering_rate = 0.0
        self.last_command_stamp = self.get_clock().now()
        self.started = False
        self.finished = False
        self.start_time = None
        self.lap_count = 0
        self.collision_count = 0
        self.off_track_count = 0
        self.off_track_duration = 0.0
        self.total_penalty_time = 0.0
        self.last_lap_time = None
        self.last_crossing_time = None
        self.in_collision = False
        self.outside_track = False
        self.front_tires_slipping = False
        self.rear_tires_slipping = False
        self.gate_contact_active = False
        self.trail: List[Point2D] = [(self.state.x, self.state.y)]
        self.last_visualization_stamp = self.get_clock().now()

        self.drive_subscriber = self.create_subscription(
            CarCommand,
            '/racing/command',
            self.drive_callback,
            10,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/racing/markers',
            10,
        )
        self.state_publisher = self.create_publisher(
            CarState,
            '/racing/state',
            10,
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.simulation_timer = self.create_timer(
            self.dt,
            self.timer_callback,
        )
        self.visualization_timer = self.create_timer(
            self.visualization_period,
            self.visualization_callback,
        )

        self.get_logger().info(
            'Racing simulator started. '
            f'Goal: finish {self.goal_laps} laps as fast as possible.'
        )

    def drive_callback(self, msg: CarCommand) -> None:
        self.command_accel = self.clamp(
            msg.acceleration,
            -self.max_decel,
            self.max_accel,
        )
        self.command_steering_rate = self.clamp(
            msg.steering_rate,
            -self.max_steering_rate,
            self.max_steering_rate,
        )
        self.last_command_stamp = self.get_clock().now()

    def timer_callback(self) -> None:
        previous_state = self.state
        has_recent_command = (
            self.get_clock().now() - self.last_command_stamp
        ).nanoseconds < int(self.command_timeout * 1e9)

        if self.finished or not has_recent_command:
            commanded_accel = 0.0
            commanded_steering_rate = 0.0
        else:
            commanded_accel = self.command_accel
            commanded_steering_rate = self.command_steering_rate

        if abs(self.state.speed) > 1e-3 or abs(commanded_accel) > 1e-3:
            self.started = True
            if self.start_time is None:
                self.start_time = self.get_clock().now()

        trial_state = self.propagate(
            previous_state,
            commanded_accel,
            commanded_steering_rate,
        )

        is_outside_track = self.is_vehicle_outside_track(trial_state)
        is_in_collision = self.is_vehicle_in_collision(trial_state)

        self.state = trial_state
        self.trail.append((self.state.x, self.state.y))
        if len(self.trail) > self.max_trail_points:
            self.trail = self.trail[-self.max_trail_points:]

        self.update_penalties(is_in_collision, is_outside_track)
        self.update_lap_counter(self.state)
        self.publish_state()
        self.publish_transform()

    def visualization_callback(self) -> None:
        self.publish_markers()
        self.last_visualization_stamp = self.get_clock().now()

    def propagate(
        self,
        state: VehicleState,
        commanded_accel: float,
        commanded_steering_rate: float,
    ) -> VehicleState:
        next_steering = self.clamp(
            state.steering + commanded_steering_rate * self.dt,
            -self.max_steering,
            self.max_steering,
        )
        (
            front_longitudinal_force,
            front_lateral_force,
            rear_longitudinal_force,
            rear_lateral_force,
        ) = self.compute_tire_forces(state, next_steering, commanded_accel)
        longitudinal_accel = (
            (front_longitudinal_force + rear_longitudinal_force) /
            self.vehicle_mass +
            state.lateral_speed * state.yaw_rate
        )
        next_speed = self.clamp(
            state.speed + longitudinal_accel * self.dt,
            0.0,
            self.max_speed,
        )
        lateral_accel = (
            (front_lateral_force + rear_lateral_force) / self.vehicle_mass -
            state.speed * state.yaw_rate
        )
        yaw_accel = (
            self.cg_to_front_axle * front_lateral_force -
            self.cg_to_rear_axle * rear_lateral_force
        ) / self.yaw_inertia

        next_lateral_speed = self.clamp(
            state.lateral_speed + lateral_accel * self.dt,
            -self.max_lateral_speed,
            self.max_lateral_speed,
        )
        yaw_rate = state.yaw_rate + yaw_accel * self.dt

        if next_speed < 0.05:
            next_lateral_speed = 0.0
            yaw_rate = 0.0
        else:
            next_lateral_speed = self.decay_toward_zero(
                next_lateral_speed,
                self.lateral_speed_damping * self.dt,
            )
            yaw_rate = self.decay_toward_zero(
                yaw_rate,
                self.yaw_rate_damping * self.dt,
            )

        next_yaw = self.normalize_angle(state.yaw + yaw_rate * self.dt)
        forward_x = math.cos(state.yaw)
        forward_y = math.sin(state.yaw)
        left_x = -math.sin(state.yaw)
        left_y = math.cos(state.yaw)
        next_x = state.x + (
            next_speed * forward_x + next_lateral_speed * left_x
        ) * self.dt
        next_y = state.y + (
            next_speed * forward_y + next_lateral_speed * left_y
        ) * self.dt
        return VehicleState(
            x=next_x,
            y=next_y,
            yaw=next_yaw,
            speed=next_speed,
            steering=next_steering,
            lateral_speed=next_lateral_speed,
            yaw_rate=yaw_rate,
        )

    def compute_tire_forces(
        self,
        state: VehicleState,
        steering: float,
        commanded_accel: float,
    ) -> Tuple[float, float, float, float]:
        safe_speed = max(state.speed, 0.25)

        front_slip = steering - math.atan2(
            state.lateral_speed + self.cg_to_front_axle * state.yaw_rate,
            safe_speed,
        )
        rear_slip = -math.atan2(
            state.lateral_speed - self.cg_to_rear_axle * state.yaw_rate,
            safe_speed,
        )

        front_load = (
            self.vehicle_mass * self.gravity *
            self.cg_to_rear_axle / self.wheelbase
        )
        rear_load = (
            self.vehicle_mass * self.gravity *
            self.cg_to_front_axle / self.wheelbase
        )
        total_load = front_load + rear_load
        commanded_longitudinal_force = self.vehicle_mass * commanded_accel
        front_longitudinal_force = (
            commanded_longitudinal_force * front_load / total_load
        )
        rear_longitudinal_force = (
            commanded_longitudinal_force * rear_load / total_load
        )
        front_lateral_force = (
            self.front_cornering_stiffness *
            front_slip
        )
        rear_lateral_force = (
            self.rear_cornering_stiffness *
            rear_slip
        )
        front_longitudinal_force, front_lateral_force, front_slipping = (
            self.apply_friction_circle(
                front_longitudinal_force,
                front_lateral_force,
                front_load,
                self.front_static_friction_coefficient,
                self.front_kinetic_friction_coefficient,
            )
        )
        rear_longitudinal_force, rear_lateral_force, rear_slipping = (
            self.apply_friction_circle(
                rear_longitudinal_force,
                rear_lateral_force,
                rear_load,
                self.rear_static_friction_coefficient,
                self.rear_kinetic_friction_coefficient,
            )
        )
        self.front_tires_slipping = front_slipping
        self.rear_tires_slipping = rear_slipping
        return (
            front_longitudinal_force,
            front_lateral_force,
            rear_longitudinal_force,
            rear_lateral_force,
        )

    def apply_friction_circle(
        self,
        longitudinal_force: float,
        lateral_force: float,
        normal_load: float,
        static_friction_coefficient: float,
        kinetic_friction_coefficient: float,
    ) -> Tuple[float, float, bool]:
        force_magnitude = math.hypot(longitudinal_force, lateral_force)
        static_limit = static_friction_coefficient * normal_load
        if force_magnitude <= static_limit:
            return longitudinal_force, lateral_force, False

        kinetic_limit = kinetic_friction_coefficient * normal_load
        scale = kinetic_limit / max(force_magnitude, 1e-6)
        return longitudinal_force * scale, lateral_force * scale, True

    def update_lap_counter(
        self,
        current_state: VehicleState,
    ) -> None:
        if not self.started or self.finished:
            return

        self.update_waypoint_state(current_state)

        if self.lap_count >= self.goal_laps:
            self.finished = True
            self.state.speed = 0.0
            self.get_logger().info(
                f'Finished {self.goal_laps} laps in '
                f'{self.elapsed_time():.2f} seconds '
                f'+ {self.total_penalty_time:.2f} seconds penalty '
                f'= {self.score_time():.2f} seconds.'
            )

    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        delta = self.get_clock().now() - self.start_time
        return delta.nanoseconds / 1e9

    def score_time(self) -> float:
        return self.elapsed_time() + self.total_penalty_time

    def update_penalties(
        self,
        is_in_collision: bool,
        is_outside_track: bool,
    ) -> None:
        if is_in_collision and not self.in_collision:
            self.collision_count += 1
            penalty = self.collision_penalty
            self.total_penalty_time += penalty
            self.get_logger().warn(
                f'Obstacle collision detected ({self.collision_count}). '
                f'+{penalty:.1f}s penalty.'
            )

        if is_outside_track and not self.outside_track:
            self.off_track_count += 1
            self.get_logger().warn(
                f'Track violation detected ({self.off_track_count}). '
                f'Penalty now accumulates while outside the track.'
            )

        if is_outside_track:
            incremental_penalty = self.dt * self.off_track_penalty_rate
            self.off_track_duration += self.dt
            self.total_penalty_time += incremental_penalty

        self.in_collision = is_in_collision
        self.outside_track = is_outside_track

    def update_waypoint_state(
        self,
        current_state: VehicleState,
    ) -> None:
        touching_gate = self.vehicle_intersects_gate(
            current_state,
            self.waypoint_progresses[self.next_waypoint_index],
        )

        if touching_gate and not self.gate_contact_active:
            passed_waypoint = self.next_waypoint_index
            self.next_waypoint_index = (
                self.next_waypoint_index + 1
            ) % len(self.waypoint_progresses)
            self.gate_contact_active = True

            if passed_waypoint == 0:
                self.lap_count += 1
                self.last_crossing_time = self.get_clock().now()
                if self.start_time is not None:
                    self.last_lap_time = (
                        self.last_crossing_time - self.start_time
                    ).nanoseconds / 1e9
                self.apply_lap_friction_loss()
                self.get_logger().info(
                    f'Lap {self.lap_count}/{self.goal_laps} completed. '
                    f'Friction: '
                    f'{100.0 * self.surface_friction_multiplier:.0f}%'
                )
        elif not touching_gate:
            self.gate_contact_active = False

    def apply_lap_friction_loss(self) -> None:
        self.friction_loss_per_lap = self.random_generator.uniform(
            *self.friction_loss_per_lap_range
        )
        self.surface_friction_loss = min(
            self.max_surface_friction_loss,
            self.surface_friction_loss + self.friction_loss_per_lap,
        )
        self.surface_friction_multiplier = 1.0 - self.surface_friction_loss
        self.refresh_tire_model()

    def refresh_tire_model(self) -> None:
        self.front_cornering_stiffness = (
            self.base_front_cornering_stiffness *
            self.surface_friction_multiplier
        )
        self.rear_cornering_stiffness = (
            self.base_rear_cornering_stiffness *
            self.surface_friction_multiplier
        )
        self.front_static_friction_coefficient = (
            self.base_static_friction_coefficient *
            self.surface_friction_multiplier
        )
        self.rear_static_friction_coefficient = (
            self.base_static_friction_coefficient *
            self.surface_friction_multiplier
        )
        self.front_kinetic_friction_coefficient = (
            self.base_kinetic_friction_coefficient *
            self.surface_friction_multiplier
        )
        self.rear_kinetic_friction_coefficient = (
            self.base_kinetic_friction_coefficient *
            self.surface_friction_multiplier
        )

    def is_vehicle_outside_track(self, state: VehicleState) -> bool:
        for corner_x, corner_y in self.vehicle_corners(state):
            inside_outer = self.is_inside_stadium(corner_x, corner_y,
                                                  self.outer_radius)
            inside_inner = self.is_inside_stadium(corner_x, corner_y,
                                                  self.inner_radius)
            if not inside_outer or inside_inner:
                return True
        return False

    def is_vehicle_in_collision(self, state: VehicleState) -> bool:
        for corner_x, corner_y in self.vehicle_corners(state):
            for obstacle in self.obstacles:
                inside_obstacle = (
                    abs(corner_x - obstacle.center_x) <= obstacle.size_x / 2.0
                    and
                    abs(corner_y - obstacle.center_y) <= obstacle.size_y / 2.0
                )
                if inside_obstacle:
                    return True
        return False

    def vehicle_corners(self, state: VehicleState) -> List[Point2D]:
        half_width = self.vehicle_width / 2.0
        front_x = self.wheelbase + self.front_overhang
        rear_x = -self.rear_overhang
        local_points = [
            (front_x, half_width),
            (front_x, -half_width),
            (rear_x, -half_width),
            (rear_x, half_width),
        ]
        return [self.transform_point(state, px, py) for px, py in local_points]

    def transform_point(
        self,
        state: VehicleState,
        local_x: float,
        local_y: float,
    ) -> Point2D:
        cos_yaw = math.cos(state.yaw)
        sin_yaw = math.sin(state.yaw)
        world_x = state.x + local_x * cos_yaw - local_y * sin_yaw
        world_y = state.y + local_x * sin_yaw + local_y * cos_yaw
        return world_x, world_y

    def publish_state(self) -> None:
        msg = CarState()
        msg.x = float(self.state.x)
        msg.y = float(self.state.y)
        msg.yaw = float(self.state.yaw)
        msg.speed = float(self.state.speed)
        msg.steering = float(self.state.steering)
        self.state_publisher.publish(msg)

    def publish_transform(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'rear_axle'
        transform.transform.translation.x = self.state.x
        transform.transform.translation.y = self.state.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation = self.quaternion_from_yaw(self.state.yaw)
        self.tf_broadcaster.sendTransform(transform)

    def publish_markers(self) -> None:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        vehicle_color, trail_color, front_highlight_color = (
            self.current_vehicle_colors()
        )

        markers.markers.append(
            self.make_line_strip(
                marker_id=0,
                stamp=stamp,
                points=self.outer_boundary,
                color=(0.95, 0.95, 0.95, 1.0),
                width=0.10,
                namespace='track',
            )
        )
        markers.markers.append(
            self.make_line_strip(
                marker_id=1,
                stamp=stamp,
                points=self.inner_boundary,
                color=(0.95, 0.95, 0.95, 1.0),
                width=0.10,
                namespace='track',
            )
        )
        markers.markers.append(
            self.make_line_strip(
                marker_id=2,
                stamp=stamp,
                points=self.centerline,
                color=(0.30, 0.80, 0.30, 0.55),
                width=0.05,
                namespace='track',
            )
        )
        markers.markers.extend(self.make_waypoint_markers(stamp))

        for index, obstacle in enumerate(self.obstacles, start=10):
            markers.markers.append(
                self.make_cube(
                    marker_id=index,
                    stamp=stamp,
                    namespace='obstacles',
                    x=obstacle.center_x,
                    y=obstacle.center_y,
                    yaw=0.0,
                    scale_x=obstacle.size_x,
                    scale_y=obstacle.size_y,
                    scale_z=0.5,
                    color=(0.85, 0.30, 0.20, 0.95),
                )
            )

        markers.markers.append(
            self.make_line_strip(
                marker_id=20,
                stamp=stamp,
                points=self.trail,
                color=trail_color,
                width=0.04,
                namespace='vehicle',
            )
        )
        markers.markers.append(self.make_vehicle_body(stamp, vehicle_color))
        markers.markers.append(
            self.make_vehicle_front_highlight(stamp, front_highlight_color)
        )
        markers.markers.extend(self.make_wheels(stamp))
        markers.markers.append(self.make_status_text(stamp))

        self.marker_publisher.publish(markers)

    def make_vehicle_body(
        self,
        stamp,
        body_color: Tuple[float, float, float, float],
    ) -> Marker:
        body_center_x = (
            self.wheelbase + self.front_overhang - self.rear_overhang
        ) / 2.0
        center_x, center_y = self.transform_point(
            self.state,
            body_center_x,
            0.0,
        )
        return self.make_cube(
            marker_id=30,
            stamp=stamp,
            namespace='vehicle',
            x=center_x,
            y=center_y,
            yaw=self.state.yaw,
            scale_x=self.wheelbase + self.front_overhang + self.rear_overhang,
            scale_y=self.vehicle_width,
            scale_z=self.body_height,
            color=body_color,
            z=self.wheel_height + self.body_height / 2.0 - 0.03,
        )

    def make_vehicle_front_highlight(
        self,
        stamp,
        highlight_color: Tuple[float, float, float, float],
    ) -> Marker:
        nose_center_x = self.wheelbase + self.front_overhang - 0.14
        center_x, center_y = self.transform_point(
            self.state,
            nose_center_x,
            0.0,
        )
        return self.make_cube(
            marker_id=35,
            stamp=stamp,
            namespace='vehicle',
            x=center_x,
            y=center_y,
            yaw=self.state.yaw,
            scale_x=0.18,
            scale_y=self.vehicle_width * 0.88,
            scale_z=self.body_height * 0.92,
            color=highlight_color,
            z=self.wheel_height + self.body_height / 2.0 - 0.02,
        )

    def current_vehicle_colors(
        self,
    ) -> Tuple[
        Tuple[float, float, float, float],
        Tuple[float, float, float, float],
        Tuple[float, float, float, float],
    ]:
        if self.in_collision or self.outside_track:
            return (
                (0.82, 0.22, 0.22, 0.55),
                (0.95, 0.28, 0.28, 0.92),
                (0.92, 0.40, 0.40, 0.96),
            )

        return (
            (0.15, 0.45, 0.90, 0.45),
            (0.25, 0.65, 1.0, 0.90),
            (0.28, 0.58, 0.98, 0.96),
        )

    def make_wheels(self, stamp) -> List[Marker]:
        wheel_layout = [
            (0.0, self.axle_track / 2.0, 31, self.state.yaw, False),
            (0.0, -self.axle_track / 2.0, 32, self.state.yaw, False),
            (
                self.wheelbase,
                self.axle_track / 2.0,
                33,
                self.state.yaw + self.state.steering,
                True,
            ),
            (
                self.wheelbase,
                -self.axle_track / 2.0,
                34,
                self.state.yaw + self.state.steering,
                True,
            ),
        ]
        markers = []
        for local_x, local_y, marker_id, wheel_yaw, is_front in wheel_layout:
            wheel_x, wheel_y = self.transform_point(self.state, local_x, local_y)
            tire_slipping = (
                self.front_tires_slipping if is_front
                else self.rear_tires_slipping
            )
            wheel_color = self.tire_color(tire_slipping)
            markers.append(
                self.make_cube(
                    marker_id=marker_id,
                    stamp=stamp,
                    namespace='vehicle',
                    x=wheel_x,
                    y=wheel_y,
                    yaw=wheel_yaw,
                    scale_x=self.wheel_length,
                    scale_y=self.wheel_width,
                    scale_z=self.wheel_height * 1.12,
                    color=wheel_color,
                    z=self.wheel_height * 0.56,
                )
            )
        return markers

    def tire_color(
        self,
        tire_slipping: bool,
    ) -> Tuple[float, float, float, float]:
        if tire_slipping:
            return (1.0, 0.05, 0.03, 1.0)
        return (0.88, 0.90, 0.92, 1.0)

    def make_status_text(self, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = 'status'
        marker.id = 40
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.6
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.55
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = (
            f'Laps: {self.lap_count}/{self.goal_laps} | '
            f'Time: {self.score_time():.1f}s | '
            f'Penalty: {self.total_penalty_time:.1f}s | '
            f'Friction: {100.0 * self.surface_friction_multiplier:.0f}%'
        )
        if self.finished:
            marker.text += ' | FINISHED'
        return marker

    def make_waypoint_markers(self, stamp) -> List[Marker]:
        markers = []
        gate_width = 2.0 * self.track_half_width - 0.10

        for index, progress in enumerate(self.waypoint_progresses):
            x, y, tangent_yaw = self.track_pose_at_progress(progress)
            if index == self.next_waypoint_index:
                color = (1.0, 0.85, 0.10, 0.95)
            else:
                color = (0.95, 0.95, 0.95, 0.22)

            markers.append(
                self.make_cube(
                    marker_id=3 + index,
                    stamp=stamp,
                    namespace='waypoints',
                    x=x,
                    y=y,
                    yaw=tangent_yaw + math.pi / 2.0,
                    scale_x=gate_width,
                    scale_y=0.05,
                    scale_z=0.02,
                    color=color,
                )
            )

        return markers

    def make_line_strip(
        self,
        marker_id: int,
        stamp,
        points: Sequence[Point2D],
        color: Tuple[float, float, float, float],
        width: float,
        namespace: str,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.points = [self.make_point(x, y, 0.02) for x, y in points]
        return marker

    def make_cube(
        self,
        marker_id: int,
        stamp,
        namespace: str,
        x: float,
        y: float,
        yaw: float,
        scale_x: float,
        scale_y: float,
        scale_z: float,
        color: Tuple[float, float, float, float],
        z: float = None,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = scale_z / 2.0 if z is None else z
        marker.pose.orientation = self.quaternion_from_yaw(yaw)
        marker.scale.x = scale_x
        marker.scale.y = scale_y
        marker.scale.z = scale_z
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        return marker

    def make_stadium_path(self, radius: float) -> List[Point2D]:
        points: List[Point2D] = []
        resolution = 48

        for index in range(resolution + 1):
            ratio = index / resolution
            x = -self.straight_half_length + (
                2.0 * self.straight_half_length * ratio
            )
            points.append((x, -radius))

        for index in range(1, resolution + 1):
            angle = -math.pi / 2.0 + math.pi * (index / resolution)
            x = self.straight_half_length + radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append((x, y))

        for index in range(1, resolution + 1):
            ratio = index / resolution
            x = self.straight_half_length - (
                2.0 * self.straight_half_length * ratio
            )
            points.append((x, radius))

        for index in range(1, resolution + 1):
            angle = math.pi / 2.0 + math.pi * (index / resolution)
            x = -self.straight_half_length + radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append((x, y))

        return points

    def is_inside_stadium(self, x: float, y: float, radius: float) -> bool:
        if -self.straight_half_length <= x <= self.straight_half_length:
            return abs(y) <= radius

        cap_center_x = (
            self.straight_half_length if x > self.straight_half_length
            else -self.straight_half_length
        )
        return (x - cap_center_x) ** 2 + y ** 2 <= radius ** 2

    def centerline_progress(self, x: float, y: float) -> float:
        if -self.straight_half_length <= x <= self.straight_half_length:
            if y <= 0.0:
                if x >= 0.0:
                    return x
                return (
                    3.0 * self.straight_half_length +
                    2.0 * math.pi * self.centerline_radius +
                    (x + self.straight_half_length)
                )

            return (
                self.straight_half_length +
                math.pi * self.centerline_radius +
                (self.straight_half_length - x)
            )

        if x > self.straight_half_length:
            angle = math.atan2(y, x - self.straight_half_length)
            if angle < -math.pi / 2.0:
                angle += 2.0 * math.pi
            return (
                self.straight_half_length +
                self.centerline_radius * (angle + math.pi / 2.0)
            )

        angle = math.atan2(y, x + self.straight_half_length)
        if angle < math.pi / 2.0:
            angle += 2.0 * math.pi
        return (
            3.0 * self.straight_half_length +
            math.pi * self.centerline_radius +
            self.centerline_radius * (angle - math.pi / 2.0)
        )

    def track_pose_at_progress(self, progress: float) -> Tuple[float, float, float]:
        normalized_progress = progress % self.track_length
        first_straight_length = self.straight_half_length
        semicircle_length = math.pi * self.centerline_radius
        top_straight_length = 2.0 * self.straight_half_length

        if normalized_progress <= first_straight_length:
            return normalized_progress, -self.centerline_radius, 0.0

        if normalized_progress <= first_straight_length + semicircle_length:
            arc_progress = normalized_progress - first_straight_length
            angle = -math.pi / 2.0 + arc_progress / self.centerline_radius
            x = self.straight_half_length + self.centerline_radius * math.cos(
                angle
            )
            y = self.centerline_radius * math.sin(angle)
            return x, y, angle + math.pi / 2.0

        if (
            normalized_progress <=
            first_straight_length + semicircle_length + top_straight_length
        ):
            straight_progress = (
                normalized_progress - first_straight_length - semicircle_length
            )
            x = self.straight_half_length - straight_progress
            return x, self.centerline_radius, math.pi

        arc_progress = (
            normalized_progress - first_straight_length -
            semicircle_length - top_straight_length
        )
        if arc_progress <= semicircle_length:
            angle = math.pi / 2.0 + arc_progress / self.centerline_radius
            x = -self.straight_half_length + self.centerline_radius * math.cos(
                angle
            )
            y = self.centerline_radius * math.sin(angle)
            return x, y, angle + math.pi / 2.0

        straight_progress = arc_progress - semicircle_length
        x = -self.straight_half_length + straight_progress
        return x, -self.centerline_radius, 0.0

    def vehicle_intersects_gate(
        self,
        state: VehicleState,
        gate_progress: float,
    ) -> bool:
        gate_start, gate_end = self.gate_segment(gate_progress)
        corners = self.vehicle_corners(state)

        for index in range(len(corners)):
            edge_start = corners[index]
            edge_end = corners[(index + 1) % len(corners)]
            if self.segments_intersect(
                edge_start,
                edge_end,
                gate_start,
                gate_end,
            ):
                return True

        return False

    def gate_segment(self, gate_progress: float) -> Tuple[Point2D, Point2D]:
        gate_x, gate_y, gate_yaw = self.track_pose_at_progress(gate_progress)
        half_width = self.track_half_width - 0.05
        normal_yaw = gate_yaw + math.pi / 2.0
        dx = half_width * math.cos(normal_yaw)
        dy = half_width * math.sin(normal_yaw)
        return (gate_x - dx, gate_y - dy), (gate_x + dx, gate_y + dy)

    def segments_intersect(
        self,
        p1: Point2D,
        p2: Point2D,
        q1: Point2D,
        q2: Point2D,
    ) -> bool:
        o1 = self.orientation(p1, p2, q1)
        o2 = self.orientation(p1, p2, q2)
        o3 = self.orientation(q1, q2, p1)
        o4 = self.orientation(q1, q2, p2)

        if o1 * o2 < 0.0 and o3 * o4 < 0.0:
            return True

        epsilon = 1e-9
        if abs(o1) <= epsilon and self.on_segment(p1, q1, p2):
            return True
        if abs(o2) <= epsilon and self.on_segment(p1, q2, p2):
            return True
        if abs(o3) <= epsilon and self.on_segment(q1, p1, q2):
            return True
        if abs(o4) <= epsilon and self.on_segment(q1, p2, q2):
            return True

        return False

    def orientation(
        self,
        a: Point2D,
        b: Point2D,
        c: Point2D,
    ) -> float:
        return (
            (b[0] - a[0]) * (c[1] - a[1]) -
            (b[1] - a[1]) * (c[0] - a[0])
        )

    def on_segment(
        self,
        a: Point2D,
        b: Point2D,
        c: Point2D,
    ) -> bool:
        epsilon = 1e-9
        return (
            min(a[0], c[0]) - epsilon <= b[0] <= max(a[0], c[0]) + epsilon and
            min(a[1], c[1]) - epsilon <= b[1] <= max(a[1], c[1]) + epsilon
        )

    def make_point(self, x: float, y: float, z: float) -> Point:
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        return point

    def quaternion_from_yaw(self, yaw: float) -> Quaternion:
        quaternion = Quaternion()
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)
        return quaternion

    def clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def decay_toward_zero(self, value: float, amount: float) -> float:
        if value > amount:
            return value - amount
        if value < -amount:
            return value + amount
        return 0.0

    def normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RacingSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
