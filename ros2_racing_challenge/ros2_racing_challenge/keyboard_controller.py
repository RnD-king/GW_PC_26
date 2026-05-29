import select
import sys
import termios
import tty

from ros2_racing_challenge_msgs.msg import CarCommand, CarState
import rclpy
from rclpy.node import Node


class KeyboardController(Node):
    def __init__(self) -> None:
        super().__init__('keyboard_controller')
        self.publisher = self.create_publisher(
            CarCommand,
            '/racing/command',
            10,
        )
        self.state_subscriber = self.create_subscription(
            CarState,
            '/racing/state',
            self.state_callback,
            10,
        )
        self.longitudinal_command = 0.0
        self.steering_rate_command = 0.0
        self.desired_speed = 0.0
        self.current_speed = 0.0
        self.desired_steering = 0.0
        self.current_steering = 0.0
        self.speed_step = 0.5
        self.steering_step = 0.2
        self.max_accel = 2.8
        self.max_decel = 4.5
        self.max_speed = 9.0
        self.max_steering_rate = 6.0
        self.max_steering = 0.6
        self.speed_kp = 2.2
        self.steering_kp = 8.0
        self.control_dt = 0.05
        self.full_brake_active = False
        self.settings = None

        if sys.stdin.isatty():
            self.settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        else:
            self.get_logger().warn('Keyboard controller requires a TTY.')

        self.print_help()
        self.timer = self.create_timer(self.control_dt, self.timer_callback)

    def print_help(self) -> None:
        self.get_logger().info(
            'Keyboard control: a/d adjust steering target, '
            'w centers steering target, r/f adjust desired speed, '
            'space full brake, x release brake, q quit.'
        )

    def timer_callback(self) -> None:
        key = self.read_key()
        if key is not None:
            self.handle_key(key)
            self.publish_status()
        self.publish_command()

    def read_key(self):
        if self.settings is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if ready:
            return sys.stdin.read(1)
        return None

    def handle_key(self, key: str) -> None:
        if key == 'r':
            self.desired_speed = self.clamp(
                self.desired_speed + self.speed_step,
                0.0,
                self.max_speed,
            )
            self.full_brake_active = False
        elif key == 'f':
            self.desired_speed = self.clamp(
                self.desired_speed - self.speed_step,
                0.0,
                self.max_speed,
            )
            self.full_brake_active = False
        elif key == 'a':
            self.desired_steering = self.clamp(
                self.desired_steering + self.steering_step,
                -self.max_steering,
                self.max_steering,
            )
        elif key == 'd':
            self.desired_steering = self.clamp(
                self.desired_steering - self.steering_step,
                -self.max_steering,
                self.max_steering,
            )
        elif key == 'w':
            self.desired_steering = 0.0
        elif key == ' ':
            self.desired_speed = 0.0
            self.full_brake_active = True
        elif key == 'x':
            self.full_brake_active = False
            self.desired_speed = self.current_speed
        elif key == 'q':
            self.get_logger().info('Keyboard controller shutting down.')
            raise KeyboardInterrupt

    def state_callback(self, msg: CarState) -> None:
        self.current_speed = msg.speed
        self.current_steering = msg.steering

    def publish_command(self) -> None:
        steering_error = self.desired_steering - self.current_steering
        self.steering_rate_command = self.clamp(
            self.steering_kp * steering_error,
            -self.max_steering_rate,
            self.max_steering_rate,
        )
        if self.full_brake_active:
            self.longitudinal_command = -self.max_decel
        else:
            speed_error = self.desired_speed - self.current_speed
            self.longitudinal_command = self.clamp(
                self.speed_kp * speed_error,
                -self.max_decel,
                self.max_accel,
            )

        msg = CarCommand()
        msg.acceleration = self.longitudinal_command
        msg.steering_rate = self.steering_rate_command
        self.publisher.publish(msg)

    def publish_status(self) -> None:
        self.get_logger().info(
            f'Target | desired_speed: {self.desired_speed:+.2f}, '
            f'desired_steer: {self.desired_steering:+.2f}'
        )

    def clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def destroy_node(self) -> bool:
        if self.settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KeyboardController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
