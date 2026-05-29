# ROS 2 Racing Challenge

Implement a vehicle controller for a ROS 2 racing simulator.

## Assignment Goal

Implement `racing_controller.py` so the vehicle can drive around the oval track.

Your controller should:

1. Drive stably without leaving the track
2. Pass the waypoint gates in order
3. Avoid the two obstacles on the track
4. Complete 10 laps as quickly as possible

## File To Modify

For the assignment, you should modify only this file:

```text
ros2_racing_challenge/racing_controller.py
```

Avoid modifying `racing_sim.py`, launch files, or message definitions so your solution stays compatible with the evaluation environment.

## Controller Interface

`racing_controller.py` already contains the basic publisher/subscriber structure.

- subscribe: `/racing/state`
- publish: `/racing/command`

The input `CarState` contains the current vehicle state.

```text
x         vehicle x position [m]
y         vehicle y position [m]
yaw       vehicle heading [rad]
speed     vehicle speed [m/s]
steering  current steering angle [rad]
```

The output `CarCommand` is the command sent to the vehicle.

```text
acceleration   acceleration command [m/s^2]
steering_rate  steering angle rate command [rad/s]
```


## Vehicle Model

For controller design, the vehicle can be approximated as a kinematic
bicycle model:

$$
\begin{aligned}
\dot{\mathbf{x}} &= f(\mathbf{x}) + B\mathbf{u} \\
\mathbf{x} &=
\begin{bmatrix}
x & y & \psi & v & \delta
\end{bmatrix}^{T} \\
\mathbf{u} &=
\begin{bmatrix}
a & \omega
\end{bmatrix}^{T} \\
f(\mathbf{x}) &=
\begin{bmatrix}
v \cos(\psi) \\
v \sin(\psi) \\
\frac{v}{L} \tan(\delta) \\
0 \\
0
\end{bmatrix}, \quad
B =
\begin{bmatrix}
0 & 0 \\
0 & 0 \\
0 & 0 \\
1 & 0 \\
0 & 1
\end{bmatrix}
\end{aligned}
$$

where `v` is `speed`, `psi` is `yaw`, `delta` is `steering`, `a` is
`acceleration`, `omega` is `steering_rate`, and `L` is the wheelbase. 

Important model parameters

```text
max speed:        9.0 m/s
max steering:     +/-0.60 rad
wheelbase:        1.0 m
vehicle width:    0.65 m
acceleration range:  [-4.5, 2.8] m/s^2
steering_rate range: [-6.0, 6.0] rad/s
```

- The tire model approximates lateral grip from tire slip and limits the
resulting force by the available road friction.

- Track friction decreases slightly as laps progress, so a controller that
is stable early in the run may need extra margin in later laps.

## How To Run

Run the simulator, your controller, and RViz together:
```bash
ros2 launch ros2_racing_challenge racing.launch.py
```

## Keyboard Control


```bash
#Terminal 1
ros2 launch ros2_racing_challenge racing_sim.launch.py

#Terminal 2
ros2 run ros2_racing_challenge keyboard_controller
```

The following keys are available when using `keyboard_controller`.

```text
a      steer left
d      steer right
w      return steering to zero
r      increase target speed
f      decrease target speed
space  maximum brake
x      release brake
q      quit
```

## Evaluation

The simulator reports the result based on the time required to complete 10 laps.

- Collision with an obstacle adds a time penalty.
- Leaving the track accumulates penalty time while the vehicle remains outside the track.
- Friction decreases randomly as laps progress, so the vehicle may become more slippery in later laps.
- The final score is the driving time plus accumulated penalty time.

You can check the lap count, time, penalty, and friction status in the RViz status text.
