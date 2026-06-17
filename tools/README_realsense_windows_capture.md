# RealSense Windows Dataset Capture

This captures YOLO images on a Windows laptop without ROS. It opens only the color stream.

Install dependencies in PowerShell:

```powershell
pip install -r requirements_realsense_windows.txt
```

Run from this directory:

```powershell
python .\realsense_windows_dataset_capture.py
```

Default output:

```text
%USERPROFILE%\Desktop\dataset\images\train
```

The script mirrors the ROS path used by `image_saver_roi_keypress.py`:

```text
RealSense RGB8 frame -> OpenCV BGR frame -> ROI 0,0,640,480 -> img_NNNN.jpg
```

Parameter source order:

```text
color-related rs_launch.py defaults and hard-coded overrides
optional YAML passed with --config overrides those values
```

If you edit color camera values in `rs_launch.py`, the Windows script follows those launch defaults and hard-coded overrides by default. Depth, gyro, and accel are intentionally ignored for this capture path.

If you want one explicit config file for both Ubuntu ROS and Windows capture, edit:

```text
src/my_cv/config/realsense_dataset_params.yaml
```

Run Windows capture with that YAML:

```powershell
python .\realsense_windows_dataset_capture.py --config ..\config\realsense_dataset_params.yaml
```

Use the same file with ROS:

```bash
ros2 launch realsense2_camera rs_launch.py config_file:=/home/noh/my_cv/src/my_cv/config/realsense_dataset_params.yaml
```
