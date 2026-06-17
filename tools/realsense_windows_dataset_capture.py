#!/usr/bin/env python3
from pathlib import Path
import glob

import cv2
import numpy as np
import pyrealsense2 as rs

# pip install pyrealsense2 opencv-python numpy

# cd $env:USERPROFILE\Desktop\realsense
# python .\realsense_windows_dataset_capture.py

# Change only these values while tuning on Windows.
WIDTH = 640
HEIGHT = 480
FPS = 15

AUTO_EXPOSURE = False
EXPOSURE = 155
AUTO_WHITE_BALANCE = False
WHITE_BALANCE = 4200.0

SATURATION = 55
CONTRAST = 55
GAMMA = 300
SHARPNESS = 30
GAIN = 64

SAVE_DIR = Path(__file__).resolve().parent / "dataset" / "images" / "train"


def set_option_if_supported(sensor, option, value):
    if sensor is not None and sensor.supports(option):
        sensor.set_option(option, value)


def find_color_sensor(profile):
    device = profile.get_device()
    for sensor in device.query_sensors():
        for stream_profile in sensor.get_stream_profiles():
            if stream_profile.stream_type() == rs.stream.color:
                return sensor
    return None


def next_image_index(save_dir):
    existing = sorted(glob.glob(str(save_dir / "img_*.jpg")))
    if not existing:
        return 0
    return int(Path(existing[-1]).stem.split("_")[1]) + 1


def apply_camera_settings(profile):
    color_sensor = find_color_sensor(profile)

    set_option_if_supported(color_sensor, rs.option.enable_auto_exposure, 1.0 if AUTO_EXPOSURE else 0.0)
    if not AUTO_EXPOSURE:
        set_option_if_supported(color_sensor, rs.option.exposure, float(EXPOSURE))

    set_option_if_supported(color_sensor, rs.option.enable_auto_white_balance, 1.0 if AUTO_WHITE_BALANCE else 0.0)
    if not AUTO_WHITE_BALANCE:
        set_option_if_supported(color_sensor, rs.option.white_balance, float(WHITE_BALANCE))

    set_option_if_supported(color_sensor, rs.option.saturation, float(SATURATION))
    set_option_if_supported(color_sensor, rs.option.contrast, float(CONTRAST))
    set_option_if_supported(color_sensor, rs.option.gamma, float(GAMMA))
    set_option_if_supported(color_sensor, rs.option.sharpness, float(SHARPNESS))
    set_option_if_supported(color_sensor, rs.option.gain, float(GAIN))


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    image_count = next_image_index(SAVE_DIR)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)

    profile = pipeline.start(config)
    apply_camera_settings(profile)

    print(f"Save directory: {SAVE_DIR}")
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

            cv2.imshow("ROI Viewer", bgr)
            key = cv2.waitKey(10)
            if key == 27:
                break
            if key == 32:
                filename = SAVE_DIR / f"img_{image_count:04}.jpg"
                cv2.imwrite(str(filename), bgr)
                print(f"Saved: {filename}")
                image_count += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
