# Windows camera and Corstone-320 VSI demonstrations

Start the Windows `vsi_video_server.py` preview server on TCP port 6000, then run
one launcher from the MLEK root. Each launcher starts an already-built U85-1024
firmware with the board button panel and HDLCD enabled. Press KEY0 to capture
and infer one frame; the result remains on the LCD until the next KEY0 press.

| Demonstration | Command | Default firmware |
| --- | --- | --- |
| Image classification | `bash scripts/vsi/run_img_class_fvp320.sh` | `build-img-class-vsi/bin/mlek_img_class.axf` |
| 10-class gesture detection | `bash scripts/vsi/run_gesture_fvp320.sh` | `build-img-class-vsi/bin/mlek_yolov8_detection.axf` |
| 80-class COCO detection | `bash scripts/vsi/run_coco_fvp320.sh` | `build-coco-vsi/bin/mlek_yolov8_detection.axf` |

The two YOLO executables have the same filename in different build directories;
the launchers select the corresponding model. Run one demonstration at a time.
Exit and restart the Windows server before reconnecting another FVP session,
since the current VSI client closes the server connection on exit.

All launchers accept these environment overrides:

| Variable | Purpose |
| --- | --- |
| `VSI_VIDEO_HOST` | Windows server IP; defaults to the IPv4 gateway for WSL NAT |
| `VSI_PYTHON_PATH` | Directory containing `arm_vsi4.py` and the modified `vsi_video.py` |
| `FVP_PATH` | Corstone-320 FVP executable |
| `IMG_CLASS_BUILD_DIR` | Image-classification firmware build directory |
| `GESTURE_BUILD_DIR` | Gesture firmware build directory |
| `COCO_BUILD_DIR` | COCO firmware build directory |

With mirrored WSL networking, explicitly set `VSI_VIDEO_HOST=127.0.0.1`.
Additional command-line arguments are forwarded to the FVP, for example
`bash scripts/vsi/run_gesture_fvp320.sh --stat`.
