# YOLOv8 integration

The MLEK-independent YOLOv8 pre-processing and post-processing API is kept in
`source/lib/mlek/use_case/yolov8_detection`.

The application supports both TFLM/TFLite and ExecuTorch. It accepts a
single-output YOLOv8 export ([1, 4 + classes, boxes] or its transposed form)
and the three/four-output DFL exports used by the Ethos-U pipeline.

The model, input image directory/file and labels are supplied through CMake
options. Labels can be an Ultralytics YAML file or a text file containing one
class name per line. The configured class count must match the label count.

With `FVP_VSI_ENABLED=ON`, `yolov8_detection_CAMERA_INPUT` defaults to `ON` and
frames come from the camera HAL instead of the generated samples. On MPS4,
`yolov8_detection_BUTTON_CAPTURE` also defaults to `ON`: press user button 0
(KEY0) to capture one RGB frame, run detection, and display the image, boxes and
labels. The result remains on the LCD until the next KEY0 press. A held button
must be released before another press is accepted; KEY1 is ignored.

Start the Windows VSI video server before the FVP and use the same `v_path` and
`VSI_VIDEO_HOST` configuration as the image-classification camera demonstration.
Enable the FVP board visualization to access its buttons, and enable the HDLCD
visualization to see the result. The default gesture model requests 320x320 RGB888
frames (307200 bytes); the camera server derives the resolution from the model.
Configure `yolov8_detection_OBB=OFF` for this ordinary gesture detector; rotated
QR-code models use their own OBB configuration.

Set `yolov8_detection_CAMERA_INPUT=OFF` and
`yolov8_detection_BUTTON_CAPTURE=OFF` to retain batch inference over generated
samples. Camera input with button capture disabled requests frames continuously.
