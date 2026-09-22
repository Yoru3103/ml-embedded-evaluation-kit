# YOLOv8 integration

The MLEK-independent YOLOv8 pre-processing and post-processing API is kept in
`source/lib/mlek/use_case/yolov8_detection`.

The application supports both TFLM/TFLite and ExecuTorch. It accepts a
single-output YOLOv8 export ([1, 4 + classes, boxes] or its transposed form)
and the three/four-output DFL exports used by the Ethos-U pipeline.

The model, input image directory/file and labels are supplied through CMake
options. Labels can be an Ultralytics YAML file or a text file containing one
class name per line. The configured class count must match the label count.
