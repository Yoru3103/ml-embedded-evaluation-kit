# YOLOv8 integration status

The MLEK-independent YOLOv8 pre-processing and post-processing API is kept in
`source/lib/mlek/use_case/yolov8_detection`.

The application glue is intentionally disabled until a model wrapper and model
resources are added. The copied Alif-specific model entry point and its
four-output application dispatch are not part of this checkout. The generated
`sample_files.h` interface remains the expected input source when the use case
is enabled again.
