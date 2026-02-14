# AI Fitness Java ONNX

Real-time fitness rep counting from webcam video using Java, ONNX Runtime, and JavaCV/OpenCV.

## Overview

This project runs a pose-estimation ONNX model on live webcam frames, draws detected keypoints, and counts upper-body reps based on elbow angle transitions.

Current implementation targets a push-up style motion:
- `down` when average elbow angle is below 70 degrees
- `up` when average elbow angle is above 140 degrees
- rep increments on a `down -> up` transition with a 500 ms cooldown

## Features

- Real-time webcam capture and display
- ONNX Runtime inference in Java
- Automatic support for model input tensor type:
- `int32` NHWC input (`[1, H, W, 3]`)
- `float32` NHWC normalized input (`[1, H, W, 3]`)
- Keypoint overlay on video frames
- Live feedback text:
- elbow angle
- pose state (`up`/`down`)
- rep count
- Quit key binding: press `Q`

## Tech Stack

- Java 17
- Maven
- JavaCV (`org.bytedeco:javacv-platform`)
- ONNX Runtime Java (`com.microsoft.onnxruntime:onnxruntime`)

## Repository Structure

```text
AiFitnessJavaOnnx/
|-- models/
|   `-- pose.onnx
|-- src/main/java/com/example/
|   `-- Main.java
|-- pom.xml
`-- README.md
```

## Requirements

- JDK 17+
- Maven 3.8+
- Webcam connected/available
- ONNX model at `models/pose.onnx`

## Setup

1. Clone the repository:

```bash
git clone https://github.com/PushkarDSU2005/AiFitnessJavaOnnx.git
cd AiFitnessJavaOnnx
```

2. Build the project:

```bash
mvn clean compile
```

## Run

Run with Maven Exec plugin:

```bash
mvn exec:java
```

On launch, a webcam window opens with keypoints and rep feedback.
Press `Q` in the window to stop.

## Model Input and Output Assumptions

Configured in `src/main/java/com/example/Main.java`:

- Model path: `models/pose.onnx`
- Input image size: `192 x 192`
- Expected keypoints: `17`

Inference output parser currently handles these output shapes:
- `float[][][][]`
- `float[][][]`
- `float[][]`

Each keypoint is interpreted as:
- `[y, x, score]` with normalized coordinates mapped to camera pixel space

## Rep Counting Logic

Elbow angle is computed using:
- left arm: keypoints `(5, 7, 9)` -> shoulder, elbow, wrist
- right arm: keypoints `(6, 8, 10)` -> shoulder, elbow, wrist

State machine:
- if angle < 70 -> `down`
- if angle > 140 -> `up`
- count rep when state changes `down -> up` and at least 500 ms passed since last change

Confidence handling:
- keypoint score must be greater than `0.2` for angle calculation and drawing

## Troubleshooting

- `Model file not found`:
- verify `models/pose.onnx` exists and path is correct
- `No webcam / camera busy`:
- close apps that are already using camera and rerun
- `UnsatisfiedLinkError` or native load issues:
- ensure Java version matches environment and rerun Maven dependency download
- poor detection quality:
- improve lighting, adjust camera angle, and keep full upper body visible

## Notes

- `target/` build output is ignored via `.gitignore`.
- `session_output.mp4` is ignored in git and not required to run the app.

## Future Improvements

- Add support for multiple exercise types (squats, curls, shoulder press)
- Add calibration per user body proportions
- Export session stats (JSON/CSV)
- Add unit tests for angle/state logic
- Add packaged runnable JAR instructions

## License

No license file is currently included. Add a `LICENSE` file if you want to define reuse terms for this repository.
