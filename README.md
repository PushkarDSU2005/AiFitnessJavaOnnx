# AI Fitness Java + ONNX

Real-time fitness rep counting with Java, ONNX Runtime, and JavaCV/OpenCV.

This app reads webcam frames, runs pose estimation using `models/pose.onnx`, draws keypoints, and counts reps based on elbow-angle transitions.

## Features

- Live webcam pose tracking
- ONNX Runtime inference from Java
- Auto handling for both `int32` and `float32` NHWC model inputs
- Keypoint overlay and on-screen feedback text
- Rep counting with debounce protection
- Quit shortcut: press `Q`

## Tech Stack

- Java 17
- Maven
- JavaCV (`org.bytedeco:javacv-platform:1.5.8`)
- ONNX Runtime (`com.microsoft.onnxruntime:onnxruntime:1.15.0`)

## Project Structure

```text
AiFitnessJavaOnnx/
|-- models/
|   `-- pose.onnx
|-- src/main/java/com/example/
|   `-- Main.java
|-- pom.xml
`-- README.md
```

## Prerequisites

- JDK 17 or newer
- Maven 3.8 or newer
- Working webcam
- Pose model file at `models/pose.onnx`

## Quick Start

```bash
git clone https://github.com/PushkarDSU2005/AiFitnessJavaOnnx.git
cd AiFitnessJavaOnnx
mvn clean compile
mvn exec:java
```

When the app starts, a camera window opens with keypoints and rep stats.

## Controls

- `Q` -> quit application
- Closing the camera window also stops execution

## How Rep Counting Works

- Left elbow angle is computed from keypoints `(5, 7, 9)`
- Right elbow angle is computed from keypoints `(6, 8, 10)`
- Average angle is used when both arms are visible
- Position logic:
- angle `< 70` -> `down`
- angle `> 140` -> `up`
- Rep is counted on `down -> up` transition
- Cooldown (`500 ms`) prevents double counting

## Model Contract

- Input size: `192 x 192`
- Keypoints expected: `17`
- Supported input tensors:
- `int32` with shape `[1, H, W, 3]`
- `float32` with shape `[1, H, W, 3]` normalized to `[0, 1]`
- Supported output parsing:
- `float[][][][]`
- `float[][][]`
- `float[][]`
- Keypoint format is treated as `[y, x, score]`
- Confidence threshold for drawing and angle logic: `0.2`

## Troubleshooting

- `Model load error`
- Verify `models/pose.onnx` exists and is readable
- `Camera not opening`
- Check if another app is using webcam
- `Native library or linkage error`
- Re-run `mvn clean compile` and confirm Java 17 is active
- `Pose quality is unstable`
- Improve lighting and keep full upper body in frame

## Notes

- `target/` is ignored by `.gitignore`
- `session_output.mp4` is ignored and not required for runtime

## Roadmap

- Add multi-exercise modes (push-ups, squats, curls)
- Add user calibration for better angle thresholds
- Save session analytics (JSON/CSV)
- Add unit tests for angle and state transitions
- Package executable distribution

## License

No `LICENSE` file is currently present.
Add one if you want explicit reuse terms.
