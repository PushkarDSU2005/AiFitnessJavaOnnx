package com.example;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.javacpp.BytePointer;
import ai.onnxruntime.*;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Collections;
import java.util.Map;

public class Main {

    private static final String MODEL_PATH = "models/pose.onnx";
    private static final int INPUT_SIZE = 192;
    private static final int NUM_KEYPOINTS = 17;

    private static class RepState {
        public String lastPos = "up";
        public int reps = 0;
        public long lastChangeTime = 0;
    }

    private static RepState repStateHolder = new RepState();

    public static void main(String[] args) {
        System.out.println("AI Fitness Java + ONNX starting...");

        OrtEnvironment env = null;
        OrtSession session = null;
        OpenCVFrameGrabber grabber = null;
        CanvasFrame canvas = null;

        try {
            // Initialize ONNX
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(MODEL_PATH, new OrtSession.SessionOptions());
            System.out.println("✅ Loaded ONNX model: " + MODEL_PATH);

            Map<String, NodeInfo> inputInfo = session.getInputInfo();
            String inputName = session.getInputNames().iterator().next();
            TensorInfo info = (TensorInfo) inputInfo.get(inputName).getInfo();
            System.out.println("📘 Model Input Info: " + info);

            boolean expectsInt = info.type == OnnxJavaType.INT32;
            long[] shape = info.getShape();

            // Camera setup
            grabber = new OpenCVFrameGrabber(0);
            grabber.start();
            int camW = grabber.getImageWidth();
            int camH = grabber.getImageHeight();
            System.out.println("📷 Camera resolution: " + camW + " x " + camH);

            // Display window
            canvas = new CanvasFrame("AI Fitness (Java + ONNX)", CanvasFrame.getDefaultGamma() / grabber.getGamma());
            canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            canvas.setCanvasSize(camW, camH);

            final boolean[] exitFlag = {false};
            canvas.addKeyListener(new KeyAdapter() {
                @Override
                public void keyPressed(KeyEvent e) {
                    if (e.getKeyCode() == KeyEvent.VK_Q) {
                        exitFlag[0] = true;
                        System.out.println("👋 Q pressed — exiting...");
                    }
                }
            });

            Java2DFrameConverter java2dConverter = new Java2DFrameConverter();
            OpenCVFrameConverter.ToMat matConverter = new OpenCVFrameConverter.ToMat();

            System.out.println("🎥 Starting webcam... Press Q to quit.");

            while (true) {
                if (exitFlag[0] || !canvas.isVisible()) break;

                Frame grabbed = grabber.grab();
                if (grabbed == null) continue;

                BufferedImage bimg = java2dConverter.getBufferedImage(grabbed);
                if (bimg == null) continue;

                // Resize to model input
                BufferedImage resized = new BufferedImage(INPUT_SIZE, INPUT_SIZE, BufferedImage.TYPE_3BYTE_BGR);
                resized.getGraphics().drawImage(bimg, 0, 0, INPUT_SIZE, INPUT_SIZE, null);

                OnnxTensor inputTensor;
                if (expectsInt) {
                    int[][][][] input = bufferedImageToNHWCInt4D(resized, INPUT_SIZE, INPUT_SIZE);
                    inputTensor = OnnxTensor.createTensor(env, input);
                } else {
                    float[][][][] input = bufferedImageToNHWCFloat4D(resized, INPUT_SIZE, INPUT_SIZE);
                    inputTensor = OnnxTensor.createTensor(env, input);
                }

                OrtSession.Result results = session.run(Collections.singletonMap(inputName, inputTensor));
                Object outObj = results.get(0).getValue();

                float[][] keypointsArr = null;
                if (outObj instanceof float[][][][]) {
                    float[][][][] raw4 = (float[][][][]) outObj;
                    keypointsArr = raw4[0][0];
                } else if (outObj instanceof float[][][]) {
                    float[][][] raw3 = (float[][][]) outObj;
                    keypointsArr = raw3[0];
                } else if (outObj instanceof float[][]) {
                    keypointsArr = (float[][]) outObj;
                }

                // Map normalized coords to pixel coords
                float[][] keypoints = new float[NUM_KEYPOINTS][3];
                if (keypointsArr != null) {
                    for (int i = 0; i < NUM_KEYPOINTS; i++) {
                        float y = keypointsArr[i][0];
                        float x = keypointsArr[i][1];
                        float score = keypointsArr[i][2];
                        keypoints[i][0] = y * camH;
                        keypoints[i][1] = x * camW;
                        keypoints[i][2] = score;
                    }
                }

                // Rep counting logic
                Float leftElbow = angleAt(keypoints, 5, 7, 9);
                Float rightElbow = angleAt(keypoints, 6, 8, 10);
                float avgElbow = -1f;
                if (leftElbow != null && rightElbow != null)
                    avgElbow = (leftElbow + rightElbow) / 2f;
                else if (leftElbow != null) avgElbow = leftElbow;
                else if (rightElbow != null) avgElbow = rightElbow;

                String feedback;
                if (avgElbow > 0) {
                    String pos = avgElbow < 70 ? "down" : (avgElbow > 140 ? "up" : repStateHolder.lastPos);
                    long now = System.currentTimeMillis();
                    if (repStateHolder.lastPos.equals("down") && pos.equals("up") &&
                        now - repStateHolder.lastChangeTime > 500) {
                        repStateHolder.reps++;
                        repStateHolder.lastChangeTime = now;
                    }
                    repStateHolder.lastPos = pos;
                    feedback = String.format("Elbow: %.1f | Pos: %s | Reps: %d",
                            avgElbow, repStateHolder.lastPos, repStateHolder.reps);
                } else {
                    feedback = "Pose not detected properly.";
                }

                Mat mat = bufferedImageToMat(bimg);
                drawKeypoints(mat, keypoints);
                opencv_imgproc.putText(mat, feedback, new Point(10, 30),
                        opencv_imgproc.FONT_HERSHEY_SIMPLEX, 0.8,
                        new Scalar(0, 255, 0, 0), 2, opencv_imgproc.LINE_AA, false);

                Frame overlayFrame = matConverter.convert(mat);
                canvas.showImage(overlayFrame);

                results.close();
                inputTensor.close();
                Thread.sleep(80);
            }

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try { if (grabber != null) grabber.stop(); } catch (Exception ignore) {}
            try { if (session != null) session.close(); } catch (Exception ignore) {}
            try { if (env != null) env.close(); } catch (Exception ignore) {}
            try { if (canvas != null) canvas.dispose(); } catch (Exception ignore) {}
            System.out.println("🛑 Shutdown complete.");
        }
    }

    // === Convert to [1, H, W, 3] int32 ===
    private static int[][][][] bufferedImageToNHWCInt4D(BufferedImage img, int w, int h) {
        int[][][][] out = new int[1][h][w][3];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int p = img.getRGB(x, y);
                out[0][y][x][0] = (p >> 16) & 0xFF; // R
                out[0][y][x][1] = (p >> 8) & 0xFF;  // G
                out[0][y][x][2] = p & 0xFF;         // B
            }
        }
        return out;
    }

    // === Convert to [1, H, W, 3] float32 ===
    private static float[][][][] bufferedImageToNHWCFloat4D(BufferedImage img, int w, int h) {
        float[][][][] out = new float[1][h][w][3];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int p = img.getRGB(x, y);
                out[0][y][x][0] = ((p >> 16) & 0xFF) / 255.0f;
                out[0][y][x][1] = ((p >> 8) & 0xFF) / 255.0f;
                out[0][y][x][2] = (p & 0xFF) / 255.0f;
            }
        }
        return out;
    }

    private static Float angleAt(float[][] k, int a, int b, int c) {
        try {
            float ay = k[a][0], ax = k[a][1], as = k[a][2];
            float by = k[b][0], bx = k[b][1], bs = k[b][2];
            float cy = k[c][0], cx = k[c][1], cs = k[c][2];
            if (as < 0.2 || bs < 0.2 || cs < 0.2) return null;
            float v1x = ax - bx, v1y = ay - by;
            float v2x = cx - bx, v2y = cy - by;
            float dot = v1x * v2x + v1y * v2y;
            float mag1 = (float) Math.sqrt(v1x * v1x + v1y * v1y);
            float mag2 = (float) Math.sqrt(v2x * v2x + v2y * v2y);
            if (mag1 * mag2 == 0) return null;
            float cos = dot / (mag1 * mag2);
            cos = Math.max(-1f, Math.min(1f, cos));
            return (float) Math.toDegrees(Math.acos(cos));
        } catch (Exception e) {
            return null;
        }
    }

    private static void drawKeypoints(Mat mat, float[][] keypoints) {
        for (float[] kp : keypoints) {
            if (kp[2] > 0.2) {
                int x = Math.round(kp[1]);
                int y = Math.round(kp[0]);
                opencv_imgproc.circle(mat, new Point(x, y), 4,
                        new Scalar(0, 0, 255, 0), -1, opencv_imgproc.LINE_AA, 0);
            }
        }
    }

    private static Mat bufferedImageToMat(BufferedImage bi) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(bi, "png", baos);
        byte[] bytes = baos.toByteArray();
        return opencv_imgcodecs.imdecode(new Mat(new BytePointer(bytes)), opencv_imgcodecs.IMREAD_COLOR);
    }
}
