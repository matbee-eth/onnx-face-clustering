import { Mat, Scalar, circle, LINE_AA, putText, FONT_HERSHEY_SIMPLEX } from "@techstark/opencv-js";
import { Point } from "../shared/Point";

export function drawLandmarks(src: Mat, landmarks: Point[], score: number) {
  // Convert the canvas to a Mat
  const colors = [
    new Scalar(255, 0, 0), // red
    new Scalar(0, 255, 0), // green
    new Scalar(0, 0, 255), // blue
    new Scalar(255, 255, 0), // yellow
    new Scalar(128, 0, 128), // purple
  ];
  const map = ["leftEye", "rightEye", "nose", "leftMouth", "rightMouth"];

  for (let i = 0; i < landmarks.length; i++) {
    const landmark = landmarks[i];
    const color = colors[i % colors.length];
    // Draw circle for the landmark
    let center = new Point(landmark.x, landmark.y);
    console.log("landmark", landmark, center);
    circle(src, center, 3, color, -1, LINE_AA, 0);

    // Put text for the landmark
    let text = `Score: ${score.toFixed(2)}, Landmark: ${map[i]}`;
    let bottomLeftCornerOfText = new Point(landmark.x, landmark.y);
    putText(src, text, bottomLeftCornerOfText, FONT_HERSHEY_SIMPLEX, 0.4, color, 1, LINE_AA);
  }
}