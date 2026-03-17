"""YOLO-based temperature change detection.

This script uses YOLO (Ultralytics) tracking to follow detected objects across
video frames and estimates each object's temperature from thermal intensity.
When an object's temperature changes beyond a threshold, an alert is emitted.

Requirements:
    pip install ultralytics opencv-python numpy

Notes:
- For true thermal cameras, adapt `pixel_to_temperature` to your sensor's
  calibration model.
- For pseudo-thermal imagery, this script linearly maps pixel intensity to a
  temperature range.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TempState:
    """Stores running temperature data for one tracked object."""

    baseline_temp: float
    last_temp: float
    last_alert_time: float = 0.0


class YoloTemperatureChangeDetector:
    """Track objects with YOLO and detect significant temperature changes."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        change_threshold_c: float = 2.0,
        min_temp_c: float = 20.0,
        max_temp_c: float = 45.0,
        alert_cooldown_s: float = 2.0,
    ) -> None:
        self.model = YOLO(model_path)
        self.change_threshold_c = change_threshold_c
        self.min_temp_c = min_temp_c
        self.max_temp_c = max_temp_c
        self.alert_cooldown_s = alert_cooldown_s
        self.states: Dict[int, TempState] = {}

    def pixel_to_temperature(self, thermal_roi: np.ndarray) -> float:
        """Convert thermal ROI pixel intensities into temperature estimate.

        Assumes intensity 0..255 maps linearly to [min_temp_c, max_temp_c].
        Replace this with your thermal sensor's calibration for production.
        """
        if thermal_roi.size == 0:
            return self.min_temp_c

        mean_intensity = float(np.mean(thermal_roi))
        normalized = np.clip(mean_intensity / 255.0, 0.0, 1.0)
        return self.min_temp_c + normalized * (self.max_temp_c - self.min_temp_c)

    def _extract_roi(
        self,
        frame_gray: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = box
        h, w = frame_gray.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0), dtype=frame_gray.dtype)
        return frame_gray[y1:y2, x1:x2]

    def process_video(
        self,
        source: Union[str, int],
        show: bool = True,
        output_path: Optional[str] = None,
        classes: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Run tracking and temperature change detection on a video source."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.model.track(
                source=frame,
                persist=True,
                verbose=False,
                classes=list(classes) if classes else None,
            )

            now = time.time()
            if results and len(results) > 0:
                res = results[0]
                boxes = res.boxes

                if boxes is not None and boxes.xyxy is not None and boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = boxes.id.cpu().numpy().astype(int)

                    for box, track_id in zip(xyxy, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        roi = self._extract_roi(gray, (x1, y1, x2, y2))
                        temp_c = self.pixel_to_temperature(roi)

                        state = self.states.get(track_id)
                        if state is None:
                            state = TempState(baseline_temp=temp_c, last_temp=temp_c)
                            self.states[track_id] = state

                        delta = temp_c - state.baseline_temp
                        abs_delta = abs(delta)
                        should_alert = (
                            abs_delta >= self.change_threshold_c
                            and now - state.last_alert_time >= self.alert_cooldown_s
                        )

                        color = (0, 255, 0)
                        label = f"ID {track_id} | {temp_c:.1f}C | Δ{delta:+.1f}C"

                        if should_alert:
                            color = (0, 0, 255)
                            state.last_alert_time = now
                            print(
                                f"[ALERT] Track {track_id}: temperature changed by "
                                f"{delta:+.2f}C (current={temp_c:.2f}C, "
                                f"baseline={state.baseline_temp:.2f}C)"
                            )

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                        state.last_temp = temp_c

            if writer is not None:
                writer.write(frame)

            if show:
                cv2.imshow("YOLO Temperature Change Detector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect object temperature changes using YOLO tracking."
    )
    parser.add_argument("--source", required=True, help="Video source path or camera index")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Temperature change threshold in Celsius for alerting",
    )
    parser.add_argument("--min-temp", type=float, default=20.0, help="Min mapped temp (C)")
    parser.add_argument("--max-temp", type=float, default=45.0, help="Max mapped temp (C)")
    parser.add_argument("--output", default=None, help="Optional output video path")
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=None,
        help="Optional class IDs to track (e.g. --classes 0 for person)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable live display window (useful for headless runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source: Union[str, int]
    if args.source.isdigit():
        source = int(args.source)
    else:
        source_path = Path(args.source)
        source = str(source_path)

    detector = YoloTemperatureChangeDetector(
        model_path=args.model,
        change_threshold_c=args.threshold,
        min_temp_c=args.min_temp,
        max_temp_c=args.max_temp,
    )

    detector.process_video(
        source=source,
        show=not args.no_show,
        output_path=args.output,
        classes=tuple(args.classes) if args.classes else None,
    )


if __name__ == "__main__":
    main()
