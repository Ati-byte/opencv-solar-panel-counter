from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2

from .detector import SolarPanelDetector


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    timestamp_seconds: float
    panel_count: int


def process_video(
    video_path: Path,
    output_dir: Path,
    detector: SolarPanelDetector,
    frame_step: int = 15,
    write_video: bool = True,
) -> list[FrameResult]:
    if frame_step < 1:
        raise ValueError("frame_step must be 1 or greater.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = None

    if write_video:
        output_video_path = output_dir / "annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps / frame_step, (width, height))

    results: list[FrameResult] = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_step == 0:
            detections = detector.detect(frame)
            annotated_frame = detector.annotate(frame, detections)
            timestamp = frame_index / fps
            results.append(
                FrameResult(
                    frame_index=frame_index,
                    timestamp_seconds=round(timestamp, 3),
                    panel_count=len(detections),
                )
            )

            if video_writer is not None:
                video_writer.write(annotated_frame)

        frame_index += 1

    capture.release()
    if video_writer is not None:
        video_writer.release()

    write_results_csv(results, output_dir / "panel_counts.csv")
    return results


def write_results_csv(results: list[FrameResult], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", "timestamp_seconds", "panel_count"])
        for result in results:
            writer.writerow([result.frame_index, result.timestamp_seconds, result.panel_count])
