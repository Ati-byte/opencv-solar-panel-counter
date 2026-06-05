from __future__ import annotations

import csv
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import PanelDetection, SolarPanelDetector


@dataclass(frozen=True)
class TrackedPanel:
    panel_id: int
    x: float
    y: float
    first_frame: int
    last_frame: int
    hits: int
    confirmed: bool


@dataclass(frozen=True)
class TrackingFrameResult:
    frame_index: int
    timestamp_seconds: float
    visible_count: int
    unique_count: int
    matched_count: int
    new_count: int


class ActivePanelTracker:
    def __init__(
        self,
        match_distance: float = 65.0,
        max_missing_frames: int = 4,
        min_confirmed_hits: int = 3,
    ) -> None:
        self.match_distance = match_distance
        self.max_missing_frames = max_missing_frames
        self.min_confirmed_hits = min_confirmed_hits
        self._next_panel_id = 1
        self._panels: list[TrackedPanel] = []
        self._confirmed_count = 0

    @property
    def unique_count(self) -> int:
        return self._confirmed_count

    def update(
        self,
        frame_index: int,
        detections: list[PanelDetection],
        previous_to_current: np.ndarray,
    ) -> tuple[list[tuple[PanelDetection, int]], int, int]:
        self._predict_active_positions(previous_to_current)

        detection_centers = [detection_center(detection) for detection in detections]
        matched_pairs = self._match_detections(detection_centers)
        matched_detection_indices = {detection_index for detection_index, _ in matched_pairs}
        assignments: list[tuple[PanelDetection, int]] = []
        new_count = 0

        for detection_index, panel_index in matched_pairs:
            center = detection_centers[detection_index]
            panel = self._panels[panel_index]
            hits = panel.hits + 1
            confirmed = panel.confirmed or hits >= self.min_confirmed_hits
            if confirmed and not panel.confirmed:
                new_count += 1
                self._confirmed_count += 1

            updated_panel = TrackedPanel(
                panel_id=panel.panel_id,
                x=(panel.x * panel.hits + float(center[0])) / hits,
                y=(panel.y * panel.hits + float(center[1])) / hits,
                first_frame=panel.first_frame,
                last_frame=frame_index,
                hits=hits,
                confirmed=confirmed,
            )
            self._panels[panel_index] = updated_panel
            if confirmed:
                assignments.append((detections[detection_index], panel.panel_id))

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue

            center = detection_centers[detection_index]
            confirmed = self.min_confirmed_hits <= 1
            panel = TrackedPanel(
                panel_id=self._next_panel_id,
                x=float(center[0]),
                y=float(center[1]),
                first_frame=frame_index,
                last_frame=frame_index,
                hits=1,
                confirmed=confirmed,
            )
            self._panels.append(panel)
            self._next_panel_id += 1
            if confirmed:
                self._confirmed_count += 1
                new_count += 1
                assignments.append((detection, panel.panel_id))

        self._panels = [
            panel
            for panel in self._panels
            if frame_index - panel.last_frame <= self.max_missing_frames
        ]
        return assignments, len(matched_pairs), new_count

    def _predict_active_positions(self, previous_to_current: np.ndarray) -> None:
        predicted: list[TrackedPanel] = []
        for panel in self._panels:
            point = transform_point(np.array([panel.x, panel.y], dtype=np.float32), previous_to_current)
            predicted.append(
                TrackedPanel(
                    panel_id=panel.panel_id,
                    x=float(point[0]),
                    y=float(point[1]),
                    first_frame=panel.first_frame,
                    last_frame=panel.last_frame,
                    hits=panel.hits,
                    confirmed=panel.confirmed,
                )
            )
        self._panels = predicted

    def _match_detections(self, centers: list[np.ndarray]) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for detection_index, center in enumerate(centers):
            for panel_index, panel in enumerate(self._panels):
                distance = float(np.linalg.norm(center - np.array([panel.x, panel.y], dtype=np.float32)))
                if distance <= self.match_distance:
                    candidates.append((distance, detection_index, panel_index))

        matched_detections: set[int] = set()
        matched_panels: set[int] = set()
        pairs: list[tuple[int, int]] = []

        for _, detection_index, panel_index in sorted(candidates, key=lambda item: item[0]):
            if detection_index in matched_detections or panel_index in matched_panels:
                continue
            matched_detections.add(detection_index)
            matched_panels.add(panel_index)
            pairs.append((detection_index, panel_index))

        return pairs


class CameraMotionEstimator:
    def __init__(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=2500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._previous_gray: np.ndarray | None = None
        self._previous_keypoints: tuple[cv2.KeyPoint, ...] | None = None
        self._previous_descriptors: np.ndarray | None = None
        self._previous_to_current = np.eye(3, dtype=np.float32)

    @property
    def previous_to_current(self) -> np.ndarray:
        return self._previous_to_current.copy()

    def update(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)

        if self._previous_gray is None or descriptors is None or self._previous_descriptors is None:
            self._set_previous(gray, keypoints, descriptors)
            self._previous_to_current = np.eye(3, dtype=np.float32)
            return self.previous_to_current

        current_to_previous = self._estimate_current_to_previous(
            keypoints,
            descriptors,
            self._previous_keypoints or (),
            self._previous_descriptors,
        )

        if current_to_previous is not None:
            self._previous_to_current = np.linalg.inv(current_to_previous).astype(np.float32)
        else:
            self._previous_to_current = np.eye(3, dtype=np.float32)

        self._set_previous(gray, keypoints, descriptors)
        return self.previous_to_current

    def _set_previous(
        self,
        gray: np.ndarray,
        keypoints: tuple[cv2.KeyPoint, ...],
        descriptors: np.ndarray | None,
    ) -> None:
        self._previous_gray = gray
        self._previous_keypoints = keypoints
        self._previous_descriptors = descriptors

    def _estimate_current_to_previous(
        self,
        current_keypoints: tuple[cv2.KeyPoint, ...],
        current_descriptors: np.ndarray,
        previous_keypoints: tuple[cv2.KeyPoint, ...],
        previous_descriptors: np.ndarray,
    ) -> np.ndarray | None:
        matches = self._matcher.match(current_descriptors, previous_descriptors)
        if len(matches) < 12:
            return None

        matches = sorted(matches, key=lambda item: item.distance)[:160]
        current_points = np.float32([current_keypoints[match.queryIdx].pt for match in matches])
        previous_points = np.float32([previous_keypoints[match.trainIdx].pt for match in matches])

        affine, inliers = cv2.estimateAffinePartial2D(
            current_points,
            previous_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=2000,
            confidence=0.99,
        )

        if affine is None or inliers is None or int(inliers.sum()) < 10:
            return None

        matrix = np.eye(3, dtype=np.float32)
        matrix[:2, :] = affine.astype(np.float32)
        return matrix


def process_video_unique(
    video_path: Path,
    output_dir: Path,
    detector: SolarPanelDetector,
    frame_step: int = 10,
    match_distance: float = 65.0,
    write_video: bool = True,
) -> list[TrackingFrameResult]:
    if frame_step < 1:
        raise ValueError("frame_step must be 1 or greater.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = None
    temp_video_path: Path | None = None

    with tempfile.TemporaryDirectory() as temp_dir:
        if write_video:
            temp_video_path = Path(temp_dir) / "tracked_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps / frame_step, (width, height))

        motion = CameraMotionEstimator()
        # A panel must be visible for ~1 s to be confirmed; scale with frame_step and fps.
        min_confirmed_hits = max(2, round(fps / frame_step))
        tracker = ActivePanelTracker(
            match_distance=match_distance,
            max_missing_frames=frame_step * 4,
            min_confirmed_hits=min_confirmed_hits,
        )
        results: list[TrackingFrameResult] = []
        frame_index = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % frame_step == 0:
                previous_to_current = motion.update(frame)
                detections = detector.detect(frame)
                assignments, matched_count, new_count = tracker.update(frame_index, detections, previous_to_current)
                timestamp = frame_index / fps

                results.append(
                    TrackingFrameResult(
                        frame_index=frame_index,
                        timestamp_seconds=round(timestamp, 3),
                        visible_count=len(detections),
                        unique_count=tracker.unique_count,
                        matched_count=matched_count,
                        new_count=new_count,
                    )
                )

                if video_writer is not None:
                    video_writer.write(annotate_tracked_frame(frame, assignments, tracker.unique_count))

            frame_index += 1

        capture.release()
        if video_writer is not None:
            video_writer.release()

        if results and results[-1].unique_count > 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_tracking_csv(results, output_dir / "unique_panel_counts.csv")
            if temp_video_path is not None and temp_video_path.exists():
                shutil.move(str(temp_video_path), str(output_dir / "tracked_video.mp4"))

        return results


def annotate_tracked_frame(
    frame: np.ndarray,
    assignments: list[tuple[PanelDetection, int]],
    unique_count: int,
) -> np.ndarray:
    output = frame.copy()
    for detection, panel_id in assignments:
        x, y, _, _ = detection.box
        box_points = np.array(detection.points, dtype=np.int32)
        cv2.polylines(output, [box_points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(
            output,
            f"ID {panel_id}",
            (x + 4, max(y + 20, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(output, (10, 10), (390, 52), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"Total unique panels: {unique_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def detection_center(detection: PanelDetection) -> np.ndarray:
    points = np.array(detection.points, dtype=np.float32)
    return points.mean(axis=0)


def transform_point(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float32)
    transformed = matrix @ homogeneous
    if transformed[2] == 0:
        return transformed[:2]
    return transformed[:2] / transformed[2]


def write_tracking_csv(results: list[TrackingFrameResult], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frame_index",
                "timestamp_seconds",
                "visible_count",
                "unique_count",
                "matched_count",
                "new_count",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.frame_index,
                    result.timestamp_seconds,
                    result.visible_count,
                    result.unique_count,
                    result.matched_count,
                    result.new_count,
                ]
            )
