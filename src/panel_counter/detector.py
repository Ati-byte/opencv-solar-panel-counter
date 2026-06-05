from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    lower_hsv: tuple[int, int, int] = (90, 70, 30)
    upper_hsv: tuple[int, int, int] = (130, 255, 245)
    min_area_ratio: float = 0.004
    max_area_ratio: float = 0.22
    min_width: int = 22
    min_height: int = 28
    min_fill_ratio: float = 0.30
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 2.50
    nms_iou_threshold: float = 0.20
    include_edge_panels: bool = True
    merge_cell_groups: bool = True
    max_cell_area_ratio: float = 0.003
    min_cell_group_size: int = 4


@dataclass(frozen=True)
class PanelDetection:
    x: int
    y: int
    width: int
    height: int
    score: float
    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


class SolarPanelDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    def detect(self, image: np.ndarray) -> list[PanelDetection]:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        mask = self._build_panel_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = image.shape[0] * image.shape[1]
        detections = [
            detection
            for contour in contours
            if (detection := self._contour_to_detection(contour, image.shape, image_area)) is not None
        ]

        detections = self._non_maximum_suppression(detections)
        if self.config.merge_cell_groups:
            detections = self._merge_cell_groups(detections, image.shape)
        return sorted(detections, key=lambda item: (item.y, item.x))

    def annotate(self, image: np.ndarray, detections: Iterable[PanelDetection]) -> np.ndarray:
        output = image.copy()
        detections = list(detections)

        for index, detection in enumerate(detections, start=1):
            x, y, _, _ = detection.box
            box_points = np.array(detection.points, dtype=np.int32)
            cv2.polylines(output, [box_points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(
                output,
                str(index),
                (x + 5, max(y + 20, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        label = f"Panel count: {len(detections)}"
        cv2.rectangle(output, (10, 10), (270, 48), (0, 0, 0), -1)
        cv2.putText(output, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return output

    def _build_panel_mask(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.config.lower_hsv), np.array(self.config.upper_hsv))

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        return mask

    def _contour_to_detection(
        self,
        contour: np.ndarray,
        image_shape: tuple[int, int, int],
        image_area: int,
    ) -> PanelDetection | None:
        x, y, width, height = cv2.boundingRect(contour)
        box_area = width * height

        if box_area == 0:
            return None

        config = self.config
        contour_area = cv2.contourArea(contour)
        fill_ratio = contour_area / box_area
        aspect_ratio = width / height
        area_ratio = box_area / image_area

        if width < config.min_width or height < config.min_height:
            return None
        if not config.min_area_ratio <= area_ratio <= config.max_area_ratio:
            return None
        if fill_ratio < config.min_fill_ratio:
            return None
        if not config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio:
            return None
        if not config.include_edge_panels and self._touches_edge(x, y, width, height, image_shape):
            return None

        rectangularity_score = min(fill_ratio, 1.0)
        size_score = min(area_ratio / 0.03, 1.0)
        score = round((0.75 * rectangularity_score) + (0.25 * size_score), 3)
        rotated_rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rotated_rect).astype(int)
        points = tuple((int(point[0]), int(point[1])) for point in box_points)
        return PanelDetection(x=x, y=y, width=width, height=height, score=score, points=points)

    def _touches_edge(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        image_shape: tuple[int, int, int],
    ) -> bool:
        image_height, image_width = image_shape[:2]
        return x <= 1 or y <= 1 or x + width >= image_width - 1 or y + height >= image_height - 1

    def _non_maximum_suppression(self, detections: list[PanelDetection]) -> list[PanelDetection]:
        selected: list[PanelDetection] = []

        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            if all(self._iou(detection, other) < self.config.nms_iou_threshold for other in selected):
                selected.append(detection)

        return selected

    def _merge_cell_groups(
        self,
        detections: list[PanelDetection],
        image_shape: tuple[int, int, int],
    ) -> list[PanelDetection]:
        image_height, image_width = image_shape[:2]
        image_area = image_height * image_width
        small_detections = [
            detection
            for detection in detections
            if (detection.width * detection.height) / image_area <= self.config.max_cell_area_ratio
        ]

        if len(small_detections) < self.config.min_cell_group_size:
            return detections

        large_detections = [detection for detection in detections if detection not in small_detections]
        grouping_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        for detection in small_detections:
            x, y, width, height = detection.box
            cv2.rectangle(grouping_mask, (x, y), (x + width, y + height), 255, -1)

        kernel_size = max(5, int(min(image_height, image_width) * 0.03))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        grouping_mask = cv2.dilate(grouping_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(grouping_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        merged: list[PanelDetection] = []
        used_ids: set[int] = set()

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            group = [
                detection
                for detection in small_detections
                if id(detection) not in used_ids and self._center_inside_box(detection, x, y, width, height)
            ]

            if len(group) < self.config.min_cell_group_size:
                continue

            for detection in group:
                used_ids.add(id(detection))
            merged.append(self._merge_detections(group, image_area))

        unused_small = [detection for detection in small_detections if id(detection) not in used_ids]
        return large_detections + merged + unused_small

    def _center_inside_box(
        self,
        detection: PanelDetection,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bool:
        center_x = detection.x + detection.width / 2
        center_y = detection.y + detection.height / 2
        return x <= center_x <= x + width and y <= center_y <= y + height

    def _merge_detections(self, detections: list[PanelDetection], image_area: int) -> PanelDetection:
        points = np.array(
            [point for detection in detections for point in detection.points],
            dtype=np.float32,
        )
        rotated_rect = cv2.minAreaRect(points)
        box_points = cv2.boxPoints(rotated_rect).astype(int)
        x, y, width, height = cv2.boundingRect(box_points)
        area_ratio = (width * height) / image_area if image_area else 0.0
        score = round(min(0.75 + area_ratio, 1.0), 3)
        return PanelDetection(
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
            score=score,
            points=tuple((int(point[0]), int(point[1])) for point in box_points),
        )

    def _iou(self, first: PanelDetection, second: PanelDetection) -> float:
        first_x2 = first.x + first.width
        first_y2 = first.y + first.height
        second_x2 = second.x + second.width
        second_y2 = second.y + second.height

        intersection_x1 = max(first.x, second.x)
        intersection_y1 = max(first.y, second.y)
        intersection_x2 = min(first_x2, second_x2)
        intersection_y2 = min(first_y2, second_y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height

        first_area = first.width * first.height
        second_area = second.width * second.height
        union_area = first_area + second_area - intersection_area
        return intersection_area / union_area if union_area else 0.0
