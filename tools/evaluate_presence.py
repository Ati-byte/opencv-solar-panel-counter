from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.panel_counter import DetectionConfig, SolarPanelDetector


@dataclass(frozen=True)
class PresenceResult:
    image_path: Path
    expected_positive: bool
    predicted_positive: bool
    predicted_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether images contain solar panels.")
    parser.add_argument("labels_csv", type=Path, help="CSV file with image_path and expected_count columns.")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/presence_evaluation.csv"))
    parser.add_argument(
        "--general-scene",
        action="store_true",
        help="Use a smaller minimum area for varied external datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectionConfig(min_area_ratio=0.001) if args.general_scene else DetectionConfig()
    rows = evaluate(args.labels_csv, SolarPanelDetector(config))
    write_report(rows, args.output_csv)
    print_summary(rows, args.output_csv)


def evaluate(labels_csv: Path, detector: SolarPanelDetector) -> list[PresenceResult]:
    rows: list[PresenceResult] = []

    with labels_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for item in reader:
            image_path = Path(item["image_path"])
            expected_count = int(item["expected_count"])
            image = cv2.imread(str(image_path))

            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")

            predicted_count = len(detector.detect(image))
            rows.append(
                PresenceResult(
                    image_path=image_path,
                    expected_positive=expected_count > 0,
                    predicted_positive=predicted_count > 0,
                    predicted_count=predicted_count,
                )
            )

    return rows


def write_report(rows: list[PresenceResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "expected_positive", "predicted_positive", "predicted_count"])
        for row in rows:
            writer.writerow([row.image_path, row.expected_positive, row.predicted_positive, row.predicted_count])


def print_summary(rows: list[PresenceResult], output_csv: Path) -> None:
    true_positive = sum(1 for row in rows if row.expected_positive and row.predicted_positive)
    true_negative = sum(1 for row in rows if not row.expected_positive and not row.predicted_positive)
    false_positive = sum(1 for row in rows if not row.expected_positive and row.predicted_positive)
    false_negative = sum(1 for row in rows if row.expected_positive and not row.predicted_positive)
    total = len(rows)

    accuracy = (true_positive + true_negative) / total * 100 if total else 0.0
    precision = true_positive / (true_positive + false_positive) * 100 if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) * 100 if true_positive + false_negative else 0.0

    print(f"Evaluated images: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"True positives: {true_positive}")
    print(f"False positives: {false_positive}")
    print(f"False negatives: {false_negative}")
    print(f"True negatives: {true_negative}")
    print(f"Report saved to: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
