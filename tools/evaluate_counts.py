from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.panel_counter import SolarPanelDetector


@dataclass(frozen=True)
class EvaluationRow:
    image_path: Path
    expected_count: int
    predicted_count: int

    @property
    def absolute_error(self) -> int:
        return abs(self.predicted_count - self.expected_count)

    @property
    def is_exact_match(self) -> bool:
        return self.absolute_error == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate panel count accuracy with labeled images.")
    parser.add_argument("labels_csv", type=Path, help="CSV file with image_path and expected_count columns.")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/evaluation.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate(args.labels_csv)
    write_report(rows, args.output_csv)
    print_summary(rows, args.output_csv)


def evaluate(labels_csv: Path) -> list[EvaluationRow]:
    detector = SolarPanelDetector()
    rows: list[EvaluationRow] = []

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
                EvaluationRow(
                    image_path=image_path,
                    expected_count=expected_count,
                    predicted_count=predicted_count,
                )
            )

    return rows


def write_report(rows: list[EvaluationRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "expected_count", "predicted_count", "absolute_error", "exact_match"])
        for row in rows:
            writer.writerow(
                [
                    row.image_path,
                    row.expected_count,
                    row.predicted_count,
                    row.absolute_error,
                    row.is_exact_match,
                ]
            )


def print_summary(rows: list[EvaluationRow], output_csv: Path) -> None:
    if not rows:
        print("No evaluation rows found.")
        return

    exact_matches = sum(1 for row in rows if row.is_exact_match)
    total_expected = sum(row.expected_count for row in rows)
    total_error = sum(row.absolute_error for row in rows)
    exact_match_rate = exact_matches / len(rows) * 100
    count_accuracy = (1 - (total_error / total_expected)) * 100 if total_expected else 0.0

    print(f"Evaluated images: {len(rows)}")
    print(f"Exact match rate: {exact_match_rate:.2f}%")
    print(f"Count accuracy: {count_accuracy:.2f}%")
    print(f"Total absolute error: {total_error}")
    print(f"Report saved to: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
