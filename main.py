from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.panel_counter import DetectionConfig, SolarPanelDetector
from src.panel_counter.tracker import process_video_unique
from src.panel_counter.video import process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count solar panels in images or videos.")
    parser.add_argument("input_path", type=Path, help="Image or video file path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for results.")
    parser.add_argument("--frame-step", type=int, default=15, help="Process every Nth video frame.")
    parser.add_argument(
        "--mode",
        choices=["frame", "unique"],
        default="frame",
        help="Use frame counts or unique panel tracking for video input.",
    )
    parser.add_argument(
        "--match-distance",
        type=float,
        default=70.0,
        help="Maximum pixel distance for matching the same panel in unique mode.",
    )
    parser.add_argument("--no-video", action="store_true", help="Do not create annotated video output.")
    parser.add_argument(
        "--exclude-edge-panels",
        action="store_true",
        help="Ignore panels touching the image border.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = SolarPanelDetector(
        DetectionConfig(include_edge_panels=not args.exclude_edge_panels)
    )

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    if args.input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        process_image(args.input_path, args.output_dir, detector)
        return

    if args.mode == "unique":
        tracking_results = process_video_unique(
            video_path=args.input_path,
            output_dir=args.output_dir,
            detector=detector,
            frame_step=args.frame_step,
            match_distance=args.match_distance,
            write_video=not args.no_video,
        )
        if tracking_results:
            print(f"Processed frames: {len(tracking_results)}")
            print(f"Final unique panel count: {tracking_results[-1].unique_count}")
            print(f"Maximum visible panel count: {max(result.visible_count for result in tracking_results)}")
            if tracking_results[-1].unique_count > 0:
                print(f"Results saved to: {args.output_dir.resolve()}")
            else:
                print("No output folder was created because no panels were detected.")
        return

    results = process_video(
        video_path=args.input_path,
        output_dir=args.output_dir,
        detector=detector,
        frame_step=args.frame_step,
        write_video=not args.no_video,
    )

    counts = [result.panel_count for result in results]
    if counts:
        print(f"Processed frames: {len(counts)}")
        print(f"Minimum panel count: {min(counts)}")
        print(f"Maximum panel count: {max(counts)}")
        print(f"Average panel count: {sum(counts) / len(counts):.2f}")
        if max(counts) > 0:
            print(f"Results saved to: {args.output_dir.resolve()}")
        else:
            print("No output folder was created because no panels were detected.")


def process_image(image_path: Path, output_dir: Path, detector: SolarPanelDetector) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    detections = detector.detect(image)
    print(f"Detected panels: {len(detections)}")
    if not detections:
        print("No output folder was created because no panels were detected.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_image = detector.annotate(image, detections)

    output_path = output_dir / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(output_path), annotated_image)

    print(f"Annotated image saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
