# OpenCV Solar Panel Counter

Detects and counts visible solar panels in images and drone videos using classical computer vision with OpenCV. No machine learning model required.

Designed for aerial or top-angle footage where panels have visible blue surfaces and clear frame boundaries.

## Features

- Single-image panel detection with annotated output
- Per-frame panel counting for video
- **Unique panel tracking** across drone video — counts each physical panel once
- Camera motion compensation between frames (ORB feature matching + affine estimation)
- Rotated bounding boxes fitted to each panel contour
- Border panel exclusion — ignore panels partially cut off at the frame edge
- Automatic frame step calibration based on video FPS
- Annotated output video and per-frame CSV export
- Desktop GUI (Tkinter)

## How It Works

The pipeline is fully rule-based:

1. Convert each frame from BGR to HSV color space.
2. Threshold the HSV image to isolate blue solar panel regions.
3. Apply morphological opening to remove small noise.
4. Extract contours and filter candidates by area ratio, fill ratio, and aspect ratio.
5. Apply non-maximum suppression to remove overlapping detections.
6. Merge small adjacent cell-level detections into single panel detections.
7. In unique tracking mode, estimate camera motion between processed frames and track panel identities across the video to count each panel exactly once.

## Project Structure

```text
opencv-solar-panel-counter/
  app_gui.py            # Desktop GUI entry point
  main.py               # CLI entry point
  requirements.txt
  src/
    panel_counter/
      __init__.py
      detector.py       # Detection pipeline and DetectionConfig
      tracker.py        # Unique panel tracker and camera motion estimator
      video.py          # Per-frame video processing
  tools/
    evaluate_counts.py  # Count accuracy evaluation against labeled CSV
    evaluate_presence.py
```

## Installation

```bash
git clone https://github.com/Ati-byte/opencv-solar-panel-counter.git
cd opencv-solar-panel-counter
pip install -r requirements.txt
```

## Usage

### Desktop GUI

```bash
python app_gui.py
```

Select an image or video, choose the analysis mode, set border panel behavior, and click **Analyze**. The frame step field is populated automatically from the video FPS when a file is selected.

### CLI — count panels in an image

```bash
python main.py path/to/image.jpg
```

### CLI — count panels in a video (per-frame mode)

```bash
python main.py path/to/video.mp4 --frame-step 10
```

### CLI — count unique panels in drone video

```bash
python main.py path/to/video.mp4 --mode unique --frame-step 10 --match-distance 80
```

This mode tracks panels across frames and counts each physical panel once, even as the drone moves. Recommended for any footage where the camera is in motion over a panel field.

### Ignore border panels

```bash
python main.py path/to/video.mp4 --mode unique --exclude-edge-panels
```

Panels whose bounding box touches the frame edge are excluded. Use this when you want to count only fully visible panels.

### Custom output directory

```bash
python main.py path/to/video.mp4 --output-dir results
```

## Frame Step

`--frame-step N` processes one out of every N frames. A smaller value gives more tracking samples but increases processing time.

The recommended value is `fps / 3` — for a 30 FPS video that is 10, for 60 FPS it is 20. The GUI sets this automatically when a video is loaded. The tracker's confirmation threshold also scales with the frame step so that a panel must be observed for approximately one second of real time before it is counted.

## Outputs

**Image input:**

```text
outputs/
  image_annotated.jpg
```

**Video — per-frame mode:**

```text
outputs/
  annotated_video.mp4
  panel_counts.csv        # frame_index, timestamp_seconds, panel_count
```

**Video — unique tracking mode:**

```text
outputs/
  tracked_video.mp4
  unique_panel_counts.csv # frame_index, timestamp_seconds, visible_count, unique_count, ...
```

## Evaluation

Create a CSV file with expected counts:

```text
image_path,expected_count
validation/frame_01.jpg,23
validation/frame_02.jpg,18
```

Then run:

```bash
python tools/evaluate_counts.py path/to/labels.csv
```

Reports exact match rate, count accuracy, and total absolute error.

## Configuration

Detection parameters are defined in `DetectionConfig` inside `src/panel_counter/detector.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lower_hsv` | `(90, 70, 30)` | Lower HSV bound for panel segmentation |
| `upper_hsv` | `(130, 255, 245)` | Upper HSV bound for panel segmentation |
| `min_area_ratio` | `0.004` | Minimum panel area as fraction of frame area |
| `max_area_ratio` | `0.22` | Maximum panel area as fraction of frame area |
| `min_fill_ratio` | `0.30` | Minimum contour fill of the bounding box |
| `min_aspect_ratio` | `0.25` | Minimum width/height ratio |
| `max_aspect_ratio` | `2.50` | Maximum width/height ratio |
| `include_edge_panels` | `True` | Whether to count panels touching the frame edge |
| `nms_iou_threshold` | `0.20` | IoU threshold for non-maximum suppression |

## Limitations

- Works best on blue or dark-blue panels shot from above with visible borders.
- Heavy glare, strong shadows, or panels that blend with the background reduce accuracy.
- Unique tracking assumes reasonably smooth drone footage. Abrupt cuts or very fast camera rotation can cause the same panel to be counted more than once.
- Parameter tuning may be needed for significantly different panel colors or camera angles.
