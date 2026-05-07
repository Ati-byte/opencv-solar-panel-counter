# OpenCV Solar Panel Counter

This project detects and counts visible solar panels in images and videos using classical computer vision methods with OpenCV.

The detector is designed for aerial or top-angle footage where solar panels have visible blue surfaces and clear frame boundaries.

## Features

- Counts solar panels in a single image
- Processes video files frame by frame
- Draws rotated bounding boxes around detected panels
- Saves an annotated output video
- Exports per-frame panel counts to CSV
- Supports ignoring panels that touch the image border

## How It Works

The project uses a rule-based image processing pipeline:

1. Read the input image or video frame with OpenCV.
2. Convert the frame from BGR to HSV color space.
3. Segment blue solar panel regions with an HSV color threshold.
4. Remove small noise with morphological operations.
5. Find panel candidates using contour detection.
6. Filter candidates by area, width, height, fill ratio, and aspect ratio.
7. Draw rotated boxes around the remaining detections.
8. Count the detections and save the result.

This approach does not require a trained machine learning model. It works best when the panel color and panel boundaries are visually clear.

## Project Structure

```text
opencv-solar-panel-counter/
  main.py
  requirements.txt
  README.md
  src/
    panel_counter/
      __init__.py
      detector.py
      video.py
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Ati-byte/opencv-solar-panel-counter.git
cd opencv-solar-panel-counter
pip install -r requirements.txt
```

## Usage

### Desktop GUI

Run the Tkinter desktop interface:

```bash
python app_gui.py
```

The interface lets you select an image or video, choose frame counting or unique tracking mode, and open the output folder after analysis.

### Count panels in an image

```bash
python main.py path/to/image.jpg
```

The annotated image is saved in the `outputs` directory.

### Count panels in a video

```bash
python main.py path/to/video.mp4
```

By default, the program processes every 15th frame. For more frequent analysis, use `--frame-step`:

```bash
python main.py path/to/video.mp4 --frame-step 5
```

To process every frame:

```bash
python main.py path/to/video.mp4 --frame-step 1
```

### Ignore border panels

Panels that touch the image border are included by default. To ignore partial panels near the frame edges:

```bash
python main.py path/to/video.mp4 --exclude-edge-panels
```

### Custom output directory

```bash
python main.py path/to/video.mp4 --output-dir results
```

### Count unique panels in drone video

For moving drone footage, use total unique tracking mode:

```bash
python main.py path/to/video.mp4 --mode unique --frame-step 10
```

This mode estimates camera motion between processed frames with OpenCV feature matching and avoids counting the same panel repeatedly while it remains in view. This is the correct mode when you need the total number of panels seen during the video.

Useful parameters:

- `--frame-step`: lower values process more frames and improve tracking continuity.
- `--match-distance`: maximum distance for assigning a detection to an existing tracked panel.

For smooth drone footage, a good starting point is:

```bash
python main.py path/to/video.mp4 --mode unique --frame-step 10 --match-distance 80
```

## Evaluation

If you have labeled images, create a CSV file with this format:

```text
image_path,expected_count
validation/images/frame_01.jpg,23
validation/images/frame_02.jpg,18
```

Then run:

```bash
python tools/evaluate_counts.py path/to/labels.csv
```

The script prints exact match rate, count accuracy, total absolute error, and saves a detailed CSV report.

For datasets where annotations represent panel groups instead of individual modules, presence evaluation is usually more appropriate:

```bash
python tools/evaluate_presence.py path/to/labels.csv --general-scene
```

This reports whether the image contains at least one detected solar panel region.

## Outputs

For video input, the program creates:

```text
outputs/
  annotated_video.mp4
  panel_counts.csv
```

`annotated_video.mp4` contains the processed video with detected panels marked.

`panel_counts.csv` contains frame-level count data:

```text
frame_index,timestamp_seconds,panel_count
0,0.0,18
15,0.5,19
30,1.0,20
```

For image input, the program saves an annotated image in the selected output directory.

In unique tracking mode, the program creates:

```text
outputs/
  tracked_video.mp4
  unique_panel_counts.csv
```

## Configuration

Detection parameters are stored in `DetectionConfig` inside `src/panel_counter/detector.py`.

Important parameters include:

- `lower_hsv` and `upper_hsv`: HSV color range for panel segmentation
- `min_area_ratio` and `max_area_ratio`: accepted candidate size range
- `min_fill_ratio`: how much of a candidate box must be filled by the detected mask
- `min_aspect_ratio` and `max_aspect_ratio`: accepted panel shape range
- `include_edge_panels`: whether border-touching panels are counted

These values can be tuned for different camera angles, lighting conditions, or panel colors.

## Limitations

This is a classical computer vision solution, so performance depends on the visual quality of the input.

The detector works best when:

- Panels are blue or dark blue
- The camera angle is from above or near above
- Panel borders are visible
- Lighting is not extremely overexposed

It may need parameter tuning for very different panel colors, heavy glare, strong shadows, or low-resolution footage.

Unique video counting assumes that the drone footage is reasonably continuous. Very fast camera motion, abrupt cuts, severe motion blur, or panels disappearing and reappearing after a long time can reduce tracking accuracy.

## Possible Improvements

- Add a simple desktop or web interface
- Add frame preview and parameter controls
- Use object tracking to avoid counting the same panel across multiple video frames
- Generate summary charts from the CSV output
