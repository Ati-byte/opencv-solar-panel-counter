from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from tkinter import (
    BOTH,
    DISABLED,
    END,
    LEFT,
    NORMAL,
    RIGHT,
    X,
    Button,
    Entry,
    Frame,
    Label,
    LabelFrame,
    Radiobutton,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)
from tkinter.ttk import Progressbar

import cv2

from src.panel_counter import DetectionConfig, SolarPanelDetector
from src.panel_counter.tracker import process_video_unique
from src.panel_counter.video import process_video


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SolarPanelCounterApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("OpenCV Solar Panel Counter")
        self.root.geometry("760x520")
        self.root.minsize(720, 500)

        self.input_path = StringVar()
        self.mode = StringVar(value="frame")
        self.edge_mode = StringVar(value="include")
        self.frame_step = StringVar(value="10")
        self.match_distance = StringVar(value="80")
        self.status = StringVar(value="Select an image or video to start.")
        self.last_output_dir: Path | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        container = Frame(self.root, padx=18, pady=16)
        container.pack(fill=BOTH, expand=True)

        input_group = LabelFrame(container, text="Input", padx=12, pady=12)
        input_group.pack(fill=X)

        Entry(input_group, textvariable=self.input_path).pack(side=LEFT, fill=X, expand=True)
        Button(input_group, text="Browse", command=self._browse_file, width=12).pack(side=RIGHT, padx=(10, 0))

        options = LabelFrame(container, text="Options", padx=12, pady=12)
        options.pack(fill=X, pady=(14, 0))

        mode_row = Frame(options)
        mode_row.pack(fill=X)
        Label(mode_row, text="Video mode:", width=16, anchor="w").pack(side=LEFT)
        Radiobutton(mode_row, text="Frame count", variable=self.mode, value="frame").pack(side=LEFT)
        Radiobutton(mode_row, text="Unique tracking", variable=self.mode, value="unique").pack(side=LEFT, padx=(16, 0))

        edge_row = Frame(options)
        edge_row.pack(fill=X, pady=(8, 0))
        Label(edge_row, text="Border panels:", width=16, anchor="w").pack(side=LEFT)
        Radiobutton(edge_row, text="Include", variable=self.edge_mode, value="include").pack(side=LEFT)
        Radiobutton(edge_row, text="Exclude", variable=self.edge_mode, value="exclude").pack(side=LEFT, padx=(16, 0))

        numeric_row = Frame(options)
        numeric_row.pack(fill=X, pady=(8, 0))
        Label(numeric_row, text="Frame step:", width=16, anchor="w").pack(side=LEFT)
        Entry(numeric_row, textvariable=self.frame_step, width=8).pack(side=LEFT)
        Label(numeric_row, text="Match distance:", padx=18).pack(side=LEFT)
        Entry(numeric_row, textvariable=self.match_distance, width=8).pack(side=LEFT)

        actions = Frame(container)
        actions.pack(fill=X, pady=(14, 0))
        self.run_button = Button(actions, text="Analyze", command=self._start_analysis, width=16)
        self.run_button.pack(side=LEFT)
        self.open_button = Button(actions, text="Open output folder", command=self._open_output_folder, width=18)
        self.open_button.pack(side=LEFT, padx=(10, 0))

        self.progress = Progressbar(container, mode="indeterminate")
        self.progress.pack(fill=X, pady=(14, 0))

        result_group = LabelFrame(container, text="Result", padx=12, pady=12)
        result_group.pack(fill=BOTH, expand=True, pady=(14, 0))

        Label(result_group, textvariable=self.status, anchor="w", justify=LEFT).pack(fill=X)
        self.output_text = Entry(result_group)
        self.output_text.pack(fill=X, pady=(10, 0))

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image or video",
            filetypes=[
                ("Images and videos", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.mov *.avi *.mkv"),
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Videos", "*.mp4 *.mov *.avi *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_path.set(path)

    def _start_analysis(self) -> None:
        try:
            input_path = Path(self.input_path.get())
            if not input_path.exists():
                messagebox.showerror("Input error", "Please select a valid image or video file.")
                return

            frame_step = int(self.frame_step.get())
            match_distance = float(self.match_distance.get())
            if frame_step < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input error", "Frame step must be an integer and match distance must be a number.")
            return

        self._set_running(True)
        worker = threading.Thread(
            target=self._run_analysis,
            args=(input_path, frame_step, match_distance),
            daemon=True,
        )
        worker.start()

    def _run_analysis(self, input_path: Path, frame_step: int, match_distance: float) -> None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("app_outputs") / f"{input_path.stem}_{timestamp}"
            detector = SolarPanelDetector(
                DetectionConfig(include_edge_panels=self.edge_mode.get() == "include")
            )

            if input_path.suffix.lower() in IMAGE_EXTENSIONS:
                result_message = self._process_image(input_path, output_dir, detector)
            elif self.mode.get() == "unique":
                results = process_video_unique(
                    video_path=input_path,
                    output_dir=output_dir,
                    detector=detector,
                    frame_step=frame_step,
                    match_distance=match_distance,
                    write_video=True,
                )
                final_count = results[-1].unique_count if results else 0
                max_visible = max((result.visible_count for result in results), default=0)
                result_message = (
                    f"Unique tracking completed.\n"
                    f"Processed frames: {len(results)}\n"
                    f"Final unique panel count: {final_count}\n"
                    f"Maximum visible panel count: {max_visible}"
                )
            else:
                results = process_video(
                    video_path=input_path,
                    output_dir=output_dir,
                    detector=detector,
                    frame_step=frame_step,
                    write_video=True,
                )
                counts = [result.panel_count for result in results]
                average = sum(counts) / len(counts) if counts else 0
                result_message = (
                    f"Frame count completed.\n"
                    f"Processed frames: {len(results)}\n"
                    f"Minimum panel count: {min(counts) if counts else 0}\n"
                    f"Maximum panel count: {max(counts) if counts else 0}\n"
                    f"Average panel count: {average:.2f}"
                )

            self.root.after(0, self._show_success, output_dir, result_message)
        except Exception as exc:
            self.root.after(0, self._show_error, str(exc))

    def _process_image(
        self,
        image_path: Path,
        output_dir: Path,
        detector: SolarPanelDetector,
    ) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        detections = detector.detect(image)
        annotated = detector.annotate(image, detections)
        output_path = output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(output_path), annotated)
        return f"Image analysis completed.\nDetected panels: {len(detections)}"

    def _show_success(self, output_dir: Path, message: str) -> None:
        self.last_output_dir = output_dir.resolve()
        self.status.set(message)
        self._set_output_text(str(self.last_output_dir))
        self._set_running(False)

    def _show_error(self, message: str) -> None:
        self.status.set("Analysis failed.")
        self._set_running(False)
        messagebox.showerror("Analysis failed", message)

    def _set_output_text(self, text: str) -> None:
        self.output_text.config(state=NORMAL)
        self.output_text.delete(0, END)
        self.output_text.insert(0, text)

    def _set_running(self, running: bool) -> None:
        self.run_button.config(state=DISABLED if running else NORMAL)
        if running:
            self.status.set("Analysis is running. Please wait.")
            self.progress.start(12)
        else:
            self.progress.stop()

    def _open_output_folder(self) -> None:
        if self.last_output_dir is None or not self.last_output_dir.exists():
            messagebox.showinfo("Output folder", "No output folder is available yet.")
            return
        os.startfile(self.last_output_dir)


def main() -> None:
    root = Tk()
    SolarPanelCounterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
