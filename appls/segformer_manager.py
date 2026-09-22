"""SegFormer demo with fractional-FPS export and frame-count checks.

Example:
  python segformer_export_fixed.py --video-file input.mp4 --save-file output.mp4 --no-show

Video-file export preserves frame order and source FPS for constant-rate input.
Variable-frame-rate timestamps and audio require a timestamp-aware media workflow.
The FPS overlay measures preprocessing, inference, mask transfer and rendering;
it excludes video decoding and encoding. No RobotEnv initialization is required.
"""
import time
import argparse
import math
import cv2
import numpy as np
import torch
from pathlib import Path

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

class SegFormerManager:
    """
    SegFormer-B0 Semantic Segmentation Manager
    """

    def __init__(
        self, 
        is_sim=True,
        robot=None,
        model_path=None):

        self.robot = robot
        self.is_sim = is_sim
        if is_sim:
            raise ValueError("SegFormerManager requires is_sim=False for inference.")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading SegFormer on {self.device}")
        self.processor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
        print("SegFormer loaded.")

        # reference FPS at max frequency
        self.max_fps = 45
        self.target_fps = 30

        self.last_time = time.time()
        self.total_frames = 0
        self.total_time = 0

        self.palette = self._create_palette()

        self._warmup()
        self.reset_statistics()

    ####################################################################
    # Warmup
    ####################################################################

    def _warmup(self):

        dummy = np.zeros((512, 512, 3), dtype=np.uint8)

        for _ in range(5):
            self.process(dummy)

    ####################################################################
    # Main Process
    ####################################################################

    @torch.no_grad()
    def process(self, frame: np.ndarray):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        # 1. Downscale & convert color on CPU efficiently
        resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 2. Convert to PyTorch Tensor & push straight to CUDA
        # Shape: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True).float()

        # 3. Fast GPU Normalization (ImageNet stats)
        tensor = tensor / 255.0
        tensor = (tensor - self.mean) / self.std

        # 4. Modern autocast API (FP16 mixed precision)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
            outputs = self.model(pixel_values=tensor)
            logits = outputs.logits

            # Upsample logits to original image resolution on GPU
            prediction = torch.nn.functional.interpolate(
                logits,
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            mask_tensor = prediction.argmax(dim=1)[0]

        # Convert final mask to CPU for display
        mask = mask_tensor.cpu().numpy().astype(np.uint8)
        overlay = self._draw_mask(frame, mask)

        # Measure exact CUDA execution time
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        latency = (time.perf_counter() - start) * 1000.0
        fps = 1000.0 / latency if latency > 0 else 0.0

        # Update rolling statistics
        self.total_frames += 1
        self.total_time += latency

        gpu_mem = None
        if self.device.type == "cuda":
            try:
                gpu_mem = torch.cuda.memory_allocated(self.device) / 1024.0 / 1024.0
            except Exception:
                gpu_mem = None

        return {
            "frame": overlay,
            "mask": mask,
            "fps": fps,
            "latency": latency,
            "device": self.device,
            "num_classes": logits.shape[1] if logits is not None else None,
            "gpu_memory": gpu_mem,
        }

    def run_source(self, source=0, window_name="SegFormer", show=True,
                   save_file=None, export_fps=None):
        """Write one annotated frame per decoded frame at the source FPS.

        Fractional FPS is preserved. Overlay FPS measures processing speed,
        not playback speed. OpenCV writes constant-rate video without audio;
        variable-frame-rate source timestamps are not preserved.
        export_fps is an explicit override, not the measured inference FPS.
        """
        if isinstance(source, Path):
            source = str(source)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if save_file is not None:
            output = Path(save_file).expanduser().resolve()
            if isinstance(source, str) and output == Path(source).expanduser().resolve():
                raise ValueError("Input and output paths must be different.")
            output.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open source: {source}")
        writer = None
        frames_read = frames_written = 0
        stopped_early = False
        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            declared_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            expected = int(declared_count) if math.isfinite(declared_count) and declared_count > 0 else 0
            fps_out = float(export_fps) if export_fps is not None else source_fps
            if save_file is not None and (not math.isfinite(fps_out) or fps_out <= 0):
                raise ValueError("Invalid source FPS. Supply a known --export-fps value.")
            print(f"Source FPS: {source_fps:.9g}; reported frames: {expected}")
            if save_file is not None:
                print(f"Export FPS: {fps_out:.9g} (independent of processing FPS)")
            output_size = None
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames_read += 1
                result = self.process(frame)
                vis = np.ascontiguousarray(result["frame"], dtype=np.uint8)
                height, width = vis.shape[:2]
                if vis.ndim != 3 or vis.shape[2] != 3:
                    raise ValueError("Expected a BGR output frame with three channels.")
                label = f"FPS:{result['fps']:.1f}"
                # scale = max(0.4, min(width / 1280.0, 1.5))
                scale = 3
                margin = max(4, round(width * 0.012))
                # thickness = max(1, round(2 * scale))
                thickness = 5
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                if tw > width - 2 * margin:
                    scale *= (width - 2 * margin) / tw
                    thickness = max(1, round(2 * scale))
                    (_, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                origin = (margin, margin + th)
                # cv2.putText(vis, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                #             (0, 0, 0), thickness + 2, cv2.LINE_AA)
                # cv2.putText(vis, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                #             (0, 255, 0), thickness, cv2.LINE_AA)
                if save_file is not None:
                    if writer is None:
                        output_size = (width, height)
                        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps_out, output_size)
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot open video writer: {output}")
                    if (width, height) != output_size:
                        raise RuntimeError("Output frame dimensions changed.")
                    writer.write(vis)
                    frames_written += 1
                if show:
                    cv2.imshow(window_name, vis)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        stopped_early = True
                        break
            if frames_read == 0:
                raise RuntimeError("No frames could be decoded from the input.")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        if expected and frames_read != expected and not stopped_early:
            print(f"WARNING: decoded {frames_read} of {expected} reported frames. "
                  "Check input decoding and frame-count metadata.")
        if stopped_early:
            print("Stopped by user; the saved video contains only the processed portion.")
        if save_file is not None:
            # Reopen after release to detect obvious failed/truncated exports.
            check = cv2.VideoCapture(str(output))
            try:
                if not check.isOpened():
                    raise RuntimeError(f"Cannot reopen exported video: {output}")
                actual_fps = float(check.get(cv2.CAP_PROP_FPS))
                raw_count = check.get(cv2.CAP_PROP_FRAME_COUNT)
                actual_count = int(raw_count) if math.isfinite(raw_count) and raw_count > 0 else 0
                if actual_count and actual_count != frames_written:
                    raise RuntimeError(f"Export reports {actual_count} frames, expected {frames_written}.")
                if not math.isfinite(actual_fps) or actual_fps <= 0 or abs(actual_fps - fps_out) > 0.001 * fps_out:
                    raise RuntimeError(f"Export FPS {actual_fps} does not match requested {fps_out}.")
            finally:
                check.release()
            print(f"Saved: {output}\nFrames: {frames_written}; FPS: {actual_fps:.9g}; "
                  f"nominal duration: {frames_written / actual_fps:.3f} s")
        return {"frames_read": frames_read, "frames_written": frames_written,
                "source_fps": source_fps, "export_fps": fps_out if save_file is not None else None,
                "stopped_early": stopped_early}

    ####################################################################
    # Visualization
    ####################################################################

    def _draw_mask(self, image, mask):

        color_mask = np.zeros_like(image)

        for label in np.unique(mask):

            # Ensure label is a Python int to avoid numpy uint8 overflow when indexing
            idx = int(label) % len(self.palette)
            color = self.palette[idx]

            color_mask[mask == label] = color

        overlay = cv2.addWeighted(
            image,
            0.45,
            color_mask,
            0.55,
            0,
        )

        return overlay

    ####################################################################
    # Color Palette
    ####################################################################

    def _create_palette(self):

        np.random.seed(0)

        palette = np.random.randint(
            0,
            255,
            size=(256, 3),
            dtype=np.uint8,
        )

        palette[0] = [0, 0, 0]

        return palette

    ####################################################################
    # Statistics
    ####################################################################

    def get_average_latency(self):

        if self.total_frames == 0:
            return 0

        return self.total_time / self.total_frames

    def reset_statistics(self):

        self.total_frames = 0
        self.total_time = 0

    ####################################################################
    # Benchmark
    ####################################################################

    def benchmark(self, video_path):

        cap = cv2.VideoCapture(video_path)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            result = self.process(frame)

            text = (
                f"FPS:{result['fps']:.1f} "
            )

            cv2.putText(
                result["frame"],
                text,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3,
            )

            cv2.imshow("SegFormer", result["frame"])

            key = cv2.waitKey(1)

            if key == 27:
                break

        cap.release()

        cv2.destroyAllWindows()
    
# ---------------------------------------
# Demo
# ---------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer manager demo")
    parser.add_argument("--model", default="./checkpoints/appls_models/segformer-b0-finetuned-ade-512-512", help="Path to SegFormer model")
    parser.add_argument("--type", choices=["video", "camera"], default="video", help="Input source type")
    parser.add_argument("--video-file", default="videos/people_detection.mp4", help="Path to video file when --type video")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index when --type camera")
    parser.add_argument("--save-file", default="segformer_results.mp4", help="Optional path to save annotated video")
    parser.add_argument("--no-show", action="store_true", help="Export without a display window")
    parser.add_argument("--export-fps", type=float, default=None, help="Optional known frame-rate override")
    args = parser.parse_args()

    print("=" * 60)
    print("SegFormer Demo")
    print("=" * 60)

    segformer = SegFormerManager(
        is_sim=False,
        robot=None,
        model_path=args.model)

    if args.type == "video":
        print(f"Running on video file: {args.video_file}")
        segformer.run_source(args.video_file, save_file=args.save_file,
                             show=not args.no_show, export_fps=args.export_fps)
    else:
        print(f"Running on camera index: {args.camera_index}")
        segformer.run_source(args.camera_index, save_file=args.save_file,
                             show=not args.no_show, export_fps=args.export_fps)