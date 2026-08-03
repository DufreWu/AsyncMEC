import time
import cv2
import numpy as np
import torch

from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)
from robot_env import RobotEnv

class SegFormerManager:
    """
    SegFormer-B0 Semantic Segmentation Manager

    Returned dictionary:
    {
        "frame": overlay image,
        "mask": prediction mask,
        "fps": float,
        "latency": ms,
        "num_classes": int,
        "gpu_memory": MB,
        "device": "cuda"/"cpu"
    }
    """

    def __init__(
        self,
        model_path="./checkpoints/segformer-b0-finetuned-ade-512-512",
        device="cuda",
        max_fps=45,
        confidence_threshold=0.0,
    ):

        self.max_fps = max_fps
        self.confidence_threshold = confidence_threshold

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Loading SegFormer... in {self.device}")

        self.processor = SegformerImageProcessor.from_pretrained(model_path)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_path
        )

        self.model.to(self.device)
        self.model.eval()

        print("SegFormer loaded.")

        self.last_time = time.time()

        self.total_frames = 0
        self.total_time = 0

        self.palette = self._create_palette()

        self._warmup()

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
        start = time.perf_counter()

        # 1. Downscale & convert color on CPU efficiently
        resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 2. Convert to PyTorch Tensor & push straight to CUDA
        # Shape: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True).float()

        # 3. Fast GPU Normalization (ImageNet stats)
        tensor = tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        # 4. Modern autocast API (FP16 mixed precision)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(pixel_values=tensor)
            logits = outputs.logits

            # Upsample logits to original image resolution on GPU
            prediction = torch.nn.functional.interpolate(
                logits,
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            # mask_tensor = prediction.argmax(dim=1)[0]

        # Convert final mask to CPU for display
        # mask = mask_tensor.cpu().numpy().astype(np.uint8)
        # overlay = self._draw_mask(frame, mask)

        # Measure exact CUDA execution time
        if self.device == "cuda":
            torch.cuda.synchronize()

        latency = (time.perf_counter() - start) * 1000.0
        fps = 1000.0 / latency if latency > 0 else 0.0

        return {
            "fps": fps,
            "latency": latency,
            "device": self.device
        }

    ####################################################################
    # Visualization
    ####################################################################

    def _draw_mask(self, image, mask):

        color_mask = np.zeros_like(image)

        for label in np.unique(mask):

            color = self.palette[label % len(self.palette)]

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
                f"Latency:{result['latency']:.1f}ms"
            )

            cv2.putText(
                result["frame"],
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
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
    # ---------------------------------------
    # Simulation Example
    # ---------------------------------------
    try:
        import yaml
        from pathlib import Path

        with open("configs/robot.yaml", "r") as f:
            robot_config = yaml.safe_load(f)

        robot = RobotEnv(robot_config, is_sim=False)
    except Exception:
        robot = None

    print("=" * 60)
    print("Test")
    print("=" * 60)

    segformer = SegFormerManager(
        model_path="./checkpoints/segformer-b0-finetuned-ade-512-512",
        max_fps=45,
    )

    video = "videos/people_detection.mp4"

    if not Path(video).exists():
        raise FileNotFoundError(video)

    cap = cv2.VideoCapture(video)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = segformer.process(
            frame
        )
        print(result)

    cap.release()
    cv2.destroyAllWindows()