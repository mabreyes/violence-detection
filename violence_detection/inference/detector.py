"""Violence detector module for inference with trained models."""

import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torchvision import transforms

from violence_detection.models import create_model


class ViolenceDetector:
    """Class for violence detection in videos using the trained model.

    Optimized for Philippine context with limited compute resources.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        clip_length: int = 16,
        frame_stride: int = 2,
        threshold: float = 0.5,
        mobile_optimized: bool = False,
        spatial_size: Tuple[int, int] = (112, 112),
    ):
        """Initialize the detector.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            clip_length: Number of frames per clip
            frame_stride: Stride between consecutive frames
            threshold: Decision threshold for violence detection
            mobile_optimized: Whether to use mobile-optimized model
            spatial_size: Input spatial size (height, width)

        """
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.threshold = threshold
        self.spatial_size = spatial_size

        # Load model
        self.model = self._load_model(model_path, mobile_optimized)

        # Setup transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
                ),
            ]
        )

    def _load_model(self, model_path: str, mobile_optimized: bool) -> nn.Module:
        """Load the trained model.

        Args:
            model_path: Path to model checkpoint
            mobile_optimized: Whether to load mobile-optimized model

        Returns:
            Loaded model

        """
        # Create model architecture
        model = create_model(
            num_classes=2,
            in_channels=3,
            clip_length=self.clip_length,
            dropout=0.0,  # No dropout during inference
            device=str(self.device),
            optimize_for_mobile=mobile_optimized,
        )

        # Load saved weights
        if model_path.endswith(".onnx"):
            # ONNX loading
            try:
                import onnxruntime as ort

                self.onnx_session = ort.InferenceSession(model_path)
                self.use_onnx = True
                print(f"Loaded ONNX model from {model_path}")
                # Return PyTorch model structure, but we'll use ONNX for inference
                return model
            except ImportError:
                print("ONNX Runtime not available. Using PyTorch model instead.")
                self.use_onnx = False
                # Fall back to PyTorch
                try:
                    model_path = model_path.replace(".onnx", ".pth")
                    print(f"Trying to load PyTorch model from {model_path}")
                except Exception as e:
                    raise ValueError("Could not find a PyTorch model file.") from e
        else:
            self.use_onnx = False

        # Load PyTorch checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        print(f"Loaded PyTorch model from {model_path}")

        return model

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame.

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame tensor

        """
        # Resize
        frame = cv2.resize(frame, (self.spatial_size[1], self.spatial_size[0]))

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        frame = self.transform(frame)

        return frame

    def _sample_clips(self, video_path: str, uniform_sampling: bool = True) -> List[torch.Tensor]:
        """Sample clips from a video.

        Args:
            video_path: Path to the video
            uniform_sampling: Whether to sample clips uniformly or densely

        Returns:
            List of clips as tensors of shape [T, C, H, W]

        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Get fps for potential later use but we're not using it directly now
        _ = cap.get(cv2.CAP_PROP_FPS)

        # Determine clip sampling strategy
        if uniform_sampling:
            # Sample clips uniformly across the video
            num_clips = max(1, frame_count // (self.clip_length * self.frame_stride))
            clip_starts = np.linspace(
                0, frame_count - self.clip_length * self.frame_stride, num_clips, dtype=int
            )
        else:
            # Dense sampling with overlap
            stride = max(1, self.clip_length // 2)  # 50% overlap
            clip_starts = list(range(0, frame_count - self.clip_length * self.frame_stride, stride))
            if len(clip_starts) > 50:  # Limit the number of clips for efficiency
                clip_starts = np.linspace(0, len(clip_starts) - 1, 50, dtype=int)
                clip_starts = [int(clip_starts[i]) for i in range(len(clip_starts))]

        # Sample clips
        clips = []

        for start_idx in clip_starts:
            # Get frames for this clip
            frames = []

            # Set position to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            # Read frames
            for _i in range(self.clip_length):
                # Read frame
                ret, frame = cap.read()

                if not ret:
                    # If we run out of frames, duplicate the last frame
                    if frames:
                        frames.append(frames[-1])
                    else:
                        # Skip this clip if we couldn't read any frames
                        break
                else:
                    # Process frame
                    frame = self._preprocess_frame(frame)
                    frames.append(frame)

                # Skip frames according to stride
                for _ in range(self.frame_stride - 1):
                    cap.read()  # Skip frame

            # Stack frames into a clip
            if len(frames) == self.clip_length:
                clip = torch.stack(frames)  # [T, C, H, W]
                clips.append(clip)

        # Release video
        cap.release()

        return clips

    def predict_video(
        self,
        video_path: str,
        save_visualization: bool = False,
        output_dir: Optional[str] = None,
        uniform_sampling: bool = True,
    ) -> Dict[str, Union[float, List[float]]]:
        """Predict violence in a video.

        Args:
            video_path: Path to the video
            save_visualization: Whether to save visualization
            output_dir: Directory to save visualization
            uniform_sampling: Whether to sample clips uniformly

        Returns:
            Dictionary with prediction results

        """
        # Sample clips from video
        clips = self._sample_clips(video_path, uniform_sampling)

        if len(clips) == 0:
            print(f"Warning: Could not extract any clips from {video_path}")
            return {
                "score": 0.0,
                "class": "non-violent",
                "clip_scores": [],
                "attention_weights": [],
            }

        # Batch predictions
        all_probs = []
        all_attention_weights = []

        with torch.no_grad():
            for clip in clips:
                # Add batch dimension: [T, C, H, W] -> [1, T, C, H, W]
                clip = clip.unsqueeze(0).to(self.device)

                if self.use_onnx:
                    # ONNX inference
                    ort_inputs = {self.onnx_session.get_inputs()[0].name: clip.cpu().numpy()}
                    ort_outs = self.onnx_session.run(None, ort_inputs)

                    logits = torch.tensor(ort_outs[0])
                    probs = fn.softmax(logits, dim=1)
                    all_probs.append(probs[0, 1].item())  # Probability of violence
                    all_attention_weights.append(
                        None
                    )  # ONNX doesn't support attention visualization
                else:
                    # PyTorch inference
                    logits, extras = self.model(clip)
                    probs = fn.softmax(logits, dim=1)
                    all_probs.append(probs[0, 1].item())  # Probability of violence

                    # Store attention weights for visualization
                    attn = extras["attention_weights"].squeeze(0).cpu().numpy()
                    all_attention_weights.append(attn)

        # Aggregate clip predictions
        violence_score = np.mean(all_probs)
        predicted_class = "violent" if violence_score >= self.threshold else "non-violent"

        # Save visualization if requested
        if save_visualization and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Create visualization
            self._save_visualization(
                video_path=video_path,
                clip_scores=all_probs,
                attention_weights=all_attention_weights,
                output_dir=output_dir,
            )

        return {
            "score": float(violence_score),
            "class": predicted_class,
            "clip_scores": [float(p) for p in all_probs],
            "attention_weights": all_attention_weights if not self.use_onnx else None,
        }

    def _save_visualization(
        self,
        video_path: str,
        clip_scores: List[float],
        attention_weights: List[np.ndarray],
        output_dir: str,
    ) -> None:
        """Save visualization of prediction results.

        Args:
            video_path: Path to the video
            clip_scores: List of violence scores for each clip
            attention_weights: List of attention weights for each clip
            output_dir: Directory to save visualization

        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Create score plot
        self._plot_scores(video_name, clip_scores, output_dir)

        # Extract key frames for clips with highest scores
        if len(clip_scores) > 0:
            self._visualize_key_frames(
                video_path, video_name, clip_scores, attention_weights, output_dir
            )

        # Save numerical results in a text file
        self._save_text_results(video_path, video_name, clip_scores, output_dir)

    def _plot_scores(self, video_name: str, clip_scores: List[float], output_dir: str) -> None:
        """Plot scores for each clip.

        Args:
            video_name: Name of the video
            clip_scores: List of violence scores for each clip
            output_dir: Directory to save visualization

        """
        plt.figure(figsize=(12, 6))
        plt.plot(clip_scores, marker="o")
        plt.axhline(y=self.threshold, color="r", linestyle="--", alpha=0.5)
        plt.title(f"Violence Detection Scores - {video_name}")
        plt.xlabel("Clip Index")
        plt.ylabel("Violence Score")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{video_name}_scores.png"))
        plt.close()

    def _visualize_key_frames(
        self,
        video_path: str,
        video_name: str,
        clip_scores: List[float],
        attention_weights: List[np.ndarray],
        output_dir: str,
    ) -> None:
        """Visualize key frames from the video with highest scores.

        Args:
            video_path: Path to the video
            video_name: Name of the video
            clip_scores: List of violence scores for each clip
            attention_weights: List of attention weights for each clip
            output_dir: Directory to save visualization

        """
        # Find top 5 clips with highest scores
        top_indices = np.argsort(clip_scores)[-5:][::-1]

        # Create montage of key frames with attention highlights
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path} for visualization")
            return

        fig, axes = plt.subplots(len(top_indices), 3, figsize=(15, 5 * len(top_indices)))

        if len(top_indices) == 1:
            axes = np.array([axes])

        for i, idx in enumerate(top_indices):
            # Get clip start frame
            start_frame = idx * self.clip_length * self.frame_stride

            # Get frames from beginning, middle, and end of clip
            frames = []
            positions = [0, self.clip_length // 2, self.clip_length - 1]

            for pos in positions:
                frame_idx = start_frame + pos * self.frame_stride
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    frames.append(np.zeros((256, 256, 3), dtype=np.uint8))

            # Display frames
            for j, frame in enumerate(frames):
                axes[i, j].imshow(frame)
                axes[i, j].set_title(f"Frame {positions[j]}")
                axes[i, j].axis("off")

            # Add score
            fig.text(
                0.5,
                0.92 - i * 0.2,
                f"Clip {idx} - Score: {clip_scores[idx]:.4f}",
                ha="center",
                fontsize=12,
                color="red" if clip_scores[idx] >= self.threshold else "black",
            )

            # Add attention visualization if available
            if attention_weights[idx] is not None:
                self._visualize_attention(attention_weights[idx], axes[i, 2])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_name}_keyframes.png"))
        plt.close()

        cap.release()

    def _visualize_attention(self, attn_weights: np.ndarray, ax) -> None:
        """Visualize attention weights.

        Args:
            attn_weights: Attention weights for a clip
            ax: Matplotlib axis to plot on

        """
        # Average across heads
        attn = attn_weights.mean(axis=0)

        # Normalize for visualization
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)

        # Display as a curve
        time_axis = np.arange(attn.shape[1])
        for j in range(min(3, attn.shape[0])):
            ax.plot(time_axis, attn[j], label=f"Head {j}")

        ax.set_title("Attention Weights")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Attention")
        ax.legend()

    def _save_text_results(
        self, video_path: str, video_name: str, clip_scores: List[float], output_dir: str
    ) -> None:
        """Save text results of the analysis.

        Args:
            video_path: Path to the video
            video_name: Name of the video
            clip_scores: List of violence scores for each clip
            output_dir: Directory to save results

        """
        with open(os.path.join(output_dir, f"{video_name}_results.txt"), "w") as f:
            f.write(f"Video: {video_path}\n")
            f.write(f"Average Violence Score: {np.mean(clip_scores):.4f}\n")
            f.write(f"Max Violence Score: {np.max(clip_scores):.4f}\n")

            # Break up the long line for linting
            is_violent = "violent" if np.mean(clip_scores) >= self.threshold else "non-violent"
            f.write(f"Predicted Class: {is_violent}\n\n")

            f.write("Clip Scores:\n")
            for i, score in enumerate(clip_scores):
                f.write(f"Clip {i}: {score:.4f}\n")
