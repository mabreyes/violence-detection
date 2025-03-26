"""Data preprocessing utilities for violence detection."""

import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_rate: int = 5,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> Tuple[int, List[str]]:
    """Extract frames from a video file.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_rate: Number of frames per second to extract
        max_frames: Maximum number of frames to extract
        resize: Tuple of (width, height) to resize frames

    Returns:
        Number of frames extracted and list of frame paths

    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0, []

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval
    if frame_rate <= 0:
        frame_interval = 1
    else:
        frame_interval = max(1, int(fps / frame_rate))

    # Extract frames
    count = 0
    frame_paths = []
    frame_idx = 0

    # Get video filename without extension for naming frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    pbar = tqdm(
        total=min(total_frames, max_frames if max_frames else total_frames),
        desc=f"Extracting frames from {video_name}",
    )

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Resize frame if requested
            if resize:
                frame = cv2.resize(frame, resize)

            # Save frame
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

            count += 1
            pbar.update(1)

            # Stop if max_frames reached
            if max_frames and count >= max_frames:
                break

        frame_idx += 1

    # Release video
    video.release()
    pbar.close()

    return count, frame_paths


def find_video_files(video_dir: str) -> List[str]:
    """Find all video files in a directory.

    Args:
        video_dir: Directory containing video files

    Returns:
        List of video file paths

    """
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob(os.path.join(video_dir, ext)))

    print(f"Found {len(video_files)} video files")
    return video_files


def load_annotations(
    annotations: Optional[str], video_files: List[str]
) -> Tuple[List[str], Optional[Dict[str, int]]]:
    """Load annotations and filter video files.

    Args:
        annotations: Path to annotations file
        video_files: List of video file paths

    Returns:
        Filtered video files and labels dictionary

    """
    if not annotations or not os.path.exists(annotations):
        if annotations:
            print(f"Warning: Annotations file {annotations} not found")
        return video_files, None

    annot_df = pd.read_csv(annotations)
    print(f"Loaded {len(annot_df)} annotations")

    # Filter video files to only those in annotations
    video_ids = [os.path.basename(v) for v in video_files]
    video_map = dict(zip(video_ids, video_files))
    filtered_videos = []
    labels = {}

    for _, row in annot_df.iterrows():
        video_id = row["video_id"]
        if video_id in video_map:
            filtered_videos.append(video_map[video_id])
            labels[video_id] = row["label"]

    print(f"After filtering, using {len(filtered_videos)} videos")
    return filtered_videos, labels


def process_videos(
    video_files: List[str],
    frames_dir: str,
    labels: Optional[Dict[str, int]],
    frame_rate: int,
    max_frames: Optional[int],
    resize: Optional[Tuple[int, int]],
) -> pd.DataFrame:
    """Process videos and extract frames.

    Args:
        video_files: List of video file paths
        frames_dir: Directory to save frames
        labels: Dictionary mapping video IDs to labels
        frame_rate: Number of frames per second to extract
        max_frames: Maximum number of frames to extract per video
        resize: Tuple of (width, height) to resize frames

    Returns:
        DataFrame with video data

    """
    video_data = []

    for video_path in tqdm(video_files, desc="Processing videos"):
        video_id = os.path.basename(video_path)
        video_name = os.path.splitext(video_id)[0]

        # Create directory for this video's frames
        video_frames_dir = os.path.join(frames_dir, video_name)
        os.makedirs(video_frames_dir, exist_ok=True)

        # Extract frames
        num_frames, _ = extract_frames(
            video_path=video_path,
            output_dir=video_frames_dir,
            frame_rate=frame_rate,
            max_frames=max_frames,
            resize=resize,
        )

        # Get label if available
        label = labels.get(video_id, None) if labels else None

        # Store video data
        video_data.append({"video_id": video_id, "num_frames": num_frames, "label": label})

    return pd.DataFrame(video_data)


def infer_labels_from_dirs(df: pd.DataFrame, video_dir: str) -> pd.DataFrame:
    """Infer labels from directory structure if not provided.

    Args:
        df: DataFrame with video data
        video_dir: Directory containing video files

    Returns:
        DataFrame with inferred labels

    """
    for i, row in df.iterrows():
        video_path = os.path.join(video_dir, row["video_id"])
        parent_dir = os.path.basename(os.path.dirname(video_path)).lower()
        if "violen" in parent_dir:
            df.at[i, "label"] = 1
        elif "non" in parent_dir or "normal" in parent_dir:
            df.at[i, "label"] = 0

    return df


def process_dataset(
    video_dir: str,
    output_dir: str,
    annotations: Optional[str] = None,
    test_size: float = 0.2,
    frame_rate: int = 5,
    max_frames_per_video: Optional[int] = 300,
    resize: Optional[Tuple[int, int]] = (224, 224),
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process a dataset of videos and create train/val splits.

    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted frames and labels
        annotations: Path to annotations file (CSV with video_id,label,path columns)
        test_size: Fraction of data to use for validation
        frame_rate: Number of frames per second to extract
        max_frames_per_video: Maximum number of frames to extract from each video
        resize: Tuple of (width, height) to resize frames
        random_seed: Random seed for train/val split

    Returns:
        Train and validation dataframes with video_id,label columns

    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    if annotations:
        # Use annotations with path column for more complex structures
        df = pd.read_csv(annotations)
        print(f"Loaded {len(df)} annotations")
        
        # Check if path column exists
        use_path = "path" in df.columns
        
        # Process videos using the annotations directly
        video_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
            video_id = row['video_id']
            label = row['label']
            
            # Get the video path - either directly from the path column or by building it
            if use_path:
                video_path = os.path.join(video_dir, row['path'])
            else:
                video_path = os.path.join(video_dir, video_id)
            
            # Skip if video doesn't exist
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                continue
            
            # Create directory for this video's frames
            video_frames_dir = os.path.join(frames_dir, video_id.replace(".", "_"))
            os.makedirs(video_frames_dir, exist_ok=True)
            
            # Extract frames
            num_frames, _ = extract_frames(
                video_path=video_path,
                output_dir=video_frames_dir,
                frame_rate=frame_rate,
                max_frames=max_frames_per_video,
                resize=resize,
            )
            
            # Store video data
            video_data.append({"video_id": video_id, "num_frames": num_frames, "label": label})
        
        # Convert to DataFrame
        new_df = pd.DataFrame(video_data)
        
        # Make sure we found and processed at least some videos
        if len(new_df) == 0:
            raise ValueError("No videos were processed. Check video paths and annotations.")
        
        # Use the new dataframe with processed videos
        df = new_df
    else:
        # Find all video files and load annotations
        video_files = find_video_files(video_dir)
        print(f"Found {len(video_files)} video files")
        
        if annotations and os.path.exists(annotations):
            video_files, labels = load_annotations(annotations, video_files)
            print(f"After filtering, using {len(video_files)} videos")
        else:
            labels = None
            
        # Process videos
        video_data = []
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            video_id = os.path.basename(video_path)
            
            # Create directory for this video's frames
            video_frames_dir = os.path.join(frames_dir, video_id.replace(".", "_"))
            os.makedirs(video_frames_dir, exist_ok=True)
            
            # Extract frames
            num_frames, _ = extract_frames(
                video_path=video_path,
                output_dir=video_frames_dir,
                frame_rate=frame_rate,
                max_frames=max_frames_per_video,
                resize=resize,
            )
            
            # Get label if available
            label = labels.get(video_id, None) if labels else None
            
            # Store video data
            video_data.append({"video_id": video_id, "num_frames": num_frames, "label": label})
        
        # Convert to DataFrame
        df = pd.DataFrame(video_data)
        
        # If labels were not provided but we need them for training
        if labels is None:
            print("No annotations provided. Using directory structure for labels.")
            df = infer_labels_from_dirs(df, video_dir)

    # Check if we have labels
    if "label" not in df.columns or df["label"].isna().any():
        raise ValueError(
            "Labels are missing for some videos. "
            "Please provide annotations or organize videos in labeled directories."
        )

    # Split into train and validation sets
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=df["label"]
    )

    print(f"Train set: {len(train_df)} videos, Val set: {len(val_df)} videos")

    # Save splits
    train_df.to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_labels.csv"), index=False)

    print(f"Saved train/val splits to {output_dir}")

    return train_df, val_df


def analyze_dataset(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str) -> None:
    """Analyze the dataset and save statistics.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        output_dir: Directory to save analysis

    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Combine dataframes for overall statistics
    df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val")])

    # Get class distribution
    class_counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    class_counts.columns = ["non-violent", "violent"]

    # Get frame count statistics
    frame_stats = df.groupby("split")["num_frames"].agg(["min", "max", "mean", "median", "std"])

    # Save statistics
    with open(os.path.join(output_dir, "dataset_analysis.txt"), "w") as f:
        f.write("=== Class Distribution ===\n")
        f.write(str(class_counts) + "\n\n")

        f.write("=== Frame Count Statistics ===\n")
        f.write(str(frame_stats) + "\n\n")

        f.write("=== Overall Dataset ===\n")
        f.write(f"Total videos: {len(df)}\n")
        f.write(f"Violent videos: {df['label'].sum()} ({df['label'].mean() * 100:.2f}%)\n")
        f.write(f"Non-violent videos: {len(df) - df['label'].sum()}\n")
        f.write(f"Total frames: {df['num_frames'].sum()}\n")
        f.write(f"Average frames per video: {df['num_frames'].mean():.2f}\n")

    # Plot class distribution
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x="label", hue="split")
        plt.title("Class Distribution")
        plt.xlabel("Label (0: Non-violent, 1: Violent)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "class_distribution.png"))

        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x="num_frames", hue="split", bins=30)
        plt.title("Frame Count Distribution")
        plt.xlabel("Number of Frames")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "frame_distribution.png"))

        print(f"Saved analysis plots to {output_dir}")
    except ImportError:
        print("Matplotlib or seaborn not available. Skipping plot generation.")
