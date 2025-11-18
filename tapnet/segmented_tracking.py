# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Point tracking from YAML-defined points using the PyTorch TAPIR implementation."""

import os
import sys
import ctypes
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tree
import yaml

from tapnet.torch_tapir import tapir_model


def load_points_from_yaml(yaml_path: str, max_points: Optional[int] = None) -> List[Tuple[int, int]]:
  """Load points from a YAML file and optionally subsample.
  
  Args:
    yaml_path: Path to YAML file containing points.
    max_points: Maximum number of points to return. If None, returns all points.
                If the file has more points, they will be subsampled uniformly.
  
  Returns:
    List of (x, y) tuples representing the points.
  """
  with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
  
  points = data.get('points', [])
  
  # Convert to list of tuples if needed
  if isinstance(points, list) and len(points) > 0:
    # Handle YAML tuple format: [[x, y], ...] or [(x, y), ...]
    if isinstance(points[0], (list, tuple)):
      points = [tuple(p) for p in points]
    else:
      raise ValueError(f"Unexpected point format in YAML: {points[0]}")
  else:
    raise ValueError(f"No points found in YAML file: {yaml_path}")
  
  # Subsample if too many points
  if max_points is not None and len(points) > max_points:
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    points = [points[i] for i in indices]
    print(f"Subsampled {len(data.get('points', []))} points to {len(points)} points")
  
  return points


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], uint8/float.

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], float32.
  """
  frames = frames.float()
  frames = frames / 255 * 2 - 1
  return frames


def online_model_init(
    model: tapir_model.TAPIR,
    frames: torch.Tensor,
    points: torch.Tensor,
) -> Any:
  """Initialize query features for the query points."""
  frames = preprocess_frames(frames)
  feature_grids = model.get_feature_grids(frames, is_training=False)
  features = model.get_query_features(
      frames,
      is_training=False,
      query_points=points,
      feature_grids=feature_grids,
  )
  return features


def postprocess_occlusions(occlusions: torch.Tensor,
                           expected_dist: torch.Tensor) -> torch.Tensor:
  """Post-process occlusions to get visibility mask."""
  visibles = (1 - torch.sigmoid(occlusions)) * (
      1 - torch.sigmoid(expected_dist)
  ) > 0.5
  return visibles


def online_model_predict(
    model: tapir_model.TAPIR,
    frames: torch.Tensor,
    features: Any,
    causal_context: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
  """Compute point tracks and occlusions given frames and query points."""
  frames = preprocess_frames(frames)
  feature_grids = model.get_feature_grids(frames, is_training=False)
  trajectories = model.estimate_trajectories(
      frames.shape[-3:-1],
      is_training=False,
      feature_grids=feature_grids,
      query_features=features,
      query_points_in_video=None,
      query_chunk_size=64,
      causal_context=causal_context,
      get_causal_context=True,
  )
  causal_context = trajectories["causal_context"]
  del trajectories["causal_context"]
  # Take only the predictions for the final resolution.
  tracks = trajectories["tracks"][-1]
  occlusions = trajectories["occlusion"][-1]
  uncertainty = trajectories["expected_dist"][-1]
  visibles = postprocess_occlusions(occlusions, uncertainty)
  return tracks, visibles, causal_context


def track_points_from_yaml(
    initial_frame: np.ndarray,
    yaml_path: Optional[str] = None,
    checkpoint_path: str = "tapnet/checkpoints/causal_bootstapir_checkpoint.pt",
    max_points: Optional[int] = 64,
    provided_points: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[tapir_model.TAPIR, Any, Any, List[Tuple[int, int]], torch.device]:
  """Initialize PyTorch TAPIR tracking with points from a YAML file or provided points.

  Args:
    initial_frame: Initial RGB/BGR image as numpy array of shape (H, W, 3), uint8.
    yaml_path: Path to YAML file containing points to track. Ignored if provided_points is given.
    checkpoint_path: Path to PyTorch TAPIR checkpoint file.
    max_points: Maximum number of points to track. If None, tracks all points.
    provided_points: Optional list of (x, y) tuples to use directly. If provided, yaml_path is ignored.

  Returns:
    Tuple of:
      - model: Initialized PyTorch TAPIR model
      - query_features: Initial query features for tracking
      - causal_state: Initial causal state for online tracking
      - selected_points: List of (x, y) tuples selected for tracking
      - device: torch.device used by the model
  """
  # Load points from YAML or use provided points
  if provided_points is not None:
    points = provided_points
    print(f"Tracking {len(points)} provided points")
  elif yaml_path is not None:
    points = load_points_from_yaml(yaml_path, max_points=max_points)
    print(f"Tracking {len(points)} points from {yaml_path}")
  else:
    raise ValueError("Either yaml_path or provided_points must be provided")

  if len(points) == 0:
    raise ValueError("No points to track")

  # Select device and warm up cuDNN if available
  if torch.cuda.is_available():
    device = torch.device("cuda")
    try:
      dummy = torch.randn(1, 1, 5, 5, device=device)
      dummy_conv = torch.nn.Conv2d(1, 1, 3, padding=1).to(device)
      _ = dummy_conv(dummy)
      torch.cuda.synchronize()
      print("cuDNN initialized successfully")
    except Exception as e:
      print(f"Warning: cuDNN initialization failed: {e}")
      print("Disabling cuDNN - using fallback CUDA operations")
      torch.backends.cudnn.enabled = False
      torch.backends.cudnn.allow_tf32 = False
  else:
    device = torch.device("cpu")

  # Initialize TAPIR model
  print("Creating PyTorch TAPIR model...")
  model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
  print("Loading checkpoint...")
  model.load_state_dict(torch.load(checkpoint_path, map_location=device))
  model = model.to(device).eval()
  torch.set_grad_enabled(False)

  # Convert points to TAPIR format: (t, y, x) where t=0 for initial frame.
  # YAML points are (x, y).
  num_points = len(points)
  query_points = torch.zeros([num_points, 3], dtype=torch.float32, device=device)
  for i, (x, y) in enumerate(points):
    query_points[i, 0] = 0.0
    query_points[i, 1] = float(y)
    query_points[i, 2] = float(x)

  # Convert initial frame to tensor on device
  # Shape: (H, W, 3) -> treat as is; model expects [..., H, W, 3]
  frame_tensor = torch.tensor(initial_frame, device=device)

  # Initialize query features
  query_features = online_model_init(
      model=model,
      frames=frame_tensor[None, None],  # [1, 1, H, W, 3]
      points=query_points[None, :],     # [1, N, 3]
  )

  # Construct initial causal state and move it to device
  causal_state = model.construct_initial_causal_state(
      num_points, len(query_features.resolutions) - 1
  )
  causal_state = tree.map_structure(
      lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, causal_state
  )

  # Warm up prediction
  _tracks, _visibles, causal_state = online_model_predict(
      model=model,
      frames=frame_tensor[None, None],
      features=query_features,
      causal_context=causal_state,
  )

  return model, query_features, causal_state, points, device


def track_frame(
    model: tapir_model.TAPIR,
    device: torch.device,
    query_features: Any,
    causal_state: Any,
    frame: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Any]:
  """Track points in a new frame using the PyTorch TAPIR model.

  Args:
    model: Initialized PyTorch TAPIR model.
    device: torch.device on which the model and state reside.
    query_features: Query features from initialization.
    causal_state: Current causal state for online tracking.
    frame: New frame as numpy array of shape (H, W, 3), uint8 (BGR from cv2).

  Returns:
    Tuple of:
      - prediction: Dictionary with 'tracks', 'occlusion', 'expected_dist', 'visibles'
      - causal_state: Updated causal state for next frame
  """
  # Convert frame to tensor on device
  frame_tensor = torch.tensor(frame, device=device)

  tracks_t, visibles_t, updated_causal_state = online_model_predict(
      model=model,
      frames=frame_tensor[None, None],  # [1, 1, H, W, 3]
      features=query_features,
      causal_context=causal_state,
  )

  # Move to CPU numpy for visualization
  # tracks_t shape: [1, num_points, 1, 2]
  tracks_np = tracks_t[0, :, 0].detach().cpu().numpy()  # [num_points, 2]
  visibles_np = visibles_t[0, :, 0].detach().cpu().numpy().astype(bool)

  return {
      "tracks": tracks_np,            # (num_points, 2) - (x, y)
      "visibles": visibles_np,        # (num_points,)
  }, updated_causal_state


def crop_frame_to_square(image: np.ndarray) -> Tuple[np.ndarray, int, int]:
  """Crop image to square by removing pixels from the longer dimension.
  
  This matches the behavior of get_frame in pytorch_live_demo.py.
  
  Args:
    image: Image array of shape (H, W, 3).
    
  Returns:
    Tuple of:
      - cropped_image: Square cropped image
      - x_offset: Number of pixels removed from left (0 if height > width)
      - y_offset: Number of pixels removed from top (0 if width > height)
  """
  h, w = image.shape[:2]
  trunc = abs(w - h) // 2
  if w > h:
    # Remove from left and right
    cropped = image[:, trunc:-trunc]
    return cropped, trunc, 0
  elif h > w:
    # Remove from top and bottom
    cropped = image[trunc:-trunc, :]
    return cropped, 0, trunc
  else:
    # Already square
    return image, 0, 0


def draw_tracks_on_frame(
    frame: np.ndarray,
    tracks: np.ndarray,
    visibles: np.ndarray,
) -> np.ndarray:
  """Draw tracked points on a frame similar to live_demo."""
  output = frame.copy()
  for idx, (x, y) in enumerate(tracks):
    if np.isnan(x) or np.isnan(y):
      continue
    color = (0, 255, 0) if visibles[idx] else (0, 0, 255)
    cv2.circle(output, (int(x), int(y)), 4, color, -1)
  return output


if __name__ == "__main__":
  # Configuration
  yaml_path = os.path.expanduser("~/repos/ligo-ml/mask_points.yaml")
  checkpoint_path = "tapnet/checkpoints/causal_bootstapir_checkpoint.pt"
  max_points = 30  # Subsample if more points than this
  
  # Load initial image from file (same directory as YAML)
  yaml_dir = os.path.dirname(yaml_path)
  image_path = os.path.join(yaml_dir, "original_image.png")
  
  print(f"Loading initial image from: {image_path}")
  if not os.path.exists(image_path):
    raise FileNotFoundError(f"Initial image not found: {image_path}")
  
  initial_frame_raw = cv2.imread(image_path)
  if initial_frame_raw is None:
    raise ValueError(f"Failed to load image from: {image_path}")

  print(f"Initial frame shape (raw): {initial_frame_raw.shape}")
  
  # Apply the same cropping as get_frame in pytorch_live_demo.py
  initial_frame, x_offset, y_offset = crop_frame_to_square(initial_frame_raw)
  print(f"Initial frame shape (cropped): {initial_frame.shape}, offset: (x={x_offset}, y={y_offset})")
  
  # Load points and adjust for cropping
  print(f"Loading points from: {yaml_path}")
  points_raw = load_points_from_yaml(yaml_path, max_points=None)  # Load all first
  
  # Adjust points for cropping: subtract offset from x if width was cropped, y if height was cropped
  selected_points = [(x - x_offset, y - y_offset) for x, y in points_raw]
  
  # Filter out points that are now outside the cropped image bounds
  h_cropped, w_cropped = initial_frame.shape[:2]
  selected_points = [(x, y) for x, y in selected_points 
                     if 0 <= x < w_cropped and 0 <= y < h_cropped]
  
  # Subsample if needed
  if max_points is not None and len(selected_points) > max_points:
    indices = np.linspace(0, len(selected_points) - 1, max_points, dtype=int)
    selected_points = [selected_points[i] for i in indices]
    print(f"Subsampled to {len(selected_points)} points after cropping")
  
  if len(selected_points) == 0:
    raise ValueError("No points remain after cropping adjustment. Check image dimensions and point coordinates.")
  
  print(f"Using {len(selected_points)} points after cropping adjustment")

  # Initialize PyTorch TAPIR tracking with points from YAML
  model, query_features, causal_state, _, device = track_points_from_yaml(
      initial_frame=initial_frame,
      yaml_path=None,  # Pass None since we're providing points directly
      checkpoint_path=checkpoint_path,
      max_points=None,  # Already subsampled
      provided_points=selected_points,  # Pass the adjusted points
  )
  
  print(f"Initialized tracking for {len(selected_points)} points")

  # Run tracking on the initial image and display it before starting the camera stream
  initial_prediction, causal_state = track_frame(
      model=model,
      device=device,
      query_features=query_features,
      causal_state=causal_state,
      frame=initial_frame,
  )

  # Debug: compare YAML points vs model-predicted points on the initial frame.
  init_tracks = initial_prediction["tracks"]  # (num_points, 2) - (x_pred, y_pred)
  print("Initial frame point comparison (YAML vs prediction):")
  for i, (x_gt, y_gt) in enumerate(selected_points):
    x_pred, y_pred = init_tracks[i]
    dx = float(x_pred) - float(x_gt)
    dy = float(y_pred) - float(y_gt)
    dist = (dx ** 2 + dy ** 2) ** 0.5
    print(
        f"  Point {i:02d}: "
        f"GT=({x_gt:.1f}, {y_gt:.1f}) "
        f"Pred=({x_pred:.1f}, {y_pred:.1f}) "
        f"Δ=({dx:+.2f}, {dy:+.2f}), |Δ|={dist:.2f}px"
    )

  initial_display = draw_tracks_on_frame(
      initial_frame.copy(),
      initial_prediction["tracks"],
      initial_prediction["visibles"],
  )
  cv2.imshow("Point Tracking", initial_display)
  cv2.waitKey(1)

  # Initialize camera for subsequent frames
  print("Initializing camera for tracking...")
  vc = cv2.VideoCapture(0)
  vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

  if not vc.isOpened():
    raise ValueError("Unable to open camera. Please ensure a camera is connected.")

  print("Press ESC to exit, SPACE to pause/resume")

  paused = False
  frame_count = 0
  t_start = time.time()
  current_display = initial_display
  total_points = len(selected_points)

  while True:
    if not paused:
      ret, frame_raw = vc.read()
      if not ret:
        print("Failed to read frame from camera")
        break

      # Apply the same cropping as initial frame
      frame, _, _ = crop_frame_to_square(frame_raw)

      prediction, causal_state = track_frame(
          model=model,
          device=device,
          query_features=query_features,
          causal_state=causal_state,
          frame=frame,
      )

      current_display = draw_tracks_on_frame(
          frame,
          prediction["tracks"],
          prediction["visibles"],
      )

      frame_count += 1
      if frame_count % 30 == 0:
        elapsed = time.time() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        visible_count = int(np.sum(prediction["visibles"]))
        print(
            f"Frame {frame_count}: {fps:.1f} fps, "
            f"{visible_count}/{total_points} points visible"
        )

    cv2.imshow("Point Tracking", current_display)

    key = cv2.waitKey(1 if not paused else 50) & 0xFF
    if key == 27:  # ESC
      break
    if key == 32:  # SPACE
      paused = not paused
      state = "Paused" if paused else "Resumed"
      print(state)

  vc.release()
  cv2.destroyAllWindows()
  print("Tracking stopped.")
