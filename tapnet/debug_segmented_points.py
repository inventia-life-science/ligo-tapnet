"""Debug script to visualize YAML-defined points on the initial image.

This helps verify that the points loaded from the YAML file align with the
image coordinates exactly as `segmented_tracking.py` expects them.
"""

import os
from typing import Optional, Tuple, List

import cv2
import numpy as np
import yaml


def load_points_from_yaml(
    yaml_path: str, max_points: Optional[int] = None
) -> List[Tuple[int, int]]:
  """Load points from a YAML file and optionally subsample."""
  with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

  points = data.get("points", [])

  if isinstance(points, list) and len(points) > 0:
    if isinstance(points[0], (list, tuple)):
      points = [tuple(p) for p in points]
    else:
      raise ValueError(f"Unexpected point format in YAML: {points[0]}")
  else:
    raise ValueError(f"No points found in YAML file: {yaml_path}")

  if max_points is not None and len(points) > max_points:
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    points = [points[i] for i in indices]
    print(
        f"Subsampled {len(data.get('points', []))} points to {len(points)} points"
    )

  return points


def main():
  # These paths should match those used in `segmented_tracking.py`.
  yaml_path = os.path.expanduser("~/repos/ligo-ml/mask_points.yaml")
  image_path = os.path.join(os.path.dirname(yaml_path), "original_image.png")

  print(f"Loading image from: {image_path}")
  image = cv2.imread(image_path)
  if image is None:
    raise ValueError(f"Failed to load image from: {image_path}")

  print(f"Image shape: {image.shape}")

  print(f"Loading points from: {yaml_path}")
  points = load_points_from_yaml(yaml_path, max_points=None)
  print(f"Loaded {len(points)} points")

  # Draw points exactly as `segmented_tracking.py` expects them:
  # YAML points are (x, y) in pixel coordinates on this image.
  vis = image.copy()
  for idx, (x, y) in enumerate(points):
    cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
    cv2.putText(
        vis,
        str(idx),
        (int(x) + 5, int(y) - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

  cv2.imshow("YAML Points Overlay", vis)
  print("Close the window or press any key in the window to exit.")
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()



