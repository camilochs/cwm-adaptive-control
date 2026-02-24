"""Storage utilities for trajectories."""

import json
import pickle
import gzip
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime

from .collector import Trajectory


class TrajectoryStorage:
    """Save and load trajectory data."""

    @staticmethod
    def save_json(
        trajectories: List[Trajectory],
        filepath: Union[str, Path],
        compress: bool = False,
        metadata: Optional[dict] = None,
    ):
        """Save trajectories to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "count": len(trajectories),
            "metadata": metadata or {},
            "trajectories": [t.to_dict() for t in trajectories],
        }

        if compress:
            if not filepath.suffix == ".gz":
                filepath = filepath.with_suffix(filepath.suffix + ".gz")
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return filepath

    @staticmethod
    def load_json(filepath: Union[str, Path]) -> List[Trajectory]:
        """Load trajectories from JSON file."""
        filepath = Path(filepath)

        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

        return [Trajectory.from_dict(t) for t in data["trajectories"]]

    @staticmethod
    def save_pickle(
        trajectories: List[Trajectory],
        filepath: Union[str, Path],
        compress: bool = True,
    ):
        """Save trajectories to pickle file (faster, more compact)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "trajectories": trajectories,
        }

        if compress:
            if not filepath.suffix == ".gz":
                filepath = filepath.with_suffix(filepath.suffix + ".gz")
            with gzip.open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return filepath

    @staticmethod
    def load_pickle(filepath: Union[str, Path]) -> List[Trajectory]:
        """Load trajectories from pickle file."""
        filepath = Path(filepath)

        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rb") as f:
                data = pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        return data["trajectories"]

    @staticmethod
    def merge_files(
        filepaths: List[Union[str, Path]],
        output_path: Union[str, Path],
        format: str = "json",
    ):
        """Merge multiple trajectory files into one."""
        all_trajectories = []

        for filepath in filepaths:
            filepath = Path(filepath)
            if filepath.suffix in [".json", ".gz"] and "json" in str(filepath):
                trajectories = TrajectoryStorage.load_json(filepath)
            else:
                trajectories = TrajectoryStorage.load_pickle(filepath)
            all_trajectories.extend(trajectories)

        if format == "json":
            return TrajectoryStorage.save_json(all_trajectories, output_path)
        else:
            return TrajectoryStorage.save_pickle(all_trajectories, output_path)

    @staticmethod
    def get_info(filepath: Union[str, Path]) -> dict:
        """Get metadata about a trajectory file without loading all data."""
        filepath = Path(filepath)

        if filepath.suffix == ".gz":
            opener = gzip.open
            mode = "rt" if "json" in str(filepath) else "rb"
        else:
            opener = open
            mode = "r" if filepath.suffix == ".json" else "rb"

        with opener(filepath, mode) as f:
            if "json" in str(filepath) or filepath.suffix == ".json":
                data = json.load(f)
                return {
                    "version": data.get("version"),
                    "timestamp": data.get("timestamp"),
                    "count": data.get("count"),
                    "metadata": data.get("metadata", {}),
                }
            else:
                data = pickle.load(f)
                return {
                    "version": data.get("version"),
                    "timestamp": data.get("timestamp"),
                    "count": len(data.get("trajectories", [])),
                }
