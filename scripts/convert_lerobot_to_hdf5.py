"""Export preprocessed LeRobot v2.1 RGB episodes for the existing EXPO HDF5 loader.

No camera selection, reflection, resize, or normalization is performed. Requires
pyarrow, Pillow, NumPy and h5py; no LeRobot/Torch, GPU or robot SDK is imported.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pyarrow.parquet as pq


IMAGE_KEYS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")


def decode_image(value, root):
    encoded = value.get("bytes")
    source = io.BytesIO(encoded) if encoded is not None else root / value["path"]
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB")).copy()


def episode_arrays(table, root, features):
    """Map stored, unnormalized dataset values to the legacy HDF5 field names."""
    arrays = {}
    for key in IMAGE_KEYS:
        images = np.stack([decode_image(value, root) for value in table[key].to_pylist()])
        if images.shape[1:] != tuple(features[key]["shape"]):
            raise ValueError(f"Unexpected image shape for {key}: {images.shape}")
        arrays[f"saved_observation/{key}"] = images
    for key, width in (("cartesian_position", 6), ("gripper_position", 1), ("actions", 7)):
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        # LeRobot stores some shape-[1] features as scalar Arrow columns.
        if width == 1 and values.shape == (len(table),):
            values = values[:, None]
        if values.shape != (len(table), width) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key}: expected finite (T, {width}) values")
        if key == "actions":
            arrays["action/cartesian_velocity"] = values[:, :6]
            arrays["action/gripper_velocity"] = values[:, 6]
        else:
            arrays[f"saved_observation/{key}"] = values[:, 0] if width == 1 else values
    return arrays


def write_verified_episode(path, arrays, episode_index, fps):
    path.parent.mkdir(parents=True)
    partial = path.with_suffix(".partial")
    with h5py.File(partial, "x") as file:
        file.attrs["episode_index"] = episode_index
        file.attrs["fps"] = fps
        for key, value in arrays.items():
            options = dict(compression="lzf", chunks=(1, *value.shape[1:])) if value.ndim == 4 else {}
            file.create_dataset(key, data=value, **options)
    with h5py.File(partial, "r") as file:
        for key, value in arrays.items():
            np.testing.assert_array_equal(file[key][:], value, err_msg=f"Export mismatch: {path}/{key}")
    partial.rename(path)


def convert(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or any(info["features"][key]["dtype"] != "image" for key in IMAGE_KEYS):
        raise ValueError("Expected LeRobot v2.1 with image features (external video is unsupported)")
    for key in ("cartesian_position", "gripper_position", "actions"):
        if info["features"][key]["dtype"] != "float32":
            raise ValueError(f"Expected float32 {key}; refusing implicit precision conversion")
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    if len(episodes) != info["total_episodes"] or sum(ep["length"] for ep in episodes) != info["total_frames"]:
        raise ValueError("Episode metadata counts do not match info.json")
    output.mkdir(parents=True, exist_ok=False)
    exports, total_frames = [], 0
    for index, episode in enumerate(episodes):
        if episode["episode_index"] != index or episode["length"] <= 0:
            raise ValueError("Expected consecutive nonempty episodes")
        source = dataset / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
        table = pq.read_table(source, use_threads=False)
        if len(table) != episode["length"]:
            raise ValueError(f"Episode length mismatch: {source}")
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(len(table), index))
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(len(table)))
        np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(total_frames, total_frames + len(table)))
        arrays = episode_arrays(table, dataset, info["features"])
        target = output / str(index) / "traj.hdf5"
        write_verified_episode(target, arrays, index, info["fps"])
        exports.append(dict(episode_index=index, frames=len(table), source_parquet=str(source),
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            hdf5=str(target.relative_to(output))))
        total_frames += len(table)
        print(f"{dataset.name}: verified {index + 1}/{len(episodes)} episodes ({total_frames} frames)", flush=True)
    report = dict(source_dataset=str(dataset), episodes=len(exports), frames=total_frames, fps=info["fps"],
                  verification="All decoded RGB/state/action arrays equal source; episode/frame order checked",
                  source_episodes=exports)
    (output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.dataset, args.output)
