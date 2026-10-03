import logging
import pathlib
from datetime import datetime

import imageio


def save_episode_video(frames, video_dir, ep_count, fps=30, quality=8, prefix="raw", encoder_threads=2):
    """Save buffered raw frames to disk. Call when episode ends."""
    if encoder_threads < 1:
        raise ValueError("encoder_threads must be positive")
    if not video_dir or not frames:
        return
    ts = datetime.now().strftime("%m%d_%H%M%S")
    out_path = pathlib.Path(video_dir) / f"{prefix}_{ts}_train_ep{ep_count:06d}.mp4"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out_path), frames, fps=fps, quality=quality, codec="libx264", macro_block_size=1,
                         output_params=["-threads", str(encoder_threads)])
        logging.getLogger(__name__).info("Saved raw video to %s", out_path)
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to save raw video: %s", e)
