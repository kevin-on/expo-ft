"""One background video encoder per environment, with bounded frame ownership."""

from concurrent.futures import ThreadPoolExecutor
import logging
import time


class EpisodeVideoWriter:
    """Own completed episode buffers until saved; never access cameras or robots.

    At most one completed episode is retained by this writer. If encoding falls
    behind an entire rollout/update cycle, the next submission waits instead of
    allowing full-resolution frame buffers to accumulate without a bound.
    """

    def __init__(self, save_video):
        self._save_video = save_video
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode-video")
        self._pending = None
        self._closed = False

    def submit(self, videos, video_dir, episode):
        """Take ownership of [(prefix, frame_list), ...]; caller must not mutate them."""
        if self._closed:
            raise RuntimeError("Video writer is closed")
        if self._pending is not None:
            if not self._pending.done():
                logging.getLogger(__name__).warning(
                    "Video encoder backlog: waiting for previous episode before queueing %d", episode
                )
            self._pending.result()
        self._pending = self._executor.submit(self._save, videos, video_dir, episode)

    def _save(self, videos, video_dir, episode):
        started = time.monotonic()
        for prefix, frames in videos:
            try:
                self._save_video(frames, video_dir, episode, prefix=prefix)
            except Exception:
                # Match synchronous video's nonfatal logging; still attempt other views.
                logging.getLogger(__name__).exception(
                    "Background video save failed: episode=%d prefix=%s", episode, prefix
                )
        logging.getLogger(__name__).info(
            "VIDEO_SAVE_FINISHED episode=%d elapsed_seconds=%.3f", episode, time.monotonic() - started
        )

    def close(self):
        """Drain queued encoding on normal disconnect/shutdown. Idempotent."""
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True)
