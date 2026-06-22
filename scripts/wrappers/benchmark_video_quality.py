"""Compatibility wrapper for the video quality benchmark pipeline."""

from pipelines.evaluation.benchmark_video_quality import *  # noqa: F401,F403
from pipelines.evaluation.benchmark_video_quality import main


if __name__ == "__main__":
    main()
