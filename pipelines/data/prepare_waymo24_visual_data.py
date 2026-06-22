from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "waymo24-121f-visual-data-prep")
PROCESSED_V2_ROOT = os.environ.get("WAYMO24_PROCESSED_V2_ROOT", "")
DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
MODELS_VOLUME_NAME = "models"
GCP_SECRET_NAME = os.environ.get("GCP_MODAL_SECRET", "gcp-cs231n-waymo")

DATA_ROOT = Path("/data")
MODELS_ROOT = Path("/models")
REPO = Path("/workspace/LTX-Video")

FPS = 24
SOURCE_FPS = 10
TOTAL_FRAMES = 121
CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
WINDOW_STARTS = [0, 120, 240, 360]
WIDTH = 512
HEIGHT = 512
MAX_STAGE_CONTAINERS = int(os.environ.get("WAYMO24_MAX_STAGE_CONTAINERS", "64"))
DEFAULT_STAGE_WORKERS = int(os.environ.get("WAYMO24_STAGE_WORKERS", str(MAX_STAGE_CONTAINERS)))
GCS_FRAME_DOWNLOAD_WORKERS = int(os.environ.get("WAYMO24_GCS_FRAME_DOWNLOAD_WORKERS", "8"))
WINDOWS_PER_SCENARIO = 4
MAX_LATENT_CONTAINERS = int(os.environ.get("WAYMO24_MAX_LATENT_CONTAINERS", "4"))
DEFAULT_LATENT_WORKERS = int(os.environ.get("WAYMO24_LATENT_WORKERS", str(MAX_LATENT_CONTAINERS)))
LATENT_COMMIT_EVERY = int(os.environ.get("WAYMO24_LATENT_COMMIT_EVERY", "10"))

CKPT_2B = os.environ.get("LTX_CKPT_2B", "ltxv-2b-0.9.6-dev-04-25.safetensors")
DEFAULT_LATENT_PREFIX = os.environ.get("WAYMO24_LATENT_PREFIX", "latents")

if not PROCESSED_V2_ROOT:
    raise RuntimeError("Set WAYMO24_PROCESSED_V2_ROOT to the processed Waymo source root before running data prep.")


@dataclass(frozen=True)
class WindowRecord:
    split: str
    scenario_id: str
    window_idx: int
    window_id: str
    fps: int
    source_fps: int
    num_frames: int
    context_frames: int
    future_frames: int
    start_frame_24fps: int
    end_frame_24fps: int
    source_min_frame_id_10fps: int
    source_max_frame_id_10fps: int
    mp4_relpath: str
    latent_relpath: str


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME)
gcp_secret = modal.Secret.from_name(GCP_SECRET_NAME)

base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "av",
        "google-cloud-storage",
        "imageio",
        "imageio-ffmpeg",
        "imageio[ffmpeg]",
        "safetensors",
    )
)

latent_image = (
    base_image.pip_install(
        "torch",
        "torchvision",
        "huggingface_hub",
    )
    .run_commands(
        "git clone https://github.com/Lightricks/LTX-Video.git /workspace/LTX-Video",
        "cd /workspace/LTX-Video && python -m pip install -e '.[inference-script]'",
    )
)


def run_checked(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
    )


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        raise RuntimeError("ffmpeg not found; install ffmpeg or imageio-ffmpeg") from exc


def gcs_join(base: str, *parts: str) -> str:
    return base.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts if p.strip("/"))


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    bucket, _, blob = uri[5:].partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def configure_gcp_credentials() -> None:
    raw_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if raw_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        cred_path = Path("/tmp/gcp_service_account.json")
        cred_path.write_text(raw_json, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)


def gcs_client():
    configure_gcp_credentials()
    from google.cloud import storage
    from google.oauth2.credentials import Credentials
    from google.oauth2 import service_account

    access_token = os.environ.get("GCP_ACCESS_TOKEN")
    if access_token:
        return storage.Client(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or "cs231n-496521",
            credentials=Credentials(token=access_token),
        )

    raw_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if raw_json:
        info = json.loads(raw_json)
        credentials = service_account.Credentials.from_service_account_info(info)
        return storage.Client(project=info.get("project_id"), credentials=credentials)

    if os.environ.get("private_key") and os.environ.get("client_email"):
        service_account_keys = [
            "type",
            "project_id",
            "private_key_id",
            "private_key",
            "client_email",
            "client_id",
            "auth_uri",
            "token_uri",
            "auth_provider_x509_cert_url",
            "client_x509_cert_url",
            "universe_domain",
        ]
        info = {key: os.environ[key] for key in service_account_keys if os.environ.get(key)}
        credentials = service_account.Credentials.from_service_account_info(info)
        return storage.Client(project=info.get("project_id"), credentials=credentials)

    return storage.Client()


def download_gcs_file(uri: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_name, blob_name = parse_gs_uri(uri)
    blob = gcs_client().bucket(bucket_name).blob(blob_name)
    blob.download_to_filename(str(local_path))


def default_scenarios_uri(split: str) -> str:
    return gcs_join(PROCESSED_V2_ROOT, f"front_512_{split}", "metadata_clean", f"scenarios_{split}.csv")


def default_frames_prefix(split: str) -> str:
    return gcs_join(PROCESSED_V2_ROOT, f"front_512_{split}", "frames_front_512")


def download_scenarios_csv(split: str, tmp_dir: Path) -> list[dict[str, str]]:
    local_csv = tmp_dir / f"scenarios_{split}.csv"
    download_gcs_file(default_scenarios_uri(split), local_csv)
    with local_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def eligible_scenarios(rows: list[dict[str, str]], *, max_scenarios: int) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("is_contiguous") == "True"]
    return selected[:max_scenarios] if max_scenarios > 0 else selected


def download_scenario_frames(row: dict[str, str], split: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_id = row["scenario_id"]
    min_frame = int(row["min_frame_id"])
    max_frame = int(row["max_frame_id"])
    client = gcs_client()
    bucket_name, prefix = parse_gs_uri(gcs_join(default_frames_prefix(split), scenario_id))
    bucket = client.bucket(bucket_name)

    def fetch(frame_id: int) -> None:
        blob = bucket.blob(f"{prefix}/{frame_id:06d}.jpg")
        blob.download_to_filename(str(out_dir / f"{frame_id:06d}.jpg"))

    with ThreadPoolExecutor(max_workers=GCS_FRAME_DOWNLOAD_WORKERS) as pool:
        futures = [pool.submit(fetch, frame_id) for frame_id in range(min_frame, max_frame + 1)]
        for future in as_completed(futures):
            future.result()


def count_video_frames(path: Path) -> int:
    try:
        import imageio

        reader = imageio.get_reader(str(path))
        try:
            return int(reader.count_frames())
        finally:
            reader.close()
    except ModuleNotFoundError:
        import imageio_ffmpeg

        n_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
        return int(n_frames)


def build_24fps_full_video(frames_dir: Path, min_frame: int, out_mp4: Path) -> None:
    run_checked(
        [
            ffmpeg_bin(),
            "-y",
            "-framerate",
            str(SOURCE_FPS),
            "-start_number",
            str(min_frame),
            "-i",
            str(frames_dir / "%06d.jpg"),
            "-an",
            "-vf",
            (
                f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{HEIGHT},"
                "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
                "setsar=1"
            ),
            "-r",
            str(FPS),
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
    )


def extract_window(full_mp4: Path, start_frame: int, out_mp4: Path) -> None:
    run_checked(
        [
            ffmpeg_bin(),
            "-y",
            "-i",
            str(full_mp4),
            "-an",
            "-vf",
            f"trim=start_frame={start_frame}:end_frame={start_frame + TOTAL_FRAMES},setpts=N/({FPS}*TB)",
            "-frames:v",
            str(TOTAL_FRAMES),
            "-r",
            str(FPS),
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
    )
    n = count_video_frames(out_mp4)
    if n != TOTAL_FRAMES:
        raise RuntimeError(f"{out_mp4} has {n} frames, expected {TOTAL_FRAMES}")


def adaptive_window_starts(interpolated_frame_count: int) -> list[int]:
    if interpolated_frame_count < TOTAL_FRAMES:
        return []
    max_start = interpolated_frame_count - TOTAL_FRAMES
    return [round(idx * max_start / (WINDOWS_PER_SCENARIO - 1)) for idx in range(WINDOWS_PER_SCENARIO)]


def upload_file(local_path: Path, remote_relpath: str) -> None:
    dst = DATA_ROOT / remote_relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, dst)


def replace_latent_prefix(latent_relpath: str, latent_prefix: str) -> str:
    path = PurePosixPath(latent_relpath)
    parts = path.parts
    if not parts:
        raise ValueError("Empty latent_relpath")
    return str(PurePosixPath(latent_prefix, *parts[1:]))


def write_manifest(rows: list[WindowRecord], split: str, tmp_dir: Path) -> None:
    manifest_csv = tmp_dir / f"{split}_windows_24fps_121f.csv"
    manifest_jsonl = tmp_dir / f"{split}_windows_24fps_121f.jsonl"
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(WindowRecord.__dataclass_fields__.keys())
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    upload_file(manifest_csv, f"manifests/{manifest_csv.name}")
    upload_file(manifest_jsonl, f"manifests/{manifest_jsonl.name}")


def stage_one_scenario_mp4_windows(split: str, row: dict[str, str], scenario_idx: int) -> list[WindowRecord]:
    rows: list[WindowRecord] = []
    scenario_id = row["scenario_id"]
    min_frame = int(row["min_frame_id"])
    max_frame = int(row["max_frame_id"])
    with tempfile.TemporaryDirectory(prefix=f"waymo24_{split}_{scenario_id[:12]}_") as tmp:
        scenario_root = Path(tmp)
        frames_dir = scenario_root / "frames"
        download_scenario_frames(row, split, frames_dir)
        full_mp4 = scenario_root / f"{scenario_id}_full20s_24fps.mp4"
        build_24fps_full_video(frames_dir, min_frame, full_mp4)
        interpolated_frame_count = count_video_frames(full_mp4)
        starts = adaptive_window_starts(interpolated_frame_count)
        if len(starts) != WINDOWS_PER_SCENARIO:
            raise RuntimeError(
                f"{scenario_id} produced {interpolated_frame_count} interpolated frames; "
                f"cannot make {WINDOWS_PER_SCENARIO} windows of {TOTAL_FRAMES} frames."
            )

        for window_idx, start in enumerate(starts):
            window_id = f"{split}_{scenario_id[:12]}_w{window_idx:02d}_24fps_121f_ctx49_fut72"
            local_window = scenario_root / f"{window_id}.mp4"
            extract_window(full_mp4, start, local_window)
            mp4_relpath = f"mp4_windows/{split}/{scenario_id}/{window_id}.mp4"
            latent_relpath = f"latents/{split}/{scenario_id}/{window_id}.safetensors"
            upload_file(local_window, mp4_relpath)
            rows.append(
                WindowRecord(
                    split=split,
                    scenario_id=scenario_id,
                    window_idx=window_idx,
                    window_id=window_id,
                    fps=FPS,
                    source_fps=SOURCE_FPS,
                    num_frames=TOTAL_FRAMES,
                    context_frames=CONTEXT_FRAMES,
                    future_frames=FUTURE_FRAMES,
                    start_frame_24fps=start,
                    end_frame_24fps=start + TOTAL_FRAMES - 1,
                    source_min_frame_id_10fps=min_frame,
                    source_max_frame_id_10fps=max_frame,
                    mp4_relpath=mp4_relpath,
                    latent_relpath=latent_relpath,
                )
            )
    return rows


def stage_split_mp4_windows(split: str, *, max_scenarios: int) -> list[WindowRecord]:
    rows: list[WindowRecord] = []
    with tempfile.TemporaryDirectory(prefix=f"waymo24_{split}_") as tmp:
        tmp_dir = Path(tmp)
        scenarios = eligible_scenarios(download_scenarios_csv(split, tmp_dir), max_scenarios=max_scenarios)
        for scenario_idx, row in enumerate(scenarios):
            rows.extend(stage_one_scenario_mp4_windows(split, row, scenario_idx))

            if (scenario_idx + 1) % 25 == 0:
                print(f"[{split}] staged {scenario_idx + 1}/{len(scenarios)} scenarios, {len(rows)} windows")

        write_manifest(rows, split, tmp_dir)
    return rows


@app.function(
    image=base_image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    secrets=[gcp_secret],
)
def list_eligible_scenarios_modal(split: str, max_scenarios: int = 0) -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix=f"waymo24_manifest_{split}_") as tmp:
        return eligible_scenarios(download_scenarios_csv(split, Path(tmp)), max_scenarios=max_scenarios)


@app.function(
    image=base_image,
    cpu=4,
    memory=12288,
    timeout=30 * 60,
    max_containers=MAX_STAGE_CONTAINERS,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[gcp_secret],
)
def stage_one_scenario_mp4_windows_modal(split: str, row: dict[str, str], scenario_idx: int) -> dict[str, Any]:
    data_volume.reload()
    records = stage_one_scenario_mp4_windows(split, row, scenario_idx)
    data_volume.commit()
    return {
        "split": split,
        "scenario_id": row["scenario_id"],
        "scenario_idx": scenario_idx,
        "windows": len(records),
        "records": [asdict(record) for record in records],
    }


@app.function(
    image=base_image,
    cpu=2,
    memory=4096,
    timeout=20 * 60,
    volumes={str(DATA_ROOT): data_volume},
)
def write_manifest_modal(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    data_volume.reload()
    records = [WindowRecord(**row) for row in rows]
    with tempfile.TemporaryDirectory(prefix=f"waymo24_manifest_{split}_") as tmp:
        write_manifest(records, split, Path(tmp))
    data_volume.commit()
    return {
        "split": split,
        "windows": len(records),
        "scenarios": len({row["scenario_id"] for row in rows}),
        "manifest_csv": f"manifests/{split}_windows_24fps_121f.csv",
    }


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


@app.function(
    image=base_image,
    cpu=16,
    memory=32768,
    timeout=8 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[gcp_secret],
)
def stage_split_mp4_windows_modal(split: str, max_scenarios: int = 0) -> dict[str, Any]:
    data_volume.reload()
    rows = stage_split_mp4_windows(split, max_scenarios=max_scenarios)
    data_volume.commit()
    return {
        "split": split,
        "windows": len(rows),
        "scenarios": len({row.scenario_id for row in rows}),
        "manifest_csv": f"manifests/{split}_windows_24fps_121f.csv",
    }


def stage_split_mp4_windows_parallel(split: str, *, max_scenarios: int, stage_workers: int) -> dict[str, Any]:
    scenarios = list_eligible_scenarios_modal.remote(split, max_scenarios=max_scenarios)
    if not scenarios:
        manifest = write_manifest_modal.remote([], split)
        return {"split": split, "windows": 0, "scenarios": 0, "manifest_csv": manifest["manifest_csv"]}

    workers = max(1, min(stage_workers, MAX_STAGE_CONTAINERS))
    all_records: list[dict[str, Any]] = []
    for start_idx, chunk in enumerate(batched(scenarios, workers)):
        offset = start_idx * workers
        inputs = [(split, row, offset + idx) for idx, row in enumerate(chunk)]
        for result in stage_one_scenario_mp4_windows_modal.starmap(
            inputs,
            order_outputs=False,
            return_exceptions=True,
        ):
            if isinstance(result, Exception):
                raise result
            all_records.extend(result["records"])
            if len(all_records) % 100 == 0:
                print(f"[{split}] staged {len(all_records)} windows")

    all_records.sort(key=lambda row: (row["scenario_id"], row["window_idx"]))
    return write_manifest_modal.remote(all_records, split)


def ensure_checkpoint_symlink(ckpt_name: str = CKPT_2B) -> Path:
    src = MODELS_ROOT / "ltx" / ckpt_name
    dst = REPO / ckpt_name
    if not src.exists():
        raise FileNotFoundError(f"Missing checkpoint in Modal volume: {src}")
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    return dst


@app.function(
    image=latent_image,
    gpu=os.environ.get("LTX_MODAL_GPU", "H100"),
    cpu=8,
    memory=49152,
    timeout=8 * 60 * 60,
    max_containers=MAX_LATENT_CONTAINERS,
    volumes={str(DATA_ROOT): data_volume, str(MODELS_ROOT): models_volume},
)
def cache_latents_for_split_shard(
    split: str,
    shard_idx: int,
    num_shards: int,
    limit: int = 0,
    overwrite: bool = False,
    latent_prefix: str = DEFAULT_LATENT_PREFIX,
    ckpt_name: str = CKPT_2B,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(REPO))
    import torch
    from safetensors.torch import save_file
    from ltx_video.inference import load_media_file
    from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
    from ltx_video.models.autoencoders.vae_encode import vae_encode

    data_volume.reload()
    ckpt = ensure_checkpoint_symlink(ckpt_name)
    vae = CausalVideoAutoencoder.from_pretrained(ckpt).to("cuda", dtype=torch.bfloat16)
    vae.eval().requires_grad_(False)

    manifest_path = DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if limit > 0:
        records = records[:limit]
    shard_records = [row for idx, row in enumerate(records) if idx % num_shards == shard_idx]

    done = 0
    skipped = 0
    for row in shard_records:
        latent_relpath = replace_latent_prefix(row["latent_relpath"], latent_prefix)
        latent_path = DATA_ROOT / latent_relpath
        if latent_path.exists() and not overwrite:
            skipped += 1
            continue
        mp4_path = DATA_ROOT / row["mp4_relpath"]
        if not mp4_path.exists():
            raise FileNotFoundError(f"Missing MP4 window: {mp4_path}")

        media = load_media_file(
            media_path=str(mp4_path),
            height=HEIGHT,
            width=WIDTH,
            max_frames=TOTAL_FRAMES,
            padding=(0, 0, 0, 0),
            just_crop=True,
        ).to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            latents = vae_encode(media, vae, vae_per_channel_normalize=True).squeeze(0).cpu()
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"latents": latents},
            str(latent_path),
            metadata={
                "window_id": row["window_id"],
                "fps": str(FPS),
                "num_frames": str(TOTAL_FRAMES),
                "context_frames": str(CONTEXT_FRAMES),
                "future_frames": str(FUTURE_FRAMES),
                "vae_checkpoint": ckpt_name,
                "latent_prefix": latent_prefix,
            },
        )
        done += 1
        if done % LATENT_COMMIT_EVERY == 0:
            data_volume.commit()
            print(f"[{split} shard {shard_idx + 1}/{num_shards}] cached {done} latents, skipped {skipped}")

    data_volume.commit()
    return {
        "split": split,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "cached": done,
        "skipped": skipped,
        "total_seen": len(records),
        "shard_seen": len(shard_records),
        "latent_prefix": latent_prefix,
        "vae_checkpoint": ckpt_name,
    }


def cache_latents_for_split_parallel(
    split: str,
    *,
    limit: int,
    overwrite: bool,
    latent_workers: int,
    latent_prefix: str,
    ckpt_name: str,
) -> dict[str, Any]:
    workers = max(1, min(latent_workers, MAX_LATENT_CONTAINERS))
    inputs = [
        (split, shard_idx, workers, limit, overwrite, latent_prefix, ckpt_name)
        for shard_idx in range(workers)
    ]
    shard_results = []
    for result in cache_latents_for_split_shard.starmap(
        inputs,
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(result, Exception):
            raise result
        shard_results.append(result)
    shard_results.sort(key=lambda row: row["shard_idx"])
    return {
        "split": split,
        "workers": workers,
        "cached": sum(row["cached"] for row in shard_results),
        "skipped": sum(row["skipped"] for row in shard_results),
        "total_seen": max((row["total_seen"] for row in shard_results), default=0),
        "latent_prefix": latent_prefix,
        "vae_checkpoint": ckpt_name,
        "shards": shard_results,
    }


@app.local_entrypoint()
def main(
    splits_csv: str = "train,val",
    max_train_scenarios: int = 0,
    max_val_scenarios: int = 0,
    stage_mp4s: bool = True,
    cache_latents: bool = True,
    latent_limit: int = 0,
    overwrite_latents: bool = False,
    latent_workers: int = DEFAULT_LATENT_WORKERS,
    stage_workers: int = DEFAULT_STAGE_WORKERS,
    latent_prefix: str = DEFAULT_LATENT_PREFIX,
    ckpt_name: str = CKPT_2B,
) -> None:
    splits = [item.strip() for item in splits_csv.split(",") if item.strip()]
    staged: dict[str, int] = {}
    if stage_mp4s:
        for split in splits:
            max_scenarios = max_train_scenarios if split == "train" else max_val_scenarios
            result = stage_split_mp4_windows_parallel(
                split,
                max_scenarios=max_scenarios,
                stage_workers=stage_workers,
            )
            staged[split] = result["windows"]
            print(
                f"[{split}] staged {result['windows']} MP4 windows from "
                f"{result['scenarios']} scenarios on Modal with up to {min(stage_workers, MAX_STAGE_CONTAINERS)} workers"
            )

    cached: dict[str, Any] = {}
    if cache_latents:
        for split in splits:
            cached[split] = cache_latents_for_split_parallel(
                split,
                limit=latent_limit,
                overwrite=overwrite_latents,
                latent_workers=latent_workers,
                latent_prefix=latent_prefix,
                ckpt_name=ckpt_name,
            )

    print(json.dumps({"staged": staged, "cached": cached}, indent=2, sort_keys=True))
