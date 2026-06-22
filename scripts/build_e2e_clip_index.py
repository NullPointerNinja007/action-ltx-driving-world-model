import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--future", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.frames_csv)

    # Make sure frame ids are numeric and sorted
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df = df.dropna(subset=["frame_id"])
    df["frame_id"] = df["frame_id"].astype(int)

    rows = []
    clip_len = args.context + args.future

    for scenario_id, g in df.groupby("scenario_id"):
        g = g.sort_values("frame_id").reset_index(drop=True)

        # Only use contiguous frame sequences
        for start in range(0, len(g) - clip_len + 1, args.stride):
            window = g.iloc[start:start + clip_len]

            frame_ids = window["frame_id"].tolist()
            expected = list(range(frame_ids[0], frame_ids[0] + clip_len))

            if frame_ids != expected:
                continue

            context_rows = window.iloc[:args.context]
            future_rows = window.iloc[args.context:]

            rows.append({
                "clip_id": f"{scenario_id}_{frame_ids[0]:06d}",
                "scenario_id": scenario_id,
                "start_frame": frame_ids[0],
                "end_frame": frame_ids[-1],
                "context_paths": "|".join(context_rows["image_path"].tolist()),
                "future_paths": "|".join(future_rows["image_path"].tolist()),
                "context_frame_ids": "|".join(map(str, context_rows["frame_id"].tolist())),
                "future_frame_ids": "|".join(map(str, future_rows["frame_id"].tolist())),

                # Keep ego/action-ish data from the last context frame
                "past_pos_x": context_rows.iloc[-1].get("past_pos_x", "[]"),
                "past_pos_y": context_rows.iloc[-1].get("past_pos_y", "[]"),
                "future_pos_x": context_rows.iloc[-1].get("future_pos_x", "[]"),
                "future_pos_y": context_rows.iloc[-1].get("future_pos_y", "[]"),
            })

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Input frames: {len(df)}")
    print(f"Wrote clips: {len(out)}")
    print(f"Output: {args.out_csv}")


if __name__ == "__main__":
    main()