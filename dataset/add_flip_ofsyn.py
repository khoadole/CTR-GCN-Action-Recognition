"""Add horizontally flipped rows to an OF-SYN keypoint CSV.

For each original row, a flipped copy is appended with:
  - video key:  <original_key>_flip
  - x columns:  1 - x  (coordinates are in [0,1] relative to frame size)
  - everything else (y, score, frame, label) unchanged

Only direction-sensitive classes are flipped. Gravity-dependent classes
(lying, lie_down, fall, fallen) are skipped — flipping them creates patterns
that overlap with top-down-camera "sitting", causing misclassification.

Usage:
  python dataset/add_flip_ofsyn.py
  python dataset/add_flip_ofsyn.py --input path/to/ofsyn.csv --output path/to/ofsyn_selective_flip.csv
"""

import argparse
import csv
import os


INPUT_CSV = (
    "data/le2i_ofsyn/combine/csv/10_to_8_100_v2/csv_v2/ofsyn_yolo11m.csv"
)
OUTPUT_CSV = (
    "data/le2i_ofsyn/combine/csv/10_to_8_100_v2/csv_v2/ofsyn_yolo11m_selective_flip.csv"
)

# Only these classes get a flipped copy — they are direction-sensitive.
# lying, lie_down, fall, fallen are excluded: flipping them creates patterns
# visually similar to sitting from a top-down camera angle.
FLIP_CLASSES = {"walk", "walking", "standing", "sit_down", "sitting", "stand_up"}

# OF-SYN video keys start with "ofsyn_<class>_..."
OFSYN_CLASS_PREFIXES = sorted(
    ["sit_down", "lie_down", "stand_up", "walking", "walk",
     "fall", "fallen", "sitting", "lying", "standing", "other"],
    key=len, reverse=True,
)


def get_ofsyn_class(video_key: str) -> str:
    if video_key.startswith("ofsyn_"):
        suffix = video_key[len("ofsyn_"):]
        for cls in OFSYN_CLASS_PREFIXES:
            if suffix.startswith(cls + "_") or suffix == cls:
                return cls
    return ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=INPUT_CSV)
    p.add_argument("--output", default=OUTPUT_CSV)
    return p.parse_args()


def flip_row(row: dict, x_cols: list[str]) -> dict:
    flipped = dict(row)
    flipped["video"] = row["video"] + "_flip"
    for col in x_cols:
        val = row[col]
        if val == "" or val.lower() == "nan":
            flipped[col] = val
        else:
            flipped[col] = str(1.0 - float(val))
    return flipped


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.input, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames)
        x_cols = [c for c in fieldnames if c.endswith("_x")]

        with open(args.output, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            orig_rows = 0
            flip_rows = 0
            skipped_classes: dict[str, int] = {}

            for row in reader:
                writer.writerow(row)
                orig_rows += 1

                cls = get_ofsyn_class(str(row.get("video", "")))
                if cls in FLIP_CLASSES:
                    writer.writerow(flip_row(row, x_cols))
                    flip_rows += 1
                else:
                    skipped_classes[cls] = skipped_classes.get(cls, 0) + 1

                if orig_rows % 200_000 == 0:
                    print(f"  processed {orig_rows:,} rows...")

    print(f"Done. original={orig_rows:,}  flipped={flip_rows:,}  total={orig_rows + flip_rows:,}")
    print(f"Skipped flip by class: { {k: v for k, v in sorted(skipped_classes.items())} }")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
