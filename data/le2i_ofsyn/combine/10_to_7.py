"""Create merged train/test CSV from remapped le2i + sampled ofsyn.

Output inside one folder:
- train.csv = le2i_train + ofsyn(sample by video, default 30%)
- test.csv  = le2i_test  + ofsyn(sample by video, default 10%)

Sampling is done per class prefix in `video` column, not by row.
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = [
	"Sit down",
	"Lying Down",
	"Walking",
	"Stand up",
	"Standing",
	"Fall Down",
	"Sitting",
]

SOURCE_TO_TARGET_NAME_10_TO_7 = {
	"sit_down": "Sit down",
	"lying": "Lying Down",
	"lie_down": "Lying Down", # optional
	"walk": "Walking",
	"stand_up": "Stand up",
	"standing": "Standing",
	"fall": "Fall Down",
	"fallen": "Lying Down", # optional
	"sitting": "Sitting",
	"other": "Other",
}


# Handle schema mismatch between files produced by different pipelines.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
	"_left_eye_y": ("left_eye_y",),
	"left_eye_y": ("_left_eye_y",),
}

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
MAX_VALID_LABEL_ID_7CLS = len(CLASS_NAMES) - 1

# OF-SYN uses "walking" in video prefixes while some mappings use "walk".
SOURCE_TO_TARGET_NAME_10_TO_7_NORMALIZED = {
	key.strip().lower(): value for key, value in SOURCE_TO_TARGET_NAME_10_TO_7.items()
}
SOURCE_TO_TARGET_NAME_10_TO_7_NORMALIZED["walking"] = "Walking"

# Fallback mapping by OF-SYN label id in case a row has unusual/empty video prefix.
OFSYN_LABEL_ID_TO_NAME = {
	0: "walking",
	1: "fall",
	2: "fallen",
	3: "sit_down",
	4: "sitting",
	5: "lie_down",
	6: "lying",
	7: "stand_up",
	8: "standing",
	9: "other",
}


@dataclass
class ClassSampleStat:
	class_name: str
	total_videos: int
	train_videos: int
	test_videos: int
	train_rows: int = 0
	test_rows: int = 0


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Split and merge le2i + ofsyn CSV at video-level")
	parser.add_argument(
		"--le2i-train-csv",
		default="../../dataset/le2i_train_kpts.csv",
		help="Remapped le2i train CSV",
	)
	parser.add_argument(
		"--le2i-test-csv",
		default="../../dataset/le2i_test_kpts.csv",
		help="Remapped le2i test CSV",
	)
	parser.add_argument(
		"--ofsyn-csv",
		default="../../dataset/yolo11s_ofsyn.csv",
		help="OF-SYN CSV",
	)
	parser.add_argument(
		"--output-dir",
		default="../../dataset/le2i_ofsyn_split_7classes",
		help="Output folder path",
	)
	parser.add_argument(
		"--train-ofsyn-ratio",
		type=float,
		default=0.80,
		help="Per-class OF-SYN video ratio for train",
	)
	parser.add_argument(
		"--test-ofsyn-ratio",
		type=float,
		default=0.20,
		help="Per-class OF-SYN video ratio for test",
	)
	parser.add_argument(
		"--max-label-id",
		type=int,
		default=MAX_VALID_LABEL_ID_7CLS,
		help="Keep LE2I labels in range [0, max-label-id] and drop the rest",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--dry-run", action="store_true", help="Only compute sampling summary")
	return parser.parse_args()


def get_header(csv_path: Path) -> list[str]:
	with csv_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		return list(reader.fieldnames or [])


def normalize_row(row: dict[str, str], out_header: list[str]) -> dict[str, str]:
	out: dict[str, str] = {}
	for col in out_header:
		if col in row:
			out[col] = row[col]
			continue

		value = ""
		for alt in COLUMN_ALIASES.get(col, ()):  # pragma: no branch
			if alt in row:
				value = row[alt]
				break
		out[col] = value
	return out


def get_video_class(video_key: str) -> str:
	key = video_key.strip()
	if not key:
		return "unknown"
	if "/" not in key:
		return "unknown"
	return key.split("/", 1)[0]


def parse_label_id(raw: str) -> int | None:
	try:
		return int(float(raw))
	except (TypeError, ValueError):
		return None


def is_label_kept(label_id: int | None, max_label_id: int) -> bool:
	return label_id is not None and 0 <= label_id <= max_label_id


def get_ofsyn_source_class(row: dict[str, str]) -> str | None:
	video = str(row.get("video", "")).strip()
	video_class = get_video_class(video).strip().lower()
	if video_class:
		return video_class

	label_id = parse_label_id(row.get("label", ""))
	if label_id is None:
		return None
	return OFSYN_LABEL_ID_TO_NAME.get(label_id)


def map_ofsyn_to_le2i(row: dict[str, str]) -> tuple[str, int] | None:
	source_class = get_ofsyn_source_class(row)
	if source_class is None:
		return None

	target_class = SOURCE_TO_TARGET_NAME_10_TO_7_NORMALIZED.get(source_class)
	if target_class is None:
		return None

	target_label = CLASS_NAME_TO_ID[target_class]
	return target_class, target_label


def rewrite_video_prefix(video: str, target_prefix: str) -> str:
	if not video:
		return video
	if "/" in video:
		_, rest = video.split("/", 1)
		return f"{target_prefix}/{rest}"
	return f"{target_prefix}/{video}"


def collect_ofsyn_videos(ofsyn_csv: Path) -> tuple[dict[str, list[str]], int]:
	class_to_videos: dict[str, set[str]] = defaultdict(set)
	dropped_rows = 0
	with ofsyn_csv.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		if "video" not in (reader.fieldnames or []):
			raise ValueError(f"Missing `video` column in {ofsyn_csv}")

		for row in reader:
			mapped = map_ofsyn_to_le2i(row)
			if mapped is None:
				dropped_rows += 1
				continue
			target_class, _ = mapped

			video = str(row.get("video", "")).strip()
			if not video:
				continue
			class_to_videos[target_class].add(video)

	return {k: sorted(v) for k, v in class_to_videos.items()}, dropped_rows


def sample_videos_by_class(
	class_to_videos: dict[str, list[str]],
	train_ratio: float,
	test_ratio: float,
	seed: int,
) -> tuple[set[str], set[str], dict[str, ClassSampleStat]]:
	if not (0.0 <= train_ratio <= 1.0 and 0.0 <= test_ratio <= 1.0):
		raise ValueError("Ratios must be in [0, 1]")
	if train_ratio + test_ratio > 1.0:
		raise ValueError("train_ratio + test_ratio must be <= 1.0")

	rng = random.Random(seed)
	train_set: set[str] = set()
	test_set: set[str] = set()
	stats: dict[str, ClassSampleStat] = {}

	for class_name in sorted(class_to_videos):
		videos = list(class_to_videos[class_name])
		rng.shuffle(videos)

		total = len(videos)
		n_train = int(round(total * train_ratio))
		n_train = max(0, min(n_train, total))

		remain = total - n_train
		n_test = int(round(total * test_ratio))
		n_test = max(0, min(n_test, remain))

		train_videos = videos[:n_train]
		test_videos = videos[n_train:n_train + n_test]

		train_set.update(train_videos)
		test_set.update(test_videos)

		stats[class_name] = ClassSampleStat(
			class_name=class_name,
			total_videos=total,
			train_videos=len(train_videos),
			test_videos=len(test_videos),
		)

	return train_set, test_set, stats


def append_csv(
	src_path: Path,
	writer: csv.DictWriter,
	out_header: list[str],
	label_counter: Counter,
	max_label_id: int,
) -> tuple[int, int]:
	rows = 0
	dropped_rows = 0
	with src_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			out = normalize_row(row, out_header)

			label_id = parse_label_id(out.get("label", ""))
			if not is_label_kept(label_id, max_label_id):
				dropped_rows += 1
				continue

			writer.writerow(out)
			rows += 1
			label_counter[label_id] += 1
	return rows, dropped_rows


def append_ofsyn_selected(
	ofsyn_csv: Path,
	train_writer: csv.DictWriter,
	test_writer: csv.DictWriter,
	out_header: list[str],
	train_videos: set[str],
	test_videos: set[str],
	stats: dict[str, ClassSampleStat],
	train_label_counter: Counter,
	test_label_counter: Counter,
) -> tuple[int, int, int]:
	train_rows = 0
	test_rows = 0
	dropped_rows = 0

	with ofsyn_csv.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			video = str(row.get("video", "")).strip()
			if not video:
				continue

			out = normalize_row(row, out_header)
			mapped = map_ofsyn_to_le2i(out)
			if mapped is None:
				dropped_rows += 1
				continue
			target_class, target_label = mapped
			out["label"] = str(target_label)
			out["video"] = rewrite_video_prefix(str(out.get("video", "")), target_class)

			if video in train_videos:
				train_writer.writerow(out)
				train_rows += 1
				if target_class in stats:
					stats[target_class].train_rows += 1
				train_label_counter[target_label] += 1
			elif video in test_videos:
				test_writer.writerow(out)
				test_rows += 1
				if target_class in stats:
					stats[target_class].test_rows += 1
				test_label_counter[target_label] += 1

	return train_rows, test_rows, dropped_rows


def write_sampling_summary(path: Path, stats: dict[str, ClassSampleStat]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fields = [
		"class_name",
		"total_videos",
		"train_videos",
		"test_videos",
		"train_video_ratio",
		"test_video_ratio",
		"train_rows",
		"test_rows",
	]
	with path.open("w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		for class_name in sorted(stats):
			s = stats[class_name]
			total = max(1, s.total_videos)
			w.writerow(
				{
					"class_name": s.class_name,
					"total_videos": s.total_videos,
					"train_videos": s.train_videos,
					"test_videos": s.test_videos,
					"train_video_ratio": f"{s.train_videos / total:.4f}",
					"test_video_ratio": f"{s.test_videos / total:.4f}",
					"train_rows": s.train_rows,
					"test_rows": s.test_rows,
				}
			)


def print_label_counts(title: str, counter: Counter) -> None:
	print(title)
	if not counter:
		print("  (empty)")
		return
	for label in sorted(counter):
		print(f"  label {label:>2}: {counter[label]}")


def main() -> None:
	args = parse_args()
	max_label_id = int(args.max_label_id)
	if max_label_id < 0:
		raise ValueError("max-label-id must be >= 0")

	le2i_train_csv = Path(args.le2i_train_csv).resolve()
	le2i_test_csv = Path(args.le2i_test_csv).resolve()
	ofsyn_csv = Path(args.ofsyn_csv).resolve()
	out_dir = Path(args.output_dir).resolve()
	out_train = out_dir / "train.csv"
	out_test = out_dir / "test.csv"
	out_summary = out_dir / "ofsyn_sampling_summary.csv"

	for p in (le2i_train_csv, le2i_test_csv, ofsyn_csv):
		if not p.exists():
			raise FileNotFoundError(f"Input CSV not found: {p}")

	out_header = get_header(le2i_train_csv)
	test_header = get_header(le2i_test_csv)
	if out_header != test_header:
		raise ValueError("le2i train/test headers are different")

	class_to_videos, ofsyn_dropped_before_sampling = collect_ofsyn_videos(
		ofsyn_csv,
	)
	ofsyn_train_videos, ofsyn_test_videos, stats = sample_videos_by_class(
		class_to_videos=class_to_videos,
		train_ratio=float(args.train_ofsyn_ratio),
		test_ratio=float(args.test_ofsyn_ratio),
		seed=int(args.seed),
	)

	print(f"le2i_train_csv: {le2i_train_csv}")
	print(f"le2i_test_csv : {le2i_test_csv}")
	print(f"ofsyn_csv     : {ofsyn_csv}")
	print(f"output_dir    : {out_dir}")
	print(f"keep labels   : [0..{max_label_id}]")
	print(f"ofsyn train/test ratio: {args.train_ofsyn_ratio:.2f}/{args.test_ofsyn_ratio:.2f}")
	print(f"ofsyn sampled videos  : train={len(ofsyn_train_videos)}, test={len(ofsyn_test_videos)}")
	print(f"ofsyn dropped rows (class not in LE2I-7 before sampling): {ofsyn_dropped_before_sampling}")

	if args.dry_run:
		write_sampling_summary(out_summary, stats)
		print(f"[DRY-RUN] Saved summary: {out_summary}")
		return

	out_dir.mkdir(parents=True, exist_ok=True)
	train_label_counter: Counter = Counter()
	test_label_counter: Counter = Counter()

	with out_train.open("w", newline="", encoding="utf-8") as f_train, out_test.open(
		"w", newline="", encoding="utf-8"
	) as f_test:
		train_writer = csv.DictWriter(f_train, fieldnames=out_header)
		test_writer = csv.DictWriter(f_test, fieldnames=out_header)
		train_writer.writeheader()
		test_writer.writeheader()

		le2i_train_rows, le2i_train_dropped = append_csv(
			src_path=le2i_train_csv,
			writer=train_writer,
			out_header=out_header,
			label_counter=train_label_counter,
			max_label_id=max_label_id,
		)
		le2i_test_rows, le2i_test_dropped = append_csv(
			src_path=le2i_test_csv,
			writer=test_writer,
			out_header=out_header,
			label_counter=test_label_counter,
			max_label_id=max_label_id,
		)
		ofsyn_train_rows, ofsyn_test_rows, ofsyn_dropped_rows = append_ofsyn_selected(
			ofsyn_csv=ofsyn_csv,
			train_writer=train_writer,
			test_writer=test_writer,
			out_header=out_header,
			train_videos=ofsyn_train_videos,
			test_videos=ofsyn_test_videos,
			stats=stats,
			train_label_counter=train_label_counter,
			test_label_counter=test_label_counter,
		)

	write_sampling_summary(out_summary, stats)

	print("\nRows written")
	print(f"  train: le2i={le2i_train_rows}, ofsyn={ofsyn_train_rows}, total={le2i_train_rows + ofsyn_train_rows}")
	print(f"  test : le2i={le2i_test_rows}, ofsyn={ofsyn_test_rows}, total={le2i_test_rows + ofsyn_test_rows}")
	print(f"  dropped by filter: le2i_train={le2i_train_dropped}, le2i_test={le2i_test_dropped}, ofsyn_not_in_le2i7={ofsyn_dropped_rows}")
	print(f"  summary: {out_summary}")
	print(f"  train csv: {out_train}")
	print(f"  test  csv: {out_test}")

	print_label_counts("\nFinal TRAIN label counts:", train_label_counter)
	print_label_counts("\nFinal TEST label counts:", test_label_counter)


if __name__ == "__main__":
	main()
