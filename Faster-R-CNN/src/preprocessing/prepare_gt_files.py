import argparse
import csv
import fnmatch
import re
from pathlib import Path


OUTPUT_FILES = {
    "7_cls": "gt_7_cls.txt",
    "6_cls": "gt_6_cls.txt",
    "5_cls": "gt_5_cls.txt",
    "4_cls": "gt_4_cls.txt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare class-specific GT files for Faster R-CNN experiments. "
            "The script reads the original MOT gt.txt file and gameinfo.ini, "
            "adds class IDs, and creates 7-, 6-, 5- and 4-class annotation files."
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing SNMOT sequence folders. The search is recursive.",
    )

    parser.add_argument(
        "--sequence-pattern",
        type=str,
        default="SNMOT-*",
        help="Folder name pattern for sequence directories.",
    )

    parser.add_argument(
        "--gt-filename",
        type=str,
        default="gt.txt",
        help="Name of the original GT file inside the gt/ directory.",
    )

    parser.add_argument(
        "--gameinfo-filename",
        type=str,
        default="gameinfo.ini",
        help="Name of the gameinfo file inside each sequence directory.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if a track ID or class name cannot be mapped.",
    )

    return parser.parse_args()


def normalize_class_name(class_name):
    class_name = class_name.strip().lower()
    class_name = re.sub(r"\s+", " ", class_name)
    return class_name


def class_name_to_7cls_id(class_name):
    """
    Mapping for the 7-class setup:

    1 = player_left
    2 = goalkeeper_left
    3 = player_right
    4 = goalkeeper_right
    5 = ball
    6 = referee
    7 = other
    """
    name = normalize_class_name(class_name)

    if name.startswith("goalkeepers team left") or name.startswith("goalkeeper team left"):
        return 2

    if name.startswith("goalkeepers team right") or name.startswith("goalkeeper team right"):
        return 4

    if name.startswith("player team left"):
        return 1

    if name.startswith("player team right"):
        return 3

    if name.startswith("ball"):
        return 5

    if name.startswith("referee"):
        return 6

    if name.startswith("other"):
        return 7

    return None


def remap_7cls_to_5cls(class_id):
    """
    Remapping from 7-class setup to 5-class setup:

    player_left      -> player
    goalkeeper_left  -> goalkeeper
    player_right     -> player
    goalkeeper_right -> goalkeeper
    ball             -> ball
    referee          -> referee
    other            -> other
    """
    class_map = {
        1: 1,  # player_left -> player
        2: 2,  # goalkeeper_left -> goalkeeper
        3: 1,  # player_right -> player
        4: 2,  # goalkeeper_right -> goalkeeper
        5: 3,  # ball -> ball
        6: 4,  # referee -> referee
        7: 5,  # other -> other
    }

    return class_map[class_id]


def read_track_classes(gameinfo_path):
    """
    Reads mapping track_id -> class_name from gameinfo.ini.

    Expected line format:
        trackletID_123 = player team left;
    """
    id_to_class = {}

    with open(gameinfo_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            match = re.match(r"trackletID_(\d+)\s*=\s*([^;]+)", line)

            if match:
                track_id = int(match.group(1))
                class_name = normalize_class_name(match.group(2))
                id_to_class[track_id] = class_name

    return id_to_class


def find_sequence_dirs(data_root, sequence_pattern, gt_filename, gameinfo_filename):
    """
    Recursively finds sequence folders containing:
        gt/<gt_filename>
        <gameinfo_filename>
    """
    data_root = Path(data_root)
    sequence_dirs = []

    for gt_path in data_root.rglob(f"gt/{gt_filename}"):
        sequence_dir = gt_path.parent.parent

        if not fnmatch.fnmatch(sequence_dir.name, sequence_pattern):
            continue

        gameinfo_path = sequence_dir / gameinfo_filename

        if not gameinfo_path.exists():
            print(f"WARNING: Missing gameinfo.ini for sequence: {sequence_dir}")
            continue

        sequence_dirs.append(sequence_dir)

    return sorted(sequence_dirs)


def prepare_rows_for_sequence(sequence_dir, gt_filename, gameinfo_filename, strict=False):
    gt_path = sequence_dir / "gt" / gt_filename
    gameinfo_path = sequence_dir / gameinfo_filename

    id_to_class = read_track_classes(gameinfo_path)

    rows_7_cls = []
    rows_6_cls = []
    rows_5_cls = []
    rows_4_cls = []

    total_rows = 0
    skipped_rows = 0
    unknown_rows = 0

    with open(gt_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            total_rows += 1

            try:
                track_id = int(row[1])
            except (ValueError, IndexError):
                skipped_rows += 1
                message = f"Invalid row format in {gt_path}: {row}"

                if strict:
                    raise ValueError(message)

                print(f"WARNING: {message}")
                continue

            class_name = id_to_class.get(track_id)

            if class_name is None:
                unknown_rows += 1
                message = f"Track ID {track_id} not found in {gameinfo_path}"

                if strict:
                    raise ValueError(message)

                print(f"WARNING: {message}")
                continue

            class_id_7 = class_name_to_7cls_id(class_name)

            if class_id_7 is None:
                unknown_rows += 1
                message = f"Unknown class name '{class_name}' for track ID {track_id}"

                if strict:
                    raise ValueError(message)

                print(f"WARNING: {message}")
                continue

            row_7 = row + [str(class_id_7)]
            rows_7_cls.append(row_7)

            # 6-class setup: same as 7-class setup, but without 'other'
            if class_id_7 != 7:
                rows_6_cls.append(row_7)

            class_id_5 = remap_7cls_to_5cls(class_id_7)
            row_5 = row + [str(class_id_5)]
            rows_5_cls.append(row_5)

            # 4-class setup: same as 5-class setup, but without 'other'
            if class_id_7 != 7:
                rows_4_cls.append(row_5)

    stats = {
        "total_rows": total_rows,
        "skipped_rows": skipped_rows,
        "unknown_rows": unknown_rows,
        "rows_7_cls": len(rows_7_cls),
        "rows_6_cls": len(rows_6_cls),
        "rows_5_cls": len(rows_5_cls),
        "rows_4_cls": len(rows_4_cls),
    }

    return rows_7_cls, rows_6_cls, rows_5_cls, rows_4_cls, stats


def write_rows(output_path, rows):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def process_sequence(sequence_dir, gt_filename, gameinfo_filename, strict=False):
    rows_7_cls, rows_6_cls, rows_5_cls, rows_4_cls, stats = prepare_rows_for_sequence(
        sequence_dir=sequence_dir,
        gt_filename=gt_filename,
        gameinfo_filename=gameinfo_filename,
        strict=strict,
    )

    gt_dir = sequence_dir / "gt"

    output_paths = {
        "7_cls": gt_dir / OUTPUT_FILES["7_cls"],
        "6_cls": gt_dir / OUTPUT_FILES["6_cls"],
        "5_cls": gt_dir / OUTPUT_FILES["5_cls"],
        "4_cls": gt_dir / OUTPUT_FILES["4_cls"],
    }

    write_rows(output_paths["7_cls"], rows_7_cls)
    write_rows(output_paths["6_cls"], rows_6_cls)
    write_rows(output_paths["5_cls"], rows_5_cls)
    write_rows(output_paths["4_cls"], rows_4_cls)

    return output_paths, stats


def main():
    args = parse_args()

    data_root = Path(args.data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    sequence_dirs = find_sequence_dirs(
        data_root=data_root,
        sequence_pattern=args.sequence_pattern,
        gt_filename=args.gt_filename,
        gameinfo_filename=args.gameinfo_filename,
    )

    if not sequence_dirs:
        raise RuntimeError(
            f"No sequence folders matching '{args.sequence_pattern}' were found under {data_root}"
        )

    print(f"Found {len(sequence_dirs)} sequences.")

    total_sequences = 0

    for sequence_dir in sequence_dirs:
        print(f"\nProcessing: {sequence_dir}")

        output_paths, stats = process_sequence(
            sequence_dir=sequence_dir,
            gt_filename=args.gt_filename,
            gameinfo_filename=args.gameinfo_filename,
            strict=args.strict,
        )

        total_sequences += 1

        print(f"  Total rows:     {stats['total_rows']}")
        print(f"  gt_7_cls rows:  {stats['rows_7_cls']}")
        print(f"  gt_6_cls rows:  {stats['rows_6_cls']}")
        print(f"  gt_5_cls rows:  {stats['rows_5_cls']}")
        print(f"  gt_4_cls rows:  {stats['rows_4_cls']}")

        if stats["skipped_rows"] > 0:
            print(f"  Skipped rows:   {stats['skipped_rows']}")

        if stats["unknown_rows"] > 0:
            print(f"  Unknown rows:   {stats['unknown_rows']}")

        for key, output_path in output_paths.items():
            print(f"  Saved {key}: {output_path}")

    print("\nDone.")
    print(f"Processed sequences: {total_sequences}")


if __name__ == "__main__":
    main()