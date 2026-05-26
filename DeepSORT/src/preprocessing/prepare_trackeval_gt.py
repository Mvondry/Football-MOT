from __future__ import annotations

import argparse
import configparser
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a flat TrackEval GT structure from SoccerNet-Tracking sequences. "
            "The script copies GT files and seqinfo.ini files into a structure expected "
            "by sn-trackeval / TrackEval."
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "Root directory containing SoccerNet-Tracking sequences. "
            "It can contain train/SNMOT-xxx and test/SNMOT-xxx subdirectories."
        ),
    )

    parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help=(
            "Path to split file. Each line should contain either an absolute path, "
            "a path relative to data-root, or a sequence name such as SNMOT-060."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="trackeval_data",
        help="Output directory where TrackEval-compatible data will be created.",
    )

    parser.add_argument(
        "--seqmap-name",
        type=str,
        default=None,
        help=(
            "Name of the generated seqmap file. If not set, the split filename stem is used."
        ),
    )

    parser.add_argument(
        "--gt-filename",
        type=str,
        default="gt.txt",
        help=(
            "Name of the source GT file inside each sequence gt/ directory. "
            "For HOTA tracking evaluation, the original gt.txt is usually used."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )

    return parser.parse_args()


def read_split_file(split_file: str | Path) -> List[Path]:
    split_file = Path(split_file)

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    sequence_paths = []

    with open(split_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().strip('"').strip("'")

            if not line:
                continue

            sequence_paths.append(Path(line))

    return sequence_paths


def resolve_sequence_path(data_root: str | Path, sequence_path: str | Path) -> Path:
    data_root = Path(data_root)
    sequence_path = Path(sequence_path)

    if sequence_path.is_absolute():
        return sequence_path

    direct_path = data_root / sequence_path

    if direct_path.exists():
        return direct_path

    matches = [
        path
        for path in data_root.rglob(sequence_path.name)
        if path.is_dir() and path.name == sequence_path.name
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple folders named '{sequence_path.name}' found under {data_root}. "
            f"Use a more specific path in the split file, e.g. 'train/{sequence_path.name}'."
        )

    raise FileNotFoundError(
        f"Sequence directory not found: {sequence_path} under {data_root}"
    )


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []

    image_paths = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    def sort_key(path: Path):
        return int(path.stem) if path.stem.isdigit() else path.stem

    return sorted(image_paths, key=sort_key)


def read_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size

    return width, height


def read_existing_seqinfo(seqinfo_path: Path) -> Optional[configparser.ConfigParser]:
    if not seqinfo_path.exists():
        return None

    config = configparser.ConfigParser()
    config.read(seqinfo_path, encoding="utf-8")

    if "Sequence" not in config:
        return None

    return config


def create_seqinfo_from_images(sequence_dir: Path, sequence_name: str) -> configparser.ConfigParser:
    image_dir = sequence_dir / "img1"
    images = list_images(image_dir)

    if images:
        width, height = read_image_size(images[0])
        seq_length = len(images)
    else:
        width, height = 0, 0
        seq_length = 0

    config = configparser.ConfigParser()
    config["Sequence"] = {
        "name": sequence_name,
        "imDir": "img1",
        "frameRate": "25",
        "seqLength": str(seq_length),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ".jpg",
    }

    return config


def write_seqinfo(config: configparser.ConfigParser, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        config.write(f)


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        return

    shutil.copy2(src, dst)


def prepare_sequence_gt(
    sequence_dir: Path,
    output_gt_root: Path,
    gt_filename: str,
    overwrite: bool = False,
) -> str:
    sequence_name = sequence_dir.name

    source_gt = sequence_dir / "gt" / gt_filename
    source_seqinfo = sequence_dir / "seqinfo.ini"

    if not source_gt.exists():
        raise FileNotFoundError(f"GT file not found: {source_gt}")

    target_sequence_dir = output_gt_root / sequence_name
    target_gt = target_sequence_dir / "gt" / "gt.txt"
    target_seqinfo = target_sequence_dir / "seqinfo.ini"

    copy_file(
        src=source_gt,
        dst=target_gt,
        overwrite=overwrite,
    )

    existing_seqinfo = read_existing_seqinfo(source_seqinfo)

    if existing_seqinfo is not None:
        copy_file(
            src=source_seqinfo,
            dst=target_seqinfo,
            overwrite=overwrite,
        )
    else:
        seqinfo = create_seqinfo_from_images(
            sequence_dir=sequence_dir,
            sequence_name=sequence_name,
        )
        if overwrite or not target_seqinfo.exists():
            write_seqinfo(seqinfo, target_seqinfo)

    return sequence_name


def write_seqmap(sequence_names: List[str], seqmap_path: Path) -> None:
    seqmap_path.parent.mkdir(parents=True, exist_ok=True)

    with open(seqmap_path, "w", encoding="utf-8") as f:
        f.write("name\n")

        for sequence_name in sequence_names:
            f.write(f"{sequence_name}\n")


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    output_root = Path(args.output_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    sequence_paths = read_split_file(split_file)

    if not sequence_paths:
        raise RuntimeError(f"No sequences found in split file: {split_file}")

    seqmap_name = args.seqmap_name
    if seqmap_name is None:
        seqmap_name = f"{split_file.stem}_seqmap.txt"

    if not seqmap_name.endswith(".txt"):
        seqmap_name = f"{seqmap_name}.txt"

    output_gt_root = output_root / "gt"
    seqmap_path = output_root / "seqmaps" / seqmap_name

    sequence_names = []
    seen_names = set()

    print("=== TrackEval GT preparation ===")
    print(f"Data root:    {data_root}")
    print(f"Split file:   {split_file}")
    print(f"Output root:  {output_root}")
    print(f"GT filename:  {args.gt_filename}")
    print(f"Seqmap file:  {seqmap_path}")
    print()

    for sequence_path in sequence_paths:
        sequence_dir = resolve_sequence_path(
            data_root=data_root,
            sequence_path=sequence_path,
        )

        sequence_name = sequence_dir.name

        if sequence_name in seen_names:
            raise RuntimeError(
                f"Duplicate sequence name '{sequence_name}' in split. "
                "TrackEval requires unique sequence names."
            )

        prepared_name = prepare_sequence_gt(
            sequence_dir=sequence_dir,
            output_gt_root=output_gt_root,
            gt_filename=args.gt_filename,
            overwrite=args.overwrite,
        )

        sequence_names.append(prepared_name)
        seen_names.add(prepared_name)

        print(f"[OK] {sequence_dir} -> {output_gt_root / prepared_name}")

    write_seqmap(sequence_names, seqmap_path)

    print()
    print("Done.")
    print(f"Prepared sequences: {len(sequence_names)}")
    print(f"GT folder:          {output_gt_root}")
    print(f"Seqmap file:        {seqmap_path}")
    print()
    print("Use these values in TrackEval:")
    print(f"  GT_FOLDER       = {output_gt_root}")
    print(f"  SEQMAP_FILE     = {seqmap_path}")
    print("  SKIP_SPLIT_FOL  = True")


if __name__ == "__main__":
    main()