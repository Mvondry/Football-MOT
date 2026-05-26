from __future__ import annotations

import configparser
from pathlib import Path
from typing import List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def list_images(image_dir: str | Path) -> List[Path]:
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    def sort_key(path: Path):
        return int(path.stem) if path.stem.isdigit() else path.stem

    return sorted(image_paths, key=sort_key)


def read_seqinfo(sequence_dir: str | Path) -> dict:
    sequence_dir = Path(sequence_dir)
    seqinfo_path = sequence_dir / "seqinfo.ini"

    if not seqinfo_path.exists():
        return {
            "frameRate": 25.0,
            "seqLength": None,
        }

    config = configparser.ConfigParser()
    config.read(seqinfo_path, encoding="utf-8")

    if "Sequence" not in config:
        return {
            "frameRate": 25.0,
            "seqLength": None,
        }

    sequence_section = config["Sequence"]

    return {
        "frameRate": float(sequence_section.get("frameRate", 25)),
        "seqLength": int(sequence_section.get("seqLength", 0)),
    }


def write_seqmap(sequence_names: list[str], seqmap_path: str | Path) -> None:
    seqmap_path = Path(seqmap_path)
    seqmap_path.parent.mkdir(parents=True, exist_ok=True)

    with open(seqmap_path, "w", encoding="utf-8") as f:
        f.write("name\n")

        for sequence_name in sequence_names:
            f.write(f"{sequence_name}\n")