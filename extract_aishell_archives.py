"""Safely extract the per-speaker AISHELL-1 tar.gz archives.

The downloaded data_aishell package stores one speaker per archive under
data_aishell/wav. Extraction is resumable through completion records. Archives
are deleted only when --delete-archives-after-success is explicitly supplied
and every regular member has been verified after extraction.
"""

import argparse
import json
import os
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath


DEFAULT_SOURCE = Path("../dataset/data_aishell/wav")
DEFAULT_OUTPUT = Path("../dataset/data_aishell/wav_extracted")
SAFETY_FREE_BYTES = 512 * 1024 * 1024


def archive_id(path):
    name = path.name
    return name[:-7] if name.endswith(".tar.gz") else path.stem


def safe_members(handle, archive):
    members = handle.getmembers()
    if not members:
        raise ValueError(f"Archive is empty: {archive}")
    regular = []
    wav_count = 0
    total_bytes = 0
    for member in members:
        posix = PurePosixPath(member.name.replace("\\", "/"))
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"Unsafe member path in {archive}: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Links/devices are not allowed in {archive}: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise ValueError(f"Unsupported member type in {archive}: {member.name}")
        if member.isfile():
            regular.append(member)
            total_bytes += member.size
            if posix.suffix.lower() == ".wav":
                wav_count += 1
    if not regular or not wav_count:
        raise ValueError(f"Archive contains no WAV files: {archive}")
    return members, regular, wav_count, total_bytes


def verify_files(output_dir, regular):
    for member in regular:
        path = output_dir.joinpath(*PurePosixPath(member.name).parts)
        if not path.is_file():
            raise FileNotFoundError(f"Missing extracted file: {path}")
        if path.stat().st_size != member.size:
            raise ValueError(
                f"Size mismatch for {path}: {path.stat().st_size} != {member.size}"
            )


def completion_path(state_dir, archive):
    return state_dir / f"{archive_id(archive)}.done.json"


def load_completion(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def append_log(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def completion_matches(record, archive):
    stat = archive.stat()
    return (
        record.get("archive_name") == archive.name
        and record.get("archive_size") == stat.st_size
        and record.get("archive_mtime_ns") == stat.st_mtime_ns
    )


def extract_one(archive, output_dir, state_dir, log_path, args):
    done_path = completion_path(state_dir, archive)
    done = load_completion(done_path)
    with tarfile.open(archive, "r:gz") as handle:
        members, regular, wav_count, total_bytes = safe_members(handle, archive)
        if done and completion_matches(done, archive):
            verify_files(output_dir, regular)
            if args.delete_archives_after_success:
                archive.unlink()
                action = "verified_skip_deleted_archive"
            else:
                action = "verified_skip"
            append_log(log_path, {"archive": archive.name, "action": action})
            return action, wav_count, total_bytes

        free_bytes = shutil.disk_usage(output_dir).free
        required = total_bytes + args.safety_free_bytes
        if free_bytes < required:
            raise OSError(
                f"Not enough free space for {archive.name}: need at least "
                f"{required / (1024 ** 3):.2f} GiB, have {free_bytes / (1024 ** 3):.2f} GiB"
            )
        if args.dry_run:
            return "dry_run", wav_count, total_bytes

        started = time.time()
        handle.extractall(output_dir, members=members)
        verify_files(output_dir, regular)
        stat = archive.stat()
        record = {
            "archive_name": archive.name,
            "archive_size": stat.st_size,
            "archive_mtime_ns": stat.st_mtime_ns,
            "regular_files": len(regular),
            "wav_files": wav_count,
            "uncompressed_bytes": total_bytes,
            "completed_unix": time.time(),
        }
        write_json_atomic(done_path, record)
        if args.delete_archives_after_success:
            archive.unlink()
            action = "extracted_verified_deleted_archive"
        else:
            action = "extracted_verified"
        append_log(
            log_path,
            {
                **record,
                "archive": archive.name,
                "action": action,
                "seconds": time.time() - started,
            },
        )
        return action, wav_count, total_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-archives", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-archives-after-success", action="store_true")
    parser.add_argument(
        "--safety-free-gib",
        type=float,
        default=SAFETY_FREE_BYTES / (1024 ** 3),
        help="Free space to retain in addition to the current archive's extracted size.",
    )
    args = parser.parse_args()
    if args.max_archives < 0:
        raise ValueError("--max-archives must be non-negative")
    if args.safety_free_gib < 0:
        raise ValueError("--safety-free-gib must be non-negative")
    args.safety_free_bytes = int(args.safety_free_gib * (1024 ** 3))

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"AISHELL archive directory not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / ".extract_state"
    log_path = output_dir / "extract_log.jsonl"
    archives = sorted(source_dir.glob("*.tar.gz"))
    if args.max_archives:
        archives = archives[: args.max_archives]
    if not archives:
        print(f"No tar.gz archives found in {source_dir}")
        return

    print(f"source={source_dir}")
    print(f"output={output_dir}")
    print(f"archives={len(archives)} dry_run={args.dry_run}")
    print(f"delete_after_success={args.delete_archives_after_success}")
    counts = {}
    total_wavs = 0
    total_bytes = 0
    for index, archive in enumerate(archives, start=1):
        action, wav_count, uncompressed_bytes = extract_one(
            archive, output_dir, state_dir, log_path, args
        )
        counts[action] = counts.get(action, 0) + 1
        total_wavs += wav_count
        total_bytes += uncompressed_bytes
        print(
            f"[{index:03d}/{len(archives):03d}] {archive.name}: {action}, "
            f"wav={wav_count}, size={uncompressed_bytes / (1024 ** 2):.1f} MiB"
        )
    print(
        json.dumps(
            {
                "archives": len(archives),
                "actions": counts,
                "wav_members": total_wavs,
                "uncompressed_gib": total_bytes / (1024 ** 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
