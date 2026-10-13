"""
Step 2: download the bottle-detection dataset from Roboflow Universe.

Two modes:
  --list      list every published version of the project (number, date,
              image counts per split, license) so we can pick one deliberately.
  --version N download that exact version in YOLOv8 label format, verify it,
              fix its data.yaml paths, and record provenance.

The Roboflow API key is read only from the ROBOFLOW_API_KEY environment
variable (optionally via a local .env file). It is never printed or written
to disk.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from roboflow import Roboflow
from roboflow.config import API_URL
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = "tpicbxaiproject"
PROJECT = "bottle-detection-conveyor"
PROJECT_URL = f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}"

load_dotenv(ROOT / ".env")  # no-op if the file doesn't exist


def get_client() -> Roboflow:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit(
            "ROBOFLOW_API_KEY is not set. Set it as a Windows user environment "
            "variable, or put ROBOFLOW_API_KEY=... in a .env file in the project "
            "root, then restart your terminal/session."
        )
    return Roboflow(api_key=key)


def get_project():
    rf = get_client()
    try:
        return rf.workspace(WORKSPACE).project(PROJECT)
    except Exception as e:
        sys.exit(
            f"STOP: could not access {PROJECT_URL} with the configured API key "
            f"({type(e).__name__}: {e}). Per instructions, not substituting a "
            "different dataset. Check the workspace/project IDs and that your "
            "key has access to this project."
        )


def split_counts(version) -> dict:
    """Best-effort split image counts from the Roboflow API response."""
    counts = {}
    # roboflow-python's Version object exposes .splits on newer SDKs; fall back
    # to the raw dict on __dict__ if the attribute isn't there.
    raw = getattr(version, "splits", None) or getattr(version, "__dict__", {}).get("splits")
    if isinstance(raw, dict):
        counts = {k: v for k, v in raw.items()}
    return counts


def cmd_list(project):
    print(f"Versions available for {WORKSPACE}/{PROJECT}:\n")
    versions = project.versions()
    if not versions:
        sys.exit("STOP: no versions found for this project.")
    for v in versions:
        num = v.version.split("/")[-1] if "/" in str(v.version) else v.version
        created = getattr(v, "created", "unknown date")
        counts = split_counts(v)
        license_name = getattr(project, "license", "unknown")
        print(f"  version {num}: created={created} license={license_name} "
              f"splits={counts or '(not exposed by API, will count after download)'}")
    print(f"\nProject-level license field: {getattr(project, 'license', 'unknown')}")


def fix_data_yaml(dataset_root: Path):
    """Rewrite data.yaml to use an absolute `path:` root, independent of cwd."""
    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Roboflow's default is relative paths like '../train/images'. Replace
    # with an explicit root plus split-relative paths.
    data["path"] = str(dataset_root.resolve())
    for split, default in (("train", "train/images"), ("val", "valid/images"), ("test", "test/images")):
        if split in data:
            # keep whatever subfolder name Roboflow used, just drop the '../' prefix
            existing = str(data[split]).replace("\\", "/").lstrip("./")
            if existing.startswith("../"):
                existing = existing[3:]
            data[split] = existing

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return data


def verify_dataset(dataset_root: Path, data: dict, license_name: str):
    problems = []
    if data.get("names") != ["bottle"]:
        problems.append(f"class list is {data.get('names')!r}, expected ['bottle']")
    if "CC BY 4.0" not in (license_name or ""):
        problems.append(f"license is {license_name!r}, expected 'CC BY 4.0'")
    for split_key, default_dir in (("train", "train"), ("val", "valid"), ("test", "test")):
        split_rel = data.get(split_key, default_dir)
        img_dir = dataset_root / split_rel
        if not img_dir.exists() or not any(img_dir.iterdir()):
            problems.append(f"split '{split_key}' missing or empty at {img_dir}")

    if problems:
        msg = "STOP: dataset verification failed:\n  - " + "\n  - ".join(problems)
        sys.exit(msg)
    print("Verification passed: class list is ['bottle'], license is CC BY 4.0, "
          "train/valid/test splits all present.")


def cmd_download(project, version_num: int):
    version = project.version(version_num)
    license_name = getattr(project, "license", None)

    location = ROOT / "datasets" / f"bottle-detection-conveyor-v{version_num}"
    print(f"Downloading version {version_num} (yolov8 format) to {location} ...")
    dataset = version.download("yolov8", location=str(location))

    dataset_root = Path(dataset.location)
    data = fix_data_yaml(dataset_root)
    verify_dataset(dataset_root, data, license_name)

    counts = {}
    for split_key, default_dir in (("train", "train"), ("val", "valid"), ("test", "test")):
        img_dir = dataset_root / data.get(split_key, default_dir)
        counts[split_key] = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0

    info = {
        "workspace": WORKSPACE,
        "project": PROJECT,
        "version": version_num,
        "url": PROJECT_URL,
        "license": license_name,
        "download_date_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "split_image_counts": counts,
    }
    info_path = ROOT / "datasets" / "DATASET_INFO.json"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"\nSaved provenance to {info_path}")
    print(f"Split counts: {counts}")
    print(f"data.yaml rewritten with absolute path: {dataset_root / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list available dataset versions")
    ap.add_argument("--version", type=int, help="download this exact version number")
    args = ap.parse_args()

    project = get_project()

    if args.list:
        cmd_list(project)
    elif args.version is not None:
        cmd_download(project, args.version)
    else:
        ap.error("pass --list or --version N")


if __name__ == "__main__":
    main()
