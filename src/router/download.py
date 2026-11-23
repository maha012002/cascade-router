"""Download RouterBench pickles from the Hugging Face Hub at a pinned revision."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from router.config import PROJECT_ROOT, load_config


def download_files(repo_id: str, revision: str, files: list[str], raw_dir: Path) -> list[Path]:
    """Download each file into ``raw_dir`` (skips files already present) and return local paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in files:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=raw_dir,
        )
        paths.append(Path(local))
        print(f"ok  {filename} -> {local}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    download_files(
        repo_id=ds["repo_id"],
        revision=ds["revision"],
        files=ds["files"],
        raw_dir=PROJECT_ROOT / cfg["paths"]["raw_dir"],
    )


if __name__ == "__main__":
    main()
