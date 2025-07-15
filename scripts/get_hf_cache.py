#!/usr/bin/env python3

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import duckdb
import fire
import pandas as pd
from huggingface_hub.utils._cache_manager import scan_cache_dir

# Optional: make sure column display doesn't truncate if printing
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ────────────────────────────────────────────────────────────────────────────────
# Dataclass representing a single repo + revision row


@dataclass
class HFRepo:
    repo_id: str
    repo_type: str
    revision: str
    size_on_disk: str  # Human-readable string, e.g. '1.2G'
    size_bytes: int  # Actual numerical size (in bytes)
    nb_files: int
    last_modified: pd.Timestamp
    refs: str
    local_path: str


def format_size_bytes(size_bytes: int) -> str:
    """Convert byte count into human-readable size string."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f}G"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f}M"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f}K"
    else:
        return f"{size_bytes}B"


def flatten_cache_info(repo_filter: Optional[List[str]] = None) -> List[HFRepo]:
    """Scan the local HF cache and flatten to a list of HFRepo objects."""
    cache_info = scan_cache_dir()
    flat_rows = []

    for repo in cache_info.repos:
        if repo_filter and repo.repo_type not in repo_filter:
            continue

        for revision in repo.revisions:
            flat_rows.append(
                HFRepo(
                    repo_id=repo.repo_id,
                    repo_type=repo.repo_type,
                    revision=revision.commit_hash,
                    size_on_disk=format_size_bytes(repo.size_on_disk),
                    size_bytes=repo.size_on_disk,
                    nb_files=repo.nb_files,
                    last_modified=pd.to_datetime(repo.last_modified, unit="s"),
                    refs=",".join(revision.refs) if revision.refs else "",
                    local_path=str(repo.repo_path),
                )
            )
    return flat_rows


def export_hf_cache_to_csv(
    out: str = "hf_cache_summary.csv",
    top: Optional[int] = None,
    only_types: Optional[str] = None,  # e.g. 'model', 'dataset'
    sort_by: str = "size_bytes",
):
    """
    Export Hugging Face cache scan to CSV using DuckDB.

    Args:
        out: Output CSV path
        top: Limit to top N rows (by size or other column)
        only_types: Comma-separated filter, e.g. 'model,dataset'
        sort_by: Column to sort by, e.g. 'size_bytes', 'last_modified'
    """
    # Split comma-separated value (if passed)
    repo_filter = [t.strip() for t in only_types.split(",")] if only_types else None

    print("🔍 Scanning Hugging Face cache...")
    flat_data = flatten_cache_info(repo_filter)
    if not flat_data:
        print("⚠️  No matching cache entries found.")
        return

    df = pd.DataFrame([asdict(repo) for repo in flat_data])

    if sort_by not in df.columns:
        raise ValueError(f"Invalid sort column: '{sort_by}'")

    if top:
        print(f"📊 Selecting top {top} entries by '{sort_by}'")
    else:
        print(f"📊 Exporting all {len(df)} entries")

    # Register DataFrame in DuckDB and use SQL to sort + export
    con = duckdb.connect(database=":memory:")
    con.register("df", df)

    top_clause = f"LIMIT {top}" if top else ""
    query = f"""
        COPY (
            SELECT repo_id, repo_type, revision, size_on_disk, nb_files,
                   last_modified, refs, local_path
            FROM df
            ORDER BY {sort_by} DESC
            {top_clause}
        ) TO '{out}' (HEADER, DELIMITER ',')
    """

    con.execute(query)
    con.close()

    print(f"✅ CSV saved to: {Path(out).resolve()}")
    print(f"📁 Rows exported: {min(len(df), top) if top else len(df)}")


def main():
    """
    CLI entrypoint using fire.

    Examples:
        uv run scripts/get_hf_cache.py
        uv run scripts/get_hf_cache.py --top 20 --only_types model --sort_by size_bytes
    """
    fire.Fire(export_hf_cache_to_csv)


if __name__ == "__main__":
    main()
