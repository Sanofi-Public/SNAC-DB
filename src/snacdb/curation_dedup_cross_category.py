"""
Cross-category heavy-chain deduplication.

A single deposited heavy chain can be emitted into different SNAC-DB categories
depending on the biological assembly it is read from. In one assembly the heavy
chain pairs with a light chain (a Fab/scFv -> an `ab_*` VH-VL entry); in another
assembly the light chain is absent or unresolved, so the same heavy chain is
emitted on its own as a nanobody (`nb_*` VHH entry). The lone-VHH copy is an
assembly artifact: the chain is genuinely an antibody heavy chain, not a nanobody.

This module removes the artifactual `nb_*` entries whenever the exact same heavy
chain (matched by PDB id + VH sequence) is also present as a VH-VL pair in an
`ab_*` category, and writes an auditable report of every dropped entry.

It MUST be run after split+annotate but BEFORE FoldSeek clustering, otherwise the
nb_complexes cluster files would still reference the dropped names.

Usage:
    python -m snacdb.curation_dedup_cross_category <curated_structures_dir> \
        --report <report_csv_path> [--dry-run]

`curated_structures_dir` must contain, for each category, both
`<category>_curation_summary.csv` and a `<category>/` folder of PDB + NPY files.
"""

import argparse
import os
from pathlib import Path

import pandas as pd


AB_CATEGORIES = ["ab_complexes", "ab_unbound"]
NB_CATEGORIES = ["nb_complexes", "nb_unbound"]


def _build_ab_vh_vl_heavy_keys(curated_dir):
    """
    Collects (PDB_ID, Sequence_VH) keys for every paired VH-VL heavy chain across
    the antibody categories, with an example antibody Name per key for reporting.

    Args:
        curated_dir (Path): Directory containing the category summary CSVs.

    Returns:
        dict: {(PDB_ID, Sequence_VH): example_ab_name}
    """
    ab_keys = {}
    for cat in AB_CATEGORIES:
        csv_path = curated_dir / f"{cat}_curation_summary.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            vl = row.get("Chain_VL")
            seq = str(row.get("Sequence_VH", ""))
            # A genuine VH-VL pair has a resolved light chain.
            if pd.notna(vl) and seq not in ("", "nan"):
                key = (row["PDB_ID"], seq)
                ab_keys.setdefault(key, row["Name"])
    return ab_keys


def find_cross_category_duplicates(curated_dir):
    """
    Identifies nb_* entries whose heavy chain (PDB_ID + Sequence_VH) is also
    present as a VH-VL pair in an ab_* category.

    Args:
        curated_dir (Path): Directory containing the category summary CSVs.

    Returns:
        pd.DataFrame: One row per nb_* entry to drop, with columns
         [Name, PDB_ID, Category, Sequence_VH, Matched_Ab_Entry].
    """
    curated_dir = Path(curated_dir)
    ab_keys = _build_ab_vh_vl_heavy_keys(curated_dir)

    records = []
    for cat in NB_CATEGORIES:
        csv_path = curated_dir / f"{cat}_curation_summary.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            seq = str(row.get("Sequence_VH", ""))
            key = (row["PDB_ID"], seq)
            if key in ab_keys:
                records.append({
                    "Name": row["Name"],
                    "PDB_ID": row["PDB_ID"],
                    "Category": cat,
                    "Sequence_VH": seq,
                    "Matched_Ab_Entry": ab_keys[key],
                })
    return pd.DataFrame(records,
                        columns=["Name", "PDB_ID", "Category",
                                 "Sequence_VH", "Matched_Ab_Entry"])


def apply_cross_category_dedup(curated_dir, report_path, dry_run=False):
    """
    Drops nb_* entries that duplicate an ab_* VH-VL heavy chain and writes a report.

    For each dropped entry the PDB file, the -atom37.npy file, and the row in the
    category summary CSV are removed. The report CSV lists every dropped entry and
    the antibody entry that justified the removal.

    Args:
        curated_dir (str | Path): Directory with category CSVs and structure folders.
        report_path (str | Path): Path to write the deduplication report CSV.
        dry_run (bool): If True, only writes the report; no files/rows are removed.

    Returns:
        pd.DataFrame: The report of dropped entries.
    """
    curated_dir = Path(curated_dir)
    report_path = Path(report_path)
    dropped = find_cross_category_duplicates(curated_dir)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    dropped.to_csv(report_path, index=False)

    print(f"Cross-category dedup: {len(dropped)} nb_* entries duplicate an "
          f"ab_* VH-VL heavy chain "
          f"({dropped['PDB_ID'].nunique() if len(dropped) else 0} PDB ids).")
    for cat in NB_CATEGORIES:
        n = int((dropped["Category"] == cat).sum()) if len(dropped) else 0
        print(f"  {cat}: {n} to drop")
    print(f"Report written to: {report_path}")

    if dry_run:
        print("Dry run: no files or CSV rows removed.")
        return dropped

    # Remove dropped names per category: prune CSV rows and delete PDB + NPY files.
    drop_by_cat = {cat: set(dropped.loc[dropped["Category"] == cat, "Name"])
                   for cat in NB_CATEGORIES}
    for cat in NB_CATEGORIES:
        names = drop_by_cat.get(cat, set())
        if not names:
            continue
        csv_path = curated_dir / f"{cat}_curation_summary.csv"
        df = pd.read_csv(csv_path)
        before = len(df)
        df = df[~df["Name"].isin(names)].reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        struct_dir = curated_dir / cat
        removed_files = 0
        for name in names:
            for fname in (f"{name}.pdb", f"{name}-atom37.npy"):
                fpath = struct_dir / fname
                if fpath.exists():
                    os.remove(fpath)
                    removed_files += 1
        print(f"  {cat}: removed {before - len(df)} rows, {removed_files} files "
              f"(now {len(df)} entries)")

    return dropped


def main():
    parser = argparse.ArgumentParser(
        description="Drop nb_* entries whose heavy chain also appears as an "
                    "ab_* VH-VL pair, and write a deduplication report.")
    parser.add_argument("curated_dir",
                        help="Directory with <category>_curation_summary.csv and "
                             "<category>/ structure folders.")
    parser.add_argument("--report", required=True,
                        help="Path to write the deduplication report CSV "
                             "(ship this in the release).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only write the report; do not remove any entries.")
    args = parser.parse_args()
    apply_cross_category_dedup(args.curated_dir, args.report, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
