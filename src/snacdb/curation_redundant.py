import numpy as np
import os
import shutil
from glob import glob
import pandas as pd
import argparse
import subprocess
from pathlib import Path
from snacdb.utils.parallelize import ParallelProcessorForCPUBoundTasks


def get_structure_completeness(npy_file):
    """
    Calculate structure completeness metrics.
    
    Returns:
        tuple: (total_residues, missing_residues)
    """
    try:
        data = np.load(npy_file, allow_pickle=True).item()
        
        total_residues = 0
        missing_residues = 0
        
        for chain_id, chain_data in data.items():
            if 'atom37' not in chain_data:
                continue
            
            coords = chain_data['atom37']
            total_residues += len(coords)
            
            # Count residues with missing CA atoms
            ca_coords = coords[:, 1, :]  # CA is index 1
            missing_ca = np.isnan(ca_coords).any(axis=1)
            missing_residues += missing_ca.sum()
        
        return total_residues, missing_residues
    except Exception as e:
        print(f"Error reading {npy_file}: {e}")
        return 0, 0


def structure_priority(struct_name, path):
    """
    Assign priority for keeping structures.
    Higher priority = keep this one.
    
    Priority order:
    1. More total residues (more complete structure) - weight 500
    2. Fewer missing residues (better quality) - weight 100
    3. Bioassembly over ASU - bonus 1000 vs 500
    4. Lower bioassembly number
    """
    npy_file = Path(path) / f"{struct_name}-atom37.npy"
    total_res, missing_res = get_structure_completeness(npy_file)
    
    # Primary: more residues is better (weight 500)
    score = total_res * 500
    
    # Secondary: fewer missing residues is better (weight 100)
    score -= missing_res * 100
    
    # Tertiary: prefer bioassembly over ASU
    if '-ASU0' in struct_name:
        score += 500
    elif '-ASU' in struct_name:
        score += 1000
        # Lower ASU number preferred
        try:
            bio_num = int(struct_name.split('-ASU')[1].split('-')[0])
            score -= bio_num
        except:
            pass
    
    return score


def run_foldseek_comparison(struct_list, path, tmp_dir, tm_threshold=0.999):
    """
    Run FoldSeek multimersearch on all structures for a PDB and identify redundant pairs.
    """
    if len(struct_list) <= 1:
        return set()
    
    # Create temporary directory for this PDB group
    pdb_id = struct_list[0].split('-ASU')[0].split('-BIO')[0]
    pdb_tmp = Path(tmp_dir) / pdb_id
    pdb_tmp.mkdir(parents=True, exist_ok=True)
    
    # Create a subdirectory with symlinks to the PDB files
    pdb_dir = pdb_tmp / "pdbs"
    pdb_dir.mkdir(exist_ok=True)
    
    abs_path = Path(path).resolve()
    valid_structures = []
    
    for struct in struct_list:
        pdb_file = abs_path / f"{struct}.pdb"
        if pdb_file.exists():
            # Create symlink in pdb_dir
            symlink = pdb_dir / f"{struct}.pdb"
            symlink.symlink_to(pdb_file)
            valid_structures.append(struct)
    
    if len(valid_structures) <= 1:
        shutil.rmtree(pdb_tmp, ignore_errors=True)
        return set()
    
    # Run FoldSeek easy-multimersearch on the directory
    result_prefix = pdb_tmp / "result"
    foldseek_tmp = pdb_tmp / "foldseek_tmp"
    foldseek_tmp.mkdir(exist_ok=True)
    
    cmd = [
        "foldseek",
        "easy-multimersearch",
        str(pdb_dir),  # Pass directory instead of list file
        str(pdb_dir),  # Search against itself
        str(result_prefix),
        str(foldseek_tmp),
        "--format-output", "query,target,complexqtmscore,complexttmscore,complexassignid"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else "Unknown error"
        print(f"FoldSeek error for {pdb_id}: {stderr}")
        shutil.rmtree(pdb_tmp, ignore_errors=True)
        return set()
    
    # Parse the complex report file
    report_file = str(result_prefix) + "_report"
    
    if not Path(report_file).exists():
        shutil.rmtree(pdb_tmp, ignore_errors=True)
        return set()
    
    # Read report file
    redundant_pairs = []
    
    with open(report_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            
            query_id = Path(parts[0]).stem
            target_id = Path(parts[1]).stem
            
            # Skip self-comparisons
            if query_id == target_id:
                continue
            
            try:
                qtm = float(parts[4])
                ttm = float(parts[5])
                max_tm = max(qtm, ttm)
                
                if max_tm > tm_threshold:
                    redundant_pairs.append((query_id, target_id, max_tm))
            except (ValueError, IndexError):
                continue
    
    # Build redundancy graph and determine which structures to remove
    to_remove = set()
    struct_priority_map = {s: structure_priority(s, path) for s in valid_structures}
    
    for query, target, tm_score in redundant_pairs:
        if query in to_remove or target in to_remove:
            continue
        
        # Keep higher priority, remove lower priority
        if struct_priority_map.get(query, 0) > struct_priority_map.get(target, 0):
            to_remove.add(target)
        else:
            to_remove.add(query)
    
    # Clean up temporary directory
    shutil.rmtree(pdb_tmp, ignore_errors=True)
    
    return to_remove


def remove_redundant_data(items, path, tmp_base_dir):
    """
    Remove redundant structures using FoldSeek multimersearch.
    
    Args:
        items: Tuple of (pdb_id, list_of_structures)
        path: Path to directory containing structures
        tmp_base_dir: Base temporary directory for FoldSeek
    
    Returns:
        str: Summary message
    """
    pdb, struct_list = items
    
    if len(struct_list) <= 1:
        return f"{pdb}: Only 1 structure, nothing to compare"
    
    # Run FoldSeek comparison
    to_remove = run_foldseek_comparison(struct_list, path, tmp_base_dir)
    
    # Remove identified redundant structures
    removed_details = []
    for struct in to_remove:
        try:
            pdb_file = Path(path) / f"{struct}.pdb"
            npy_file = Path(path) / f"{struct}-atom37.npy"
            
            if pdb_file.exists():
                os.remove(pdb_file)
            if npy_file.exists():
                os.remove(npy_file)
            
            removed_details.append(struct)
        except Exception as e:
            print(f"Error removing {struct}: {e}")
    
    if removed_details:
        return f"{pdb}: removed {len(removed_details)}/{len(struct_list)} - {', '.join(removed_details)}"
    else:
        return f"{pdb}: no redundancy found ({len(struct_list)} structures)"


def main():
    """
    Remove redundant structures using FoldSeek multimersearch.
    Keeps more complete structures and prefers bioassembly over ASU.
    """
    parser = argparse.ArgumentParser(
        description="Remove redundant structures using FoldSeek"
    )
    parser.add_argument("-d", "--input_dir", required=True,
                       help="Path to the input directory")
    parser.add_argument("-c", "--input_csv", required=True,
                       help="Path to the input CSV")
    parser.add_argument("-m", "--make_new_dir", 
                       default=False, action='store_true',
                       help="Make new directory")
    parser.add_argument("--tm_threshold", type=float, default=0.999,
                       help="TM-score threshold for redundancy (default: 0.999)")
    parser.add_argument("--workers", type=int, default=48,
                       help="Number of parallel workers (default: 48)")
    
    args = parser.parse_args()
    pdb_folder = args.input_dir.rstrip('/')
    pdb_folder = pdb_folder if "/" in pdb_folder else "./" + pdb_folder
    csv_file = args.input_csv
    csv_file = csv_file if "/" in csv_file else "./" + csv_file
    
    original = "_".join(pdb_folder.split("_")[:-1])
    
    if args.make_new_dir:
        new_folder = f"{original}_curated"
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)
        os.makedirs(new_folder, exist_ok=True)
        
        pdb_files = sorted(glob(os.path.join(pdb_folder, "*")))
        for file in pdb_files:
            struct = os.path.basename(file)
            shutil.copy(file, f"{new_folder}/{struct}")
        
        new_csv = f"{original}_curation_summary.csv"
        if os.path.exists(new_csv):
            os.remove(new_csv)
    else:
        new_folder = pdb_folder
        new_csv = csv_file
    
    # Create temporary directory for FoldSeek
    tmp_base_dir = Path(new_folder) / "foldseek_tmp"
    tmp_base_dir.mkdir(exist_ok=True)
    
    # Group structures by PDB ID
    pdb_files = sorted(glob(os.path.join(new_folder, "*.pdb")))
    master_dict = {}
    for file in pdb_files:
        struct = Path(file).stem
        # Handle both ASU and BIO
        if '-ASU' in struct:
            pdb = struct.split("-ASU")[0]
        elif '-BIO' in struct:
            pdb = struct.split("-BIO")[0]
        else:
            pdb = struct
        
        if pdb in master_dict:
            master_dict[pdb].append(struct)
        else:
            master_dict[pdb] = [struct]
    
    from functools import partial
    
    print(f"Found {len(master_dict)} unique PDB IDs")
    print(f"Total structures: {sum(len(v) for v in master_dict.values())}")
    print(f"TM-score threshold: {args.tm_threshold}")
    print(f"Priority: More residues > Fewer missing > Bioassembly > ASU\n")
    
    # Process in parallel - use partial instead of lambda
    worker_func = partial(remove_redundant_data, path=new_folder, tmp_base_dir=tmp_base_dir)
    cpu_parallel = ParallelProcessorForCPUBoundTasks(
        worker_func,
        max_workers=args.workers
    )
    processed_rows = cpu_parallel.process(master_dict.items())
    
    for row in processed_rows:
        print(row)
    
    # Clean up temporary directory
    shutil.rmtree(tmp_base_dir, ignore_errors=True)
    
    # Update CSV with remaining structures
    leftover_files = [
        Path(struct).stem
        for struct in glob(os.path.join(new_folder, "*.pdb"))
    ]
    
    print(f"\nRemaining structures: {len(leftover_files)}")
    
    processed_df = pd.read_csv(csv_file)
    initial_count = len(processed_df)
    processed_df = processed_df[processed_df["Name"].isin(leftover_files)]
    removed_count = initial_count - len(processed_df)
    
    processed_df = processed_df.sort_values(by=["Name", "Bioassembly"])
    processed_df.to_csv(new_csv, index=False)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"CSV updated: {initial_count} → {len(processed_df)} entries")
    print(f"Removed {removed_count} redundant entries from CSV")
    print(f"{'='*80}")
    processed_df.info()


if __name__ == "__main__":
    main()