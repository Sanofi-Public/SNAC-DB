import os
import argparse
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from Bio import pairwise2
from glob import glob
from snacdb.utils.pdb_utils_clean_parse import PDBUtils


def fasta_parser(fasta_file):
    """
    This script is able to read and convert a fasta file
    and converts it as a dictionary where the chain_id is
    the keys of the dictionary and the values are the
    sequences.

    Args:
        fasta_file (str): path to fasta file

    Returns
        struct_dict (dict): dictionary form of the fasta file
    """

    with open(fasta_file, "r") as file:
        struct_dict = {}
        key = None
        for line in file:
            # solving for the chain ID of the sequence
            if line.startswith(">"):
                key = line[1:].strip()
            elif line.strip():
                # grabbing sequence and getting rid of white space
                struct_dict[key] = line.strip()
    return struct_dict


def create_fasta(fasta_dict, filename):
    """
    Create a fasta file based on a inputted dictionary. The keys
    will be the header of the sequences and the values are the 
    sequences.

    Args:
        fasta_dict (dict): dictionary containing information chain ids
         and sequences of the complex
        filename (str): name and path of the new fasta file
    """
    
    with open(filename, "w") as file:
        for chain_id, seq in fasta_dict.items():
            file.write(f">{chain_id}\n{seq}\n")


def add_pdb_header_fc(pdb_file, path):
    """
    Ensuring that the newly created pdb file has the same header
    information as the original pdb file.

    Args:
        pdb_file (str): name of the pdb file
        path (str): path to the cleaning directory
    """

    # getting the content of the new pdb file
    with open(pdb_file, 'r') as f:
        content = "".join(f.readlines()[9:])
    # getting the header of the original pdb file
    with open(f"{path}/input_complexes/{os.path.basename(pdb_file)}", "r") as f:
        header = "".join(f.readlines()[:9])
    # combine the content and header
    with open(pdb_file, "w") as f:
        f.write(header)
        f.write(content)


def get_chains_from_name(path, name):
    """
    Solving for a list of the chain IDs of the structure file that
    has been processed by the SNAC-DB pipeline.

    Args:
        path (str): path to the cleaning directory
        name (str): name of input pdb file

    Returns:
        all_chains (list): list of all the chain IDs in the complex
    """
    
    all_chains = []
    if "-VH_" in name:  # Antibody
        all_chains = ["H", "L"]
    else:  # Nanobody
        all_chains = ["H"]
    # finding all the Ag chains in the complex from npy file
    if "-Ag_" in name:
        if "-replicate" in name:  # getting rid of replicate keyword
            ag_name = (name.split("-replicate")[0]).split("-Ag_")[1]
        else:
            ag_name = name.split("-Ag_")[1]

        # converting chain ids in file name to new chain id of the complex
        chain_dict = np.load(f"{path}/input_complexes/{name}-atom37.npy", allow_pickle=True).item()
        new_to_old_dict = {c: chain_dict[c]["old_id"] for c in chain_dict if c != "H" and c != "L"}
        old_to_new_dict = {val: key for key, val in new_to_old_dict.items()}
        all_chains += [old_to_new_dict[c] for c in ag_name.split("_")]
    return all_chains


def run_mmseqs_search_and_extract(query_db, target_db, prefix, output_tsv, tmp, threshold, path):
    """
    The purpose of this script is to find the highest matches between
    two inputted databases, create an alignment based off the two, and
    finally summarize the results in a TSV file.

    Args:
        query_db (str): path to query database
        target_db (str): path to target database
        prefix (str): keyword of the target database
        output_tsv (str): path to the output tsv file
        tmp (str): path to temporary directory to house information from the
         mmseq function calls
        threshold (int): Percent match needed to consider a comparison in the
         mmseq search
        path (str): path to the cleaning directory
    """
    
    result_db = f"{path}/{prefix}_result"  # search output file
    aln_db = f"{path}/{prefix}_aln"  # alignment output file

    # trying to run search of mmseqs
    try:
        subprocess.run([
            "mmseqs", "search",
            query_db, target_db, result_db, tmp,
            "--max-seqs", "1",
            "--min-seq-id", str(threshold)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error on Search done for database: {prefix}")
        print("Command:", e.cmd)
        print("Return Code:", e.returncode)
        raise subprocess.CalledProcessError(e.returncode, e.cmd)

    # trying to run the alignment using mmseq
    try:
        subprocess.run([
            "mmseqs", "align",
            query_db, target_db, result_db, aln_db,
            "-a"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error on Alignment done for database: {prefix}")
        print("Command:", e.cmd)
        print("Return Code:", e.returncode)
        print("Standard Output:\n", e.stdout)
        print("Standard Error:\n", e.stderr)
        raise subprocess.CalledProcessError(e.returncode, e.cmd)

    # converting the results to a TSV file
    try:
        subprocess.run([
            "mmseqs", "convertalis",
            query_db, target_db, aln_db, output_tsv,
            "--format-output", "query,target,qseq,tseq,qaln,taln"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error on Creating TSV done for database: {prefix}")
        print("Command:", e.cmd)
        print("Return Code:", e.returncode)
        print("Standard Output:\n", e.stdout)
        print("Standard Error:\n", e.stderr)
        raise subprocess.CalledProcessError(e.returncode, e.cmd)


def ensure_mmseqs_db(db_path, db_name, fallback_dir):
    """
    Checks to see if the function requires the database (inputted value is
    not None). Once that is done then it checks to see if the database
    is properly downloaded, if not it will download at the specified path.
    If not path is specified then the fallback path is utilized.

    Args:
        db_path (str): path to database
        db_name (str): name of database
        fallback_dir (str): path to backup location to download database

    Returns:
        db_path (str): path of the newly downloaded database
    """

    # check if the None output is provided
    if db_path is None or db_path == "None":
        return None  # if so then don't download database

    # use fallback directory if db_path is empty or False
    db_path = Path(db_path) if db_path else Path(fallback_dir) / db_name
    if not db_path.exists():
        # downloading database since it doesn't exist
        print(f"Database {db_name} not found at {db_path}. Downloading...")
        tmp_dir = db_path.parent / f"tmp_{db_name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # trying to download the database
        try:
            subprocess.run(["mmseqs", "databases", db_name, str(db_path), str(tmp_dir)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error with creating Database: {db_name}")
            print("Command:", e.cmd)
            print("Return Code:", e.returncode)
            print("Standard Output:\n", e.stdout)
            print("Standard Error:\n", e.stderr)
            raise subprocess.CalledProcessError(e.returncode, e.cmd)
    return str(db_path)


def parse_args():
    """
    The purpose of this function is to parse the arguements provided to this script.

    Returns:
        (parser.parse_args()): Arguements fed to script
    """
    
    parser = argparse.ArgumentParser(description="Clean PDB structures and align sequences with optional DB fetching.")
    parser.add_argument("-q", "--input_dir", type=str, help="Path to input structure directory.")
    parser.add_argument("-s", "--swissprot", default=False, help="SwissProt DB path or fallback to download.")
    parser.add_argument("-u", "--uniref", default=False, help="UniRef DB path or fallback to download. Use 'None' to skip.")
    return parser.parse_args()


def main():
    """
    This is the main function of the script. The purpose of the script is
    to find matches between an inputted series of complexes with unresolved
    residues and the Swissprot and UniRef database. The goal is to find
    exact matches to then fill in those unresolved residues.
    """
    
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    path = Path(f"{input_dir}_cleaning_complexes")
    if path.exists():  # delete directory if it exists
        shutil.rmtree(path)
    path.mkdir()  # create directory
    (path / "input_complexes").mkdir()

    # move the files to the new directory: input_complexes
    for file in sorted(glob(f"{input_dir}/*")):
        shutil.copy(file, f"{path}/input_complexes/{os.path.basename(file)}")

    # ensure existence of databases
    swissprot_db = ensure_mmseqs_db(args.swissprot, "UniProtKB/Swiss-Prot", path.parent)
    uniref_db = ensure_mmseqs_db(args.uniref, "UniRef90", path.parent)

    # specify path to testdata_setup
    project_root = Path(__file__).resolve().parents[0]
    script_path = project_root / "testdata_setup.py"
    try:  # create fasta files of the complexes in input_complexes
        subprocess.run(["python", script_path, "-q", f"{path}/input_complexes", "-k", "curated_fasta", "-s", "fasta"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in utilizing testdata_setup script: {path}/input_complexes")
        print("Command:", e.cmd)
        print("Return Code:", e.returncode)
        print("Standard Output:\n", e.stdout)
        print("Standard Error:\n", e.stderr)
        raise subprocess.CalledProcessError(e.returncode, e.cmd)

    # create directory to store the complexes with unresolved residues
    if os.path.exists(f"{path}/problem_complexes"):
        shutil.rmtree(f"{path}/problem_complexes")
    os.makedirs(f"{path}/problem_complexes")

    # storing all the fasta files into the complex_dict
    complex_dict = {}
    for fasta_file in glob(f"{path}/curated_fasta_query/*"):
        name = os.path.splitext(os.path.basename(fasta_file))[0]
        complex_dict[name] = fasta_parser(fasta_file)

    # new_complex_dict will house all the complexes that have an unresolved residue
    new_complex_dict = {s: d for s, d in complex_dict.items() if any("X" in seq for c, seq in d.items() if c not in ["H", "L"])}

    # move problem complexes to the proper dictionary
    for name in new_complex_dict:
        shutil.copy(f"{path}/curated_fasta_query/{name}.fasta", f"{path}/problem_complexes/{name}.fasta")

    # convert problem antigens to a fasta file
    chains = []
    with open(f"{path}/input_fasta.fasta", "w") as f:
        for fasta, fasta_dict in new_complex_dict.items():
            for chain_id, seq in fasta_dict.items():
                if chain_id not in ["H", "L"]:
                    tag = f">{fasta}_fasta_{chain_id}"
                    chains.append(tag)  # store the problem antigne chain IDs
                    f.write(f"{tag}\n{seq}\n")

    query_db = f"{path}/query_db"
    tmp_dir = f"{path}/tmp_mmseqs"
    os.makedirs(tmp_dir, exist_ok=True)
    # convert input fasta file to a database
    try:
        subprocess.run(["mmseqs", "createdb", f"{path}/input_fasta.fasta", query_db], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in converting fasta to database: input_fasta.fasta")
        print("Command:", e.cmd)
        print("Return Code:", e.returncode)
        print("Standard Output:\n", e.stdout)
        print("Standard Error:\n", e.stderr)
        raise subprocess.CalledProcessError(e.returncode, e.cmd)

    updated_complexes = {}
    # finding all the antigen chains that were updated
    if swissprot_db:
        swissprot_out = f"{path}/output_swissprot.tsv"
        run_mmseqs_search_and_extract(query_db, swissprot_db, "swissprot", swissprot_out, tmp_dir, 0.8, path)
        df_1 = pd.read_csv(swissprot_out, sep="\t", header=None, names=["Query", "Target", "Q_Seq", "T_Seq", "Q_Aln", "T_Aln"])
        # see if the match is an exact match, if not disregard
        for _, row in df_1.iterrows():
            query, qseq, tseq = row["Query"], row["Q_Aln"], row["T_Aln"].replace("-", "X")
            seq = ""
            continue_flag = False
            for q, t in zip(qseq, tseq):
                if q != "X" and q != t:
                    if t == "X":
                        seq += q
                    else:
                        continue_flag = True
                        break
                elif q == "X" or q == t:
                    seq += t
            if continue_flag:
                continue
            final_seq = seq
            # ensure that the initial sequence is the proper length
            if len(row["Q_Seq"]) > len(seq):
                # add all the sequences that may be missing from the aligned sequence
                q_seq_original = "".join([res for res in row["Q_Seq"] if res != "X"])
                alignments = pairwise2.align.globalms(q_seq_original, seq, 5, -5, -0.5, -0.5, gap_char="X")
                aligned_seq_master = alignments[0].seqA
                aligned_seq = alignments[0].seqB
                final_seq = ""
                for q, t in zip(aligned_seq, aligned_seq_master):
                    if q != "X" and q != t:
                        if t == "X":
                            final_seq += q
                        else:
                            raise ValueError(f"Error with {query}")
                    elif q == "X" or q == t:
                        final_seq += t
            updated_complexes[query] = final_seq

    if uniref_db:  # if uniref_db is provided
        # creating new input fasta file
        missing_chains = list(set(chains) - set(df_1["Query"].values))
        with open(f"{path}/input_fasta_2.fasta", "w") as f:
            for fasta, fasta_dict in new_complex_dict.items():
                for chain_id, seq in fasta_dict.items():
                    tag = f">{fasta}_fasta_{chain_id}"
                    if tag in missing_chains:
                        f.write(f"{tag}\n{seq}\n")
        query_db_2 = f"{path}/query_db_2"
        # converting new input fasta file to a database
        try:
            subprocess.run(["mmseqs", "createdb", f"{path}/input_fasta_2.fasta", query_db_2], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in converting fasta to database: input_fasta_2.fasta")
            print("Command:", e.cmd)
            print("Return Code:", e.returncode)
            print("Standard Output:\n", e.stdout)
            print("Standard Error:\n", e.stderr)
            raise subprocess.CalledProcessError(e.returncode, e.cmd)
        
        uniref_out = f"{path}/output_uniref.tsv"
        # find all matches from the problem complexes
        run_mmseqs_search_and_extract(query_db_2, uniref_db, "uniref", uniref_out, tmp_dir, 0.9, path)
        df_2 = pd.read_csv(uniref_out, sep="\t", header=None, names=["Query", "Target", "Q_Seq", "T_Seq", "Q_Aln", "T_Aln"])

        # see if the match is an exact match, if not disregard
        for _, row in df_2.iterrows():
            query, qseq, tseq = row["Query"], row["Q_Aln"], row["T_Aln"]
            seq = ""
            continue_flag = False
            for q, t in zip(qseq, tseq):
                if q != "X" and q != t:
                    if t == "X":
                        seq += q
                    else:
                        continue_flag = True
                        break
                elif q == "X" or q == t:
                    seq += t
            if continue_flag:
                continue

            # ensure that the initial sequence is the proper length
            final_seq = seq
            if len(row["Q_Seq"]) > len(seq):
                # add all the sequences that may be missing from the aligned sequence
                q_seq_original = "".join([res for res in row["Q_Seq"] if res != "X"])
                alignments = pairwise2.align.globalms(q_seq_original, seq, 5, -5, -0.5, -0.5, gap_char="X")
                aligned_seq_master = alignments[0].seqA
                aligned_seq = alignments[0].seqB
                final_seq = ""
                for q, t in zip(aligned_seq, aligned_seq_master):
                    if q != "X" and q != t:
                        if t == "X":
                            final_seq += q
                        else:
                            raise ValueError(f"Error with {query}")
                    elif q == "X" or q == t:
                        final_seq += t
            updated_complexes[query] = final_seq

    query_dict = {}
    # adding all the corrected antigen chain to query_dict
    for query, seq in updated_complexes.items():
        name, chain_id = query.split("_fasta_")
        if name not in query_dict:
            query_dict[name] = {}
        query_dict[name][chain_id] = seq

    # next goal is to ensure that all the chains are added back to the complex
    for name, fasta_dict in query_dict.items():
        all_chains = get_chains_from_name(path, name)
        curr_chains = list(fasta_dict.keys())
        lf_chains = list(set(all_chains) - set(curr_chains))
        if lf_chains:
            original_fasta_dict = new_complex_dict[name]
            for chain in lf_chains:
                query_dict[name][chain] = original_fasta_dict[chain]

    # double checking that all the chains for a complex were properly added
    for name, fasta_dict in query_dict.items():
        expected_ag = get_chains_from_name(path, name)
        if len(expected_ag) != len(fasta_dict):
            raise ValueError("problem (need to rerun):", name, expected_ag, fasta_dict.keys())

    # creating final corrected fasta directory
    if os.path.exists(f"{path}/corrected_complexes_fasta"):
        shutil.rmtree(f"{path}/corrected_complexes_fasta")
    os.makedirs(f"{path}/corrected_complexes_fasta")
    # moving corrected fasta files to the new directory
    for file_name, fasta_dict in query_dict.items():
        create_fasta(fasta_dict, filename=f"{path}/corrected_complexes_fasta/{file_name}.fasta")

    # creating final structure directory
    # if os.path.exists(f"{path}/corrected_complexes_structure"):
    #     shutil.rmtree(f"{path}/corrected_complexes_structure")
    # os.makedirs(f"{path}/corrected_complexes_structure")
    # pdb_utils = PDBUtils()
    # editing structures based off of corrected SEQRES
    # for structure_file in glob(f"{path}/input_complexes/*.pdb"):
    #     name = os.path.splitext(os.path.basename(structure_file))[0]
    #     if os.path.exists(f"{path}/corrected_complexes_fasta/{name}.fasta"):
    #         seqres = fasta_parser(f"{path}/corrected_complexes_fasta/{name}.fasta")
    #         output_file = f"{path}/corrected_complexes_structure/{name}.pdb"
    #         pdb_utils(structure_file, output_pdb_file=output_file, custom_seqres=seqres)
    #         add_pdb_header_fc(output_file, path)
        # else:
        #     shutil.copy(structure_file, f"{path}/corrected_complexes_structure/{name}.pdb")

    # move the final complexes to the directory cleaned
    # final_path = input_dir.parent / f"{input_dir.name}_cleaned"
    # if final_path.exists():
    #     shutil.rmtree(final_path)
    # os.makedirs(final_path)
    # for file in glob(f"{path}/corrected_complexes_structure/*"):
    #     shutil.copy(file, f"{final_path}/{os.path.basename(file)}")


if __name__ == "__main__":
    main()