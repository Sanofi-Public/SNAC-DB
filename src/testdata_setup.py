import argparse
import os
import shutil
from glob import glob
from Bio import PDB
import sys
from snacdb.utils.pdb_utils_clean_parse import PDBUtils, merge_chains, dict_to_structure
from snacdb.utils.sequence_utils import chain_dict_to_VH_VL_Ag_categories, get_region_splits_anarci
from snacdb.utils.parallelize import ParallelProcessorForCPUBoundTasks


def create_sc_ag(old_dir, new_dir):
    """
    This function serves to move a series of structure files located
    in the input directory and moves them to the output directory.
    Before moving them, the function uses the process_row function
    to ensure that the structure is marked as a single chain, and
    erases any chain that is not an antigen chain.

    Args:
        old_dir (str): Path to input directory
        new_dir (str): Path to output directory
    """

    os.makedirs(new_dir, exist_ok=True)  # create new output directory
    if os.path.splitext(old_dir)[1] == "":
        pdb_files = glob(f"{old_dir}/*.pdb")
    else:
        pdb_files = [old_dir]
    
    # creating tools to help parse pdb structure
    parser = PDB.PDBParser(QUIET=True)
    pdb_utils = PDBUtils(verbose=False)

    # parallelizing parsing and moving the structures
    cpu_parallel = ParallelProcessorForCPUBoundTasks(process_row, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_files, new_dir, parser, pdb_utils, just_Ag=True)


def create_sc_ligand(old_dir, new_dir):
    """
    This function serves to move a series of structure files located
    in the input directory and moves them to the output directory.
    Before moving them, the function uses the process_row function
    to ensure that the structure is marked as a single chain, and
    to earse any chain that is not a ligand chain.

    Args:
        old_dir (str): Path to input directory
        new_dir (str): Path to output directory
    """

    os.makedirs(new_dir, exist_ok=True)  # create new output directory
    if os.path.splitext(old_dir)[1] == "":
        pdb_files = glob(f"{old_dir}/*.pdb")
    else:
        pdb_files = [old_dir]
    
    # creating tools to help parse pdb structure
    parser = PDB.PDBParser(QUIET=True)
    pdb_utils = PDBUtils(verbose=False)

    # parallelizing parsing and moving the structures
    cpu_parallel = ParallelProcessorForCPUBoundTasks(process_row, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_files, new_dir, parser, pdb_utils, just_Ag=None)


def create_just_ag(old_dir, new_dir):
    """
    This function serves to move a series of structure files located
    in the input directory and moves them to the output directory.
    Before moving them, the function uses the process_row function
    to ensure that the structure only consists of antigen chains.

    Args:
        old_dir (str): Path to input directory
        new_dir (str): Path to output directory
    """

    os.makedirs(new_dir, exist_ok=True)  # create new output directory
    if os.path.splitext(old_dir)[1] == "":
        pdb_files = glob(f"{old_dir}/*.pdb")
    else:
        pdb_files = [old_dir]
    
    # creating tools to help parse pdb structure
    parser = PDB.PDBParser(QUIET=True)
    pdb_utils = PDBUtils(verbose=False)

    # parallelizing parsing and moving the structures
    cpu_parallel = ParallelProcessorForCPUBoundTasks(process_row, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_files, new_dir, parser, pdb_utils, just_Ag=True, single_chain=False)


def create_just_ligand(old_dir, new_dir):
    """
    This function serves to move a series of structure files located
    in the input directory and moves them to the output directory.
    Before moving them, the function uses the process_row function
    to ensure that the structure only composed of ligand chains.

    Args:
        old_dir (str): Path to input directory
        new_dir (str): Path to output directory
    """

    os.makedirs(new_dir, exist_ok=True)  # create new output directory
    if os.path.splitext(old_dir)[1] == "":
        pdb_files = glob(f"{old_dir}/*.pdb")
    else:
        pdb_files = [old_dir]
    
    # creating tools to help parse pdb structure
    parser = PDB.PDBParser(QUIET=True)
    pdb_utils = PDBUtils(verbose=False)

    # parallelizing parsing and moving the structures
    cpu_parallel = ParallelProcessorForCPUBoundTasks(process_row, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_files, new_dir, parser, pdb_utils, just_Ag=None, single_chain=False)


def create_sc(old_dir, new_dir):
    """
    This function serves to move a series of structure files located
    in the input directory and moves them to the output directory.
    Before moving them, the function uses the process_row function
    to ensure that the structure is marked as a single chain.

    Args:
        old_dir (str): Path to input directory
        new_dir (str): Path to output directory
    """

    os.makedirs(new_dir, exist_ok=True)  # create new output directory
    if os.path.splitext(old_dir)[1] == "":
        pdb_files = glob(f"{old_dir}/*.pdb")
    else:
        pdb_files = [old_dir]
    
    # creating tools to help parse pdb structure
    parser = PDB.PDBParser(QUIET=True)
    pdb_utils = PDBUtils(verbose=False)
    
    # parallelizing parsing and moving the structures
    cpu_parallel = ParallelProcessorForCPUBoundTasks(process_row, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_files, new_dir, parser, pdb_utils)


def complex_with_sc_ag(old_dir, new_dir):
    """
    This function serves to move a series of structure files located
    in the input directory and moves them to the output directory.
    Before moving them, the function uses the process_row function
    to modify the structure so that the antigens are labeled under
    one chain IDs, but the ligands (if multiple) maintain their
    original chain IDs.

    Args:
        old_dir (str): Path to input directory
        new_dir (str): Path to output directory
    """
    os.makedirs(new_dir, exist_ok=True)  # create new output directory
    if os.path.splitext(old_dir)[1] == "":
        pdb_files = glob(f"{old_dir}/*.pdb")
    else:
        pdb_files = [old_dir]
    
    # creating tools to help parse pdb structure
    parser = PDB.PDBParser(QUIET=True)
    pdb_utils = PDBUtils(verbose=False)
    
    # parallelizing parsing and moving the structures
    cpu_parallel = ParallelProcessorForCPUBoundTasks(process_row, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_files, new_dir, parser, pdb_utils, single_chain=None)


def process_row(pdb, out_folder, parser, pdb_utils, just_Ag=False, single_chain=True):
    """
    Purpose of this function is to filter out specific chains or change the format of
    a structure based on a series of specifications.

    Args:
        pdb (str): path of input structure file
        out_folder (str): path to output directory
        parser (PDB.PDBParser): Biopython tool used to convert a structure file
         into a biopython structure
        pdb_utils (PDBUtils): A class that stores a function to merge all the
         chains in a structure into one
        just_Ag (bool): Whether the structures should just contain antigens
        single_chain (bool): Whethere the structure should contain just one chain

    Returns:
        str: String saying that the structure ran successfully
    """

    pdb_name = os.path.basename(pdb)  # obtain just the pdb name
    _, structure = pdb_utils(pdb, return_biopython_structure=True, write_file=False)

    if just_Ag:  # If we want to only deal with Ag chains and not whole complexes
        try:
            structure[0].detach_child("H")  # removing heavy chain
        except:
            raise ValueError(f"Heavy chain is not found in the structure. This struture was most likely not generated using SNAC-DB pipeline. Please reveiw this structure: {os.path.basename(pdb)}")
        if "VHH" not in pdb_name:
            try:
                structure[0].detach_child("L")  # removing light chain
            except:
                raise ValueError(f"Light Chain not found in structure despite structure saying it has an L chain. Please review this structure: {os.path.basename(pdb)}")   
    elif just_Ag is None:
        chain_ids = set([chain.id for chain in structure[0]]) - set(["H", "L"])
        [structure[0].detach_child(chain) for chain in chain_ids]
    elif just_Ag == False and single_chain is None:
        structure_ligand = structure.copy()
        chain_ids = set([chain.id for chain in structure_ligand[0]]) - set(["H", "L"])
        [structure_ligand[0].detach_child(chain) for chain in chain_ids]
        structure[0].detach_child("H")  # removing heavy chain
        if "VHH" not in pdb_name:
            try:
                structure[0].detach_child("L")  # removing light chain
            except:
                print("L Chain not found in structure despite structure saying it has an L chain. Please review structure.")

    chain_ids = [chain.id for chain in structure[0]]
    if len(chain_ids) != 1 and single_chain:  #structure has more than one chain
        structure_ag = merge_chains(structure, chains_to_merge=chain_ids, merged_chain_id='Z')
    elif single_chain is None and len(chain_ids) != 1:  # for the case of modified Ag but regular ligand
        structure = merge_chains(structure, chains_to_merge=chain_ids, merged_chain_id='Z')
        for chain in structure[0]:
            structure_ligand[0].add(chain)
        structure_ag = structure_ligand
    elif single_chain is None and len(chain_ids) == 1: # modified Ag (except already single-chain) and regular ligand
        for chain in structure[0]:
            structure_ligand[0].add(chain)
        structure_ag = structure_ligand
    else: # structure already has only one chain
        structure_ag = structure

    # Write the output file
    out_file = f"{out_folder}/{pdb_name}"
    pdb_utils.write_pdb_with_seqres_from_structure(structure_ag[0], out_file)
    return f"Processed {pdb_name} successfully"


def fasta_conversion(file, new_dir, pdb_utils):
    """
    Utilizes the PDBUtils class to convert a structure file
    into a fasta file.

    Args:
        file (str): path to structure file
        new_dir (str): path to new directory
        pdb_utils (PDBUtils): an instance of the PDBUtils class
    
    Returns:
        str: String saying that the structure ran successfully
    """
    
    new_file = os.path.join(new_dir, os.path.basename(file))
    pdb_utils.write_fasta_for_pdb(file, output_file=new_file)
    return f"Passed: {os.path.splitext(os.path.basename(file))[0]}"


def convert_to_fasta(old_dir, new_dir):
    """
    This function serves to move a series of structure files located
    in the input directory and moves them to the output directory.
    Before moving them, the function uses the fasta_conversion function
    to obtain the fasta files for each structure file.

    Args:
        old_dir (str): Path to input directory
        new_dir (str): Path to output directory
    """
    pdb_utils = PDBUtils(verbose=False)
    os.makedirs(new_dir)  # create new output directory
    if os.path.splitext(old_dir)[1] == "":
        pdb_files = glob(f"{old_dir}/*.pdb")
    else:
        pdb_files = [old_dir]
    # parallelizing parsing and moving the structures
    cpu_parallel = ParallelProcessorForCPUBoundTasks(fasta_conversion, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_files, new_dir, pdb_utils)


def obtain_cdrs(file, new_dir, pdb_utils, parser, cdr, fwr):
    """
    Purpose of the function is to isolate the CDR, FWR, or both for a specific
    structure file. Once that region is isolated then a fasta file can be
    created to detail the sequences of that region.

    Args:
        file (str): path to structure file
        new_dir (str): path to new directory
        pdb_utils (PDBUtils): an instance of the PDBUtils class
        parser (PDB.PDBParser): Biopython tool used to convert a structure file
         into a biopython structure
        cdr (bool): Whether to include CDRs
        fwr (bool): Whether to include FWRs
        

    Returns:
        str: String saying that the structure ran successfully
    """
    
    # naming the final fasta file
    new_file = f"{new_dir}/{(file.split('/')[-1]).split('.')[0]}.fasta"

    # converting file to biopython structure using PDBUtils class
    struct_name = os.path.splitext(os.path.basename(file))[0]
    _, structure = pdb_utils(file, return_biopython_structure=True, write_file=False)
    struct_dict = pdb_utils.extract_structure_to_dict(structure[0])
    seq_dict = {}
    region_of_interest = []
    
    # determining the regions of interest
    if cdr:
        region_of_interest += ['cdr1', 'cdr2', 'cdr3']
    if fwr:
        region_of_interest += ['fwr1', 'fwr2', 'fwr3', 'fwr4']
    

    # obtaining the sequences for the heavy chain in the region of interest
    VH_seq = struct_dict["H"]["seq"] # grabbing sequence of heavy chain
    # checking if anarci can recognize chain as a heavy chain
    out_anar = get_region_splits_anarci(VH_seq)
    for key in region_of_interest:
        if key in out_anar[0]:
            seq_dict[f"H_{key}"] = out_anar[0][key]

    # check if there is a light chain in the structure
    VL_seq = struct_dict["L"]["seq"] if "L" in struct_dict else None

    # if there is a ligiht chain get the region of interest
    if VL_seq:
        try:
            out_anar = get_region_splits_anarci(VL_seq)
            for key in region_of_interest:
                if key in out_anar[0]:
                    seq_dict[f"L_{key}"] = out_anar[0][key]
        except:
            print(f"Failed: {os.path.splitext(os.path.basename(file))[0]}")

    # add the complex and its CDR sequence to the dictionary
    if len(seq_dict.keys()) > 0:
        with open(new_file, "w") as f:
            for chain_id, seq in seq_dict.items():
                f.write(f">{chain_id}\n")
                f.write(f"{seq}\n")
        return f"Passed: {os.path.splitext(os.path.basename(file))[0]}"
    else:
        return f"Failed: {os.path.splitext(os.path.basename(file))[0]}"


def convert_to_cdrs(old_dir, new_dir, cdr=False, fwr=False):
    """
    This function serves to move a series of structure files located
    in the input directory and moves them to the output directory.
    Before moving them, the function uses the obtain_cdrs function
    to isolate the CDRs, FWRs, or both into fasta files.

    Args:
        old_dir (str): Path to input directory
        new_dir (str): Path to output directory
        cdr (bool): Whether to include CDRs
        fwr (bool): Whether to include FWRs
    """
    pdb_utils = PDBUtils(verbose=False)
    parser = PDB.PDBParser(QUIET=True)
    os.makedirs(new_dir, exist_ok=True)  # create new output directory
    if os.path.splitext(old_dir)[1] == "":
        pdb_files = glob(f"{old_dir}/*.pdb")
    else:
        pdb_files = [old_dir]
    # parallelizing parsing and moving the structures
    cpu_parallel = ParallelProcessorForCPUBoundTasks(obtain_cdrs, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_files, new_dir, pdb_utils, parser, cdr, fwr, struct)


def main():
    """
    This is the main function for the Setup script. The main purpose of this
    script is to create input directories to do single chain analysis with
    foldseek. It either creates a directory filled with structures consisting
    only of antigen chains, or with the whole structure, but all under one
    chain ID.
    """

    # define inputs
    parser = argparse.ArgumentParser(description="Creating Antigen Directories to do Single-Chain Analysis with Foldseek")
    parser.add_argument("-q", "--query_dir", help="Path to the directory holding the query structures",
                        required=True)
    parser.add_argument("-t", "--target_dir", help="Path to the directory holding the target structures",
                        required=False)
    parser.add_argument("-o", "--no_query", help="Remove the query keyword that is added when creating a directory (not available when inputting a value for target_dir", default=False, required=False)
    parser.add_argument("-k", "--keyword", help="Keyword used for naming the output directories",
                        required=True)
    parser.add_argument("-s", "--setup", help="The type of setup to preform, either for single_chain_ag, single_chain, or multimer", required=True)
    args = parser.parse_args()

    query_dir = args.query_dir  # path to query directory
    query_dir = query_dir if query_dir[-1] != "/" else query_dir[:-1]
    target_dir = args.target_dir  # path to target directory
    no_query = args.no_query
    if target_dir:
        target_dir = target_dir if target_dir[-1] != "/" else target_dir[:-1]
        no_query = False

    dir_path = os.path.dirname(query_dir)
    dir_path = "./" if dir_path == "" else dir_path
    keyword = args.keyword
    setup = args.setup

    # Need to delete the directories and csv if a previous run exists in the space
    if os.path.exists(f"{dir_path}/{keyword}_query"):
        shutil.rmtree(f"{dir_path}/{keyword}_query")
    if os.path.exists(f"{dir_path}/{keyword}_target"):
        shutil.rmtree(f"{dir_path}/{keyword}_target")
        
    # Determining what setup is required for single chain analysis
    new_dir = f"{dir_path}/{keyword}" if no_query else f"{dir_path}/{keyword}_query"
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    if target_dir:
        new_target = f"{dir_path}/{keyword}_target"
        if os.path.exists(new_target):
            shutil.rmtree(new_target)
    
    if setup == "single_chain_ag":  # comparisons between just the Ag chains
        print("Preparing single chain antigen analysis:")
        print("Doing setup for Query")
        create_sc_ag(query_dir, new_dir)
        if target_dir:
            print("Doing setup for Target")
            create_sc_ag(target_dir, new_target)
    elif setup == "single_chain_ligand":
        print("Preparing single chain ligand analysis:")
        print("Doing setup for Query")
        create_sc_ligand(query_dir, new_dir)
        if target_dir:
            print("Doing setup for Target")
            create_sc_ligand(target_dir, new_target)
    elif setup == "single_chain":  # comparisons of the complex
        print("Preparing single chain analysis")
        print("Doing setup for Query")
        create_sc(query_dir, new_dir)
        if target_dir:
            print("Doing setup for Target")
            create_sc(target_dir, new_target)
    elif setup == "ag_chains":
        print("Preparing antigen chain analysis")
        print("Doing setup for Query")
        create_just_ag(query_dir, new_dir)
        if target_dir:
            print("Doing setup for Target")
            create_just_ag(target_dir, new_target)
    elif setup == "ligand_chains":
        print("Preparing ligand chain analysis")
        print("Doing setup for Query")
        create_just_ligand(query_dir, new_dir)
        if target_dir:
            print("Doing setup for Target")
            create_just_ligand(target_dir, new_target)
    elif setup == "complex_with_sc_ag":
        print("Preparing complex with modified single antigen chain")
        print("Doing setup for Query")
        complex_with_sc_ag(query_dir, new_dir)
        if target_dir:
            print("Doing setup for Target")
            complex_with_sc_ag(target_dir, new_target)
    elif setup == "fasta":
        print("Converting Structure Files into Fasta File")
        print("Doing setup for Query")
        convert_to_fasta(query_dir, new_dir)
        if target_dir:
            print("Doing setup for Target")
            convert_to_fasta(target_dir, new_target)
    elif setup == "cdr":
        print("Converting Structure Files into Fasta File of the CDR")
        print("Doing setup for Query")
        convert_to_cdrs(query_dir, new_dir, cdr=True)
        if target_dir:
            print("Doing setup for Target")
            convert_to_cdrs(target_dir, new_target, cdr=True)
    elif setup == "fw":
        print("Converting Structure Files into Fasta File of the FWR")
        print("Doing setup for Query")
        convert_to_cdrs(query_dir, new_dir, fwr=True)
        if target_dir:
            print("Doing setup for Target")
            convert_to_cdrs(target_dir, new_target, fwr=True)
    elif setup == "cdr_and_fwr":
        print("Converting Structure Files into Fasta File of the CDR and FWR")
        print("Doing setup for Query")
        convert_to_cdrs(query_dir, new_dir, cdr=True, fwr=True)
        if target_dir:
            print("Doing setup for Target")
            convert_to_cdrs(target_dir, new_target, cdr=True, fwr=True)
    else:  # setup parameter is not an expected input
        raise ValueError(f"Unknown value provided for setup: {setup}")


if __name__ == "__main__":
    main()