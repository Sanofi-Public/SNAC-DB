import pandas as pd
import os
import numpy as np
import copy
import argparse
import shutil
from glob import glob
from snacdb.utils.pdb_utils_clean_parse import PDBUtils, dict_to_structure
from snacdb.utils.sequence_utils import chain_dict_to_VH_VL_Ag_categories
from snacdb.utils.structure_utils import identify_VH_VL_pairs_in_contact
from snacdb.utils.parallelize import ParallelProcessorForCPUBoundTasks
from snacdb.curation_process_PDBs import parse_pdb, parse_cif
from snacdb.curation_identify_complexes import identify_pairing
from snacdb.curation_filter_complexes import filter_func_VHH, filter_func_VH_VL


def add_pdb_header(pdb_file, info, chain_dict):
    """
    Adds all the header information from the original PDB file into the new one.

    Args:
        pdb_file (str): Newly created PDB file
        input_dir (str): Path to input directory.
        chain_dict (dict): Dictionary containing information on the structure
        df (pd.Series): Series containing information about the specific
         structures
    """
    pdb_name = pdb_file.split("/")[-1][:-4]
    pdb_id = pdb_name.split("ASU")[0]

    # Reading the original file content
    with open(pdb_file, "r") as f_read:
        content = "".join(f_read.readlines()[1:])

    with open(pdb_file, "w") as f_new:
        title, classification, resol, rel_date, dep_date, method = info
        f_new.write(f"NAME       {title}\n")
        f_new.write(f"HEAD       {classification}\n")
        f_new.write(f"IDCODE     {pdb_id}\n")
        f_new.write(f"DEPOSITION_DATE     {dep_date}\n")
        f_new.write(f"RELEASE_DATE     {rel_date}\n")
        f_new.write(f"STRUCTURE_METHOD     {method}\n")
        if resol == "Resolution is Missing":
            f_new.write(f"RESOLUTION     {resol}\n")
        else:
            f_new.write(f"RESOLUTION     {resol}Å\n")

        remark = []
        if "VHH" in pdb_name:
            VHH = pdb_name.split('-VHH_')[-1][0]
            remark.append(f"The VHH Chain ID is H, with its old ID being {VHH}")
        else:
            VH = pdb_name.split('-VH_')[-1][0]
            VL = pdb_name.split('-VL_')[-1][0]
            remark.append(f"The VH Chain ID is H, with its old ID being {VH}")
            remark.append(f"The VL Chain ID is L, with its old ID being {VL}")
        Ag_chain = True if "Ag" in pdb_name else False
        if Ag_chain:
            Ag_chain = sorted(set(chain_dict.keys()) - set(["H", "L"]))
            Ag = ", ".join(Ag_chain)
            Ag_old_id = ", ".join([chain_dict[Ag_id]["old_id"] for Ag_id in Ag_chain])
            remark.append(f"The Antigen Chain ID is {Ag}, with its old ID being {Ag_old_id}")
        remark = "; ".join(remark)
        f_new.write(f"STRUCTURE_IDENTITY  {remark}\n")
        f_new.write("REMARK 99 Modified by Bryan Munoz Rivero, Sanofi\n")
        f_new.write(content)  # Adding the original content


def autofill_row(pdb_file, name, idx, info, chain_dict, comments):
    """
    Returns information about the complex in a way that it can be
     stored into a summary csv that contains details on the
     complex.

    Args:
        pdb_file (str): Name of the original structure file
        name (str): Basename of the complex.
        idx (str): String that details the labeling of the complex.
        info (list): Information collected from the metadata file
        chain_dict (dict): Contains information about the complex and
         its structure.
        comments (str): Contains information about how the complex is created.

    Returns:
        complex_dict (dict): Dictionary containing information in structure that
         will be used to be inputted into summary csv.
    """

    complex_dict = {}  # defining the dictionary that will hold all the data
    complex_dict["Name"] = f"{name}-{idx}"
    complex_dict["Parent_File"] = os.path.basename(pdb_file)
    asu = ""
    # captures asu value if it more than one digit
    for i in name.split("-ASU")[1]:
        if not i.isdigit():
            break
        asu += i
    complex_dict["Bioassembly"] = asu

    # stores the information from the metadata file
    title, classification, resol, rel_date, dep_date, method = info
    complex_dict['Structure_Title'] = title
    complex_dict['Structure_Classification'] = classification
    complex_dict["Resolution"] = resol
    complex_dict['Method'] = method
    complex_dict['Date_Deposited'] = dep_date
    complex_dict['Date_Released'] = rel_date
    complex_dict["PDB_ID"] = name.split("-ASU")[0]

    # store information about the labeling of the structure
    complex_dict["Complex"] = idx
    complex_dict["Chain_VH"] = "H"
    complex_dict["Chain_VH_old_id"] = chain_dict["H"]["old_id"][0]

    if "L" in chain_dict.keys():  # antibody case
        complex_dict["Chain_VL"] = "L"
        complex_dict["Chain_VL_old_id"] = chain_dict["L"]["old_id"][0]
    else:  # nanobody case
        complex_dict["Chain_VL"] = None
        complex_dict["Chain_VL_old_id"] = None

    complex_dict["Chain_Ag"] = sorted(set(chain_dict.keys()) - set(["H", "L"]))
    complex_dict["Chain_Ag_old_id"] = [chain_dict[Ag]['old_id'] for Ag in complex_dict["Chain_Ag"]]
    complex_dict["Interaction_Type"] = [chain_dict[Ag]['interaction_type'] for Ag in complex_dict["Chain_Ag"]]
    complex_dict["Antigen_Type"] = [chain_dict[Ag]['antigen_type'] for Ag in complex_dict["Chain_Ag"]]

    # save the comments made when creating and refining the complex
    complex_dict["comments"] = comments
    return complex_dict


def identify_complex(chain_dict, ligand, Ags, rep=False):
    """
    The purpose of this script is identify what antigens are potentially
     interacting with the specified ligand. This check uses a loose criteria
     to determine interactions. 

    Args:
        chain_dict (dict): Contains information about the complex and
         its structure.
        ligands (tuple or str): Contains the chain ID(s) for the ligand.
        Ags (list): A list of all the antigen-chain IDs
        rep (bool): Whether the complex has identical ligands to another
         complex in the same structure file

    Returns:
        chain_dict_new (dict): Dictionary detailing information about the newly
         created complex and its structure.
        comments (list): comments detailing how the antigen chains in the complex
         were determined
        idx (str): string detailing the labeling of the complex based on chain IDs
    """
    # Removing VH-VL/VHH from antigen list
    Ag_chains = list(set(Ags) - set([lig.split('_')[0] for lig in ligand]))
    if type(ligand) is tuple:  # for VH-VL
        chain_dict_new, comments = identify_pairing(ligand[0], ligand[1], Ag_chains, chain_dict)
        idx = f"VH_{ligand[0].split('_')[0]}-VL_{ligand[1].split('_')[0]}"
    else:  # for VHH
        chain_dict_new, comments = identify_pairing(ligand, None, Ag_chains, chain_dict)
        idx = f"VHH_{ligand.split('_')[0]}"

    # Adding antigen chains to idx
    Ag_chains = set(chain_dict_new[Ag]["old_id"] for Ag in chain_dict_new.keys() if Ag not in ["H", "L"])
    if len(Ag_chains) > 0:
        idx += "-Ag"
        for Ag in sorted(Ag_chains):
            idx += f"_{Ag}"

    # if there is another complex with the same ligand ID then place the replicate keyword
    if rep:
        f"-replicate{chain_dict_new['H']['old_id'].split('_')[1]}"

    return chain_dict_new, comments, idx


def filter_complex(chain_dict, ligand, Ags, idx):
    """
    The purpose of this script is identify what antigens are potentially
     interacting with the specified ligand. This check uses a loose criteria
     to determine interactions. 

    Args:
        chain_dict (dict): Contains information about the complex and
         its structure.
        ligands (tuple or str): Contains the chain ID(s) for the ligand.
        Ags (list): A list of all the antigen-chain IDs
        idx (str): string detailing information about the complex

    Returns:
        chain_dict_new (dict): Dictionary detailing information about the newly
         refined complex and its structure.
        comments (dict): comments detailing how the antigen chains in the complex
         were determined
        idx_new (str): string detailing the labeling of the complex based on chain
         IDs
    """
    if "VHH" in idx:  # nanobody
        chain_dict_new, comments = filter_func_VHH(ligand, Ags, chain_dict)
    else:  # antibody
        chain_dict_new, comments = filter_func_VH_VL(list(ligand), Ags, chain_dict)

    # Adding antigen chains to idx
    idx_new = idx.split("-Ag_")[0]
    Ag_chains = set(chain_dict_new[Ag]["old_id"] for Ag in chain_dict_new.keys() if Ag not in ["H", "L"])
    if len(Ag_chains) > 0:
        idx_new += "-Ag"
        for Ag in sorted(Ag_chains):
            idx_new += f"_{Ag}"

    # if there is a replicate then add back in the keyword
    if "replicate" in idx:
        rep = idx.split("replicate")[-1]
        idx_new += f"-replicate{rep}"

    # getting dictionary form (row) that will be added to csv
    return chain_dict_new, comments, idx_new


def pipeline_func(pdb_file, out_folder, pdb_utils):
    """
    The purpose of this function to identify all possible complexes in a
    specified structure file. This includes finding all the complexes for
    each frame in the structure file, if it has more than one frame.

    Args:
        pdb_file (str): Path and name of structure file to parse.
        out_folder (str): folder to save new structures to.
        pdb_utils (PDBUtils): An instance of the PDBUtils class that is used
         to process and label the chains.

    Returns:
        row_list (list): list of dictionary detailing each complex from the
         structure file
    """

    # getting the name and path of the initial structure file
    file_name_and_path, ext = os.path.splitext(pdb_file)
    file_name = os.path.basename(file_name_and_path)
    file_path = os.path.dirname(file_name_and_path)

    # determing the bioassembly number of the structure
    pdb_base = file_name
    if "pdb" in ext:
        bioassembly = ""
        # ensures we capture asu values higher than one digit
        for i in reversed(ext):
            if not i.isdigit():
                break
            bioassembly += i
        if len(bioassembly) > 0:
            bioassembly = bioassembly[::-1]
        else:
            bioassembly = "0"
    elif "cif" in ext:
        if "-assembly" in file_name:
            pdb_base, bioassembly = file_name.split("-assembly")
        else:
            bioassembly = "0"

    # need to check for a base file as either cif or pdb
    asymmetric = pdb_file
    if bioassembly != "0":  # file is not the asymmetric unit cell
        asymmetric_pdb = os.path.join(file_path, pdb_base)
        if os.path.exists(asymmetric_pdb + ".pdb"):
            asymmetric = asymmetric_pdb + ".pdb"
        elif os.path.exists(asymmetric_pdb + ".cif"):
            asymmetric = asymmetric_pdb + ".cif"
        elif os.path.exists(asymmetric_pdb + ".mmcif"):
            asymmetric = asymmetric_pdb + ".mmcif"

    # collect information from the metadata file
    if "pdb" in os.path.splitext(asymmetric)[1]:
        info = parse_pdb(asymmetric)
    else:
        info = parse_cif(asymmetric)

    name = f"{pdb_base}-ASU{bioassembly}"  # pdb id w/ bioassembly
    output_pdb_file = os.path.join(out_folder, f'{name}.pdb')

    # determining if the structure has multiple frames
    num_of_structures, structure = pdb_utils(pdb_file, output_pdb_file=output_pdb_file, return_biopython_structure=True, write_file=False)
    # converting each structure to a dictionary containing information about each structure
    chain_dict_list = [pdb_utils.extract_structure_to_dict(model) for model in structure]
    chain_dict_list = [chain_dict_to_VH_VL_Ag_categories(chain_dict) for chain_dict in chain_dict_list]
    row_list = []

    for i in range(num_of_structures):  # loop through all the frames
        # ensures that changes made on chain_dict will not be saved
        chain_dict = copy.deepcopy(chain_dict_list[i])

        # adding frame keyword if there are multiple frames
        if num_of_structures != 1:
            name = f'{pdb_base}-ASU{bioassembly}-frame{i}'

        # determining the Ab and Nb complexes in the structure
        VH_VL_pairs, VHH_chains = identify_VH_VL_pairs_in_contact(chain_dict, return_VHH_chains=True)
        if len(VH_VL_pairs) == 0 and len(VHH_chains) == 0:
            continue  # skip analysis if there are no Ab/Nb
        for ag in chain_dict["Ag"]:
            ab_like = any([ag in [c1[0], c2[0]] for c1, c2 in VH_VL_pairs])
            nb_like = ag in [c[0] for c in VHH_chains]
            if ab_like and nb_like:
                chain_dict["Ag"][ag]["antigen_type"] = "Ab-like & Nb-like Antigen"
            elif ab_like:
                chain_dict["Ag"][ag]["antigen_type"] = "Ab-like Antigen"
            elif nb_like:
                chain_dict["Ag"][ag]["antigen_type"] = "Nb-like Antigen"
            else:
                chain_dict["Ag"][ag]["antigen_type"] = "Antigen"

        ligands = VH_VL_pairs + VHH_chains
        # looping through all the ligands in the structure to solve for each complex
        for lig in ligands:
            Ags = list(chain_dict['Ag'].keys())  # get Ag chain IDs

            # test for replicate ligand IDS
            if lig in VH_VL_pairs:
                lig_id = (lig[0].split("_")[0], lig[1].split("_")[0])
                rep_check = sum([lig_id == (H.split("_")[0], L.split("_")[0]) for H, L in VH_VL_pairs])
            else:
                lig_id = lig.split("_")[0]
                rep_check = sum([lig_id == H.split("_")[0] for H in VHH_chains])
            rep = True if rep_check >= 2 else False

            # identify and then refine the complex for the specified ligand
            chain_dict_complex, comments_complex, idx_old = identify_complex(chain_dict, lig, Ags, rep)
            lig = "H" if "VHH" in idx_old else ("H", "L")
            Ags = list(set(chain_dict_complex.keys()) - set(["H", "L"]))
            chain_dict_filter, comments_filter, idx = filter_complex(copy.deepcopy(chain_dict_complex), lig, Ags, idx_old)

            # writing info about the complex into a dictionary that can be saved into summary file
            comments = {key: comments_complex[key] + comments_filter[key] for key in comments_complex}
            row = autofill_row(pdb_file, name, idx, info, chain_dict_filter, comments)

            # save final complex in structure and npy file
            output_file = os.path.join(out_folder, f'{name}-{idx}')
            structure = dict_to_structure(chain_dict_filter)
            pdb_utils.write_pdb_with_seqres_from_structure(structure[0], output_file + ".pdb")
            add_pdb_header(output_file + ".pdb", info, chain_dict_filter)
            np.save(output_file + "-atom37.npy", chain_dict_filter)
            row_list.append(row)  # add info about complex to row_list

    return row_list


def main():
    """
    This is the main function for curation_SNAC_DB_Pipeline.py. The purpose of this
    script is to run the SNAC-DB pipeline on series of structures. The goal of the
    pipeline is to extract all possible Ab/Nb complexes from these structure files.
    Once done, metadata about this analysis and pipeline is saved in the form of new
    structure files of the complexes, npy files detailing information about the
    complexes, and a summary csv file containing more metadata information.
    """

    # define inputs
    parser = argparse.ArgumentParser(description="Process and clean PDB/CIF Structures")
    # Add arguments
    parser.add_argument("input_dir", help="Path to the input PDB or CIF directory.")

    # ensures that the input directory has the correct path setup
    args = parser.parse_args()
    pdb_folder = args.input_dir
    if pdb_folder[-1] == "/":
        pdb_folder = pdb_folder[:-1]
    pdb_folder = pdb_folder if "/" in pdb_folder else "./" + pdb_folder

    # remove previous runs in the same directory
    if os.path.exists(f"{pdb_folder}_curated"):
        shutil.rmtree(f"{pdb_folder}_curated")
    if os.path.exists(f"{pdb_folder}_curation_summary.csv"):
        os.remove(f"{pdb_folder}_curation_summary.csv")

    # define input and output folders, and output csv
    out_folder = f'{pdb_folder}_curated'
    output_csv_name = f'{pdb_folder}_curation_summary.csv'
    os.makedirs(out_folder, exist_ok=True)

    # running process_pdb through parallel function
    pdb_utils = PDBUtils()
    pdb_files = sorted(glob(os.path.join(pdb_folder, "*")))
    cpu_parallel = ParallelProcessorForCPUBoundTasks(pipeline_func, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_files, out_folder, pdb_utils)

    # saving the recorded information for each structure into the csv
    flat_processed_rows = []
    for parsed_row in processed_rows:
        for row in parsed_row:
            flat_processed_rows.append(row)
    processed_df = pd.DataFrame(flat_processed_rows)
    processed_df = processed_df.sort_values(by=["Name", "Bioassembly"])
    processed_df.to_csv(output_csv_name, index=False)
    processed_df.info()


if __name__ == "__main__":
    main()