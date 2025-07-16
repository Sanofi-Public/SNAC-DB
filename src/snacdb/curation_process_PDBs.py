import pandas as pd
import shutil
import sys
import os
import numpy as np
from glob import glob
from pathlib import Path
import argparse
from Bio import PDB
from dateutil import parser as date_parser
from snacdb.utils.pdb_utils_clean_parse import PDBUtils, dict_to_structure
from snacdb.utils.sequence_utils import chain_dict_to_VH_VL_Ag_categories
from snacdb.utils.structure_utils import *
from snacdb.utils.parallelize import ParallelProcessorForCPUBoundTasks

def isdate(date):
    """
    Determine if the input is in datetime format

    Args:
        date (str): string that is potentially a date.

    Returns:
        (bool): Whether the string is a date or not
    """
    
    try:
        date_parser.parse(date)
        return True
    except:
        return False

def isfloat(num):
    """
    Determine if the input is a float.

    Args:
        num (str): string that is potentially a float.

    Returns:
        (bool): Whether the string is a float or not
    """

    try:
        float(num)
        return True
    except:
        return False

def parse_pdb(asymmetric):
    """
    Parse through a PDB file to get information about the structure.

    Args:
        asymmetric (str): name of file to parse.

    Returns:
        (list): Information derived from the parsed structure file
    """

    # providing some initial default values
    title = ""
    classification = "Classification is Missing"
    resol = "Resolution is Missing"
    rel_date = "Release Date is Missing"
    dep_date = "Deposition Date is Missing"
    method = "Method is Missing"

    with open(asymmetric, 'r') as pdb_file:
        for line in pdb_file:
            split_line = line.split()
            if len(split_line) <= 1:
                continue
            # catches resolution value
            if "REMARK   2 RESOLUTION" in line:
                for num in split_line[3:]:
                    if isfloat(num):
                        resol = num
                        
            # catching deposite date and classification
            elif "HEADER" in split_line[0]:
                classification = " ".join(split_line[1:-2])
                possible_date = split_line[-2]
                while len(possible_date) >= 8:
                    if isdate(possible_date):
                        dep_date = possible_date
                        break
                    else:
                        possible_date = possible_date[1:]

            # catching release date
            elif "REVDAT   1" in line:
                possible_date = split_line[2]
                while len(possible_date) >= 8:
                    if isdate(possible_date):
                        rel_date = possible_date
                        break
                    else:
                        possible_date = possible_date[1:]

            # catching experiment used
            elif "EXPDTA" in split_line[0]:
                method = " ".join(split_line[1:])

            # catching title of the structure file
            elif "TITLE" in split_line[0]:
                title += " ".join(split_line[1:])

    # if title is empty then place default value
    title = "Title is Missing" if title == "" else title
    return title, classification, resol, rel_date, dep_date, method


def parse_cif(asymmetric):
    """
    Parse through a CIF file to get information about the structure.

    Args:
        asymmetric (str): name of file to parse.

    Returns:
        (list): Information derived from the parsed structure file
    """

    # providing some initial default values
    title = ""
    classification = "Classification is Missing"
    resol = "Resolution is Missing"
    rel_date = "Release Date is Missing"
    dep_date = "Deposition Date is Missing"
    method = "Method is Missing"

    with open(asymmetric, 'r') as pdb_file:
        flag = False
        title_flag = False
        for line in pdb_file:
            split_line = line.split()
            if len(split_line) > 0:
                # catching resolutions
                if "_reflns.d_resolution_high" in split_line[0] or "_em_3d_reconstruction.resolution" == split_line[0]:
                    if isfloat(split_line[-1]):
                        resol = split_line[-1]
                
                # catching deposit date
                elif "_pdbx_database_status.recvd_initial_deposition_date" in split_line[0]:
                    if isdate(line.split()[-1]):
                        dep_date = split_line[-1]

                # catching classification 
                elif "_struct_keywords.pdbx_keywords" in split_line[0]:
                    classification = " ".join(split_line[1:])

                # catching title of the structure file
                elif "_struct.title" in split_line[0] or title_flag == True:
                    if title_flag:
                        if line[0] == "_":
                            title_flag = False
                            continue
                        title += line[:-1]
                    else:
                        title += " ".join(split_line[1:])
                        title_flag = True

                # catching release date of structure
                elif "_pdbx_audit_revision_history.revision_date" in line:
                    if len(split_line) > 1:
                        rel_date = split_line[-1]
                    else:
                        flag = True
                elif flag:
                    if line[0] == "1":
                        flag = False
                        for info in split_line[::-1]:
                            if isdate(info):
                                rel_date = info
                                break
                    else:
                        continue

                # catching experimental method used
                elif "_refine_hist.pdbx_refine_id" in split_line[0] or "_exptl.method" == split_line[0]:
                    method = " ".join(split_line[1:])

    # placing default value for title
    title = "Title is Missing" if title == "" else title
    return title, classification, resol, rel_date, dep_date, method

    
def process_pdb(pdb_file, out_folder, pdb_utils):
    """
    Returns a dictionary containing information on the Heavy Chain, Light
    Chain, and Antigens contained in the PDB/CIF structure.

    Args:
        pdb_file (str): PDB/CIF file name.
        out_folder (str): Path to folder where PDB and NPY file will be
         saved to.
        pdb_utils (PDBUtils): An instance of the PDBUtils class that is used
         to process and label the chains.

    Returns:
        row (dict): Dictionary summarizing information about the structure
    """

    # getting file name and file path
    file_name_and_path, ext = os.path.splitext(pdb_file)
    file_name = os.path.basename(file_name_and_path)
    file_path = os.path.dirname(file_name_and_path)

    pdb_base = file_name
    # break up file name to determine bioassembly
    if "pdb" in ext:
        bioassembly = ""
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
    
    # determine which structure is the asymmetric unit cell for the structure
    asymmetric = pdb_file
    if bioassembly != "0":  # structure is not ASU
        asymmetric_pdb = os.path.join(file_path, pdb_base)
        # searching for potential ASU file to get all the metadataa
        if os.path.exists(asymmetric_pdb + ".pdb"):
            asymmetric = asymmetric_pdb + ".pdb"
        elif os.path.exists(asymmetric_pdb + ".cif"):
            asymmetric = asymmetric_pdb + ".cif"
        elif os.path.exists(asymmetric_pdb + ".mmcif"):
            asymmetric = asymmetric_pdb + ".mmcif"

    name = f"{pdb_base}-ASU{bioassembly}" #pdb id w/ bioassembly
    output_pdb_file = os.path.join(out_folder, f'{name}.pdb')

    # determine the biopython structures and chain_dict for all the frames in the model (if there is more than one frame)
    num_of_structures, structure = pdb_utils(pdb_file, output_pdb_file=output_pdb_file, return_biopython_structure=True)
    try:
        chain_dict_list = [pdb_utils.extract_structure_to_dict(model) for model in structure]
        chain_dict_list = [chain_dict_to_VH_VL_Ag_categories(chain_dict) for chain_dict in chain_dict_list]
    except Exception as e:
        for i in range(num_of_structures):
            if num_of_structures != 1:
                name = f'{pdb_base}-ASU{bioassembly}-frame{i}'
            os.remove(f"{name}.pdb")
        raise RuntimeError("There was an error in creating the npy file") from e
    row_list = []

    # loop will be length of 1 if there is not multiple frames (not nmr)
    for i in range(num_of_structures):
        # checking if there are multiple models in the structure file
        if num_of_structures != 1:
            name = f'{pdb_base}-ASU{bioassembly}-frame{i}'
        chain_dict = chain_dict_list[i]
        #save npy file
        npy_file = os.path.join(out_folder, name + '-atom37.npy')
        np.save(npy_file, chain_dict)

        # identify the Ab and Nb in structure
        VH_VL_pairs, VHH_chains = identify_VH_VL_pairs_in_contact(chain_dict, return_VHH_chains=True)
    
        # solves for the metadata in the ASU file
        if "pdb" in os.path.splitext(asymmetric)[1]:
            title, classification, resol, rel_date, dep_date, method = parse_pdb(asymmetric)
        else:
            title, classification, resol, rel_date, dep_date, method = parse_cif(asymmetric)
    
        # defining row dictionary that houses information on structure
        row = {}
        row['Name'] = name
        row['Parent_File'] = os.path.basename(pdb_file)
        row['PDB_ID'] = pdb_base
        row['Bioassembly'] = bioassembly
        row['Structure_Title'] = title
        row['Structure_Classification'] = classification
        row['Resolution'] = resol
        row['Method'] = method
        row['Date_Deposited'] = dep_date
        row['Date_Released'] = rel_date
        row['Chain_VH'] = list(chain_dict['VH'].keys())
        row['Chain_VL'] = list(chain_dict['VL'].keys())
        row['Chain_Ag'] = list(chain_dict['Ag'].keys())
        row['VH_VL_Pairs'] = VH_VL_pairs
        row['Chain_VHH'] = VHH_chains
        row_list.append(row)

        print("Successfully Completed:", name)
    return row_list
    

def main():
    """
    The main function of the first part of the pipeline. The goal of this function
    is to process, clean, and label a series of PDB/CIF files located in the input
    directory. A new PDB file is created that contains the newly cleaned structures
    and a NPY file is created that holds a dictionary containing information about
    the structure, sequence, and labeling of the chains.
    """

    # define inputs
    parser = argparse.ArgumentParser(description="Process and clean PDB/CIF Structures")
    # Add arguments
    parser.add_argument(
        "input_dir",
        help="Path to the input PDB or CIF directory. If omitted, the --test option must be used.")
    args = parser.parse_args()
    pdb_folder = args.input_dir
    if pdb_folder[-1] == "/":
        pdb_folder = pdb_folder[:-1]
    pdb_folder = pdb_folder if "/" in pdb_folder else "./" + pdb_folder

    # deleting pre-existing directories
    if os.path.exists(f"{pdb_folder}_parsed"):
        shutil.rmtree(f"{pdb_folder}_parsed")
    if os.path.exists(f"{pdb_folder}_parsed_file_chains.csv"):
        os.remove(f"{pdb_folder}_parsed_file_chains.csv")

    # define input and output folders, and output csv
    out_folder = f'{pdb_folder}_parsed'
    output_csv_name = f'{pdb_folder}_parsed_file_chains.csv'
    os.makedirs(out_folder, exist_ok=True)

    # running process_pdb through parallel function
    pdb_utils = PDBUtils()
    pdb_files = sorted(glob(os.path.join(pdb_folder, "*")))
    cpu_parallel = ParallelProcessorForCPUBoundTasks(process_pdb, max_workers=48)
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