import pandas as pd
import sys
import os
import ast
import numpy as np
import itertools
import copy
import argparse
import shutil
from itertools import chain as Chain
from snacdb.utils.pdb_utils_clean_parse import PDBUtils, dict_to_structure
from snacdb.utils.sequence_utils import chain_dict_to_VH_VL_Ag_categories, get_region_splits_anarci
from snacdb.utils.structure_utils import *
from snacdb.utils.parallelize import ParallelProcessorForCPUBoundTasks


def nan_any(contact_map, axis=-1):
    """
    Performs np.any while preserving NaN values.

    Args:
        contact_map (np.ndarray): Input boolean/numeric array.
        axis (int): Axis along which to apply np.any.

    Returns:
        result (np.ndarray): Result after applying np.any with 
         NaN preservation.
    """
    nan_mask = np.isnan(contact_map)  # Identify NaNs
    result = np.any(contact_map==1, axis=axis)  # Apply np.any

    # If all values along the axis were NaN, keep NaN in result
    all_nan_mask = np.all(nan_mask, axis=axis)
    result = result.astype(float)  # Convert to float to accommodate NaNs
    result[all_nan_mask] = np.nan  # Assign NaN where all values were NaN

    return result


def get_contact_map_atom37(coord_chain1, coord_chain2, cutoff=8.0, alpha_carbon=True):
    """
    Determines contact map between two different chains with a specified cutoff
     distance. As well, you can specify to do the contact map using all the heavy
     atoms from atom37 coordinate space or to just use alpha carbons. 

    Args:
        coord_chain1 (np.ndarray): Atom37 coordinate for structure 1.
        coord_chain2 (np.ndarray): Atom37 coordinate for structure 2.
        cutoff (float): The maximum distance where contact is considered
        alpha_carbon (bool): Whether to calculate the contact map for only the 
         alpha carbon of the structure

    Returns:
        contact_map (np.ndarray): Matrix showing contact between structure 1 
         and 2.
    """

    # goal for contact map is to just do contact map over the cdrs of the ligand and all heavy atoms for antigen
    if alpha_carbon: # only considering alpha carbons for atom37 structures
        coord_chain1 = coord_chain1[:, 1:2, :]
        coord_chain2 = coord_chain2[:, 1:2, :]

    # distance matrix of the two structures
    dist_mat = np.sqrt(np.sum((coord_chain1[:, None, :, None, :] - coord_chain2[None, :, None, :, :])**2, axis=-1))
    # Determining contact within cutoff distance and reshaping matrix
    contact_map = np.where(np.isnan(dist_mat), np.nan, np.where(dist_mat < cutoff, 1, 0)) 
    contact_map = contact_map.reshape(list(contact_map.shape[:2]) + [-1])
    contact_map = nan_any(contact_map, axis=-1)
    return contact_map


def check_cdr_contact(V_chain_coords, target_chain_coords, imgt, cutoff=8.0):
    """
    Determines whether there is contact in the CDR region between 
     V_chain_coords and target_chain_coords.

    Args:
        V_chain_coords (np.ndarray): Atom37 coordinate for structure 1.
        target_chain_coords (np.ndarray): Atom37 coordinate for antigen.
        imgt (list): The imgt numbering.
        cutoff (float): The maximum distance where contact is considered.

    Returns:
        contact_dict (dict): Dictionary containing information about if there is contact
        with each CDR loop.
    """

    # Check if the contact happens in the CDR regions
    contact_dict = {'CDR3' : False, 'CDR2' : False, 'CDR1' : False}
    cdrs = [range(27, 38), range(56, 65), range(105, 117)]
    # determining if there is contact for each region in the CDR loop
    for i, region in enumerate(contact_dict.keys()):

        imgt_num = []
        for idx in cdrs[i]:  # section of FR1
            if str(idx) in imgt:
                imgt_num.append(imgt.index(str(idx)))

        contact_map = get_contact_map_atom37(V_chain_coords, target_chain_coords, cutoff=cutoff)
        
        # total_contacts = np.nansum(contact_map[crd_lims[0]: crd_lims[1], :])
        interaction = contact_map[imgt_num, :]
        total_contacts = np.nansum(interaction)

        if total_contacts > 0:
            contact_dict[region] = True

    return contact_dict


# All possible chain ids that can be taken
allowed_chain_ids = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
allowed_chain_ids = list(set(allowed_chain_ids) - set(['H', 'L']))
allowed_chain_ids.sort()


def identify_complex(chain_dict, Ab, Ags, pdb_id, df, rep=False):
    """
    This function determines which antigens are in complex with the
     specified Antibody/Nanobody. Returns information on the structure
     as two different dictionaries and a string (one for the summary
     csv) and the other to create both a pdb and npy file on the newly
     determined complex.

    Args:
        chain_dict (dict): Dictionary containing information on the
         structure.
        Ab (tuple/str): Antigen/Nanobdy Chain id.
        Ags (list): List of all the antigens in the structure.
        pdb_id (str): 4 letter PDB id of structure with its
         bioassembly number.
        df (pd.DataFrame): Dataframe of processed Input Structures

    Returns:
        new_row (dict): Dictionary containing information on the structure that
         is outputted from the autofill_row_ic function.
        chain_dict_new (dict): Dictionary containing information on the structure
         that is will be inputted into the autofill_row_ic function.
        idx (str): Details the VH, VL, and Ag invovled in structure.
    """

    # Removing VH-VL/VHH from antigen list
    Ag_chains = list(set(Ags) - set([ab.split('_')[0] for ab in Ab]))
    if type(Ab) == tuple:  # for VH-VL
        chain_dict_new, comments = identify_pairing(Ab[0], Ab[1], Ag_chains, chain_dict)
        idx = f"VH_{Ab[0].split('_')[0]}-VL_{Ab[1].split('_')[0]}"
    else:  # for VHH
        chain_dict_new, comments = identify_pairing(Ab, None, Ag_chains, chain_dict)
        idx = f"VHH_{Ab.split('_')[0]}"
    # Adding antigen chains to idx
    Ag_chains = set(chain_dict_new[Ag]["old_id"] for Ag in chain_dict_new.keys() if Ag not in ["H", "L"])
    if len(Ag_chains) > 0:
        idx += "-Ag"
        for Ag in sorted(Ag_chains):
            idx += f"_{Ag}"

    if rep:
        idx += f"-replicate{chain_dict_new['H']['old_id'].split('_')[1]}"
    
    # getting dictionary form (new_row) that will be added to csv
    new_row = autofill_row_ic(pdb_id, idx, chain_dict_new, comments, df)
    return new_row, chain_dict_new, idx


def identify_pairing(VH, VL, Ags, chain_dict):
    """
    Determines if the complex is an Antibody or Nanobody. Then from there
     defines the antigens involved in the complex using the criteria of
     contact with the CDR3 region, and if no contacts are found then it
     uses a less stringent definition of CDR2 contact. Dictionary that
     contains information on the complex is then returned, along with a
     string that has information on the complex identification.

    Args:
        VH (str): Chain ID for Heavy chain.
        VL (str): Chain ID for Light Chain.
        Ags (list): List of all the antigens in the structure.
        chain_dict (dict): Dictionary containing information on the
         structure.

    Returns:
        chain_dict_new (dict): Contains information about the complex and
         its labeling.
        comments (str): Contains information about how the complex is created.
    """

    chain_dict_new = {}
    comments = {"Complex_Identity": [], "Single-Chain_Antigens": [], "Multi-Chain_Antigen": [], "Secondary_Check": [], "Replicate_Chains": [], "Complex_Summary": []}
    chain_dict_new['H'] = chain_dict['VH'][VH]
    chain_dict_new['H']["old_id"] = VH
    #finding all antigens that are VH-VL pairs
    VH_VL_edges = identify_VH_VL_pairs_in_contact(chain_dict)
    VH_VL_edges = [(pair[0].split("_")[0], pair[1].split("_")[0]) for pair in VH_VL_edges]
    if VL:  # for Antibodies
        chain_dict_new['L'] = chain_dict['VL'][VL]
        chain_dict_new['L']["old_id"] = VL
        comments["Complex_Identity"] = ["Antibody Complex"]
        Ab = [("VH", VH), ("VL", VL)]
        VH_VL_edges.remove((VH.split("_")[0], VL.split("_")[0]))
    else:  # for nanobodies
        comments["Complex_Identity"] = ["NANOBODY VHH Complex"]
        Ab = [("VH", VH)]

    pair_list = list(set(list(itertools.chain.from_iterable(VH_VL_edges))))

    # Determing Antigen contact if there is contact with CDR3 region
    ag_c_idx = 0
    Ag_list = []
    for c in Ags:
        if c not in Ag_list:
            for chain_id, chain in Ab:
                c_id = chain_id[1]
                Ab_dict = chain_dict[chain_id][chain]
                Ag_atom37 = chain_dict['Ag'][c]['atom37']
                cdr_contact_dict = check_cdr_contact(Ab_dict['atom37'], Ag_atom37, Ab_dict['imgt'], cutoff=12.0)
                contact_cdr3 = cdr_contact_dict["CDR3"]

                if contact_cdr3 and c in pair_list:
                    for pair in VH_VL_edges:
                        if c in pair:
                            c2 = pair[0] if c != pair[0] else pair[1]
                    comments["Multi-Chain_Antigen"].append(f'Adding Multi-Chain Antigen {(c, c2)} because found contact in {c_id}CDR3 for {chain} and {c}')
                    
                    new_c1 = allowed_chain_ids[ag_c_idx]
                    new_c2 = allowed_chain_ids[ag_c_idx + 1]
                    ag_c_idx += 2
                    chain_dict_new[new_c1] = chain_dict['Ag'][c]
                    chain_dict_new[new_c2] = chain_dict['Ag'][c2]
                    chain_dict_new[new_c1]["old_id"] = c
                    chain_dict_new[new_c2]["old_id"] = c2
                    chain_dict_new[new_c1]["pair"] = new_c2
                    chain_dict_new[new_c2]["pair"] = new_c1
                    Ag_list.append(c)
                    Ag_list.append(c2)
                    break
                elif contact_cdr3:
                    comments["Single-Chain_Antigens"].append(f'Added because found contact in {c_id}CDR3 for {chain} and {c}')
                    chain_dict_new[allowed_chain_ids[ag_c_idx]] = chain_dict['Ag'][c]
                    chain_dict_new[allowed_chain_ids[ag_c_idx]]["old_id"] = c
                    ag_c_idx += 1
                    Ag_list.append(c)
                    break
                elif contact_cdr3 is None: #Residue at CDR3 is not resolved
                    comments["Single-Chain_Antigens"].append(f'Could not find {c_id}CDR3 in {chain}')

    # if no antigens were determined for complex then use less stringent conditions
    for c in Ags:
        if c not in Ag_list:
            for chain_id, chain in Ab:
                c_id = chain_id[1]
                Ab_dict = chain_dict[chain_id][chain]
                Ag_atom37 = chain_dict['Ag'][c]['atom37']
                cdr_contact_dict = check_cdr_contact(Ab_dict['atom37'], Ag_atom37, Ab_dict['imgt'], cutoff=12.0)
                contact_cdr2 = cdr_contact_dict["CDR2"]

                if contact_cdr2 and c in pair_list:
                    for pair in VH_VL_edges:
                        if c in pair:
                            c2 = pair[0] if c != pair[0] else pair[1]
                    comments["Multi-Chain_Antigen"].append(f'Adding Multi-Chain Antigen {(c, c2)} because found contact in {c_id}CDR2 for {chain} and {c}')
                    
                    new_c1 = allowed_chain_ids[ag_c_idx]
                    new_c2 = allowed_chain_ids[ag_c_idx + 1]
                    ag_c_idx += 2
                    chain_dict_new[new_c1] = chain_dict['Ag'][c]
                    chain_dict_new[new_c2] = chain_dict['Ag'][c2]
                    chain_dict_new[new_c1]["old_id"] = c
                    chain_dict_new[new_c2]["old_id"] = c2
                    chain_dict_new[new_c1]["pair"] = new_c2
                    chain_dict_new[new_c2]["pair"] = new_c1
                    Ag_list.append(c)
                    Ag_list.append(c2)
                    break
                elif contact_cdr2 is None: # residue at CDR2 is not resolved
                    comments["Single-Chain_Antigens"].append(f'Could not find {c_id}CDR2 in {chain}')
                elif contact_cdr2: # contact is made at CDR2
                    comments["Single-Chain_Antigens"].append(f'Added because found contact in {c_id}CDR2 for {chain} and {c}')
                    chain_dict_new[allowed_chain_ids[ag_c_idx]] = chain_dict['Ag'][c]
                    chain_dict_new[allowed_chain_ids[ag_c_idx]]["old_id"] = c
                    ag_c_idx += 1
                    Ag_list.append(c)
                    break

    for c in Ags:
        if c not in Ag_list:
            for chain_id, chain in Ab:
                c_id = chain_id[1]
                Ab_dict = chain_dict[chain_id][chain]
                Ag_atom37 = chain_dict['Ag'][c]['atom37']
                cdr_contact_dict = check_cdr_contact(Ab_dict['atom37'], Ag_atom37, Ab_dict['imgt'], cutoff=12.0)
                contact_cdr1 = cdr_contact_dict["CDR1"]

                if contact_cdr1 and c in pair_list:
                    for pair in VH_VL_edges:
                        if c in pair:
                            c2 = pair[0] if c != pair[0] else pair[1]
                    comments["Multi-Chain_Antigen"].append(f'Adding Multi-Chain Antigen {(c, c2)} because found contact in {c_id}CDR1 for {chain} and {c}')

                    new_c1 = allowed_chain_ids[ag_c_idx]
                    new_c2 = allowed_chain_ids[ag_c_idx + 1]
                    ag_c_idx += 2
                    chain_dict_new[new_c1] = chain_dict['Ag'][c]
                    chain_dict_new[new_c2] = chain_dict['Ag'][c2]
                    chain_dict_new[new_c1]["old_id"] = c
                    chain_dict_new[new_c2]["old_id"] = c2
                    chain_dict_new[new_c1]["pair"] = new_c2
                    chain_dict_new[new_c2]["pair"] = new_c1
                    Ag_list.append(c)
                    Ag_list.append(c2)
                    break
                elif contact_cdr1 is None: # residue at CDR2 is not resolved
                    comments["Single-Chain_Antigens"].append(f'Could not find {c_id}CDR1 in {chain}')
                elif contact_cdr1: # contact is made at CDR2
                    comments["Single-Chain_Antigens"].append(f'Added because found contact in {c_id}CDR1 for {chain} and {c}')
                    chain_dict_new[allowed_chain_ids[ag_c_idx]] = chain_dict['Ag'][c]
                    chain_dict_new[allowed_chain_ids[ag_c_idx]]["old_id"] = c
                    ag_c_idx += 1
                    Ag_list.append(c)
                    break

    return chain_dict_new, comments


def autofill_row_ic(pdb_id, idx, chain_dict, comments, df):
    """
    Returns information about the complex in a way that it can be
     stored into a summary csv that contains details on the
     complex.

    Args:
        pdb_id (str): 4 letter PDB id of structure with its
         bioassembly number.
        idx (str): Details the VH, VL, and Ag invovled in structure.
        chain_dict (dict): Contains information about the complex and
         its labeling.
        comments (dict): Contains information about how the complex is created.
        df (pd.DataFrame): DataFrame of the processed Input Structures

    Returns:
        complex_dict (dict): Dictionary containing information in structure that
         will be used to be inputted into summary csv.
    """

    # supplying all the information needed of complex for summary csv
    parent_struct = df[df["Name"] == pdb_id]
    complex_dict = {}
    complex_dict["Name"] = f"{pdb_id}-{idx}"
    complex_dict["Parent_File"] = parent_struct["Parent_File"].values[0]
    if pdb_id[-2:].isdigit():
        complex_dict["Bioassembly"] = pdb_id[-2:]
    else:
        complex_dict["Bioassembly"] = pdb_id[-1]
    complex_dict['Structure_Title'] = parent_struct["Structure_Title"].values[0]
    complex_dict['Structure_Classification'] = parent_struct["Structure_Classification"].values[0]
    complex_dict["Resolution"] = parent_struct["Resolution"].values[0]
    complex_dict['Method'] = parent_struct["Method"].values[0]
    complex_dict['Date_Deposited'] = parent_struct["Date_Deposited"].values[0]
    complex_dict['Date_Released'] = parent_struct["Date_Released"].values[0]
    complex_dict["PDB_ID"] = parent_struct["PDB_ID"].values[0]
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
    complex_dict["Antigen_Type"] = [chain_dict[Ag]['antigen_type'] for Ag in complex_dict["Chain_Ag"]]
    complex_dict["comments"] = comments
    return complex_dict


def add_pdb_header_ic(pdb_file, input_dir, chain_dict, df):
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
    pdb_id = pdb_name.split("-")[0]
    # Reading the original file content
    with open(pdb_file, "r") as f_read:
        content = "".join(f_read.readlines()[1:])

    with open(pdb_file, "w") as f_new:
        title = df["Structure_Title"].values[0]
        classification = df["Structure_Classification"].values[0]
        resol = df["Resolution"].values[0]
        rel_date = df["Date_Released"].values[0]
        dep_date = df["Date_Deposited"].values[0]
        method = df["Method"].values[0]
        
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
        Ag_chain = sorted(set(chain_dict.keys()) - set(["H", "L"]))
        if Ag_chain:
            Ag = ", ".join(Ag_chain)
            Ag_old_id = ", ".join([chain_dict[Ag_id]["old_id"] for Ag_id in Ag_chain])
            remark.append(f"The Antigen Chain ID is {Ag}, with its old ID being {Ag_old_id}")
        remark = "; ".join(remark)
        f_new.write(f"STRUCTURE_IDENTITY  {remark}\n")
        f_new.write("REMARK 99 Modified by Bryan Munoz Rivero, Sanofi\n")
        f_new.write(content)  # Adding the original content


def pipeline_func(pdb_id, df, pdb_curated, complex_dir, pdb_utils):
    """
    Determines all the structures invovled in a processed PDB structure.
     Returns this information as a list of dictionaries, where each
     dictionary contains information about a complex.

    Args:
        pdb_id (str): 4 letter PDB id of structure with its
         bioassembly number.
        df (pd.DataFrame): Dataframe containing information on the
         processed PDB structures.
        pdb_curated (str): path to directory that houses the processed
         PDB files
        complex_dir (str): path to directory where the complexes will
         be saved to
        pdb_utils (PDBUtils): An instance of the PDBUtils class that is used
         to process and label the chains.

    Returns:
        ab_rows (list): List of dictionaries that contains all the identified
         complexes from a specified PDB structrue.
    """

    # determining if there is a VH chain in the structure
    pdb = df.loc[df["Name"] == pdb_id]
    vh_chains = str(pdb['Chain_VH'].values[0])
    Chain_VH = ast.literal_eval(vh_chains)
    if len(Chain_VH) == 0:
        return None

    # determine the antigens invovled in complex
    npy_file = f'{pdb_curated}/{pdb_id}-atom37.npy'
    chain_dict = np.load(npy_file, allow_pickle=True).item()
    ag_chains = str(pdb['Chain_Ag'].values[0])
    Ags = ast.literal_eval(ag_chains)

    # Combining VH-VL and VHH into Ab complexes list
    vh_vl_pairs = str(pdb['VH_VL_Pairs'].values[0])
    VH_VL_pairs = ast.literal_eval(vh_vl_pairs)
    chain_vhhs = str(pdb['Chain_VHH'].values[0])
    Chain_VHHs = ast.literal_eval(chain_vhhs)
    Ab_complexes = VH_VL_pairs + Chain_VHHs

    for ag in chain_dict["Ag"]:
        ab_like = any([ag in [c1[0], c2[0]] for c1, c2 in VH_VL_pairs])
        nb_like = ag in [c[0] for c in Chain_VHHs]
        if ab_like and nb_like:
            chain_dict["Ag"][ag]["antigen_type"] = "Ab-like & Nb-like Antigen"
        elif ab_like:
            chain_dict["Ag"][ag]["antigen_type"] = "Ab-like Antigen"
        elif nb_like:
            chain_dict["Ag"][ag]["antigen_type"] = "Nb-like Antigen"
        else:
            chain_dict["Ag"][ag]["antigen_type"] = "Antigen"

    # solving for all the complexes in the structure
    ab_rows = []
    for Ab in Ab_complexes:  # determing all the complexes in a structure
        # test for replicate ligand IDS
        if Ab in VH_VL_pairs:
            if chain_dict["VH"][Ab[0]]["seq"] == "" or chain_dict["VL"][Ab[1]]["seq"] == "":
                raise ValueError(f"There is an error with the npy file: {npy_file}, where the ligand chains are not properly read.")
            Ab_id = (Ab[0].split("_")[0], Ab[1].split("_")[0])
            rep_check = sum([Ab_id == (H.split("_")[0], L.split("_")[0]) for H, L in VH_VL_pairs])
        else:
            if chain_dict["VH"][Ab]["seq"] == "":
                raise ValueError(f"There is an error with the npy file: {npy_file}, where the ligand chain is not properly read.")
            Ab_id = Ab.split("_")[0]
            rep_check = sum([Ab_id == H.split("_")[0] for H in Chain_VHHs])
        rep = True if rep_check >= 2 else False

        chain_dict_copy = copy.deepcopy(chain_dict)
        row, chain_dict_new, idx = identify_complex(chain_dict_copy, Ab, Ags, pdb_id, df, rep)
        output_file = os.path.join(complex_dir, pdb_id + f'-{idx}')
        structure = dict_to_structure(chain_dict_new)
        pdb_utils.write_pdb_with_seqres_from_structure(structure[0], output_file + ".pdb")
        add_pdb_header_ic(output_file + ".pdb", pdb_curated, chain_dict_new, df[df["Name"]==pdb_id])
        np.save(output_file + "-atom37.npy", chain_dict_new)
        print("Successfully Completed:", output_file)
        ab_rows.append(row)

    return ab_rows


def main():
    """
    The main function of the second part of the pipeline. The goal of this function
    is to identify all complexes that are located in the processed PDB files, so
    that new PDB and NPY files can be created that show the newly identified complexes.
    As well, a summary csv file is created that contains information on all of the
    complexes. The csv contains information the Heavy chain, and Light chain, and
    antigen chains invovled in complex.
    """

    # define inputs
    parser = argparse.ArgumentParser(description="Identify complexes located in structure file")
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
    if os.path.exists(f"{pdb_folder}_complexes"):
        shutil.rmtree(f"{pdb_folder}_complexes")
    if os.path.exists(f"{pdb_folder}_complexes_curated.csv"):
        os.remove(f"{pdb_folder}_complexes_curated.csv")

    # define input and output folders, and output csv, and reads previous summary csv
    pdb_utils = PDBUtils()
    pdb_processed = f'{pdb_folder}_parsed'
    complex_dir = f'{pdb_folder}_complexes'
    output_csv_name = f'{pdb_folder}_complexes_curated.csv'
    os.makedirs(complex_dir, exist_ok=True)
    df = pd.read_csv(f'{pdb_folder}_parsed_file_chains.csv')

    # running process_pdb through parallel function
    pdb_names = sorted(df["Name"].values)
    cpu_parallel = ParallelProcessorForCPUBoundTasks(pipeline_func, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_names, df, pdb_processed, complex_dir, pdb_utils)

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