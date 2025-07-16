import shutil
from glob import glob
import numpy as np
import pandas as pd
import os
import sys
import argparse
import numpy as np
import scipy
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


def get_contact_map_atom37(coord_chain1, coord_chain2, cutoff=5.0, alpha_carbon=True):
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
    # ranges = [(1, 26), (27, 38), (39, 55), (56, 65), (66, 104), (105, 117), (118, 128)]
    #cdrs = Chain(range(27, 38), range(56, 65), range(105, 117))
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


def remove_redudandant_data(items, path):
    """
    The purpose of this function is to do a contact map between the
     structures in the struct_list to see if there are redundant
     structures. This means that the contact map is the exact same
     between two complexes, so we remove it.

    Args:
        items (tuple): Contains the pdb id of the complexes and a list of
         the complexes in that pdb.
        path (str): The path to where these structure files are located.

    Returns:
        (str): A string detailing that this specific pdb passed.
    """
    
    pdb, struct_list = items
    removed_list = []
    
    # We want to compare all of the structures against each other
    for struct_1 in struct_list.copy():
        if struct_1 in removed_list:  # skip structure if it is redundant
            continue
        # loading the structure from the atom37 position stored in the npy file
        dict_1 = np.load(f"{path}/{struct_1}-atom37.npy", allow_pickle=True).item()
        coor_1 = np.empty((0,37, 3))
        for val in dict_1.values():
            for k, v in val.items():
                if k == "atom37":
                    coor_1 = np.append(coor_1, v, axis=0)

        # looping through to compare the structures against struct_1
        for struct_2 in struct_list.copy():
            # skip already redundant complexes
            if struct_2 in removed_list:
                continue
            if struct_1 != struct_2:  # avoid comparing a complex to itself
                # load coor_2 using the atom37 positions in the npy file
                dict_2 = np.load(f"{path}/{struct_2}-atom37.npy", allow_pickle=True).item()
                coor_2 = np.empty((0,37, 3))
                for val in dict_2.values():
                    for k, v in val.items():
                        if k == "atom37":
                            coor_2 = np.append(coor_2, v, axis=0)

                # now it is time to do contact map
                contact_map = get_contact_map_atom37(coor_1, coor_2, cutoff=6.0)
                contact_map_sum = np.nansum(contact_map)

                # test to see if the contact map is identical
                if contact_map_sum == 0:  # remove redundant complex
                    os.remove(f"{path}/{struct_2}.pdb")
                    os.remove(f"{path}/{struct_2}-atom37.npy")
                    removed_list.append(struct_2)
    return f"Completed Structural Comparison for {pdb}"


def main():
    """
    This is the main function for curation_redundant.py. The purpose of this script is
    to delete identical complexes for complexes that share the same base name.
    This refers to complexes found from asymmetric unit cells and bioassembly files that
    detail the same complex. The purpose of this is to eliminate redundant information
    since the complex is counted twice. 
    """
    
    # define inputs
    parser = argparse.ArgumentParser(description="Process and clean PDB/CIF Structures")
    # Add arguments
    parser.add_argument("-d", "--input_dir", help="Path to the input PDB or CIF directory.")
    parser.add_argument("-c", "--input_csv", help="Path to the input CSV.")
    parser.add_argument("-m", "--make_new_dir", help="Boolean value to make a new directory or not.", default=False, type=bool)

    # ensure the inputs have proper paths
    args = parser.parse_args()
    pdb_folder = args.input_dir
    if pdb_folder[-1] == "/":
        pdb_folder = pdb_folder[:-1]
    pdb_folder = pdb_folder if "/" in pdb_folder else "./" + pdb_folder
    csv_file = args.input_csv
    csv_file = csv_file if "/" in csv_file else "./" + csv_file

    original = "_".join(pdb_folder.split("_")[:-1])  # the base name of analysis
    if args.make_new_dir:  # if a new directory should be made to hold the new structures
        # if so then create the _curated directory and copy over all the complexes over there
        new_folder = f"{original}_curated"
        if os.path.exists(new_folder):  # if _curated already exists, gets rid of it
            shutil.rmtree(new_folder)
        os.makedirs(new_folder, exist_ok=True)
        # copying over the files to the new directory
        pdb_files = sorted(glob(os.path.join(pdb_folder, "*")))
        for file in pdb_files:
            struct = os.path.basename(file)
            shutil.copy(file, f"{new_folder}/{struct}")
        # removing the _curation_summary.csv if it already exists
        new_csv = f"{original}_curation_summary.csv"
        if os.path.exists(new_csv):
            os.remove(new_csv)
    else:  # do not create a new folder
        new_folder = pdb_folder
        new_csv = csv_file

    # master dict will place the complexes in groups by their pdb id
    pdb_files = sorted(glob(os.path.join(new_folder, "*.pdb")))
    master_dict = {}
    for file in pdb_files:
        struct = file.split("/")[-1][:-4]
        pdb = struct.split("-ASU")[0]
        if pdb in master_dict:
            master_dict[pdb].append(struct)
        else:
            master_dict[pdb] = [struct]

    # parallelize the script since order doesn't matter
    cpu_parallel = ParallelProcessorForCPUBoundTasks(remove_redudandant_data, max_workers=48)
    processed_rows = cpu_parallel.process(master_dict.items(), new_folder)
    print(processed_rows)

    # getting the complexes that are left so we know what to filter out from csv
    leftover_files = [os.path.splitext(os.path.basename(struct))[0] for struct in glob(os.path.join(new_folder, "*")) if ".pdb" in struct]
    processed_df = pd.read_csv(csv_file)
    processed_df = processed_df[processed_df["Name"].isin(leftover_files)]
    processed_df = processed_df.sort_values(by=["Name", "Bioassembly"])
    processed_df.to_csv(new_csv, index=False)
    processed_df.info()

if __name__ == "__main__":
    main()