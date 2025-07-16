import argparse
from glob import glob
import shutil
import os
import pandas as pd


def ensure_existence(path, struct):
    """
    Checks to see if a struct exists at a specified path. If
    not then check for structure with a similar name because
    foldseek can cutoff bits of a complexes name.

    Args:
        path (str): Path that structure is located
        struct (str): Name of structure

    Returns:
        struct (str): Proper name of the structure file
        """
    # check if structure name exists
    if not os.path.exists(f"{path}/{struct}.pdb"):
        # if not use glob to find structure file with name consisting of name snippet
        struct_path, ext = os.path.splitext(f"{path}/{struct}")
        struct_path, _ = os.path.splitext(glob(f"{struct_path}*")[0])
        struct = struct_path.split("/")[-1]
    return struct


def single_chain_classification(results, pdb_files, cutoff):
    """
    Reading through the output file from a foldseek analysis on
    single chain comparison. Complexes are considering failing
    when they have a higher template model score higher than
    the cutoff value, or if they are not found in the comparison
    result file.

    Args:
        results (str): Name of output file from Foldseek
        pdb_files (list): List of inputted pdb structures 
        cutoff (float): Cutoff value to determine structural similarity

    Returns:
        list: List of PDB files that failed the classification test
    """

    summary_dict = {}
    with open(results, 'r') as file:
        for line in file:
            # foldseek sometimes will double count structures and give this suffix
            if "checkpoint" in line:
                continue

            info = line.split()
            struct_inter, struct_comp = info[:2]
            if struct_inter == struct_comp:  # comparison is between same structure
                continue
            
            tm = max([float(i) for i in info[2:]])

            # writing in information about comparison if this a closest match
            if struct_inter in summary_dict:
                if tm == summary_dict[struct_inter][1]:
                    summary_dict[struct_inter][0].append(struct_comp)
                elif tm > summary_dict[struct_inter][1]:
                    summary_dict[struct_inter] = [[struct_comp], tm, tm < cutoff]
            else:
                summary_dict[struct_inter] = [[struct_comp], tm, tm < cutoff]

    # a list of the complexes that were not found in the comparison
    for lf in list(set(pdb_files) - set(summary_dict.keys())):
        summary_dict[lf] = [[], 0, False]

    return create_summary_df(summary_dict)


def multimer_classification(results, pdb_files, cutoff, query_path, target_path):
    """
    Reading through the output file from a foldseek analysis on
    multiple chain comparison. Complexes are considering failing
    when they have a higher template model score higher than
    the cutoff value.

    Args:
        results (str): Name of output file from Foldseek
        pdb_files (list): List of inputted pdb structures 
        cutoff (float): Cutoff value to determine structural similarity

    Returns:
        pdb_files_failed (list): List of PDB files that failed the
         classification test
    """

    summary_dict = {}
    already_compared = []
    with open(results, 'r') as file:
        for line in file:
            # foldseek sometimes will double count structures and give this suffix
            if "checkpoint" in line:
                continue

            # ensuring that the chain comparison is including all of the chains of the smallest structure
            info = line.split()
            struct_inter, struct_comp = info[:2]
            # comparison is between same structure
            if struct_inter == struct_comp:
                continue

            # if the comparison has already been made
            if struct_comp in already_compared:
                continue

            num_chains_q = 1 if "VHH" in struct_inter else 2
            num_chains_t = 1 if "VHH" in struct_comp else 2
            if "Ag" in struct_inter:
                num_chains_q += len((struct_inter.split("Ag_")[-1]).split("_"))
            if "Ag" in struct_comp:
                num_chains_t += len((struct_comp.split("Ag_")[-1]).split("_"))
            shorter = min([num_chains_q, num_chains_t])
            query_chain = info[2].split(',')
            target_chain = info[3].split(',')

            # skip comparison if not all the chains are properly being compared
            if not (len(query_chain) >= shorter) or not (len(target_chain) >= shorter):
                continue

            # testing that VH and VLs are compared to each other
            flag = False
            for q, t in zip(query_chain, target_chain):
                if q == "H" and t != "H":
                    flag = True
                    break
                elif q == "L" and t != "L":
                    flag = True
                    break
            if flag:
                continue

            # writing in information about comparison if this a closest match
            tm = max([float(i) for i in info[4:6]])
            if tm >= cutoff:
                already_compared.append(struct_inter)

            struct_inter = ensure_existence(query_path, struct_inter)
            struct_comp = ensure_existence(target_path, struct_comp)
            if struct_inter in summary_dict:
                if tm == summary_dict[struct_inter][1]:
                    summary_dict[struct_inter][0].append(struct_comp)
                elif tm > summary_dict[struct_inter][1]:
                    summary_dict[struct_inter] = [[struct_comp], tm, tm < cutoff]
            else:
                summary_dict[struct_inter] = [[struct_comp], tm, tm < cutoff]

    return create_summary_df(summary_dict)


def create_summary_df(input_dict):
    """
    Create a summary DataFrame detailing which complexes passed and failed,
    as well as the closest match for each complex and the TM score of that
    match.

    Args:
        input_dict (dict): Dictionary that contains all the structural comparisons

    Returns:
        (pd.DataFrame): DataFrame detailing the closest matches for each query
            complex and if it passed or not
    """
    summary_dict = {"Name": [], "Passed": [], "Closest_Match": [], "TM_Score": []}
    for key, val in input_dict.items():
        match, tm, passed = val
        summary_dict["Name"].append(key)
        summary_dict["Passed"].append(passed)
        summary_dict["Closest_Match"].append(match)
        summary_dict["TM_Score"].append(tm)

    return pd.DataFrame(summary_dict).sort_values("Name")


def main():
    """
    The main function for Classification.py. The purpose of this script
    is to read an output file from Foldseek and determine what complexes
    are structurally similar, and which are not. Then proceed to move
    those structure file to directories labelled either passed or
    failed.
    """

    # Getting inputs for the script
    parser = argparse.ArgumentParser(description="Analyze comparisons from Foldseek to determine what structures are dissimilar")
    parser.add_argument("-q", "--query_dir", help="Path to the directory holding the query structures",
                        required=True)
    parser.add_argument("-t", "--target_dir", help="Path to the directory holding the query structures",
                        required=False)
    parser.add_argument("-r", "--results", help="Path to the result file from foldseek",
                        required=True)
    parser.add_argument("-k", "--keyword", help="Keyword used to recognize this analysis",
                        required=True)
    parser.add_argument("-c", "--cutoff", help="Cutoff value to determine dissimilarity in TM score",
                        required=False, default=0.85, type=float)
    args = parser.parse_args()

    query_dir = args.query_dir  # path of the query directory
    query_dir = query_dir if query_dir[-1] != "/" else query_dir[:-1]
    if args.target_dir is None:
        target_dir = query_dir
    else:
        target_dir = args.target_dir if args.target_dir[-1] != "/" else args.target_dir[:-1]
    dir_path = "/".join(query_dir.split("/")[:-1])  # path to Test_Data_Analysis directory
    dir_path = dir_path if dir_path != "" else "."
    results = args.results
    keyword = args.keyword
    cutoff = args.cutoff
    pdb_files_orig = glob(f"{query_dir}/*.pdb")  # getting the paths to the original query files
    pdb_files = [pdb.split("/")[-1][:-4] for pdb in pdb_files_orig]  # obtaining just the pdb names

    # determing if the foldseek analysis was done on a single or multiple chain analysis
    if "report" in results:
        print('Doing Analysis on Multi-Chain Comparison')
        df = multimer_classification(results, pdb_files, cutoff, query_dir, target_dir)
    else:
        print('Doing Analysis on Single-Chain Comparison')
        df = single_chain_classification(results, pdb_files, cutoff)

    # save DataFrame and define a list of the complexes that passed
    df.to_csv(f"{dir_path}/{keyword}_summary.csv", index=False)
    pdb_passed = list(df[df["Passed"] == True]["Name"].values)

    # delete past directories if they exist and make new ones
    if os.path.exists(f"{dir_path}/{keyword}_passed"):
        shutil.rmtree(f"{dir_path}/{keyword}_passed")
    if os.path.exists(f"{dir_path}/{keyword}_failed"):
        shutil.rmtree(f"{dir_path}/{keyword}_failed")
    os.makedirs(f"{dir_path}/{keyword}_passed", exist_ok=True)
    os.makedirs(f"{dir_path}/{keyword}_failed", exist_ok=True)

    # now move the pdb files where they need to go
    for pdb in pdb_files_orig:
        pdb_name = pdb.split("/")[-1][:-4]
        if pdb_name in pdb_passed:
            print(pdb_name, "has passed")
            shutil.copy(pdb, f"{dir_path}/{keyword}_passed/{pdb_name}.pdb")
        else:
            print(pdb_name, "has failed")
            shutil.copy(pdb, f"{dir_path}/{keyword}_failed/{pdb_name}.pdb")


if __name__ == "__main__":
    main()