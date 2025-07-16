import argparse
import os
import shutil
from glob import glob
import pandas as pd


def summarize_cluster(results, directory):
    """
    The purpose of this function is to read the outputs of a foldseek easy-multimer cluster
    and organize it in a way that is easy to read and utilize. Complexes are moved into
    directories named after their representative cluster and summary files detailing
    each cluster are made.

    Args:
        results (str): The path to the foldseek output file
        directory (str): The path to the target structures
    """
    
    df = pd.read_csv(results, sep="\t", header=None, names=["Representative_Complex", "Complexes"])
    df.info()
    grouped_df = df.groupby("Representative_Complex")["Complexes"].apply(lambda x: ','.join(map(str, x))).reset_index()
    grouped_df["Size_of_Cluster"] = grouped_df["Complexes"].apply(lambda x: len(x.split(",")))
    grouped_df.info()
    
    # make directory to house the analysis
    path = "/".join(results.split("/")[:-1]) + f"/{directory}_cluster"
    os.makedirs(path, exist_ok=True)
    grouped_df.to_csv(f"{path}/Summary_Cluster.csv", index=False)


def search(results, cutoff, directory, struct_path, curated):
    """
    The purpose of the function was to read through the output of the foldseek
    multimersearch function to find hits for the query structures. The function
    makes sure to check to see if the query structure is curated and choose a
    representative structure if the query structures are the same as the
    target structures.

    Args:
        results (str): The path to the foldseek output file
        cutoff (float): The cutoff value for structural similarity of TM score
        directory (str): The path to the target structures
        struct_path (str): The path to the query structures
        curated (bool): Whether the query structures are curated
    """

    path = "/".join(results.split("/")[:-1]) + f"/{directory}_cluster"
    name = (results.split("/")[-1]).split("_report")[0] # output file name

    # reading through output file to find hits
    with open(results, "r") as file:
        storage = {}
        for line in file:
            if "checkpoint" in line:
                continue
            info = line.split()
            struct_inter, struct_comp = info[:2]
            query_chain = info[2].split(',')
            target_chain = info[3].split(',')
            # comparison is between same structure
            if struct_inter == struct_comp:
                continue

            if curated:
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

            tm = max([float(i) for i in info[4:6]])
            if tm < cutoff:
                continue

            # adds to the structure of interest
            if struct_inter in storage:
                storage[struct_inter].append((struct_comp, tm))
            else:
                storage[struct_inter] = [(struct_comp, tm)]

    if len(storage.keys()) == 0:  # no hits were found
        print("No hits were found during this analysis")
    else:  # create the directories and summary csv file for each hit 
        find_hits(name, storage, path, directory, struct_path)


def sort_struct(unsorted_list, struct_family):
    """
    Determines which target structure has the highest match to the query
    structure. First it checks if they come from the same root structure
    file (determine based off of the naming of the structure file). Then
    it compares the TM score.

    Args:
        unsorted_list (list): The unsorted list of target structures
        struct_family (str): The name of the parent structure file

    Returns:
        next_to_add (str): Name of target structure file that has the highest
         match to query structure
    """

    next_to_add = ("", 0)
    for comp_struct, tm in unsorted_list:
        # check if comp structure is in struct family
        if struct_family in comp_struct and struct_family in next_to_add[0]:
            if tm > next_to_add[1]:
                next_to_add = (comp_struct, tm)
        # max value is in struct family but comp struct isn't
        elif struct_family in next_to_add[0]:
            continue
        else:
            # compares based on tm value
            if tm > next_to_add[1]:
                next_to_add = (comp_struct, tm)
    return next_to_add


def find_hits(name, storage, path, directory, struct_path):
    """
    Create directories and summary csv files for representative structures and their hits,
    which are located in a dictionary. The directories will house both the representative
    structure and its hits from some target directory. The summary csv file will have the
    hits organized in a hierarichal fashion that describes which matches the representative
    the best.

    Args:
        name (str): Name of the foldseek output file
        storage (dict): A dictionary housing the representative structures thats values are
         the hits from some target directory
        path (str): The path of the directory to house all of the output directories and
         summary csv files
        directory (str): The path to the directory holding target structures
        struct_path (str): The path to the directory or file which contains the structure
         of interest
    """

    # create primary data directory
    os.makedirs(path, exist_ok=True)
    print("Creating new directory:", path)

    # determining both the correct path and name of structure of interest
    struct_path = directory if struct_path is None else struct_path
    for struct in storage:
        comparison_structures = storage[struct]
        potenial_struct = glob(f"{struct_path}/{struct}.*")
        if len(potenial_struct) != 1:
            raise ValueError(f"You have input files with similar names. It is difficult for the code to verifiy which complex is which: {potential_struct}")
        struct_name, struct_ext = os.path.splitext(potenial_struct[0])
        if not os.path.exists(f"{struct_path}/{struct_name}{struct_ext}"):
            potential_struct = glob(f"{struct_path}/{struct}*")
            if len(potential_struct) != 1:
                raise ValueError(f"You have input files with similar names. It is difficult for the code to verifiy which complex is which: {potential_struct}")
            struct = (potential_struct[0]).split("/")[-1]
        new_path = f"{path}/{struct.split('.')[0]}_cluster_for_{name}"

        # creating the structure directory to house the hits
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        os.makedirs(new_path)
        print("Creating new directory:", new_path)
        shutil.copy(f"{struct_path}/{struct}", f"{new_path}/{struct}")

        # create hit directory and move the comp structures (and create a list to make the summary csv)
        comp_struct_list = []
        for comp_struct, tm in comparison_structures:
            if not os.path.exists(f"{directory}/{comp_struct}.pdb"):
                comp_struct = (glob(f"{directory}/{comp_struct}*.pdb")[0]).split("/")[-1][:-4]
            # assuming the comp structures are all curated so they are pdb files
            shutil.copy(f"{directory}/{comp_struct}.pdb", f"{new_path}/{comp_struct}.pdb")
            comp_struct_list.append((comp_struct, tm))

        # goal is to organized comparison structures and write them into summary csv
        struct_family = "-".join((struct.split("-VH")[0]).split("-")[:-1])
        with open(f"{new_path}.csv", "w") as file:
            before_sort = comp_struct_list
            final_sort = []
            while len(before_sort) != 0:
                # need to order complexes in how we want it be structured
                next_to_add = sort_struct(before_sort, struct_family)
                final_sort.append(next_to_add)
                before_sort.remove(next_to_add)

            for comp_struct, tm in final_sort:
                file.write(f"{comp_struct},{tm}\n")


def main():
    """
    The main function for the analysis_finding_hits script. The purpose of this script is to read an output file from
    foldseek easy-multimersearch to find hits for the query structures. Directories are created to contain the
    structure files that have hits and a summary csv file is created to show the similarity score between the
    structures. 
    """

    # read the input arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-r", "--results", help="Path to results file", required=True)
    parser.add_argument("-c", "--cutoff", help="Cutoff for structural similarity", type=float, default=0.90, required=False)
    parser.add_argument("-d", "--directory", help="Directory containing comparison structure files", required=True)
    parser.add_argument("-s", "--struct", help="Query Structure", required=False)
    parser.add_argument("-e", "--curated", help="Is Query Structure curated (edited)", default=True, type=bool, required=False)
    args = parser.parse_args()

    # checks that inputs are written in the appropriate manner
    results = args.results if "/" in args.results else "./" + args.results
    cutoff = args.cutoff
    directory = args.directory if "/" in args.directory else "./" + args.directory
    curated = args.curated

    print("Inputs of the script:", results, cutoff, directory, curated)

    if "_cluster.tsv" in results:
        print("Performing Cluster")
        summarize_cluster(results, directory)
    else:
        print("Preforming Search")
        if os.path.splitext(args.struct)[1] == "":
            struct_path = args.struct if "/" in args.struct else "."
        else:
            struct_path = "/".join(args.struct.split("/")[:-1]) if "/" in args.struct else "."
        # create the directories and summary csv files for all the hits
        search(results, cutoff, directory, struct_path, curated)


if __name__ == "__main__":
    main()