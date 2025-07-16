import argparse
from glob import glob
import os
import shutil
import pandas as pd


def create_summary_df(df_antigen, df_ligand, df_configuration, rep_names):
    """
    Creates a summary csv file detailing where each complex passed or failed in the
    testdata pipeline. 

    Args:
        df_antigen (list): pd.DataFrame detailing the results of the antigen step
        df_ligand (list): pd.DataFrame detailing the results of the ligand step
        df_configuration (list): pd.DataFrame detailing results of the configuration
            step
        rep_name (list): A list of all complexes that passed representative
            classification

    Returns:
        pd.DataFrame: DataFrame explaining what classification each complex passed
    """

    # DataFrame header
    summary_dict = {"Name": [], "Similar_Antigen": [], "Similar_Global_Alignment": [], "Similar_Complex": [], "Representative_Structure": [], "Closest_Structure_Match": [], "TM_Score_Closest_Match": []}

    # loop through each input DataFrame to ensure we capture where each complex passes and fails
    for name, df in [df_antigen, df_ligand, df_configuration]:
        if df is None:
            continue
        # only going through the complexes that passed first
        df_passed = df[df["Passed"] == True]
        for _, row in df_passed.iterrows():
            summary_dict["Name"].append(row["Name"])
            summary_dict["Similar_Antigen"].append(name == "antigen")
            summary_dict["Similar_Global_Alignment"].append(name == "ligand")
            summary_dict["Similar_Complex"].append(name == "configuration")
            summary_dict["Representative_Structure"].append(row["Name"] in rep_names)
            summary_dict["Closest_Structure_Match"].append(row["Closest_Match"])
            summary_dict["TM_Score_Closest_Match"].append(row["TM_Score"])

    df_config = df_configuration[1]
    df_failed = df_config[df_config["Passed"] == False]
    # going through the complexes that failed
    for _, row in df_failed.iterrows():
        summary_dict["Name"].append(row["Name"])
        summary_dict["Similar_Antigen"].append(False)
        summary_dict["Similar_Global_Alignment"].append(False)
        summary_dict["Similar_Complex"].append(False)
        summary_dict["Representative_Structure"].append(False)
        summary_dict["Closest_Structure_Match"].append(row["Closest_Match"])
        summary_dict["TM_Score_Closest_Match"].append(row["TM_Score"])

    return pd.DataFrame(summary_dict).sort_values("Name")


def main():
    """
    The main function for testdata_summary script. The main purpose of
    this script is to create a directory containing all the represenative
    deduplicated structures that passed the pipeline outside the Deduplication
    directory. Then create a summary csv file explaining which
    classification each structure passed/failed.
    """

    # Getting inputs for the script 
    parser = argparse.ArgumentParser(description="Determine the representative complexes for the testdata and create a summary csv file")
    parser.add_argument("-p", "--dir_path", help="Path to the Deduplication directory",
                       required=True)
    args = parser.parse_args()
    dir_path = args.dir_path # pass of Deduplication directory
    dir_path = dir_path if dir_path[-1] != "/" else dir_path[:-1]

    antigen_df = ("antigen", pd.read_csv(f"{dir_path}/antigen_summary.csv"))
    ligand_df = ("ligand", pd.read_csv(f"{dir_path}/ligand_summary.csv"))
    configuration_df = ("configuration", pd.read_csv(f"{dir_path}/configuration_summary.csv"))
    rep_df = pd.read_csv(f"{dir_path}/representative_summary.csv")
    rep_names = list(rep_df[rep_df["Passed"] == True]["Name"].values)
    df = create_summary_df(antigen_df, ligand_df, configuration_df, rep_names)
    df.to_csv(f"{dir_path}/../Test_Data_Summary.csv", index=False)

    # moving the representative complexes to outside the Test_Data_Analysis directory
    rep_path = f"{dir_path}/representative_passed"
    final_dir = f"{dir_path}/../Test_Data"
    if os.path.exists(final_dir):  # Deleting directory containing a past run
        shutil.rmtree(final_dir)
    os.makedirs(final_dir)
    for pdb in rep_names:
        shutil.copy(f"{rep_path}/{pdb}.pdb", f"{final_dir}/{pdb}.pdb")
    print("Script ran successfully")


if __name__ == "__main__":
    main()