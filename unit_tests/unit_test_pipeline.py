import pytest
import subprocess
import os
from glob import glob
import pandas as pd
import ast
import shutil


@pytest.fixture
def path_to_pdb_files():
    """
    Initializes the path to the test directory of PDB files
    """

    return "./unit_tests/pdb_files_test"


def test_process_PDBs(path_to_pdb_files):
    # deleting the test directories and csv if they exist
    if os.path.exists(f"{path_to_pdb_files}_parsed"):
        shutil.rmtree(f"{path_to_pdb_files}_parsed")
    if os.path.exists(f"{path_to_pdb_files}_parsed_file_chains.csv"):
        os.remove(f"{path_to_pdb_files}_parsed_file_chains.csv")
    
    # Testing process_PDBs script (that it runs properly and that outputs are created)
    result = subprocess.run(["python", "./src/snacdb/curation_process_PDBs.py", path_to_pdb_files])
    assert result.returncode == 0, f"The script for process_PDBs terminated due to an error."
    assert os.path.exists(f"{path_to_pdb_files}_parsed"), "Output Directory for process_PDBs was not properly created."
    assert os.path.exists(f"{path_to_pdb_files}_parsed_file_chains.csv"), "Output CSV File for process_PDBs was not properly created."


def test_identify_complexes(path_to_pdb_files):
    # deleting the test directories and csv if they exist
    if os.path.exists(f"{path_to_pdb_files}_complexes"):
        shutil.rmtree(f"{path_to_pdb_files}_complexes")
    if os.path.exists(f"{path_to_pdb_files}_complexes_curated.csv"):
        os.remove(f"{path_to_pdb_files}_complexes_curated.csv")

    # Testing identify_complexes script (that it runs properly and that outputs are created)
    result = subprocess.run(["python", "./src/snacdb/curation_identify_complexes.py", path_to_pdb_files])
    assert result.returncode == 0, f"The script for identify_complexes terminated due to an error."
    assert os.path.exists(f"{path_to_pdb_files}_complexes"), "Output Directory for identify_complexes was not properly created."
    assert os.path.exists(f"{path_to_pdb_files}_complexes_curated.csv"), "Output CSV File for identify_complexes was not properly created."
    
    # this final test should determine that VH, VL, and Ag chains are chain ids present in csv for specific structure/bioassembly
    df_curated = pd.read_csv(f"{path_to_pdb_files}_parsed_file_chains.csv")
    df_complex = pd.read_csv(f"{path_to_pdb_files}_complexes_curated.csv")
    for pdb_id in df_curated["Name"].values:
        curated = df_curated[df_curated["Name"] == pdb_id]
        chain_VH = set(ast.literal_eval(curated["Chain_VH"].values[0]))
        chain_VL = set(ast.literal_eval(curated["Chain_VL"].values[0]))
        chain_Ag = set(ast.literal_eval(curated["Chain_Ag"].values[0]))

        # going through each generated complex for a PDB structure
        for complex_name in df_complex[df_complex["Parent_File"] == pdb_id]["Name"].values:
            complex_i = df_complex[df_complex["Name"] == complex_name]
            VH = [complex_i["Chain_VH_old_id"].values[0]]
            assert len(set(VH) - chain_VH) == 0, f"The Chain ID of the Heavy Chain for complex {complex_i['Name'].values[0]} is not found in Heavy Chain list of {pdb_id} ({VH} not in {chain_VH})"
            VL = complex_i['Chain_VL_old_id'].values[0]
            if VL == "[]":  # accounting for VHH chains
                VL = []
            else:
                VL = [VL]
            assert len(set(VL) - chain_VL) == 0, f"The Chain ID of the Light Chain for complex {complex_i['Name'].values[0]} is not found in Light Chain list of {pdb_id} ({VL} not in {chain_VL})"
            Ag = ast.literal_eval(complex_i['Chain_Ag_old_id'].values[0])
            Ag = set([chain[0] for chain in Ag])
            assert len(Ag - chain_Ag) == 0, f"The Chain ID of the Antigen Chain(s) for complex {complex_i['Name'].values[0]} is not found in Antigen Chain list of {pdb_id} ({Ag} not in {chain_Ag})"


def test_filter_complexes(path_to_pdb_files):
    # deleting the test directories and csv if they exist
    if os.path.exists(f"{path_to_pdb_files}_filter"):
        shutil.rmtree(f"{path_to_pdb_files}_filter")
    if os.path.exists(f"{path_to_pdb_files}_outputs_multichain_filter.csv"):
        os.remove(f"{path_to_pdb_files}_outputs_multichain_filter.csv")

    # Testing filter_complexes script (that it runs properly and that outputs are created)
    result = subprocess.run(["python", "./src/snacdb/curation_filter_complexes.py", path_to_pdb_files])
    assert result.returncode == 0, f"The script for filter_complexes terminated due to an error."
    assert os.path.exists(f"{path_to_pdb_files}_filter"), "Output Directory for filter_complexes was not properly created."
    assert os.path.exists(f"{path_to_pdb_files}_outputs_multichain_filter.csv"), "Output CSV File for filter_complexes was not properly created."
    assert len(glob(f"{path_to_pdb_files}_complexes/*")) == len(glob(f"{path_to_pdb_files}_filter/*")), "One or more PDB files was not processed correctly during the run of filter_complexes."
    #this final test should determine that VH and VL chain ids should be the exact same as the previous complex and that the filtered Ags were at least in the complex from before the filter
    df_complex = pd.read_csv(f"{path_to_pdb_files}_complexes_curated.csv")
    df_filter = pd.read_csv(f"{path_to_pdb_files}_outputs_multichain_filter.csv")
    for complex_name, filter_name in zip(sorted(df_complex["Name"].values), sorted(df_filter["Name"].values)):
        assert complex_name[:6] == filter_name[:6], f"PDB ID and Bioassembly do not match between the pre-filtered complex ({complex_name[:6]}) and post-filtered complex ({filter_name[:6]})"
        
        complex_i = df_complex[df_complex["Name"] == complex_name]
        filter_i = df_filter[df_filter["Name"] == filter_name]

        assert complex_i["Chain_VH_old_id"].equals(filter_i["Chain_VH_old_id"]), f"Chain IDs of the Heavy Chain do not match for PDB structure {complex_name[:6]} ({complex_i['Chain_VH_old_id']} vs {filter_i['Chain_VH_old_id']})."
        assert complex_i["Chain_VL_old_id"].equals(filter_i["Chain_VL_old_id"]), f"Chain IDs of the Light Chain do not match for PDB structure {complex_name[:6]} ({complex_i['Chain_VL_old_id']} vs {filter_i['Chain_VL_old_id']})."
        complex_Ag = ast.literal_eval(complex_i["Chain_Ag_old_id"].values[0])
        filter_Ag = ast.literal_eval(filter_i["Chain_Ag_old_id"].values[0])
        assert len(set(filter_Ag) - set(complex_Ag)) == 0, f"Chain Ids of the Antigen Chains do not match for the for PDB structure {complex_name[:6]} ({complex_Ag} vs {filter_Ag})."