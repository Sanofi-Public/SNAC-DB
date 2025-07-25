# Data Curation Pipeline for Antibodies and Nanobodies

## Overview
This repository contains pipelines used in the Structural NANOBODY® VHH and Antibody Complex Database (SNAC-DB) project. The goal of SNAC-DB is to extract all possible antibody (Ab) and NANOBODY® VHH (Nb) complexes from structural files (e.g., PDBs or CIFs), and curate and analyze them for downstream applications.

Specifically, the repository supports:

1. **SNAC-DB Pipeline:** Extract and curate non-redundant Ab/Nb complexes from structural files.
2. **Test Dataset Pipeline:** Create a structurally novel test dataset for benchmarking docking models.
3. **Finding Hits Pipeline:** Identify structures in a SNAC-DB-curated dataset based on the structural similarity with some specified structure.

All pipelines are orchestrated via bash scripts, but underlying Python modules provide granular control for customization and debugging.

#### The curated version of the ready-to-use SNAC-DB dataset is available at: [https://zenodo.org/records/16226208](https://zenodo.org/records/16226208)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Clone the Repository](#1-clone-the-repository)
  - [Set Up the Environment](#2-set-up-the-environment)
- [Usage](#usage)
  - [SNAC-DB Pipeline](#1-snac-db-pipeline)
    - [Run via Bash](#run-via-bash)
    - [Advanced Control](#advanced-control)
  - [Test Dataset Pipeline](#2-test-dataset-pipeline)
  - [Finding Hits Pipeline](#3-finding-hits-pipeline)
- [Filling in Unresolved Residues](#filling-in-unresolved-residues)
  - [Script Purpose](#script-purpose)
  - [Usage](#usage-1)
  - [Arguments](#arguments)
  - [Output](#output)
- [Naming Convention of Complexes](#naming-convention-of-complexes)
- [Pipeline Details](#pipeline-details)
  - [SNAC-DB Pipeline](#snac-db-pipeline)
    - [Process PDBs](#process-pdbs)
    - [Identify Complexes](#identify-complexes)
    - [Filter Complexes](#filter-complexes)
    - [Remove Redundancies](#remove-redundancies)
  - [Test Dataset Pipeline](#test-dataset-pipeline)
    - [Novel Antigens](#novel-antigens)
    - [Novel Epitopes](#novel-epitopes)
    - [Novel Conformations](#novel-conformations)
    - [Unique Complexes](#unique-complexes)
  - [Finding Hits](#finding-hits)
    - [Create Proper Reference Directory](#create-proper-reference-directory)
    - [Finding Hits](#finding-hits-1)
- [Unit Testing](#unit-testing)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)
- [Citations](#citations)
---

## Repository Structure

```
.
├── src/                        # Python source code
│   ├── snacdb/
│   │    ├──utils/
│   │    │  ├── parallelize.py
│   │    │  ├── pdb_utils_clean_parse.py
│   │    │  ├── residue_constants.py
│   │    │  ├── sequence_utils.py
│   │    │  ├── structure_utils.py
│   │    ├── curation_identify_complexes.py 
│   │    ├── curation_process_PDBs.py 
│   │    ├── curation_redundant.py
│   │    ├── curation_filter_complexes.py
│   │    ├── patch.py
│   ├── analysis_fill_in_unresolved_residues.py  
│   ├── analysis_finding_hits.py  
│   ├── curation_SNAC_DB_Pipeline.py
│   ├── testdata_classificaiton.py 
│   ├── testdata_setup.py 
│   ├── testdata_summary.py 
├── unit_tests/                # Unit testing scripts
│   ├── data_curation_pipeline.py
│   ├── pdb_files_test         # Sample pdb files
├── data_curation.sh           # SNAC-DB pipeline launcher
├── test_dataset.sh            # Test dataset curation launcher
├── finding_hits.sh            # Hit-finding pipeline launcher
├── environment.yml            # Conda environment setup
├── requirements.txt           # pip requirements
├── setup.py                   # Setup file
├── pyproject.toml
├── README.md
```

---

## Getting Started

1. **Clone the Repository**

    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
    
2. **Set Up the Environment**

    - Create a conda environment:

        ```bash
        conda create -n snacdb_env python=3.10 -y
        conda activate snacdb_env
        ```

    - Install pip dependencies from PyPI:

        ```bash
        pip install -r requirements.txt
        ```

    - Install the pipeline package
        ```bash
        pip install .
        ```

    - Install ANARCI from their github repository: 
        [https://github.com/oxpig/ANARCI](https://github.com/oxpig/ANARCI)

        Please note that the original ANARCI tool does not support ambiguous residues represented by the amino acid `'X'`. To accommodate structural data containing unknown residues, we provide a simple patch to modify the installed ANARCI source to allow it to handle `'X'` residues gracefully during sequence parsing and annotation.

        Run this patch (`src/snacdb/patch.py`) using the command:
        ```bash
        snacdb-patch-anarci
        ```

---

## Usage

**General Note**: All three pipelines can be run via bash scripts. For deeper customization, you may directly execute specific Python modules.


1. **SNAC-DB Pipeline**

    This pipeline allows you to curate a folder containing `.pdb` or `.cif` structure files, and find / extract antibody and NANOBODY® VHH complexes. This is what we used to curate all the structures deposited in the Protein Data Bank (PDB) and create the SNAC-DB database.

    - Run via Bash

        ```bash
        bash data_curation.sh path/to/<input_directory>
        ```

        Optional parameters:

        - Show intermediate outputs: `True`
        - Remove redundant complexes: `True`


        Example with both options:

        ```bash
        bash data_curation.sh path/to/<input_directory> True True
        ```
        
    - Advanced Control

        You can run individual Python scripts from the `src/` directory:

        - The scripts are required to be run in sequential order. If you want to run curation_identify_complex script you must first run curation_process_PDBs
        - Logs and summaries are generated at each step


2. **Pipeline to Create Test/Benchmarking Dataset**

    This is an optional pipeline which could be used to create a structurally distinct benchmarking dataset for model evaluation. 
    
    - First install **FoldSeek**: [https://github.com/steineggerlab/foldseek](https://github.com/steineggerlab/foldseek).
    
    - Create a structurally novel test set:

        ```bash
        bash test_dataset.sh path/to/<query_directory> path/to/<target_directory>
        ```
        
        Outputs include:

        - `Test_Data/`: curated non-redundant dataset
        - `Test_Data_Summary.csv`: where each complex passed and failed along with closest matches and their TM-scores


3. **Pipeline to Find Hits**

    This is an optional pipeline which could be used to search for structurally similar complexes using **FoldSeek** multimer-search. 
    
    ```bash
    bash finding_hits.sh <reference_dir> <chain_type> [<structure_of_interest> <is_curated>]
    ```
    
    
    - Required:

        - `reference_dir`: Curated dataset (output from SNAC-DB pipeline)
        - `chain_type`: Chain type to compare (`ligand`, `antigen`, or `complex`)
    
    
    - Optional:

        - `structure_of_interest`: Structure file to query against the reference
        - `is_curated`: Set to `True` if the query is SNAC-DB formatted
    
    
    - Behavior:

        - If no structure is provided, a clustering operation is performed on the dataset.
        - If a query is provided, matching structures will be identified and ranked by TM-score.

    
    - Outputs appear in:

        ```
        <reference_dir>_<chain_type>_cluster/
        ```


        Includes match directories and summary CSV files.

---

## Filling in Unresolved Residues

In addition to the three main pipelines, this repository includes a helpful script for improving the quality of curated complexes:  
`analysis_fill_in_unresolved_residues.py`.

This script attempts to resolve unknown residues (typically marked as `'X'`) by searching the SwissProt and UniRef90 databases using MMseqs2. The goal is to reduce ambiguity in antigen chains, which can improve downstream structural and sequence-based analyses.

- **Script purpose**:

    - Scans curated complexes for unresolved residues.
    - Searches SwissProt and/or UniRef90 to find high-identity matches.
    - Fills in missing residues based on sequence alignment and similarity.
    - Provides the corrected fasta files corresponding to the complexes for which making a correction was possible.
    
    
- **Usage**:

    Install MMSEQS2:
    ```conda install bioconda::mmseqs2```

    Run:
    ```bash
    python src/analysis_fill_in_unresolved_residues.py \
      --input_dir path/to/curated_complexes \
      --swissprot /path/to/swissprot_db \
      --uniref90 /path/to/uniref90_db
    ```
    
    
- **Arguments**:
    - `--input_dir` (str, required):  
      Path to a directory containing complexes curated by the SNAC-DB pipeline.
    - `--swissprot` (str, optional):  
      Path to a SwissProt MMseqs2 database.  
      If not provided or the path doesn’t exist, it will be downloaded automatically to the parent of the input directory.  
      ⚠️ Do not set this to `None` — SwissProt is required.
    - `--uniref90` (str, optional):  
      Path to a UniRef90 MMseqs2 database.  
      If not provided or doesn't exist, it will be downloaded.  
      If explicitly set to `"None"`, UniRef90 will be skipped.

    
- **Output**:
    - `<input_name>_cleaning_complexes/`:  
      Directory with corrected sequences, intermediate results, logs, temporary databases, and alignment files.

Not all unresolved residues will be filled—only those that match sufficiently to known entries. To track processing steps and potential issues, we recommend logging the output:

```bash
python src/analysis_fill_in_unresolved_residues.py \
  --input_dir path/to/my_dir > path/to/my_dir_cleaning_complexes/fill_in_unresolved_residues.log 2>&1
```

**Note:** The user can utilize fasta files with the corrected sequences to update the PDB files corresponding to those complxes. Given the intricate nature of these corrections, we do not automatically create the updated PDB files.

---

## Naming Convention of Complexes

Each PDB and NPY file follows a specific naming convention. 

1. **Base Name:** The identification name of the structure. Can be 4 letter PDB ID from the RCSB database or some unique name.
   - Ex: 8FSL in **8FSL**-ASU1-VHH_B-Ag_C
2. **Bioassembly Number:** The bioassembly number of the structure. If no bioassembly number is provided in the original structure then a default value of 0 is given. A value of 0 usually refers to the asymmetric unit cell.
   - Ex: 1 in 7zmr-**ASU1**-VHH_K-Ag_A
3. **Frame (Optional):** When structures are collected through experimental techniques such as NMR, the experimentalist can capture multiple frames of the structure that they can input into the structure file. Since there are small changes in these frames, the pipeline preserves them as different complexes.
   - Ex: 0 in 7y7m-ASU1-**frame0**-VH_F-VL_G-Ag_C
4. **Structure Information:** This section describes the type of structure and the chains associated with it. The type of potential structures are antibodies (VH_VL) and nanobodies (VHH).
   - Ex: VHH_G-Ag_B_C in 8k46-ASU0-**VHH_G-Ag_B_C**
5. **Replicate (Optional):** This occurs when multiple ligand chains have the same chain ID in the structure file. In this case, to avoid overwriting a complex in case they bind to the same antigen the replicate keyword is added to ensure the naming scheme is unique. 
   - Ex: 0 in 6ul6-ASU0-VHH_B-Ag_A-**replicate0**

As mentioned above, the keywords Frame and Replicate are optional keywords in the naming scheme, and only apply in the cases specified above. It is important to note that unlike the other keywords, the replicate keyword  is not gaurenteed to have the same integer value as part of the replicate keyword if you rerun the pipeline.

---

## Pipeline Details

### SNAC-DB Pipeline

1. **Process PDBs**

    Aim: Clean input structures, fill in missing residues, and identify VH/VL/Ag chains.

    Output: 
    - Cleaned PDB files
    - `.npy` files with annotations
    - `<input_dir>__parsed_file_chains.csv`
    

2. **Identify Complexes**

    Aim: Extracts antibody and NANOBODY® VHH complexes from processed structures.

    Output:
    - PDB files of isolated complexes
    - `.npy` annotation files
    - `<input_dir>_complexes_curated.csv`
    

3. **Filter Complexes**

    Aim: Applies more stringent filtering to exclude non-interacting chains.

    Output:
    - Filtered complex structures
    - Summary CSV: `<input_dir>_outputs_multichain_filter.csv`


4. **Remove Redundancies**

    Aim: Eliminates duplicate complexes based on contact map comparison.

    Output:
    - Updated complex directory and summary CSV

    
### Pipeline to Create Test/Benchmarking Dataset

1. **Novel Antigens**

    Aim: Identify complexes with structurally dissimilar antigens.

    Output: 
    - Passed and failed directories based on TM-score


2. **Novel Epitopes**

    Aim: Compare full complex structures to detect epitope novelty.

    Output:
    - Passed and failed directories based on TM-score


3. **Novel Conformations**

    Aim: Detect small but meaningful differences in binding conformation.

    Output:
    - Passed and failed directories based on multi-chain TM-score


4. **Unique Complexes**

    Aim: Ensure dataset is structurally diverse internally.

    Output:
    - Final dataset
    - `Test_Data_Summary.csv`


### Pipeline to Find Hits

1. **Create Proper Reference Directory**

    Aim: Extract chains (ligand/antigen) from curated complexes for clustering or comparison, unless the analysis is being done using the complex input (means looking at all the chains in a complex).

    Output: 
    - Subset directories with only relevant chains


2. **Finding Hits**

    Aim: Cluster structures or compare to a structure of interest using FoldSeek.

    Output: 
    - Clustering summary or match summary (if no structure of interest is specified)
    - Match files and summaries per structure of interest (if specified)

---

## Unit Testing

Run the SNAC-DB pipeline unit test:

```bash
pytest unit_tests/data_curation_pipeline.py
```

This checks:

- Expected outputs are created
- Naming conventions are preserved

---

## Dependencies

- Python 3.10.16 (PSF License)
- Biopython (BSD 3-Clause License)
- FoldSeek (GNU General Public License ver. 3 (GPLv3))
- ANARCI (BSD 3-Clause License)
- SciPy (BSD 3-Clause License)
- TQDM (Mozilla Public License (MPL) v. 2.0)
- Pytest (MIT License)
- networkx (BSD 3-Clause License)
- Pandas (BSD 3-Clause License)
- MMSEQS2 (MIT License)

---

## Acknowledgements

We would like to acknowledge the developers of the following tools and libraries that are integral to the functionality of this pipeline:

- **[FoldSeek](https://github.com/steineggerlab/foldseek):** Used for fast and sensitive structural alignment and clustering in the hit-finding and test dataset pipelines. FoldSeek enables high-throughput identification of structural matches, which is essential for both redundancy reduction and benchmarking tasks.
    - **Reference:** van Kempen, M., Kim, S.S., Tumescheit, C., Mirdita, M., Lee, J., Gilchrist, C.L.M., Söding, J., and Steinegger, M. Fast and accurate protein structure search with Foldseek. Nature Biotechnology, doi:10.1038/s41587-023-01773-0 (2023)
    - **License:** GNU General Public License ver. 3 (GPLv3)

- **[ANARCI](https://github.com/oxpig/ANARCI):** Used for antibody and NANOBODY® VHH sequence annotation and CDR identification.
    - **Reference:** Dunbar, J., & Deane, C. (2015). ANARCI: antigen receptor numbering and receptor classification. Bioinformatics, 32(2), 298–300.
    - **License:** BSD 3-Clause License
    
- **[MMSEQS2](https://github.com/soedinglab/MMseqs2):** Used for fast and sensitive many-against-Many sequence searching and clustering. MMSEQS2 enables us to fill missing residues based on hits against UniRef and SwissProt databases.
    - **Reference:** Steinegger, M. and Söding, J., (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature biotechnology, 35(11), pp.1026-1028.
    - **License:** MIT License

---

## Citations
This current work has been accepted as part of the workshop [Data World](https://icml.cc/virtual/2025/workshop/39966) in the International Machine Learning Conference (ICML). There is future plans to submit this work for publication as well. For now please use this citation when referencing our work:

```bibtex
@inproceedings{
  gupta2025snacdb,
  title={{SNAC}-{DB}: The Hitchhiker{\textquoteright}s Guide to Building Better Predictive Models of Antibody \& {NANOBODY}{\textregistered} {VHH}{\textendash}Antigen Complexes},
  author={Abhinav Gupta and Bryan Munoz Rivero and Jorge Roel-Touris and Ruijiang Li and Norbert Furtmann and Yves Fomekong Nanfack and Maria Wendt and Yu Qiu},
  booktitle={DataWorld: Unifying Data Curation Frameworks Across Domains, Workshop at the 42nd International Conference on Machine Learning (ICML 2025)},
  year={2025},
  address={Vancouver, Canada},
  url={https://openreview.net/forum?id=68DcIpDaHK}
}
```
