#!/bin/bash

# Pipline explanation:
# This pipeline details the curation process for the SNAC-DB Pipeline
# Purpose of the pipeline is to extract all possible Ab/Nb complexes from a structure file
# Only required input is the path to the structure files
# There are two optional inputs: whether to show each step of the curation process and whether to get rid of redundant complexes
# default options are that the pipeline only outputs the final curated complexes and removes all redundant complexes

# Check if the input directory is provided as an argument
if [ -z "$1" ]; then
  echo "Need to include an input structure directory as an arguement. Ex: bash $0 /path/to/dir/input_directory"
  exit 1
fi

# optional inputs to specify how to run the pipeline
keep_all_data=False # controls if outputs are produced at each step of pipeline
if [ ! -z $2 ]; then
    keep_all_data=$2
fi
redundant=False # controls whether the redundant data is eliminated from final dataset
if [ ! -z $3 ]; then
    redundant=$3
fi


# Defining the path to the src scripts and path to directory of interest
input_dir=$1
parent_dir=$(dirname "$input_dir")
input_dir_name=$(basename "$input_dir")
logs="$parent_dir/${input_dir_name}_logs"
pipeline_dir=$(dirname "$(realpath "$0")")

# creating a log file
rm -rf "$logs"
mkdir "$logs"

echo "Starting Data Curation Pipeline"

if [ $keep_all_data = "True" ]; then  # runs the pipeline as three separate scripts

    # Script will process and annotate input structure files
    python $pipeline_dir/src/snacdb/curation_process_PDBs.py $input_dir > "$logs/process_PDBs.log" 2>&1
    # reporting whether this section ran successfully
    exit_code_process_pdb=$?
    if [ $exit_code_process_pdb -eq 0 ]; then
        echo "Finished Processing All Input PDB/CIF Files. Job completed successfully: True"
    else
        echo "Finished Processing All Input PDB/CIF Files. Job completed successfully: False (Exit Code: $exit_code_process_pdb)"
    fi
    
    # Script will identify complexes and use a loose interaction criteria to solve for antigen chains in complex
    python $pipeline_dir/src/snacdb/curation_identify_complexes.py $input_dir > "$logs/identify_complexes.log" 2>&1
    # reporting whether this section ran successfully
    exit_code_ic=$?
    if [ $exit_code_ic -eq 0 ]; then
        echo "Finished Identifying All Complexes From Processed PDB Files. Job completed successfully: True"
    else
        echo "Finished Identifying All Complexes From Processed PDB Files. Job completed successfully: False (Exit Code: $exit_code_ic)"
    fi
    
    # Script will filter out antigen chains not invovled in the complex and create new structure files
    python $pipeline_dir/src/snacdb/curation_filter_complexes.py $input_dir > "$logs/filter_complexes.log" 2>&1
    # reporting whether this section ran successfully
    exit_code_fc=$?
    if [ $exit_code_fc -eq 0 ]; then
        echo "Finished Filtering Out Chains in Identified Complexes. Job completed successfully: True"
    else
        echo "Finished Filtering Out Chains in Identified Complexes. Job completed successfully: False (Exit Code: $exit_code_fc)"
    fi

    if [ $redundant = "False" ]; then
        # the nonredundant data is saved to a new directory so the redundant complexes are still saved in the *_filter directory
        python $pipeline_dir/src/snacdb/curation_redundant.py -d "${input_dir}_filter" -c "${input_dir}_outputs_multichain_filter.csv" -m True > "$logs/eliminate_redundant_complexes.log" 2>&1
        # reporting whether this section ran successfully
        exit_code_redundant_1=$?
        if [ $exit_code_redundant_1 -eq 0 ]; then
            echo "Finished Eliminating Redundant Complexes. Job completed successfully: True"
        else
            echo "Finished Eliminating Redundant Complexes. Job completed successfully: False (Exit Code: $exit_code_redundant_1)"
        fi
    fi

else 
    # default option, running pipeline using one script and showing final curated complexes
    python $pipeline_dir/src/curation_SNAC_DB_Pipeline.py $input_dir > "$logs/data_curation.log" 2>&1
    exit_code_pipeline=$?
    # reporting whether this section ran successfully
    if [ $exit_code_pipeline -eq 0 ]; then
        echo "Finished Performing the SNAC-DB Pipeline. Job completed successfully: True"
    else
        echo "Finished Performing the SNAC-DB Pipeline. Job completed successfully: False (Exit Code: $exit_code_pipeline)"
    fi

    if [ $redundant = "False" ]; then  # check to see if redundant data should be kept
        # is specified that redundant data is not saved
        python $pipeline_dir/src/snacdb/curation_redundant.py -d "${input_dir}_curated" -c "${input_dir}_curation_summary.csv" > "$logs/eliminate_redundant_complexes.log" 2>&1

        # reporting whether this section ran successfully
        exit_code_redundant_2=$?
        if [ $exit_code_redundant_2 -eq 0 ]; then
            echo "Finished Eliminating Redundant Complexes. Job completed successfully: True"
        else
            echo "Finished Eliminating Redundant Complexes. Job completed successfully: False (Exit Code: $exit_code_redundant_2)"
        fi
    fi

fi