# Pipline explanation:
# The purpose of this script is to search through a provided directory of structure files given and cluster hits based on factors such as antigen, ligand, or complex chains of the structure files
# Can also put an optional input of a structure file of interest which will cause the script to search for hits based on the structure file specified based on the facts of the antigen, ligand, or complex chains

# Checking if the required inputs are given
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Need to include the comparison directory, the analysis type (antigen, ligand, or complex), and the file of interest (optional) Ex: bash $0 /path/comparison_dir antigen test.pdb False"
    exit 1
fi

directory=$1  # reference directory to look for hits
comp_type=$2  # type of chains to compare
analyze_file=$3  # structures of interest
analyze_curated="True"  # if structure of interest is curated
if [ ! -z $4 ]; then
    analyze_curated=$4
fi

# Getting the paths for the provided directory and path of this script
dir_path=$(dirname $directory)
pipeline_dir=$(dirname "$(realpath "$0")")  # bash to python script

# creating the name of the summary directory that describes the analysis done
if [ -z $analyze_file ]; then
    name="$(basename $directory)_${comp_type}_all"
elif [ -d $analyze_file ]; then
    name="$(basename $directory)_${comp_type}_$(basename ${analyze_file})"
    analyze_name="$(basename ${analyze_file})_${comp_type}"
else
    name="$(basename $directory)_${comp_type}_$(basename ${analyze_file%.*})"
    analyze_name="$(basename ${analyze_file%.*})_${comp_type}"
fi

# If antigen or ligand provided as arg for comp_type then create those input directories
echo "Creating Input Directories for $comp_type Analysis"
if [ $comp_type = "antigen" ]; then
    # creating input directory based on specifed comp_type
    python $pipeline_dir/src/testdata_setup.py -q $directory -k "$(basename $directory)_${comp_type}" -s ag_chains > "$dir_path/Setup_Directory_log" 2>&1
    # reporting whether this section ran successfully
    exit_code_antigen=$?
    if [ $exit_code_antigen -eq 0 ]; then
        echo "Finished Creating Input Files for Antigen Comparison. Job completed successfully: True"
    else
        echo "Finished Creating Input Files for Antigen Comparison. Job completed successfully: False (Exit Code: $exit_code_antigen)"
    fi
    directory="$dir_path/$(basename $directory)_${comp_type}_query"
    
    if [ ! -z $analyze_file ] && [ $analyze_curated = "True" ]; then
        python $pipeline_dir/src/testdata_setup.py -q $analyze_file -k $analyze_name -s ag_chains > "$dir_path/Setup_file_log" 2>&1
        # reporting whether this section ran successfully
        exit_code_inter_antigen=$?
        if [ $exit_code_inter_antigen -eq 0 ]; then
            echo "Finished Creating Input Files for Antigen Comparison of Structure File of Interest. Job completed successfully: True"
        else
            echo "Finished Creating Input Files for Antigen Comparison of Structure File of Interest. Job completed successfully: False (Exit Code: $exit_code_inter_antigen)"
        fi
        analyze_file="$dir_path/${analyze_name}_query"
    fi

elif [ $comp_type = "ligand" ]; then
    # creating input directory based on specifed comp_type
    python $pipeline_dir/src/testdata_setup.py -q $directory -k "$(basename $directory)_${comp_type}" -s ligand_chains > "$dir_path/Setup_log" 2>&1
    # reporting whether this section ran successfully
    exit_code_ligand=$?
    if [ $exit_code_ligand -eq 0 ]; then
        echo "Finished Creating Input Files for Ligand Comparison. Job completed successfully: True"
    else
        echo "Finished Creating Input Files for Ligand Comparison. Job completed successfully: False (Exit Code: $exit_code_ligand)"
    fi
    directory="$dir_path/$(basename $directory)_${comp_type}_query"

    if [ ! -z $analyze_file ] && [ $analyze_curated = "True" ]; then
        python $pipeline_dir/src/testdata_setup.py -q $analyze_file -k $analyze_name -s ligand_chains > "$dir_path/Setup_log" 2>&1
        # reporting whether this section ran successfully
        exit_code_inter_ligand=$?
        if [ $exit_code_inter_ligand -eq 0 ]; then
            echo "Finished Creating Input Files for Ligand Comparison of Structure File of Interest. Job completed successfully: True"
        else
            echo "Finished Creating Input Files for Ligand Comparison of Structure File of Interest. Job completed successfully: False (Exit Code: $exit_code_inter_ligand)"
        fi
        analyze_file="$dir_path/${analyze_name}_query"
    fi

elif [ $comp_type = "complex" ]; then  # no need to filter chains out for this option
    :

else  # catches unrecognized inputs
    echo "Input for analysis type is not recognized"
    exit 1
fi

echo "Performing Structural Comparison and Clustering"
if [ -z $analyze_file ]; then  # no structure file of interest is specified
    # preforming foldseek clustering and then creates summary file based on the cluster
    foldseek easy-multimercluster $directory $name tmp_cluster > "$dir_path/foldseek_search_log" 2>&1
    # reporting whether this section ran successfully
    exit_code_foldseek_1=$?
    if [ $exit_code_foldseek_1 -eq 0 ]; then
        echo "Finished Performing Foldseek Clustering. Job completed successfully: True"
    else
        echo "Finished Performing Foldseek Clustering. Job completed successfully: False (Exit Code: $exit_code_foldseek_1)"
    fi

    python $pipeline_dir/src/analysis_finding_hits.py -r "$dir_path/${name}_cluster.tsv" -d $directory > "$dir_path/analysis_cluster_log" 2>&1
    # reporting whether this section ran successfully
    exit_code_analysis_1=$?
    if [ $exit_code_analysis_1 -eq 0 ]; then
        echo "Finished Performing Analysis on Clusters. Job completed successfully: True"
    else
        echo "Finished Performing Analysis on Clusters. Job completed successfully: False (Exit Code: $exit_code_analysis_1)"
    fi
else
    # preforms a foldseek search and then creates a summary file based on the search
    foldseek easy-multimersearch $analyze_file $directory $name tmp_search_single > "$dir_path/foldseek_search_log" 2>&1
    # reporting whether this section ran successfully
    exit_code_foldseek_2=$?
    if [ $exit_code_foldseek_2 -eq 0 ]; then
        echo "Finished Performing Foldseek Search. Job completed successfully: True"
    else
        echo "Finished Performing Foldseek Search. Job completed successfully: False (Exit Code: $exit_code_foldseek_2)"
    fi
    
    python $pipeline_dir/src/analysis_finding_hits.py -r "$dir_path/${name}_report" -d $directory -s $analyze_file -e $analyze_curated> "$dir_path/analysis_cluster_log" 2>&1
    # reporting whether this section ran successfully
    exit_code_analysis_2=$?
    if [ $exit_code_analysis_2 -eq 0 ]; then
        echo "Finished Performing Analysis on Search. Job completed successfully: True"
    else
        echo "Finished Performing Analysis on Search. Job completed successfully: False (Exit Code: $exit_code_analysis_2)"
    fi
fi
