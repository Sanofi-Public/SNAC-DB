# Pipline explanation:
# Consists of four different classification
# First step is preparing query and target directories to be used as inputs for foldseek (Setup.py)
# Then running foldseek on inputted structures
# Finally moving structures that are structurally similar to a directory labelled failed and dissimilar
# structures to a directory labelled passed (Classification.py)

# ensure proper inputs are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Need to include a directory path for a query and target complexes. Ex: bash $0 /path/query_dir /path/target_dir"
    exit 1
fi

# connecting inputs to variables to be used
query_dir_old=$1
target_dir_old=$2
dir_path=$(dirname $query_dir_old)"/Test_Data_Analysis"

# If a previous run's data exists with the same name, delete it
if [ -d $dir_path ]; then
    rm -rf $dir_path
fi
mkdir "$dir_path" "$dir_path/logs"  # new directory is made

# copy and move query_dir and target_dir into Test_Data_Analysis
query_dir="$dir_path/query_dir"
mkdir $query_dir
cp $query_dir_old/* "$query_dir/"  # grabbing only complexes
target_dir="$dir_path/target_dir"
mkdir $target_dir
cp $target_dir_old/* "$target_dir/"  # grabbing only complexes
pipeline_dir=$(dirname "$(realpath "$0")")  # bash to python script
query=$(ls $dir_path/query_dir/*.pdb | wc -l) # number of query structures

# Running the first classification that looks at structural similarity of antigens when labelled as single chains
python $pipeline_dir/src/testdata_setup.py -q $query_dir -t $target_dir -k antigen -s single_chain_ag > "$dir_path/logs/Antigen_Setup_log" 2>&1
antigen_query="$dir_path/antigen_query"
antigen_target="$dir_path/antigen_target"
foldseek easy-search $antigen_query $antigen_target "$dir_path/result_antigen" "$dir_path/tmp_antigen" -s 7.5 --format-output "query,target,qtmscore,ttmscore" > "$dir_path/logs/foldseek_antigen_output.log" 2>&1
python $pipeline_dir/src/testdata_classification.py -q $query_dir -r "$dir_path/result_antigen" -k antigen -c 0.6  > "$dir_path/logs/Antigen_Classification_log" 2>&1
antigen=$(ls $dir_path/antigen_passed/ | wc -l)
# reporting whether this section ran successfully
exit_code_antigen=$?
if [ $exit_code_antigen -eq 0 ]; then
    echo "Finished Structural Comparison of the Antigen Chains. Job completed successfully: True"
    echo "Finished Antigen Classification ($antigen Structures Passed)"
else
    echo "Finished Structural Comparison of the Antigen Chains. Job completed successfully: False (Exit Code: $exit_code_antigen)"
fi

# Running the second classification on complexes that are labeled as single chains
python $pipeline_dir/src/testdata_setup.py -q "$dir_path/antigen_failed" -t $target_dir -k ligand -s single_chain > "$dir_path/logs/ligand_Setup_log" 2>&1
ligand_query="$dir_path/ligand_query"
ligand_target="$dir_path/ligand_target"
foldseek easy-search $ligand_query $ligand_target "$dir_path/result_ligand" "$dir_path/tmp_ligand" -s 7.5 --format-output "query,target,qtmscore,ttmscore" > "$dir_path/logs/foldseek_ligand_output.log" 2>&1
python $pipeline_dir/src/testdata_classification.py -q "$dir_path/antigen_failed" -r "$dir_path/result_ligand" -k ligand -c 0.6 > "$dir_path/logs/Ligand_Classification_log" 2>&1
ligand=$(ls $dir_path/ligand_passed/ | wc -l)
# reporting whether this section ran successfully
exit_code_ligand=$?
if [ $exit_code_ligand -eq 0 ]; then
    echo "Finished Structural Comparison of the Modified Single-Chain Complex. Job completed successfully: True"
    echo "Finished Ligand Classification ($ligand Structures Passed)"
else
    echo "Finished Structural Comparison of the Modified Single-Chain Complex. Job completed successfully: False (Exit Code: $exit_code_ligand)"
fi

# Running the third classification on complexes using the multimer search from foldseek
configuration_query="$dir_path/configuration_query"
mkdir $configuration_query
cp $dir_path/ligand_failed/* $configuration_query
foldseek easy-multimersearch $configuration_query $target_dir "$dir_path/result_configuration" "$dir_path/tmp_configuration" -s 7.5 > "$dir_path/logs/foldseek_configuration_output.log" 2>&1
python $pipeline_dir/src/testdata_classification.py -q $configuration_query -t $target_dir -r "$dir_path/result_configuration_report" -k configuration > "$dir_path/logs/Configuration_Classification_log" 2>&1
configuration=$(ls $dir_path/configuration_passed/ | wc -l)
echo "Finished Configuration Classification ($configuration Structures Passed)"
# reporting whether this section ran successfully
exit_code_configuration=$?
if [ $exit_code_configuration -eq 0 ]; then
    echo "Finished Structural Comparison of the Complex. Job completed successfully: True"
    echo "Finished Configuration Classification ($configuration Structures Passed)"
else
    echo "Finished Structural Comparison of the Complex. Job completed successfully: False (Exit Code: $exit_code_configuration)"
fi

# Getting Representaive samples by doing search on final samples
representative_query="$dir_path/representative_query"
mkdir $representative_query
cp $dir_path/*_passed/* $representative_query
foldseek easy-multimersearch $representative_query $representative_query "$dir_path/result_representative" "$dir_path/tmp_representative" -s 7.5 > "$dir_path/logs/foldseek_representative_output.log" 2>&1
python $pipeline_dir/src/testdata_classification.py -q $representative_query -r "$dir_path/result_representative_report" -k representative > "$dir_path/logs/Representative_Classification_log" 2>&1
representative=$(ls $dir_path/representative_passed/ | wc -l)
# reporting whether this section ran successfully
exit_code_representative=$?
if [ $exit_code_representative -eq 0 ]; then
    echo "Finished Structural Comparison of the Query Dataset Against Itself. Job completed successfully: True"
    echo "Finished Choosing Replicates From Curated Test Data ($representative Structures Passed)"
else
    echo "Finished Structural Comparison of the Query Dataset Against Itself. Job completed successfully: False (Exit Code: $exit_code_representative)"
fi

# Combining all the complexes that passed into a final directory (Test_Data)
python $pipeline_dir/src/testdata_summary.py -p $dir_path > "$dir_path/logs/Testdata_Summary_log" 2>&1
# reporting whether this section ran successfully
exit_code_summary=$?
if [ $exit_code_summary -eq 0 ]; then
    echo "Finished Sorting and Moving Curated Dataset. Job completed successfully: True"
    echo "Finished Pipeline"
else
    echo "Finished Sorting and Moving Curated Dataset. Job completed successfully: False (Exit Code: $exit_code_summary)"
fi

# Summarzing results of the pipeline:
echo "Pipeline Summary:"
total=$((antigen + ligand + configuration))
echo "Total Passed: $(awk "BEGIN {printf \"%.3f\", ($total/$query)*100}")% ($total out of $query)"
echo "Antigen: $(awk "BEGIN {printf \"%.3f\", ($antigen/$total)*100}")% ($antigen out of $total)"
echo "Ligand: $(awk "BEGIN {printf \"%.3f\", ($ligand/$total)*100}")% ($ligand out of $total)"
echo "Configuration: $(awk "BEGIN {printf \"%.3f\", ($configuration/$total)*100}")% ($configuration out of $total)"
echo "Representative: $(awk "BEGIN {printf \"%.3f\", ($representative/$query)*100}")% ($representative out of $query)"
