#!/bin/bash

if [ -d "trajFiles" ]; then
    rm -r "trajFiles"
fi
                                                                                                               
seeds=$(ls -d */)                                                                                              
mkdir trajFiles     
for seed in $seeds; do                                                                                         
    echo "Running script with seed: $seed"
    dirpath="${seed}dump"
    input_file=$(find ${dirpath} -maxdepth 1 -name "*lammpstrj*"  | head -n 1)
    output_dir="${dirpath}/out/"
    mkdir -p $output_dir
    lines_per_timestep=$((1024+9))
    split -d -l $lines_per_timestep -a 4 $input_file ${output_dir}                                                             
    for file in "${dirpath}/out"/*; do
	file_name=$(basename "$file")
	id="${file_name#*_}"
	id=$(echo "$id" | sed 's/^0*//')
	if (( id > 200 )); then
		mv "$file" "trajFiles/${seed%/}_${file_name}"
	fi
    done         
    rm -r "${dirpath}/out"                                                                               
    echo "Script with seed $seed completed"                                                                                                                                     
done 
