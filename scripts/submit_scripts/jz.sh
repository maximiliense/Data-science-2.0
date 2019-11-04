#!/usr/bin/env bash

# default values
nb_gpus=1;
nb_cores=1;

walltime="00:10:00";

script_dir=".";
out="$HOME";
usage(){
    indentation="                          ";
    echo "Usage: submit python_file [-h] [-g NB_GPUS] [-c NB_CORES] [-d ROOT_DIR]";
    echo "$indentation[-w WALLTIME] [-n NAME] [--host HOST]";

    echo "";
    echo "Submit a job on Jean Zay.";

    echo "";

    echo "optional arguments:";
    small_indentation="         ";
    echo "$small_indentation -h, --help$small_indentation  show this help message and exit";
    echo "$small_indentation -g, --gpu$small_indentation  set the required number of GPUs (default: ${nb_gpus})";
    echo "$small_indentation -c, --cores$small_indentation set the required number of cores (default: ${nb_cores})";
    # echo "$small_indentation -n, --nodes$small_indentation set the required number of nodes";
    echo "$small_indentation --dir$small_indentation       SLURM scripts dir (default: ${script_dir}";
    echo "$small_indentation -o --out$small_indentation    SLURM logs dir (default: ${out}";
    echo -n "$small_indentation -w, --wt$small_indentation    set the job walltime (default: ${walltime})";
    echo ", hours if int, can be >20";
    echo "$small_indentation -n, --name$small_indentation  set the experiment name (default: \$python_file)";
    echo "$small_indentation --dev$small_indentation       run on the dev partition";
    echo "$small_indentation --[data science 2.0 options]";
    exit 1
}

execute() {
    name=NULL;

    python_file=NULL;

    dev=false;

    project_path="~/Data-science-2.0";

    options="";
    while [[ "$1" != "" ]]; do
        case $1 in
            -g | --gpu )            shift
                                    nb_gpus=$1;
                                    ;;
            --dir)                  shift;
                                    script_dir=$1;
                                    ;;
            --dev)                  dev=true;
                                    ;;
            -w | --wt )             shift;
                                    walltime=$1;
                                    ;;
            -n | --name )           shift;
                                    name="_"$1;
                                    ;;
            -o | --out )            shift;
                                    out=$1;
                                    ;;
            -h | --help )           usage;
                                    ;;
            * )                     if [[ "$python_file" = NULL ]];
                                    then
                                        python_file=$1;
                                    else
                                        options="$options $1";
                                    fi
        esac
        shift
    done
    echo $options;
    if [[ "$python_file" = NULL ]];
    then
        usage;
    fi

    # setup name
    setup_name="${python_file/projects\/}";
    setup_name="${setup_name/\.py/}";
    setup_name="${setup_name/\//_}";

    if [[ "${name}" = NULL ]];
    then
        name="";
    else
        options="${options} --name ${setup_name}${name}";
    fi

    # constructing GPU command
    gpu_command=""
    for (( i=0 ; i <$nb_gpus ; i++));
    do
        if [[ "$gpu_command" != "" ]];
        then
            gpu_command="$gpu_command,";
        fi
        gpu_command="$gpu_command$i";
    done

    if [[ "$nb_gpus" -gt "0" ]];
    then
        options="$options --gpu $gpu_command";
    fi
    # Jean Zay
    options="$options --homex /gpfswork/rech/fqg/uid61lx/output/";
    # with or without resubmission
    re='^[0-9]+$'
    JOB_ID=NULL;
    if ! [[ ${walltime} =~ $re ]] ; then
        actualWalltime=${walltime};
        actualName="$name";
        createScript;
    else
        i=1;
        while [[ ${walltime} -gt 0 ]]
        do
            actualWalltime="$((walltime>20 ? 20 : walltime)):00:00";
            actualName="${name}_part_$i";

            createScript;
            if [[ ${options} != *" -r"* ]] && [[ ${options} != *" --r"* ]];
            then
                options="$options --restart";
            fi
            walltime=$((walltime-20));
            i=$((i+1))
        done;
    fi

}

createScript() {
    # constructing script
    echo "CREATING SCRIPT $script_dir/${setup_name}$actualName.slurm"

    echo "#!/bin/bash" > ${script_dir}/${setup_name}${actualName}.slurm
    echo "#SBATCH --job-name=${setup_name}$actualName         # nom du job" >> ${script_dir}/${setup_name}${actualName}.slurm
    # if [[ ${dev} == false ]];
    # then
        # echo "#SBATCH --partition=gpu_gct3">> ${script_dir}/${setup_name}${actualName}.slurm
    # else
    #     echo "#SBATCH --partition=gpu_dev">> ${script_dir}/${setup_name}${actualName}.slurm
    # fi

    echo "#SBATCH  --mem=160G">> ${script_dir}/${setup_name}${actualName}.slurm
    echo "#SBATCH  --cpus-per-task=4">> ${script_dir}/${setup_name}${actualName}.slurm
    echo "#SBATCH --gres=gpu:$nb_gpus  # nombre de GPU à réserver">> ${script_dir}/${setup_name}${actualName}.slurm
    echo "#SBATCH --time=$actualWalltime             # (HH:MM:SS)">> ${script_dir}/${setup_name}${actualName}.slurm
    echo "#SBATCH --output=${out}/${setup_name}${actualName}_%j.out" >> ${script_dir}/${setup_name}${actualName}.slurm
    echo "#SBATCH --error=${out}/${setup_name}${actualName}_%j.err" >> ${script_dir}/${setup_name}${actualName}.slurm

    echo >> ${script_dir}/${setup_name}${actualName}.slurm

    echo "module load pytorch-gpu/py3/1.1" >> ${script_dir}/${setup_name}${actualName}.slurm
    echo "module load p7zip/16.02/gcc-9.1.0" >> ${script_dir}/${setup_name}${actualName}.slurm


    echo >> ${script_dir}/${setup_name}${actualName}.slurm

    echo "# echo des commandes lancées" >> ${script_dir}/${setup_name}${actualName}.slurm
    echo "set -x" >> ${script_dir}/${setup_name}${actualName}.slurm

    echo "# exécution du code" >> ${script_dir}/${setup_name}${actualName}.slurm

    exec_command="core=\"python $python_file$options\"";  # TODO add export folder
    echo ${exec_command} >> ${script_dir}/${setup_name}${actualName}.slurm;
    echo "export PYTHONPATH=\"${PYTHONPATH}:.\"" >> ${script_dir}/${setup_name}${actualName}.slurm;
    echo "echo \"\$core\";" >> ${script_dir}/${setup_name}${actualName}.slurm
    echo "\$core;" >> ${script_dir}/${setup_name}${actualName}.slurm


    # first job
    if [[ "${JOB_ID}" = NULL ]];
    then
        echo "sbatch ${script_dir}/${setup_name}${actualName}.slurm";
        JOB_ID=`sbatch ${script_dir}/${setup_name}${actualName}.slurm | cut -d " " -f 4`
    else
        echo "sbatch --dependency=afterany:${JOB_ID} ${script_dir}/${setup_name}${actualName}.slurm";
        JOB_ID=`sbatch --dependency=afterany:${JOB_ID} ${script_dir}/${setup_name}${actualName}.slurm | cut -d " " -f 4`;
    fi
}
