#!/usr/bin/env bash



usage(){
  indentation="                           ";
	echo "Usage: submit python_file [-h] [-g NB_GPUS] [-c NB_CORES] [-d ROOT_DIR]";
	echo "${indentation}[-w WALLTIME] [-n NAME] [--host HOST]";

	echo "";
	echo "Submit a job on NEF.";

	echo "";

	echo "optional arguments:";
	small_indentation="         ";
	echo "$small_indentation -h, --help$small_indentation show this help message and exit";
	echo "$small_indentation -g, --gpus$small_indentation set the required number of GPUs";
	echo "$small_indentation -c, --cores$small_indentation set the required number of cores to register";
	echo "$small_indentation --dir$small_indentation set the the root directory in which the script will be saved";
	echo "$small_indentation -o --out$small_indentation where to save OAR files";
	echo "$small_indentation -w, --wt$small_indentation set the job walltime";
	echo "$small_indentation -n, --name$small_indentation set the experiment name (or python_file by default)";
	echo "$small_indentation --host$small_indentation set the required host (any by default)";
	echo "$small_indentation -m, --mem$small_indentation set the minimum amount of required memory (default: 90000)";
	echo "$small_indentation --[data science 2.0 options]";
	exit 1
}

    execute() {
    nb_cores=1;
    nb_gpus=1;
    script_dir=".";
    walltime=10;
    name=NULL;
    python_file=NULL;
    min_memory=64000
    host=NULL;
    out="$HOME";

    project_path="$HOME/Data-science-2.0";

    options="";

    while [[ "$1" != "" ]]; do
        case $1 in
            -g | --gpu )            shift
                                    nb_gpus=$1;
                                    ;;
            --dir)                  shift;
                                    script_dir=$1;
                                    ;;
            -w | --wt )             shift;
                                    walltime=$1;
                                    ;;
            -n | --name )           shift;
                                    name="_"$1;
                                    ;;
            --host )                shift;
                                    host=$1;
                                    ;;
            -m | --mem )            shift;
                                    min_memory=$1;
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
    fi

    # constructing GPU command
    gpu_command=""
    # shellcheck disable=SC2004
    for (( i=0 ; i < $nb_gpus ; i++));
    do
        if [[ "$gpu_command" != "" ]];
        then
            gpu_command="$gpu_command,";
        fi
        gpu_command="$gpu_command$i";
    done

    # constructing script
    echo "CREATING SCRIPT $script_dir/${setup_name}$name.sh"
    echo "#!/usr/bin/sh" > "${script_dir}/${setup_name}${name}.sh"

    # Machine may need GPU
    if [[ "$nb_gpus" -gt "0" ]];
    then
        echo "#OAR -l /nodes=1/gpunum=$nb_gpus,walltime=$walltime" >> "${script_dir}/${setup_name}${name}.sh"
        oar_p="#OAR -p gpu='YES' and gpucapability>='6.1' and gpucapability<'7.5' and mem>'$min_memory'"
        options="$options --gpu $gpu_command";
    else
        echo "#OAR -l /nodes=1/core=$nb_cores,walltime=$walltime" >> "${script_dir}/${setup_name}${name}.sh"
        oar_p="#OAR -p mem>'$min_memory'";
    fi

    # specific host
    if [[ "${host}" != NULL ]];
    then
        if [[ "$oar_p" != "" ]];
        then
            oar_p="$oar_p and host='$host.inria.fr'";
        else
            oar_p="#OAR -p host='$host.inria.fr'";
        fi
    fi
    echo "$oar_p" >> "${script_dir}/${setup_name}${name}.sh"


    echo "#OAR --name ${setup_name}$name" >> "${script_dir}/${setup_name}${name}.sh";
    print_oar="echo \"Launching job \$OAR_JOBID on \`oarprint gpunb\` gpus on host \`oarprint host\`\"";
    echo "${print_oar}" >> "${script_dir}/${setup_name}${name}.sh";
    echo "module load conda/4.4.0-python3.6">> "${script_dir}/${setup_name}${name}.sh";

    if [[ "$nb_gpus" -gt "0" ]];
    then
        echo "module load cuda/9.2" >> "${script_dir}/${setup_name}${name}.sh";
    fi


    echo "cd ${project_path};" >> "${script_dir}/${setup_name}${name}.sh";
    echo "source activate venv3;" >> "${script_dir}/${setup_name}${name}.sh";

    if [[ "$name" != "" ]];
    then
        options="$options --name ${name#?}";
    fi

    exec_command="core=\"python $python_file$options\"";

    echo "${exec_command}" >> "${script_dir}/${setup_name}${name}.sh";

    echo "echo \"\$core\";" >> "${script_dir}/${setup_name}${name}.sh";
    echo "\$core;" >> "${script_dir}/${setup_name}${name}.sh";

    # submitting script
    chmod u+x "${script_dir}/${setup_name}${name}.sh";
    oarsub -S "${script_dir}/${setup_name}${name}.sh" -d "${out}";
}
