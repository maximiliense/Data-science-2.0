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
	echo "$small_indentation -g, --gpus$small_indentation  set the required number of GPUs (default: ${nb_gpus})";
	echo "$small_indentation -c, --cores$small_indentation set the required number of cores (default: ${nb_cores})";
	# echo "$small_indentation -n, --nodes$small_indentation set the required number of nodes";
	echo "$small_indentation --dir$small_indentation       SLURM scripts dir (default: ${script_dir}";
	echo "$small_indentation -o --out$small_indentation    SLURM logs dir (default: ${out}";
	echo "$small_indentation -w, --wt$small_indentation    set the job walltime (default: ${walltime})";
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

    # constructing script
    echo "CREATING SCRIPT $script_dir/${setup_name}$name.slurm"

    echo "#!/bin/bash" > ${script_dir}/${setup_name}${name}.slurm
    echo "#SBATCH --job-name=${setup_name}$name         # nom du job" >> ${script_dir}/${setup_name}${name}.slurm
    if [[ ${dev} == false ]];
    then
        echo "#SBATCH --partition=gpu_gct3">> ${script_dir}/${setup_name}${name}.slurm
    else
        echo "#SBATCH --partition=gpu_dev">> ${script_dir}/${setup_name}${name}.slurm
    fi

    echo "#SBATCH  --mem=160G">> ${script_dir}/${setup_name}${name}.slurm
    echo "#SBATCH  --cpus-per-task=4">> ${script_dir}/${setup_name}${name}.slurm
    echo "#SBATCH --gres=gpu:$nb_gpus  # nombre de GPU à réserver">> ${script_dir}/${setup_name}${name}.slurm
    echo "#SBATCH --time=$walltime             # (HH:MM:SS)">> ${script_dir}/${setup_name}${name}.slurm
    echo "#SBATCH --output=${out}/${setup_name}${name}_%j.out" >> ${script_dir}/${setup_name}${name}.slurm
    echo "#SBATCH --error=${out}/${setup_name}${name}_%j.err" >> ${script_dir}/${setup_name}${name}.slurm

    echo >> ${script_dir}/${setup_name}${name}.slurm

    echo "module load pytorch-gpu/py3/1.1" >> ${script_dir}/${setup_name}${name}.slurm
    echo "module load p7zip/16.02/gcc-9.1.0" >> ${script_dir}/${setup_name}${name}.slurm


    echo >> ${script_dir}/${setup_name}${name}.slurm

    echo "# echo des commandes lancées" >> ${script_dir}/${setup_name}${name}.slurm
    echo "set -x" >> ${script_dir}/${setup_name}${name}.slurm

    echo "# exécution du code" >> ${script_dir}/${setup_name}${name}.slurm

    options="$options --homex /gpfswork/rech/fqg/uid61lx/output/";

    exec_command="core=\"python $python_file$options\"";  # TODO add export folder
    echo ${exec_command} >> ${script_dir}/${setup_name}${name}.slurm;
    echo "export PYTHONPATH=\"${PYTHONPATH}:.\"" >> ${script_dir}/${setup_name}${name}.slurm;
    echo "echo \"\$core\";" >> ${script_dir}/${setup_name}${name}.slurm
    echo "\$core;" >> ${script_dir}/${setup_name}${name}.slurm

    echo "sbatch ${script_dir}/${setup_name}${name}.slurm";
    sbatch ${script_dir}/${setup_name}${name}.slurm

    # chargement des modules

    # salloc --ntasks=1 --threads-per-core=1 --gres=gpu:1 --partition=gpu_dev --time=00:05:00
    # module load ...
    # srun --ntasks=1 --gres=gpu:1 python mon_script.py
    # srun --ntasks=1 --gres=gpu:1 python -c "import torch; val=torch.cuda.is_available(); print('val = ', val);"



    # EXAMPLE PERL
    # #!/usr/bin/perl
    #my @tasks = split(',', $ARGV[0]);
    #my @nodes = `scontrol show hostnames $SLURM_JOB_NODELIST`;
    #my $node_cnt = $#nodes + 1;
    #my $task_cnt = $#tasks + 1;
    #
    #if ($node_cnt < $task_cnt) {
    #	print STDERR "ERROR: You only have $node_cnt nodes, but requested layout on $task_cnt nodes.\n";
    #	$task_cnt = $node_cnt;
    #}
    #
    #my $cnt = 0;
    #my $layout;
    #foreach my $task (@tasks) {
    #	my $node = $nodes[$cnt];
    #	last if !$node;
    #	chomp($node);
    #	for(my $i=0; $i < $task; $i++) {
    #		$layout .= "," if $layout;
    #		$layout .= "$node";
    #	}
    #	$cnt++;
    #}
    # print $layout;
}