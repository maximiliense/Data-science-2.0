#!/usr/bin/env bash

usage(){
	echo "Usage: submit python_file [-h] [other options for the framework]";
	echo "";
	echo "Submit your job.";
	echo "";
	echo "optional arguments:";
	small_indentation="         ";
	echo "  -h, --help$small_indentation   show this help message and exit";
	echo "  --screen$small_indentation  run job without screen.";
	echo "  --screen-name$small_indentation  set the screen name.";
}

execute() {
    # contains findPython that help finding the correct interpreter
    rootDir=$(dirname $0);
    source ${rootDir}/utilities/python.sh;

    # parameters
    python_file=NULL;

    runScreen=false;

    options="";
    help=false;
    name="default";
    # setting options
    while [[ "$1" != "" ]]; do
        case $1 in
            --screen )           runScreen=true;
                                    ;;
            --screen-name )           shift;
                                    name=$1;
                                    ;;
            -h | --help )           help=true;
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
    # testing if need to print usage...
    if [[ ${help} = true ]] || [[ "${python_file}" = NULL ]];
    then
        usage;
        if [[ "${python_file}" != NULL ]];
        then
            $(findPython) "${python_file}" -h | sed  '1,10d;$d';
        fi
        exit 1;
    fi
    command="$(findPython) ${python_file}${options}";

    if [[ ${runScreen} = true ]];
    then
        echo "Submitting command: ${command};";
        echo "screen -S ${name} -dm bash -c \"${command}\"";
        screen -S "${name}" -dm bash -c "${command}";
        echo "Job submitted in a screen.";
    else
        echo "Executing command: ${command};";
        ${command}
    fi
}
