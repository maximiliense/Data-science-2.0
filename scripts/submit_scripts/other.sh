#!/usr/bin/env bash

usage(){
	echo "Usage: submit python_file [-h] [other options for the framework]";
	echo "";
	echo "Submit your job.";
	echo "";
	echo "optional arguments:";
	small_indentation="         ";
	echo "  -h, --help$small_indentation   show this help message and exit";
	echo "  --no-screen$small_indentation  run job without screen.";
}

execute() {
    # contains findPython that help finding the correct interpreter
    rootDir=$(dirname $0);
    source ${rootDir}/utilities/python.sh;

    # parameters
    python_file=NULL;

    runScreen=true

    options="";
    help=false;
    # setting options
    while [[ "$1" != "" ]]; do
        case $1 in
            --no-screen )           runScreen=false;
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
            $(findPython) "${python_file}" -h | sed  '1,11d;$d';
        fi
        exit 1;
    fi
    command="$(findPython) ${python_file}${options}";
    echo "Submitting command: ${command};";
    if [[ ${runScreen} = true ]];
    then
        screen -dm bash -c "${command}";
        echo "Job submitted in a screen.";
    else
        ${command}
    fi
}
