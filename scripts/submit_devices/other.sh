#!/usr/bin/env bash

usage(){
	echo "Usage: submit.sh python_file [-h] [other options for the framework]";
	echo "";
	echo "Run a job locally.";
	echo "";
	echo "optional arguments:";
	small_indentation="         ";
	echo "  -h, --help$small_indentation show this help message and exit";
	echo "  --py-version$small_indentation python version (default: 3.7)";
}


python_file=NULL;
# project_path=".";
python_version="3.7"

options="";
help=false;

# setting options
while [[ "$1" != "" ]]; do
    case $1 in

        --py-version )            shift
                                python_version=$1;
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
        python"${python_version} ${python_file}" -h | sed  '1,12d;$d';
    fi
    exit 1;
fi
command="python${python_version} ${python_file}${options}"
echo "${command};";
${command}
