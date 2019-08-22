#!/usr/bin/env bash

usage(){
	echo "Usage: submit.sh python_file [-h] [other options for the framework]";
	echo "";
	echo "Run a job locally.";
	echo "";
	echo "optional arguments:";
	small_indentation="         ";
	echo "  -h, --help$small_indentation show this help message and exit";
}


# contains findPython that help finding the correct interpreter
rootDir=$(dirname $0);
source ${rootDir}/../utilities/python.sh;

# parameters
python_file=NULL;

options="";
help=false;

# setting options
while [[ "$1" != "" ]]; do
    case $1 in
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
        $(findPython) "${python_file}" -h | sed  '1,12d;$d';
    fi
    exit 1;
fi
command="$(findPython) ${python_file}${options}";
echo "Submitting command: ${command};";
echo;
${command}
