#!/usr/bin/env bash

usage(){
	echo "Usage: scripts/list_jobs.sh [-h] [-j job_id] [-u user]";
	echo "";
	echo "List jobs on NEF.";
	echo "";
	echo "optional arguments:";
	small_indentation="         ";
	echo "  -h, --help$small_indentation show this help message and exit";
	echo "  -j, --job$small_indentation  show only a specific job with [job_id]";
	echo "  -u, --user$small_indentation show only the jobs of a specific user [user_id]";
	echo "  -f, --full$small_indentation print details";
}

execute() {
    job_id=NULL;
    user_id=NULL;
    options="";

    while [[ "$1" != "" ]]; do
        case $1 in
            -h | --help )           usage;
                                    exit 1;
                                    ;;
            -j | --jobs )            shift
                                    job_id=$1;
                                    ;;
            -u | --user )           shift;
                                    user_id=$1;
                                    ;;
            -f | --full )           options="$options -f";
                                    ;;
            * )                     usage;
                                    exit 1;


        esac
        shift
    done

    current_user=$(whoami);

    if [[ ${user_id} != NULL ]];
    then
        oarstat -u ${user_id}${options};
    elif [[ ${job_id} != NULL ]];
    then
        oarstat -j ${job_id}${options};
    else
        oarstat -u ${current_user}${options};
    fi
}
