#!/usr/bin/env bash

usage(){
	echo "Usage: kill job_id [-h]";
	echo "";
	echo "Kill a job on NEF.";
	echo "";
	echo "optional arguments:";
	small_indentation="         ";
	echo "  -h, --help$small_indentation show this help message and exit";
}

execute() {
    job_id=NULL;

    while [[ "$1" != "" ]]; do
        case $1 in
            -h | --help )           usage;
                                    exit 1;
                                    ;;
            * )                     if [[ ${job_id} == NULL ]];
                                    then
                                        job_id=$1;
                                    else
                                        usage;
                                        exit 1;
                                    fi


        esac
        shift
    done

    if [[ ${job_id} == NULL ]];
    then
        usage;
        exit 1;
    fi

    oardel ${job_id};
}
