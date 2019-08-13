#!/usr/bin/env bash

usage(){
	echo "Usage: scripts/list_jobs.sh [-h] [-u USER]";
	echo "";
	echo "List jobs.";
	echo "";
	echo "optional arguments:";
	small_indentation="         ";
	echo "  -h, --help$small_indentation show this help message and exit";
	echo "  -u, --user$small_indentation username of the owner of the processes";
}

user_id=NULL;

while [[ "$1" != "" ]]; do
    case $1 in
        -h | --help )           usage;
                                exit 1;
                                ;;
        -u | --user )           shift;
                                user_id=$1;
                                ;;
        * )                     echo "unknown parameters $1";
                                exit 1;

    esac
    shift
done

if [[ ${user_id} == NULL ]];
then
    user_id=$(whoami | cut -c1-6)
fi
ps aux | head -n1;
ps aux | grep ${user_id} | grep python | grep -v "grep python";

