#!/usr/bin/env bash

usage(){
	echo "Usage: scripts/list_jobs.sh [-h]";
	echo "";
	echo "List jobs.";
	echo "";
	echo "optional arguments:";
	small_indentation="         ";
	echo "  -h, --help$small_indentation show this help message and exit";
}

while [[ "$1" != "" ]]; do
    case $1 in
        -h | --help )           usage;
                                exit 1;
                                ;;
        * )                     echo "unknown parameters $1";
                                exit 1;

    esac
    shift
done

current_user=$(whoami | cut -c1-6)
echo $(ps aux | head -n1);
echo $(ps aux | grep ${current_user} | grep pyth | grep -v "grep pyth");

