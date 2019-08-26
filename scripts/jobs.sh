#!/usr/bin/env bash

usage() {
    #
    # Usage of the jobs scripts
    #
    indentation="                           ";
	echo "Usage: jobs.sh [list] [submit] [kill] [-h] [other options for the framework]";
	echo "";
	echo "Job management scripts.";
	echo "";
	echo "optional arguments:";
	small_indentation="         ";
	echo "  -h, --help$small_indentation show this help message and exit";
}


# identifying script type to execute
dir=$(echo "$0" |xargs dirname);

case $1 in
    -h | --help )           usage;
                            exit 1;
                            ;;
    list )                  dir="$dir/list_scripts/";
                            ;;
    submit )                dir="$dir/submit_scripts/";
                            ;;
    kill )                  dir="$dir/kill_scripts/";
                            ;;
    * )                     usage;
                            exit 1;

esac
shift
# identifying machine
if [[ "$(hostname)" =~ "nef" ]];
then
    	cluster="NEF";
    	script="$dir/nef.sh";
 elif [[ "$(hostname)" =~ "jean" ]];
 then
        cluster="Jean Zay";
        script="$dir/jz.sh";
else
        cluster="Regular";
        script="$dir/other.sh";
fi
# echo "[${cluster}]";
# echo;

source ${script};

execute $@;
