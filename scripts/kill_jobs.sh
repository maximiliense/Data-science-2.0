#!/usr/bin/env bash

dir=$(echo "$0" |xargs dirname);
dir="$dir/kill_devices";
if [[ "$(hostname)" =~ "nef" ]];
then
    	cluster="NEF";
    	command="$dir/nef.sh";
 elif [[ "$(hostname)" =~ "jean" ]];
 then
        cluster="Jean Zay";
        command="$dir/jz.sh";
 elif [[ "$(hostname)" =~ "hardware" ]];
 then
        cluster="GPUx";
        command="$dir/other.sh";
else
        cluster="an unregistered machine...";
        command="$dir/other.sh";
fi
${command} "$@";