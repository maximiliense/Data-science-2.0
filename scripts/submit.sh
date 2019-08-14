#!/usr/bin/env bash

dir=$(echo "$0" |xargs dirname);
dir="$dir/submit_devices";
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
        command="$dir/gpu4.sh";
else
        cluster="an unregistered machine...";
        command="$dir/other.sh";
fi
echo "Welcome on ${cluster}";
${command} "$@";