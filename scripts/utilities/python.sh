#!/usr/bin/env bash

function findPython {
    # This function returns the best python interpreter found on the machine
	virtual_env=$(python -c "import sys;print(hasattr(sys, 'base_prefix') or hasattr(sys, 'real_prefix'))");

	local lastPython="python";
	if [[ ${virtual_env} == "False" ]];
	then

		for i in $(seq 2 3);
		do
		for j in $(seq 0 9);
			do
				c=$(command -v python${i}.${j});
				if [[ ${c} != "" ]];
				then
					lastPython=${c};
				fi
			done;
		done;
	fi;
	echo ${lastPython};
}

