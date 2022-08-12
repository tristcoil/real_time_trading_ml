#!/bin/bash

# run every 5 minutes between market hours mon-fri

pids=$(ps -ef | grep stream_script.py | grep -v grep | awk '{print $2}' | wc -l)

if [ ${pids} -eq 0 ]; then
    #echo $pids 
    /home/user/anaconda3/bin/conda run -n alpaca_env python /home/user/alpaca/stream_script.py &
    else
        exit 0
fi

exit 0

