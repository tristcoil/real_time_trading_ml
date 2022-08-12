#!/bin/bash

# to be run exactly at market close
# it will sleep 30 seconds and then will stop data stream
sleep 30

pids=$(ps -ef | grep sleepy.sh | grep -v grep | awk '{print $2}')

for pid in "$pids"; do
    #echo $pid
    /bin/kill $pid
done

exit 0 



