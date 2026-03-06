#!/bin/bash

find ./experiments -type f -name *.log | sort | while read i
do
    logfile=$i
    result=`tail -n 1 $logfile`
    result=($result) # convert to an array
    echo $logfile "${result[11]}" # best classification accuracy
done

exit
