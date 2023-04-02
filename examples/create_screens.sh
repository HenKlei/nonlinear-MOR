#!/bin/bash
count=1
screen_prefix="s_"
current_screens_file="current_screens.txt"
echo "Starting the following screens:" > $current_screens_file
input="run_commands_in_screens.txt"
while read -r line
do
  screen_name="$screen_prefix$count"
  echo "Starting screen $screen_name with command:" >> $current_screens_file
  echo "$line" >> $current_screens_file
  echo >> $current_screens_file
  screen -S $screen_name -d -m
  screen -S $screen_name -X stuff "$line\n"
  count=`expr $count + 1`
  sleep 2
done < "$input"
