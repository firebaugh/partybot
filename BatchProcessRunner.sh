#!/bin/bash

##
## batchProcessRunner.sh <file of commands to execute>  <number of concurrent processes>
##
## This script executes a set of commands, allowing only a limited number to run
## concurrently.  When one terminates, the next one in the series starts.
## 
## Each line of the file of commands should contain a single self-contained command
## to run and should probably begin with "nohup nice".
## 
## Here is an example of a file of commands to execute:
##   nohup nice java MyExperiment1 parameter1 parameter2 parameter3
##   nohup nice java MyExperiment2 parameter1 parameter2 parameter3
##   # comments and blank lines will be skipped
##   nohup nice java MyExperiment3 parameter1 parameter2 parameter3
##   nohup nice java MyExperiment4 parameter1 parameter2 parameter3
##   ...
##
## 
## Author Eric Eaton 2008-12-05
## Version: 0.1
## 
## Based on process_runner.sh by Donald 'Paddy' McCarthy Dec. 17 2007
## (http://paddy3118.blogspot.com/2007/12/batch-process-runner-in-bash-shell.html)
##
## 
## Copyright (c) Eric Eaton 2008
##	
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##	
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##	
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
##	

if [[ $# != 2 && $# != 3 ]]; then
  echo "Usage: `basename $0` <commands_file>  <num_processes> [<wait_time>]"
  echo "  <commands_file>  The file of commands to execute, one command per line."
  echo "  <num_processes>  The number of concurrent processes to run."
  echo "  [<wait_time>]    (optional) The wait time in seconds between checking"
  echo "                   for completed processes.  Default = 60."
  echo ""
  echo "This script executes a set of commands, allowing only a limited"
  echo "number to run concurrently.  When one terminates, the next one"
  echo "in the series starts."
  echo ""
  echo "Each line of the file of commands should contain a single"
  echo "self-contained command to run and should probably begin"
  echo "with 'nohup nice'."
  echo ""
  echo "Here is an example of a file of commands to execute:"
  echo "   nohup nice java MyExperiment1 parameter1 parameter2 parameter3"
  echo "   nohup nice java MyExperiment2 parameter1 parameter2 parameter3"
  echo "   # comments and blank lines will be skipped"
  echo "   nohup nice java MyExperiment3 parameter1 parameter2 parameter3"
  echo "   nohup nice java MyExperiment4 parameter1 parameter2 parameter3"
  echo "   ..."
  exit 1
fi



## Parse command line arguments
# file of commands
commandsFile=$1
# how many processes to run in parallel
numConcurrentProcesses=$2
# main loop wait time in seconds between checking background processes
if [[ $# == 3 ]]; then
	tick=$3
else
	tick=60
fi



## Read the commands file into an array so we don't hold a handle on the file
old_IFS=$IFS
IFS=$'\n'
lines=($(cat $commandsFile)) # array of lines from file
IFS=$old_IFS

# Extract the commands from the lines, stripping trailing semicolons and ampersands from the commands
c=0
numSkipped=0
for ((l=0; l<${#lines[*]}; l+=1 )); do
	# strip leading and trailing whitespace
	lines[l]="${lines[l]#"${lines[l]%%[![:space:]]*}"}"   # remove leading whitespace
	lines[l]="${lines[l]%"${lines[l]##*[![:space:]]}"}"   # remove trailing whitespace
	
	# keep only non blank lines that are not comments
	if [[ -z "${lines[l]}" || "${lines[l]}" =~ \[\#*\] ]]; then
		((numSkipped+=1)) # skip line
	else
		commands[c]=${lines[l]%;}     # strip trailing semicolons
		commands[c]=${commands[c]%&}  # strip trailing ampersands
		commands[c]="${commands[c]%"${commands[c]##*[![:space:]]}"}"   # remove trailing whitespace
		((c+=1))
	fi
done

# the number of command lines in the file
maxprocs=${#commands[*]}



## Print out the header of what's going to occur
echo "BATCH PROCESS RUNNER"
echo "---------------------------------------------------"
echo "Running $maxprocs commands from $commandsFile, $numConcurrentProcesses processes at a time:"
# print all the commands as read from the file
for ((c=0; c<${#commands[*]}; c+=1 )); do
	echo "  ${commands[c]}"  
done
echo ""



## Array to track the PIDs of background processes
for ((i=0; i<$numConcurrentProcesses; i+=1 )); do
	running[$i]=123456789;   # start with invalid PIDs in all slots
done



## Main loop to run all commands
ran=0   # current command number
until
  while [ $ran -lt $maxprocs ]; do
  
  	# check whether each process is running, and start a new one in its place if it is not
    for ((p=0; p<$numConcurrentProcesses; p+=1 )); do
    
      # check whether the process is running
      proc=${running[$p]}
      ps -p $proc | fgrep $proc >/dev/null
      
      # if it isn't, then start the next command
      if [ $? -ne '0' ] ; then
      
        # Run another command in the background and store the PID
        echo "Executing command $ran of $maxprocs:  ${commands[ran]} &"
        ${commands[ran]} &  # execute the command
        running[$p]=$!      # store the PID
        ((ran+=1))          # increment the command number
        
        # If we've run through all of the processes, wait for them to finish
        if [ $ran -ge $maxprocs ]; then break 1; fi
        
      fi
    done
    sleep $tick
  done

  sleep $tick
do [  `jobs -r|wc -l` -eq 0 ]
done
wait

echo ""
echo "All $maxprocs processes have finished."

exit 0
