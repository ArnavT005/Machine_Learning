#!/bin/sh
### Set the job name (for your reference)
#PBS -N FirstRun
### Set the project name, your department code by default
#PBS -P col774
### Request email when job begins and ends, don't change anything on the below line 
#PBS -m bea
### Specify email address to use for notification, don't change anything on the below line
#PBS -M $USER@iitd.ac.in
#### Request your resources, just change the numbers
#PBS -l select=1:ncpus=8:ngpus=2
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=02:00:00
#PBS -l software=PYTHON

# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

cd col774
python3 test.py
exit
