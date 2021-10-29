#!/usr/bin/env python

import socket
from subprocess import call


def generate_single_on_discovery(idx, data_name):		# northeastern discovery cluster specific code
	cmd = ''
	cmd += "#!/bin/bash\n"
	cmd += "\n#set a job name  "
	cmd += "\n#SBATCH --job-name=%d_%s"%(idx, data_name)
	cmd += "\n#################  "
	cmd += "\n#a file for job output, you can check job progress"
	cmd += "\n#SBATCH --output=./tmp/no_name/%d_%s.out"%(idx, data_name)
	cmd += "\n#################"
	cmd += "\n# a file for errors from the job"
	cmd += "\n#SBATCH --error=./tmp/no_name/%d_%s.err"%(idx, data_name)
	cmd += "\n#################"
	cmd += "\n#time you think you need; default is one day"
	cmd += "\n#in minutes in this case, hh:mm:ss"
	cmd += "\n#SBATCH --time=24:00:00"
	cmd += "\n#################"
	cmd += "\n#number of tasks you are requesting"
	cmd += "\n#SBATCH -N 1"
	cmd += "\n#SBATCH --exclusive"
	cmd += "\n#################"
	cmd += "\n#partition to use"
	#cmd += "\n#SBATCH --partition=general"
	#cmd += "\n#SBATCH --partition=ioannidis"	
	cmd += "\n#SBATCH --partition=gpu"	
	#cmd += "\n#SBATCH --constraint=E5-2680v2@2.80GHz"		# 20 cores	
	#cmd += "\n#SBATCH --exclude=c3096"
	cmd += "\n#SBATCH --mem=120Gb"
	cmd += "\n#################"
	cmd += "\n#number of nodes to distribute n tasks across"
	cmd += "\n#################"
	cmd += "\n"
	cmd += "\npython ./kN_example.py"
	
	fin = open('execute_combined.bash','w')
	fin.write(cmd)
	fin.close()


run_num = 20
data_name = 'cancer'

for idx in range(run_num):
	generate_single_on_discovery(idx, data_name)

	if socket.gethostname().find('login') != -1:
		call(["sbatch", "execute_combined.bash"])
	else:
		os.system("bash ./execute_combined.bash")

