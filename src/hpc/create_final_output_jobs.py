import os
import re
import pandas as pd
import pdb
import numpy as np
#######################################
# Jan 2021 - Jared - preprocess jobs
#         `  
#######################################



sbatch = ""
ct = 0
# start = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000]
# end = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,12416]
start = np.arange(0,186000,1000,dtype=np.int32)
end = start[:] + 1000
end[-1] = 185553
for i in range(len(start)):
    ct += 1
    #for each unique lake
    print(i)
    l = start[i]
    l2 = end[i]

    # if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAll_pball"): 
    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --gres=gpu:k40:2\n#SBATCH --output=final_output_%s.out\n#SBATCH --error=final_output_%s.err\n\n#SBATCH -p k40"%(i,i)
    script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
    # script2 = "python write_NLDAS_xy_pairs.py %s %s"%(l,l2)
    script2 = "python predict_lakes_EALSTM_final.py %s %s"%(l,l2)
    # script3 = "python singleModel_customSparse.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_finalout_%s.sh"%(i),sbatch])
    with open('./jobs/job_finalout_{}.sh'.format(i), 'w') as output:
        output.write(all)


compile_job_path= './jobs/finalout_jobs.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)

print(ct, " jobs created, run this to submit: ", compile_job_path)