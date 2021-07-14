import os
import re
import pandas as pd
import pdb
import numpy as np

rand_arr = [2,8,32,128,256,512]


n_lakes = 0
sbatch = ""
ct = 0


for name in rand_arr:
    ct += 1
    #for each unique lake
    print(name)
    l = name

    # if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAll_pball"): 
    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=CT_rand%s.out\n#SBATCH --error=CT_rand%s.err\n#SBATCH --gres=gpu:k40:2\n#SBATCH -p k40"%(l,l)
    script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
    script2 = "python CTLSTM_random.py %s"%(l)
    # script3 = "python singleModel_customSparse.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_%s_foldXGB_err.sh"%(l),sbatch])
    with open('./jobs/job_{}_foldXGB_err.sh'.format(l), 'w') as output:
        output.write(all)

compile_job_path= './jobs/sbatch_CT_rand.sh'
with open(compile_job_path, 'w') as output3:
    output3.write(sbatch)
print(ct, " jobs created, run this to submit: ", compile_job_path)



n_lakes = 0
sbatch = ""
ct = 0
for name in rand_arr:
    ct += 1
    #for each unique lake
    print(name)
    l = name

    # if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAll_pball"): 
    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EA_rand%s.out\n#SBATCH --error=EA_rand%s.err\n#SBATCH --gres=gpu:k40:2\n#SBATCH -p k40"%(l,l)
    script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
    script2 = "python EALSTM_random.py %s"%(l)
    # script3 = "python singleModel_customSparse.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_%s_foldXGB_err.sh"%(l),sbatch])
    with open('./jobs/job_{}_foldXGB_err.sh'.format(l), 'w') as output2:
        output2.write(all)

compile_job_path= './jobs/sbatch_EA_rand.sh'
with open(compile_job_path, 'w') as output4:
    output4.write(sbatch)
print(ct, " jobs created, run this to submit: ", compile_job_path)
