                    save_path = "../../models/global_model_"+str(n_hidden)+"hid_"+str(num_layers)+"layer_"+str(dropout)+"drop"
import os
import re
import pandas as pd
import pdb
import numpy as np
#######################################
# Nov 2020
# Jared - this script creates source model creation jobs to submit to msi in one script
# (note: takeme_source.sh must be custom made on users home directory on cluster for this to work: example script 
#        `
#          #!/bin/bash
#         source activate mtl_env
#         cd research/surface_temp/src/train
#         `  
#######################################

n_clusters = 16


n_lakes = 0
sbatch = ""
ct = 0
for name in range(n_clusters):
    ct += 1
    #for each unique lake
    print(name)
    l = name

    # if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAll_pball"): 
    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_cluster_%s.out\n#SBATCH --error=EALSTM_cluster_%s.err\n#SBATCH --gres=gpu:k40:2\n#SBATCH -p k40"%(l,l)
    script = "source /home/kumarv/willa099/takeme_train.sh\n" #cd to directory with training script
    script2 = "python train_conus_cluster_EA-LSTM.py %s"%(l)
    # script3 = "python singleModel_customSparse.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_%s_clusterEA.sh"%(l),sbatch])
    with open('./jobs/job_{}_clusterEA.sh'.format(l), 'w') as output:
        output.write(all)


compile_job_path= './jobs/sbatch_script_clusterEA.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)

print(ct, " jobs created, run this to submit: ", compile_job_path)