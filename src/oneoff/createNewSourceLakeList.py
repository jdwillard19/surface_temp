import re
import numpy as np
import pandas as pd


glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = np.array([re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)])

# other_source_ids = train_lakes[~np.isin(train_lakes,site_id)] #remove site id
train_lakes = train_lakes[~np.isin(train_lakes, ['121623043','121623126',\
                                                                '121860894','143249413',\
                                                                '143249864', '152335372',\
                                                                '155635994','70332223',\
                                                                '75474779'])] #remove cuz <= 1 surf temp obs



np.save("../../data/static/lists/source_lakes_wrr",train_lakes)