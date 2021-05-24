import numpy as np
import pandas as pd
from functools import reduce

def get_explicit_features(*awrgs):
        # ref = awrgs[0]
        # for awrg in awrgs[1:]:
        #         idx = ref[:, 0]
        #         attr = awrg[ idx ][:, 1]
        #         ref = np.hstack((ref, attr))
        return reduce(lambda x,y: pd.merge(x,y, on="cols_0", how='left'), awrgs)


def read_file(fpath):
        df = pd.read_csv(fpath, sep="\t", header=None)
        df.columns = ["cols_"+str(i) for i, a in enumerate(df.columns)]
        return df
