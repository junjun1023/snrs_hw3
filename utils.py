import os
import numpy as np
import pandas as pd

from functools import reduce


def read_file(fpath):
        df = pd.read_csv(fpath, sep="\t", header=None)
        df.columns = ["cols_"+str(i) for i, a in enumerate(df.columns)]
        return df


def get_explicit_features(*awrgs):
        """
        awrgs: list of path
        """
        import os
        from scipy.sparse import csr_matrix
        features = None
        indices = []
        for path in awrgs:
                df = read_file(path)
                data = df.cols_2 if 'cols_2' in df else np.array([1 for _ in range(len(df))])
                interact = csr_matrix( (data, (df.cols_0, df.cols_1))).toarray()
                interact = np.delete(interact, 0, axis=1)
                indices.append(interact.shape[1])
                if features is not None:
                        features = np.hstack((features, interact))
                else:
                        features = interact
        
        features = np.delete(features, 0, axis=0)

        return features, indices


class Data:

        def __init__(self, path, user_cnt, item_cnt):
                self.path = path
                self.user_cnt = user_cnt
                self.item_cnt = item_cnt

        def get_interact(self, fname="train.dat", path=None):
                import os
                from scipy.sparse import csr_matrix

                df = None
                if path:
                        df = read_file(path)
                else:
                        df = read_file(os.path.join(self.path, fname))

                interact = csr_matrix((df.cols_2, (df.cols_0, df.cols_1)), shape=(self.user_cnt+1, self.item_cnt+1))
                interact = interact.toarray()
                interact = np.delete(interact, 0, axis=1)
                interact = np.delete(interact, 0, axis=0)


                return interact