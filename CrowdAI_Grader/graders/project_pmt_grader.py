from common_grader import CommonGrader

import io
import numpy as np
import h5py
import pandas as pd
import scipy.stats

import itertools as it

# cache hdf5 results
files = {}


def wpdistance(df_ans, df_sub):
    '''
    do the actual grade

    return the mean Wasserstain and Poisson distances.
    '''
    dists = pois = 0

    # number of channels is 30
    e_ans = df_ans['EventID']*30 + df_ans['ChannelID']
    e_ans, i_ans = np.unique(e_ans, return_index=True)
    gl = len(e_ans)

    e_sub = df_sub['EventID']*30 + df_sub['ChannelID']
    e_sub, i_sub = np.unique(e_sub, return_index=True)

    # bad: additional memory allocation
    i_sub = np.append(i_sub, len(df_sub))

    p = 0
    ejd = e_sub[p]
    # append an additional largest eid, so that the last event is also graded
    for eid, i0, i in zip(e_ans, np.nditer(i_ans), it.chain(np.nditer(i_ans[1:]), [len(df_ans)])):
        while ejd < eid:
            p += 1
            ejd = e_sub[p]
        assert ejd == eid, 'Answer must include Event {} Channel {}.'.format(eid//30, eid % 30)

        j0 = i_sub[p]; j = i_sub[p+1]

        # scores
        wl = df_sub[j0:j]['Weight']
        dists += scipy.stats.wasserstein_distance(df_ans[i0:i]['PETime'],
                                                  df_sub[j0:j]['PETime'], v_weights=wl)
        Q = i-i0; q = np.sum(wl)
        pois += np.abs(Q - q) * scipy.stats.poisson.pmf(Q, Q)

    return dists/gl, pois/gl


class PMTGrader(CommonGrader):

    def __init__(self, *kargs):
        super(PMTGrader, self).__init__(*kargs)
        file_path = self.answer_file_path
        if files.__contains__(file_path):
            self.df_ans = files[file_path]
        else:
            with h5py.File(file_path) as f_ans:
                self.df_ans = f_ans["GroundTruth"][()]
            files[file_path] = self.df_ans

    @staticmethod
    def check_column(row_name, fields):
        if row_name not in fields:
            raise ValueError('Bad submission: column {} not found in Answer table'.format(row_name))

    def do_grade(self):
        b = io.BytesIO(self.submission_content)
        with h5py.File(b) as f_sub:
            # check for data structure in hdf5 file
            if "Answer" not in f_sub:
                raise ValueError('Bad submission: no Answer table found')
            answer_fields = f_sub['Answer'].dtype.fields
            self.check_column('PETime', answer_fields)
            self.check_column('EventID', answer_fields)
            self.check_column('ChannelID', answer_fields)
            self.check_column('Weight', answer_fields)
            return wpdistance(self.df_ans, f_sub['Answer'][()])


if __name__ == "__main__":
    import argparse
    psr = argparse.ArgumentParser()
    psr.add_argument("-r", dest='ref', help="reference")
    psr.add_argument('ipt', help="input to be graded")
    args = psr.parse_args()

    with h5py.File(args.ref) as ref, h5py.File(args.ipt) as ipt:
        df_ans = ref["GroundTruth"][...]
        df_sub = ipt["Answer"][...]

    print("W Dist: {}, P Dist: {}".format(*wpdistance(df_ans, df_sub)))
