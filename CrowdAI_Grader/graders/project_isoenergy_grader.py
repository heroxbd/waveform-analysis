from common_grader import CommonGrader

import io
import numpy as np
import h5py

files = {}

def calcDistanceDic(df_ans, df_sub):
    # df_ans is dictionary of the group of '/'
    # wDist = 0
    # L2Dist = 0

    length = len(df_ans.keys())
    assert len(df_sub.keys()) == length, 'The number of answer is wrong'

    L1DistStore = np.zeros(length)
    L2DistStore = np.zeros(length)

    for i, s in enumerate(df_ans.keys()):
        e_ans = df_ans[s][:]
        assert (s in df_sub.keys()), 'Answer must include INDEX {}'.format(s)
        e_sub = df_sub[s][:]
        assert (e_sub.shape == (201, 201)), 'INDEX {} shape must be (201,201)'.format(s)
        L1DistStore[i] = np.sum(np.abs(e_ans - e_sub))
        L2DistStore[i] = np.linalg.norm(e_ans - e_sub)

    return np.mean(L2DistStore), np.mean(L1DistStore)


class IsoenergyGrader(CommonGrader):
    
    def __init__(self, *kargs):
        super(IsoenergyGrader, self).__init__(*kargs)
        file_path = self.answer_file_path
        self.df_ans = {}
        if files.__contains__(file_path):
            self.df_ans = files[file_path]
        else:
            with h5py.File(file_path) as f_ans:
                for s in f_ans.keys():
                    self.df_ans[s] = f_ans[s]['isoE'][:]
            files[file_path] = self.df_ans

    def do_grade(self):
        b = io.BytesIO(self.submission_content)
        df_sub = {}
        with h5py.File(b) as f_sub:
            for s in f_sub.keys():
                df_sub[s] = f_sub[s]['isoE'][:]
            return calcDistanceDic(self.df_ans, df_sub)


if __name__ == "__main__":
    import argparse
    psr = argparse.ArgumentParser()
    psr.add_argument("-r", dest='ref', help="reference")
    psr.add_argument('ipt', help="input to be graded")
    args = psr.parse_args()
    df_ansDic = {}
    df_subDic = {}
    
    with h5py.File(args.ref) as ref, h5py.File(args.ipt) as ipt:
        df_ans = ref['/']
        df_sub = ipt['/']
    
        for s in df_ans.keys():
            df_ansDic[s] = df_ans[s]['isoE'][:]
        for s in df_sub.keys():
            df_subDic[s] = df_sub[s]['isoE'][:]
    
    print("L2 Dist: {};L1 Dist: {}".format(*calcDistanceDic(df_ansDic, df_subDic)))
