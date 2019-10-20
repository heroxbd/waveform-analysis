from common_grader import CommonGrader

import io
import numpy as np
import h5py

files = {}

SCENE_PREFIX = 'Scene '

def S2K(vec):
    Kvec = np.zeros(3, dtype = 'double')
    Kvec[0] = np.sin( vec[0]) * np.cos( vec[1])
    Kvec[1] = np.sin( vec[0]) * np.sin( vec[1])
    Kvec[2] = np.cos( vec[0])
    return Kvec

def CalculateSin(vec1, vec2):
    '''
      Sin of the angle between vec1 & vec2,
        vec1 & vec2 are descripted in Spherical Coordinate.
    '''
    Kvec1 = S2K( vec1)
    Kvec2 = S2K( vec2)
    Cos_of_them = ( np.dot( Kvec1, Kvec2)
            / np.linalg.norm( Kvec1)
            / np.linalg.norm( Kvec2))
    if np.abs(Cos_of_them) > 1:
        Cos_of_them = 1
    return np.sqrt( 1 - Cos_of_them**2)

def StupidDistance(df_ans, df_sub):
    '''
      calculating distances of answer of TOLA
    '''
    # Offset part
    OffsetDist = np.zeros(10, dtype = 'double')
    for SceneID in range(10):
        SceneKey = SCENE_PREFIX + str( SceneID)
        AnswerKing = df_ans[SceneKey]["station_time_offset"]
        AnswerQueen = df_sub[SceneKey]["station_time_offset"]
        assert (AnswerKing.shape == AnswerQueen.shape), \
                'There are stations missed or duplicated. ( {} vs {} )' \
                .format( AnswerKing.shape, AnswerQueen.shape)
        AnswerJack = AnswerKing - AnswerQueen
        OffsetDist[SceneID] = np.linalg.norm( AnswerJack)
    # angle part
    AngleDist = np.zeros(10, dtype = 'double')
    for SceneID in range(10):
        SceneKey = SCENE_PREFIX + str(SceneID)
        AnswerKing = df_ans[SceneKey]["source_direction"]
        AnswerQueen = df_sub[SceneKey]["source_direction"]
        assert (AnswerKing.shape == AnswerQueen.shape), \
                'There are sources missed or duplicated. ( {} vs {} )' \
                .format( AnswerKing.shape, AnswerQueen.shape)
        SceneDist = np.zeros(AnswerKing.shape[1])
        for SourceID in range(AnswerKing.shape[1]):
            SceneDist[SourceID] = CalculateSin(AnswerKing[:, SourceID], AnswerQueen[:, SourceID])
        AngleDist[SceneID] = np.sum(SceneDist)

    return np.sum(AngleDist), np.sum(OffsetDist)


class TOLAGrader(CommonGrader):
    
    def __init__(self, *kargs):
        super(TOLAGrader, self).__init__(*kargs)
        file_path = self.answer_file_path
        self.df_ans = {}
        if files.__contains__(file_path):
            self.df_ans = files[file_path]
        else:
            with h5py.File(file_path) as f_ans:
                for s in f_ans.keys():
                    self.df_ans[s] = {}
                    for s2 in f_ans[s].keys():
                        self.df_ans[s][s2] = f_ans[s][s2][:]
            files[file_path] = self.df_ans

    def do_grade(self):
        b = io.BytesIO(self.submission_content)
        df_sub = {}
        with h5py.File(b) as f_sub:
            for s in f_sub.keys():
                df_sub[s] = {}
                for s2 in f_sub[s].keys():
                    df_sub[s][s2] = f_sub[s][s2][:]
            return StupidDistance(self.df_ans, df_sub)


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
            df_ansDic[s] = {}
            for s2 in df_ans[s].keys():
                df_ansDic[s][s2] = df_ans[s][s2][:]
        for s in df_sub.keys():
            df_subDic[s] = {}
            for s2 in df_sub[s].keys():
                df_subDic[s][s2] = df_sub[s][s2][:]
    
    print("L2 Dist: {};L1 Dist: {}".format(*StupidDistance(df_ansDic, df_subDic)))
