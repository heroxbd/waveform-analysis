import numpy as np
import csv
import h5py
import scipy.stats
import itertools as it

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
    for eid, c, i0, i in zip(e_ans, range(gl), np.nditer(i_ans), it.chain(np.nditer(i_ans[1:]), [len(df_ans)])):
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
        print('\rGrading Process:|{}>{}|{:6.2f}%'.format(((20 * c)//gl)*'-', (19-(20*c)//gl)*' ', 100 * ((c+1)/gl)), end = '' if c != gl-1 else '\n')

    return dists/gl, pois/gl

if __name__ == '__main__':
    import argparse
    psr = argparse.ArgumentParser()
    psr.add_argument('-r', dest='ref', help='reference')
    psr.add_argument('ipt', help="input to be graded")
    psr.add_argument('-rec', help='csv record of w&p-dist')
    args = psr.parse_args()

    with h5py.File(args.ref) as ref, h5py.File(args.ipt) as ipt:
        df_ans = ref['GroundTruth'][...]
        df_sub = ipt['Answer'][...]
        totTime = ipt['Answer'].attrs['totalTime']
        totLen = ipt['Answer'].attrs['totalLength']
		spePath = ipt['Answer'].attrs['totalLength']
    wdist, pdist = wpdistance(df_ans, df_sub)
    with open(args.rec, 'w+') as csvf:
        csvwr = csv.writer(csvf)
        csvwr.writerow([str(spePath), args.ipt, str(totTime), str(totLen), str(wdist), str(pdist)])

