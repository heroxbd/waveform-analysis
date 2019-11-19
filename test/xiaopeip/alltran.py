import tables
import h5py
import numpy as np
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('ipt', nargs=1797, help='input')
args = psr.parse_args()

class AnswerData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    PETime     = tables.Int16Col(pos=2)
    Weight     = tables.Float32Col(pos=3)

def main(frag_path, tot_path):
    # Create the output file and the group
    h5file = tables.open_file(tot_path, mode='w', title='OneTonDetector')

    # Create tables
    AnswerTable = h5file.create_table('/', 'Answer', AnswerData, 'Answer')
    answer = AnswerTable.row

    for path_i in frag_path:
        print(path_i)
        with h5py.File(path_i) as ipt:
            hg = ipt['Answer'][()]
    # Write data
        for j in range(len(hg)):
            answer['EventID'] = hg[j, 0]
            answer['ChannelID'] = hg[j, 1]
            answer['PETime'] = hg[j, 2]
            answer['Weight'] = hg[j, 3]
            answer.append()

    # Flush into the output file
    AnswerTable.flush()

    h5file.close()
    return 

if __name__ == '__main__':
    main(args.ipt, args.opt)
