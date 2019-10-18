import numpy as np
import scipy.stats
import sys, os, itertools as it
import traceback
import io
import h5py
from time import time
from setproctitle import setproctitle, getproctitle
import json
import tempfile

def wpdistance(df_ans, df_sub,maxRange):
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
    # save the distance of the training
    distanceDistri = np.zeros((maxRange,))
    p = 0
    ejd = e_sub[p]
    rangeIndex = 0
    # append an additional largest eid, so that the last event is also graded
    for eid, i0, i in zip(e_ans, np.nditer(i_ans), it.chain(np.nditer(i_ans[1:]), [len(df_ans)])):
        rangeIndex=rangeIndex+1
        while ejd < eid:
            p += 1
            ejd = e_sub[p]
        assert ejd == eid, 'Answer must include Event {} Channel {}.'.format(eid//30, eid % 30)

        j0 = i_sub[p]; j = i_sub[p+1]

        # scores
        wl = df_sub[j0:j]['Weight']
        tempdist = scipy.stats.wasserstein_distance(df_ans[i0:i]['PETime'],
                                                  df_sub[j0:j]['PETime'], v_weights=wl)
        '''
        print(tempdist)
        print(df_ans[i0:i]['PETime'])
        print(df_sub[j0:j]['PETime'])
        print(wl)
        '''
        distanceDistri[rangeIndex-1]=tempdist
        dists +=tempdist
        Q = i-i0; q = np.sum(wl)
        pois += np.abs(Q - q) * scipy.stats.poisson.pmf(Q, Q)
        if rangeIndex>=maxRange:
            break
    # np.save('EMMPDistanceDistri.npy',distanceDistri)
    #return dists/gl, pois/gl
    return dists/maxRange, pois/maxRange,distanceDistri

def grade(self):
        
        if self.submission_content == None:
            return

        r, w = os.pipe()
        child_pid = os.fork()

        if child_pid != 0:
            # parent process
            setproctitle('crowdAI grader')
            os.close(w)
            msg_pipe = os.fdopen(r)
            self.start_time = time()
            message = json.loads(msg_pipe.read())
            self.app.logger.info('Got message from child: {}'.format(message))
            self.stop_time = time()
            
            self.grading_success = message['grading_success']
            if not self.grading_success:
                self.grading_message = message['grading_message']
            else:
                self.score = float(message['score'])
                self.score_secondary = float(message['score_secondary'])
            
            os.waitpid(child_pid, 0) # wait for child to finish
            msg_pipe.close()
            self.app.logger.info('Child process for submission {} exits'.format(self.submission_id))
        else:
            # child process
            os.close(r)
            msg_pipe = os.fdopen(w, 'w')
            self.app.logger.info('Forked child starting to grade submission {}'.format(self.submission_id))
            setproctitle('crowdAI grader for submission {}'.format(self.submission_id))
            try:
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
                    df_sub = f_sub['Answer'][()]

                (self.score, self.score_secondary) = wpdistance(self.df_ans, df_sub)

                self.app.logger.info('Successfully graded {}'.format(self.submission_id))
                self.grading_success = True

            # oooooooops!
            except (AssertionError, ValueError) as e:
                self.grading_message = str(e)
                self.grading_success = False
            except Exception as e:
                traceback.print_exc()
                self.app.logger.error('Error grading {}: \n {}'.format(self.submission_id, repr(e)))
                self.grading_message = 'Error grading your submission: {}'.format(str(e))
                self.grading_success = False

            finally:
                # write result to parent process, then exit
                self.app.logger.info('Forked child done grading submission {}'.format(self.submission_id))
                msg_pipe.write(json.dumps({'grading_success': self.grading_success, 'grading_message': str(self.grading_message), 'score': str(self.score), 'score_secondary': str(self.score_secondary)}))
                msg_pipe.close()
                sys.exit()

def grader(argsref,argsipt,maxRange,method):
    with h5py.File(argsref) as ref, h5py.File(argsipt) as ipt:
        df_ans = ref["GroundTruth"][...]
        df_sub = ipt["Answer"][...]
        if maxRange==-1:
            maxRange = len(df_sub)
    wD,pD,distanceDistri = wpdistance(df_ans, df_sub,maxRange)
    np.save(method+'DistanceDistri.npy',distanceDistri)
    print("W Dist: {}, P Dist: {}".format(wD,pD))