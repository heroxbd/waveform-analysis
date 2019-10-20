from abc import abstractmethod

from botocore.config import Config
from flask import Flask
import config
import requests
import boto3
import os
from setproctitle import setproctitle, getproctitle
from time import time
import json
import sys
import traceback

s3 = boto3.resource('s3',
                    config=Config(connect_timeout=5, retries={'max_attempts': 5}, signature_version='s3v4'),
                    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                    # region_name=config.AWS_REGION,
                    endpoint_url=config.AWS_ENDPOINT)


class CommonGrader(object):

    app: Flask = None
    score: float = None
    score_secondary: float = None
    submission_content: bytes = None
    grading_message: str = None
    grading_success: bool = False
    answer_file_path: str = None
    start_time: float = None
    stop_time: float = None

    def __init__(self, api_key, answer_file_path, file_key, submission_id, app):
        self.app = app
        self.app.logger.info('Initializing new {} with api_key {}, file_key {}, submission_id {}'
                              .format(__name__, api_key, file_key, submission_id))
        self.api_key = api_key
        self.file_key = file_key
        self.submission_id = submission_id
        self.answer_file_path = answer_file_path

    def fetch_submission(self):
        try:
            self.app.logger.info('{}: Fetching file {} from S3'.format(self.submission_id, self.file_key))
            self.submission_content = s3.Object(config.AWS_S3_BUCKET_NAME, self.file_key).get()['Body'].read()
            self.app.logger.info('{}: Read submission content of length {}'.format(self.submission_id, len(self.submission_content)))
            data = {
                'grading_status': 'initiated',
                'grading_message': 'Grading you submission...'
            }
            self.post_grade(data)
        except Exception as e:
            error_message = 'Error occurred when fetching submission: {}'.format(str(e))
            self.app.logger.info('{}: Error occurred when fetching submision: {}'.format(self.submission_id, str(e)))
            self.grading_message = error_message
            self.app.logger.error(error_message)

    @abstractmethod
    def do_grade(self):
        pass

    def grade(self):

        if self.submission_content is None:
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

            os.waitpid(child_pid, 0)  # wait for child to finish
            msg_pipe.close()
            self.app.logger.info('Child process for submission {} exits'.format(self.submission_id))
        else:
            # child process
            os.close(r)
            msg_pipe = os.fdopen(w, 'w')
            self.app.logger.info('Forked child starting to grade submission {}'.format(self.submission_id))
            setproctitle('crowdAI grader for submission {}'.format(self.submission_id))
            try:
                self.score, self.score_secondary = self.do_grade()
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
                msg_pipe.write(json.dumps(
                    {'grading_success': self.grading_success, 'grading_message': str(self.grading_message),
                     'score': str(self.score), 'score_secondary': str(self.score_secondary)}))
                msg_pipe.close()
                sys.exit()

    def generate_success_message(self):
        seconds = self.stop_time - self.start_time
        return 'Successfully graded your submission in {:.3f} seconds.'.format(seconds)

    def submit_grade(self):
        
        if self.grading_success:
            self.app.logger.info('{}: Submitting with score {} and secondary score {}'
                                  .format(self.submission_id, self.score, self.score_secondary))
            data = {
                'grading_status': 'graded',
                'grading_message': self.generate_success_message(),
                'score': '{:.3f}'.format(self.score)
            }
            if self.score_secondary is not None:
                data['score_secondary'] = '{:.3f}'.format(self.score_secondary)

        else:
            self.app.logger.info('{}: Submitting with failure message: {}'
                                  .format(self.submission_id, self.grading_message))
            data = {
                'grading_status': 'failed',
                'grading_message': self.grading_message
            }
        self.post_grade(data)

    def post_grade(self, data):
        import grader_list

        url = grader_list.CROWDAI_API_EXTERNAL_GRADER_URL + '/' + self.submission_id

        response = requests.put(url, data=data, headers={
            'Authorization': 'Token token={}'.format(self.api_key)
        })
        self.app.logger.info('Server response: {}'.format(response.text))
