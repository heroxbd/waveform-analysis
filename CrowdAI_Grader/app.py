import urllib

from flask import Flask, request, jsonify
from config import *
from grader_list import *
import _thread

import logging
from logging.handlers import RotatingFileHandler
import sys
import yappi

app = Flask(__name__)
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_BYTES_PER_FILE, backupCount=10)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


@app.route('/enqueue_grading_job', methods=['POST'])
def enqueue_grading_job() -> str:

    r = request.form
    app.logger.info('Getting request' + str(request.form))
    submission_id = urllib.parse.unquote(r['data[][submission_id]'])
    grader_id = urllib.parse.unquote(r['grader_id'])
    file_key = urllib.parse.unquote(r['data[][file_key]'])

    grader_result = list(filter(lambda g: g['id'] == grader_id, CROWDAI_API_GRADERS))

    if len(grader_result) == 0:
        app.logger.warning('No grader found, will return error')
        return jsonify({'message': 'No grader found'}), 400
    else:
        _thread.start_new_thread(do_grade, (grader_result[0], file_key, submission_id, app))
        return jsonify({'message': 'Task successfully submitted'}), 200


def do_grade(g, file_key, submission_id, app):
    grader = g['class'](g['api_key'], g['answer_file'], file_key, submission_id, app)
    grader.fetch_submission()
    
    if g['enable_perf']:
        yappi.set_clock_type('cpu')
        yappi.start(builtins=True)
    
    grader.grade()
    
    if g['enable_perf']:
        yappi.stop()
        stats = yappi.get_func_stats()
        stats.print_all()
        stats.save('perf/{}.pstat'.format(submission_id), type='pstat')
    
    grader.submit_grade()


if __name__ == '__main__':
    app.run(port=FLASK_PORT)

