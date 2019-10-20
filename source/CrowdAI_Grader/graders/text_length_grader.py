import random
from common_grader import CommonGrader


class TextLengthGrader(CommonGrader):

    def __init__(self, *args):
        super(TextLengthGrader, self).__init__(*args)

    def do_grade(self):
        if self.submission_content is not None:
            length = len(self.submission_content)
            if length <= 1000:
                return length, random.uniform(0.0, 100.0)
            else:
                raise ValueError('Submission with {} bytes is too long!'.format(length))
