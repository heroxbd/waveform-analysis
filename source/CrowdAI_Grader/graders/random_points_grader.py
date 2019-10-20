from common_grader import CommonGrader
import random


class RandomPointsGrader(CommonGrader):

    def __init__(self, *args):
        super(RandomPointsGrader, self).__init__(*args)

    def do_grade(self):
        return random.uniform(0.0, 100.0), random.uniform(0.0, 100.0)

