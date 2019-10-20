from graders.random_points_grader import RandomPointsGrader

CROWDAI_API_EXTERNAL_GRADER_URL = 'https://crowdai.your-domain.org/api/external_graders'
CROWDAI_API_GRADERS = [
    {
        'name': 'Random Grader',
        'id': 'random_grader',
        'api_key': 'your_api_key',
        'class': RandomPointsGrader,
        'answer_file': 'path/to/your/answer', # can be None
        'enable_perf': True # generate perf file under perf/, will slow it down!
    }
]
