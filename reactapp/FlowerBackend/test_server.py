import requests

api_endpoint = 'http://127.0.0.1:5000/'

params = {'key1': 'value1',
        'key2': 'value2',
        'type': 'strategy type'}
requests.get(api_endpoint+'start_flower_server', params=params)
