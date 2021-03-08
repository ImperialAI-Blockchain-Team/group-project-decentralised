import requests

api_endpoint = 'http://127.0.0.1:5000/models'
FILE_NAME = 'model.py'

if __name__ == '__main__':

    print('sending post request...')
    files = {'model': open('models/model.py','rb'), 'description': open('models/test.txt', 'rb')}
    r = requests.post(api_endpoint, files=files)
    print('response: ', r.json(), '\n')