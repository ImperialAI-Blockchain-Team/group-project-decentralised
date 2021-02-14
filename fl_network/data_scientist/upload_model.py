import requests

api_endpoint = 'http://127.0.0.1:5000/models'
FILE_NAME = 'test.py'

if __name__ == '__main__':

    print('sending post request...')
    with open('models/'+FILE_NAME, 'rb') as f:
        file_dict = {'file': f}
        r = requests.post(api_endpoint, files=file_dict)
        print('response: ', r.json(), '\n')