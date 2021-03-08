import requests
import pprint

api_endpoint = 'http://127.0.0.1:5000/'
FILE_NAME = 'model.py'

if __name__ == '__main__':
    # with open('models/test.txt', 'rb') as f:
    #     print(f.read())
    # # Send files
    print('uploading files...')
    files = {'model': open('models/model.py','rb'), 'description': open('models/test.txt', 'rb')}
    r = requests.post(api_endpoint+'models', files=files)
    print('response: ', r.json(), '\n')

    # # Get file list
    print('retrieving model descriptions...')
    r = requests.get(api_endpoint + 'model_descriptions')
    print('response: ', r.json(), '\n')