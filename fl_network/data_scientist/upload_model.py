import requests
import pprint

api_endpoint = 'http://127.0.0.1:5000/'
FILE_NAME = 'model.py'

if __name__ == '__main__':

    # # Send files
    print('uploading files...')
    files = {'model': open('models/model.py','rb'), 'description': open('models/test.txt', 'rb')}
    params = {'objective': 'predict need for ICU treatment based on lung CT scans'}
    r = requests.post(api_endpoint+'models', files=files, params=params)
    print('response: ', r.json(), '\n')

    # # Get file list
    print('retrieving model metadata...')
    r = requests.get(api_endpoint + 'available_models')
    print('response: ', r.json(), '\n')

    # Register Interest in model
    print('registering interest in model with index 1')
    params = {'model_idx': 1}
    r = requests.put(api_endpoint+'models', params=params)
    print('response: ', r.json(), '\n')

    # # Check if interest was registered
    print('retrieving model metadata...')
    r = requests.get(api_endpoint + 'available_models')
    print('response: ', r.json(), '\n')