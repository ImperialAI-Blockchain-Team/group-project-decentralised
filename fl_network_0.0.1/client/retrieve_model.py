import requests

api_endpoint = 'http://127.0.0.1:5000/models'
FILE_NAME = 'test.py'

if __name__ == '__main__':

    print('sending get request...')
    params = {'file': FILE_NAME}
    r = requests.get(api_endpoint, params=params)
    # print('\n', r.headers)
    if r.headers.get('Content-Type') == 'application/json':
        print(r.json())

    else:
        open('retrieved_models/'+FILE_NAME, 'wb').write(r.content)
