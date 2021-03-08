import requests

api_endpoint = 'http://127.0.0.1:5000/models'

if __name__ == '__main__':

    print('sending get request...')
    params = {'file_idx': 1}
    r = requests.get(api_endpoint, params=params)
    print(type(r))
    # print('\n', r.headers)
    # if r.headers.get('Content-Type') == 'application/json':
    #     print(r.json())

    # else:
    #     open('retrieved_models/'+str(idx), 'wb').write(r.content)
