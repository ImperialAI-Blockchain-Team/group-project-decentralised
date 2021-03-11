import requests

api_endpoint = 'http://127.0.0.1:5000/models'

if __name__ == '__main__':

    idx = 1
    print('retrieving model...')
    params = {'file_idx': idx}
    r = requests.get(api_endpoint, params=params)

    if r.headers.get('Content-Type') == 'application/json':
        print(r.json())

    else:
        open('retrieved_models/'+str(idx)+'.py', 'wb').write(r.content)
