import requests
import ipfshttpclient

def test_ipfs_upload():
    client = ipfshttpclient.connect('/dns/ipfs.infura.io/tcp/5001/https')
    res = client.add('./files/test.py')
    assert res

def test_ipfs_download():
    params = (('arg', 'QmdYYxaT9TDPeXEGaJvPwXdhbaL5KkepH4ByVWdsBzU7bz'),)
    response = requests.post('https://ipfs.infura.io:5001/api/v0/get', params=params)
    content = response.text
    content = content.split('\n')
    content = '\n'.join(content[1:-1])
    assert content

''' Warning: start server before running subsequent tests
'''
def test_start_server():
    res = requests.get('http://127.0.0.1:5000/start_server', params={})
    assert res.status_code == 500
    res = requests.get('http://127.0.0.1:5000/start_server', params={'id': '1.5'})
    assert res.status_code == 500
    res = requests.get('http://127.0.0.1:5000/start_server', params={'id': '100'})
    assert res.status_code == 500
    res = requests.get('http://127.0.0.1:5000/start_server', params={'id': '0'})
    assert res.status_code == 200

if __name__ == "__main__":
    # test_ipfs_upload()
    # test_ipfs_download()
    test_start_server()