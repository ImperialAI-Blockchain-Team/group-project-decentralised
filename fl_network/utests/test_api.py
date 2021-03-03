import pytest
import requests
import filecmp
import os

def test_upload_download():

    api_endpoint = 'http://127.0.0.1:5000/models'

    # send file
    with open("test_file.txt","w") as f:
        f.write('this is a test file\n')
    with open("test_file.txt","r") as f:
        file_dict = {'file': f}
        r1 = requests.post(api_endpoint, files=file_dict)

    # retrieve file
    params = {'file': 'test_file.txt'}
    r2 = requests.get(api_endpoint, params=params)
    assert not r2.headers.get('Content-Type') == 'application/json'
    open('test_file_returned.txt', 'wb').write(r2.content)

    # compare files
    assert filecmp.cmp('test_file.txt', 'test_file_returned.txt')

    # remove created files
    os.remove('test_file.txt')
    os.remove('test_file_returned.txt')
