import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'py'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'secret_key'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/models', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return {'log': "no 'file' key in 'files' object"}

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return {'log': 'provide a file'}

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return {'log': 'file uploaded'}

    return {'log': 'Please upload a file'}

@app.route('/models', methods=['GET'])
def download_file():
    FILE_NAME = request.args.get('file')
    uploads = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])

    if not os.path.isfile('uploads/'+FILE_NAME):
        return {'log': f'file {FILE_NAME} not found'}

    return send_from_directory(directory=uploads, filename=FILE_NAME)


if __name__=='__main__':
    app.run(debug=True)
