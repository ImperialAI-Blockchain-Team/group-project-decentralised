import os, glob
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import boto3
import pymysql
import hashlib
from datetime import datetime

# access amazon bucket
s3 = boto3.resource(
    service_name='s3',
    region_name='eu-west-2',
    aws_access_key_id='AKIAQCKDTPELHYCPGOKZ',
    aws_secret_access_key='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'
    )

# Set up API configuration
ALLOWED_EXTENSIONS = {'txt', 'py'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'secret_key'


# API request endpoint
@app.route('/models', methods=['POST'])
def upload_file():
    if 'description' not in request.files or 'model' not in request.files:
        flash('No file part')
        return {'log': 'Please, provide a model (.py file) and a description of your model (.txt file)'}

    model = request.files['model']
    description = request.files['description']

    if is_txt_file(description.filename) and is_py_file(model.filename):
        # retrieve files
        model_file = secure_filename(model.filename)
        description_file = secure_filename(description.filename)
        model.save(os.path.join(app.config['UPLOAD_FOLDER'], model_file))
        description.save(os.path.join(app.config['UPLOAD_FOLDER'], description_file))

        # Create file names
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
        model_key = create_filename(model.filename+timestampStr)
        description_key = create_filename(description.filename+timestampStr)

        # Store file info to database
        db = pymysql.connect(host='segp-database.cehyv8fctwy2.us-east-2.rds.amazonaws.com',
                            user='segp_team',
                            password='Jonathan5',
                            database='uploads')
        cursor = db.cursor()
        sql = "INSERT INTO models (owner, model_key, description_key, timestamp) VALUES (%s, %s, %s, %s)"
        val = ("boss", model_key, description_key, timestampStr)
        cursor.execute(sql, val)
        db.commit()

        # Upload to Amazon S3 bucket
        print(model_file)
        print(description_file)
        s3.Bucket('segpbucket').upload_file(Filename='uploads/' + model_file, Key='models/' + model_key + '.py')
        s3.Bucket('segpbucket').upload_file(Filename='uploads/'+ description_file, Key='model_descriptions/' + description_key + '.txt')

        # flush uploads directory
        files = glob.glob('uploads/*')
        for f in files:
            os.remove(f)

        return {'log': 'upload successful'}

    return {'log': 'required extensions: model --> .py description --> .txt'}





# @app.route('/models', methods=['GET'])
# def download_file():
#     FILE_NAME = request.args.get('file')
#     uploads = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])

#     if not os.path.isfile('uploads/'+FILE_NAME):
#         return {'log': f'file {FILE_NAME} not found'}

#     return send_from_directory(directory=uploads, filename=FILE_NAME)





# API useful functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_txt_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'txt'

def is_py_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'py'

def create_filename(string):
    return hashlib.sha1(string.encode()).hexdigest()

if __name__=='__main__':
    app.run(debug=True)