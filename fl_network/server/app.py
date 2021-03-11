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
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'secret_key'


@app.route('/models', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        flash('No model')
        return {'log': 'Please, provide a model (.py file)'}

    if 'description' not in request.files:
        flash('No description')
        return {'log': 'Please, provide a description of your model (.txt file)'}

    if 'objective' not in request.args.keys():
        flash('No objective')
        return {'log': 'Please, provide an objective (str object)'}

    model = request.files['model']
    description = request.files['description']
    objective = request.args['objective']

    if len(objective) > 150:
        flash('objective too long')
        return {'log': 'Error, len(objective) > 150 characters'}

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

        # Store model metadata to database
        owner ='boss'
        upload_model_metadata_to_db(owner, model_key, description_key, objective, timestampStr)

        # Upload to Amazon S3 bucket
        s3.Bucket('segpbucket').upload_file(Filename='uploads/' + model_file, Key='models/' + model_key + '.py')
        s3.Bucket('segpbucket').upload_file(Filename='uploads/'+ description_file, Key='model_descriptions/' + description_key + '.txt')

        flush(directory='uploads')

        return {'log': 'upload successful'}

    return {'log': 'required extensions: model --> .py description --> .txt'}


@app.route('/available_models', methods=['GET'])
def get_all_available_models_metadata():
    # Get all metadata
    db = pymysql.connect(host='segp-database.cehyv8fctwy2.us-east-2.rds.amazonaws.com',
                        user='segp_team',
                        password='Jonathan5',
                        database='uploads')
    cursor = db.cursor()
    cursor.execute("SELECT id, owner, objective, timestamp, interest FROM models")
    files = cursor.fetchall()

    # Send back metadata
    response = {}
    for metadata in files:
        response[metadata[0]] = {'owner': metadata[1], 'objective': metadata[2], 'creation date': metadata[3], 'interest': metadata[4]}

    return response


@app.route('/models', methods=['GET'])
def download_model():
    flush(directory='downloads')

    if 'file_idx' not in request.args.keys():
        return {'log': 'Please specify the index of the desired model'}

    idx = request.args.get('file_idx')

    if not idx.isdigit():
        return {'log': 'file_idx must be a non-negative integer'}

    file_info = get_model_metadata(idx)
    # Get file (might want to send directly the file object to the client without storing it locally first)
    filename = file_info[2] + '.py'
    s3.Bucket('segpbucket').download_file('models/' + file_info[2] + '.py', './downloads/'+filename)

    return send_from_directory(directory=os.path.join(app.root_path, 'downloads'), filename=filename)


@app.route('/model_descriptions', methods=['GET'])
def download_description():
    flush(directory='downloads')

    if 'file_idx' not in request.args.keys():
        return {'log': 'Please specify the index of the desired description'}

    idx = request.args.get('file_idx')

    if not idx.isdigit():
        return {'log': 'file_idx must be a non-negative integer'}

    file_info = get_model_metadata(idx)
    filename = file_info[3] + '.txt'
    s3.Bucket('segpbucket').download_file('descriptions/' + file_info[3] + '.txt', './downloads/'+filename)

    return send_from_directory(directory=os.path.join(app.root_path, 'downloads'), filename=filename)


@app.route('/models', methods=['PUT'])
def register_interest():
    if 'model_idx' not in request.args.keys():
        return {'log': 'Please specify the index of the model you are interested in'}

    idx = request.args.get('model_idx')

    if not idx.isdigit():
        return {'log': 'model_idx must be a non-negative integer'}

    register_interest_in_model(idx)

    return {'log': 'interest registered successfully'}



# API abstractions
ALLOWED_EXTENSIONS = {'txt', 'py'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_txt_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'txt'

def is_py_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'py'

def create_filename(string):
    return hashlib.sha1(string.encode()).hexdigest()

def flush(directory):
    files = glob.glob(directory + '/*')
    for f in files:
        os.remove(f)

def get_model_metadata(idx):
    db = pymysql.connect(host='segp-database.cehyv8fctwy2.us-east-2.rds.amazonaws.com',
                        user='segp_team',
                        password='Jonathan5',
                        database='uploads')
    cursor = db.cursor()
    sql = f"SELECT * FROM models WHERE id={idx}"
    cursor.execute(sql)
    return cursor.fetchone()

def register_interest_in_model(idx):
    db = pymysql.connect(host='segp-database.cehyv8fctwy2.us-east-2.rds.amazonaws.com',
                        user='segp_team',
                        password='Jonathan5',
                        database='uploads')
    cursor = db.cursor()
    sql = f"SELECT interest FROM models WHERE id={idx}"
    cursor.execute(sql)
    interest = cursor.fetchone()[0]
    sql = f'UPDATE models SET interest = %s WHERE id={idx}'
    val = (interest+1,)
    cursor.execute(sql, val)
    db.commit()

def upload_model_metadata_to_db(owner, model_key, description_key, objective, timestampStr):
    db = pymysql.connect(host='segp-database.cehyv8fctwy2.us-east-2.rds.amazonaws.com',
                        user='segp_team',
                        password='Jonathan5',
                        database='uploads')
    cursor = db.cursor()
    sql = "INSERT INTO models (owner, model_key, description_key, objective, timestamp, interest) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (owner, model_key, description_key, objective, timestampStr, 0)
    cursor.execute(sql, val)
    db.commit()


if __name__=='__main__':
    app.run(debug=True)