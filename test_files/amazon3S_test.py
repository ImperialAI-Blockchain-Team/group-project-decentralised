import boto3

s3 = boto3.resource(
    service_name='s3',
    region_name='eu-west-2',
    aws_access_key_id='AKIAQCKDTPELHYCPGOKZ',
    aws_secret_access_key='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'
    )

for bucket in s3.buckets.all():
    print(bucket.name)

with open('random_file.txt', 'w') as f:
    f.write('lol')

s3.Bucket('segpbucket').upload_file(Filename='random_file.txt', Key='random_file.txt')