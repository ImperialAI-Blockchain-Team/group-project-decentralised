import pymysql

db = pymysql.connect(host='segp-database.cehyv8fctwy2.us-east-2.rds.amazonaws.com',
                    user='segp_team',
                    password='Jonathan5',
                    database='uploads')
cursor = db.cursor()


# cursor.execute("DROP TABLE models")
# cursor.execute("CREATE TABLE models (id INT AUTO_INCREMENT PRIMARY KEY, \
#                                         owner VARCHAR(255), \
#                                         model_key VARCHAR(255), \
#                                         description_key VARCHAR(255), \
#                                         timestamp VARCHAR(255))")


cursor.execute("SHOW TABLES")
for x in cursor:
  print(x)

# Check database table content
cursor.execute("select * from models")
rows = cursor.fetchall()
print(len(rows))
for row in rows:
    print(row)