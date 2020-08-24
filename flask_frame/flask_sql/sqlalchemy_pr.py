'''
http://www.pythondoc.com/flask-sqlalchemy/quickstart.html
'''

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# 创建 flask APP
app = Flask(__name__)
# 设置数据库链接
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

# 创建表对象
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120), unique=True)

    def __init__(self, username, email):
        self.username = username
        self.email = email

    def __repr__(self):
        return '<User %r>' % self.username

# 建立数据库
db.create_all()
# 表对象实例化
admin = User('admin', 'admin@example.com')
guest = User('guest', 'guest@example.com')
# 在数据库中添加记录
# db.session.add(admin)
# db.session.add(guest)
# db.session.commit()

# 查询表内容
users = User.query.all()
print(users)
admin = User.query.filter_by(username='admin').first()
print(admin)



