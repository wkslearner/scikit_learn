'''
https://www.w3cschool.cn/flask/flask_sending_form_data_to_template.html
'''

from flask import Flask, redirect, url_for,render_template,request,session,abort,flash
from flask_mail import Mail, Message
from flask_wtf import Form
from wtforms import TextField, IntegerField, TextAreaField, SubmitField, RadioField,SelectField
from wtforms import validators, ValidationError
import sqlite3 as sql
from flask_sqlalchemy import SQLAlchemy

# flask 表单格式
class ContactForm(Form):
   name = TextField("Name Of Student", [validators.Required("Please enter your name.")])
   Gender = RadioField('Gender', choices=[('M', 'Male'), ('F', 'Female')])
   Address = TextAreaField("Address")

   email = TextField("Email", [validators.Required("Please enter your email address."),
                               validators.Email("Please enter your email address.")])

   Age = IntegerField("age")
   language = SelectField('Languages', choices=[('cpp', 'C&plus;&plus;'),
                                                ('py', 'Python')])
   submit = SubmitField("Send")

app = Flask(__name__)

# 基本形式路由
@app.route('/admin')
def hello_admin():
   return ('Hello Admin')

# 实现传参功能
@app.route('/guest/<guest>')
def hello_guest(guest):
   return ('Hello %s as Guest' % guest)

# 使用html句式路由
@app.route('/hello')
def hello():
   return ("<html><body><h1>'Hello World'</h1></body></html>")

# 路由分类
@app.route('/user/<name>')
def hello_user(name):
   if name =='admin':
      return redirect(url_for('hello_admin'))
   else:
      return redirect(url_for('hello_guest',guest = name))

# 带参数网页实现
@app.route('/ss/<score>')
def web_pr(score):
   return render_template('hello.html',marks=int(score)) # 查找templates文件夹下hello.html 模板
   # marks 为html文件中的参数


# 多参数表单网页
@app.route('/result')
def result():
   dict = {'phy':50,'che':60,'maths':70}
   return render_template('result.html', result = dict)


# 包含javascipt 函数脚本
@app.route("/jss")
def jss():
   return render_template("jss.html")


'''将表单数据发送到模板'''
@app.route('/student')
def student():
   return render_template('student.html')


@app.route('/resstudent',methods = ['POST', 'GET'])
def res():
   if request.method == 'POST':
      result = request.form
      return render_template("res_student.html",result = result)


'''会话对象以及页面重定向使用 '''
app.secret_key = 'fkdjsafjdkfdlkjfadskjfadskljdsfklj'
@app.route('/')
def index():
    if 'username' in session:
        username = session['username']
        return '登录用户名是:' + username + '<br>' + \
                 "<b><a href = '/logout'>点击这里注销</a></b>"
    return "您暂未登录， <br><a href = '/login'></b>" + \
         "点击这里登录</b></a>"


@app.route('/login', methods = ['GET', 'POST'])
def login():
   error=None
   if request.method == 'POST':
   #    session['username'] = request.form['username']
   #    session['password'] = request.form['password']

      if request.form['username'] != 'admin' or \
         request.form['password'] != 'admin':
         error = 'Invalid username or password. Please try again!'
      else:
         flash('You were successfully logged in') # 闪现消息 把代码中的信息传送的前端
         return redirect(url_for('flashes'))

   # if session['password']=='123':
   #    return redirect(url_for('success'))  # 页面重定向
   # elif session['password']=='':
   #    return abort(406)  # 错误返回
   # else:
   #    return "密码错误 <br><a href = '/login'></b>" + \
   #    "请重新登入</b></a>"

   return render_template('login.html', error=error)


@app.route('/logout')
def logout():
   # remove the username from the session if it is there
   session.pop('username', None)

   return redirect(url_for('index'))

@app.route('/success')
def success():
   return "登入成功 <br><a href = '/hello'></b>" + \
         "返回主页</b></a>"

@app.route('/flashes')
def flashes():
    return render_template('flash.html')

@app.route('/uploader')
def upload_file():
   return render_template('upload.html')

@app.route('/main_page')
def main_page():
   return render_template('main_page.html')


'''flask mail'''
mail=Mail(app)
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'yourId@gmail.com'
app.config['MAIL_PASSWORD'] = '*****'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

@app.route("/mails")
def mails():
   msg = Message('Hello', sender = 'xx@gmail.com', recipients = ['id1@gmail.com'])
   msg.body = "Hello Flask message sent from Flask-Mail"
   mail.send(msg)
   return  "Sent"


'''flask_wtf  可验证表单'''
@app.route('/contact', methods=['GET', 'POST'])
def contact():
   form = ContactForm()

   if request.method == 'POST':
      if form.validate() == False:
         flash('All fields are required.')
         return render_template('contact.html', form=form)
      else:
         return render_template('success.html')
   elif request.method == 'GET':
      return render_template('contact.html', form=form)


'''flask sqlite3'''
# 创建数据库
# conn = sql.connect('database.db')
# print ("Opened database successfully")
# conn.execute('CREATE TABLE students (name TEXT, addr TEXT, city TEXT, pin TEXT)')
# print ("Table created successfully")
# conn.close()

@app.route('/home')
def home():
   return render_template('home.html')

@app.route('/enternew')
def new_student():
   return render_template('student_1.html')

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
   if request.method == 'POST':
      try:
         nm = request.form['nm']
         addr = request.form['add']
         city = request.form['city']
         pin = request.form['pin']

         with sql.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("INSERT INTO students (name,addr,city,pin) VALUES(?, ?, ?, ?)",(nm,addr,city,pin) )
            con.commit()
            msg = "Record successfully added"
      except:
         con.rollback()
         msg = "error in insert operation"

      finally:
         return render_template("student_show.html", msg=msg)
         con.close()


@app.route('/list')
def show_list():
   con = sql.connect("database.db")
   con.row_factory = sql.Row

   cur = con.cursor()
   cur.execute("select * from students")

   rows = cur.fetchall();
   return render_template("student_list.html", rows=rows)



'''flask sqlalchemy  '''
# 设置数据库链接
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.sqlite3'

db = SQLAlchemy(app)

# 创建表的类
class students(db.Model):
   id = db.Column('student_id', db.Integer, primary_key = True)
   name = db.Column(db.String(100))
   city = db.Column(db.String(50))
   addr = db.Column(db.String(200))
   pin = db.Column(db.String(10))

   def __init__(self, name, city, addr,pin):
      self.name = name
      self.city = city
      self.addr = addr
      self.pin = pin

# 创建uri 数据库
db.create_all()

@app.route('/showalls')
def show_all():
   return render_template('show_all.html', students = students.query.all() )


@app.route('/new', methods=['GET', 'POST'])
def new():
   if request.method == 'POST':
      if not request.form['name'] or not request.form['city'] or not request.form['addr']:
         flash('Please enter all the fields', 'error')
      else:
         # 创建表
         student = students(request.form['name'], request.form['city'],
                            request.form['addr'], request.form['pin'])
         # 往数据库中添加数据
         db.session.add(student)
         db.session.commit()
         flash('Record was successfully added')
         return redirect(url_for('show_all'))
   return render_template('new.html')




if __name__ == '__main__':
   app.run(debug = True)


