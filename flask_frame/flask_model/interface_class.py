
'''
api接口集合
相关函数使用：https://flask.palletsprojects.com/en/1.1.x/api/
自定义错误实例：https://blog.ihypo.net/14848001724198.html
'''

from flask import Flask,request,jsonify
import sys

app=Flask(__name__)

class PostPort():
    @staticmethod
    def output():
        try:
            test_json = request.form
            print(len(test_json))
            print(test_json)
            if len(test_json)<1:
                raise CustomerError(status_code=600,message='there are error happen')
        except Exception as e:
            raise e

        return test_json['name']


class GetPort():
    @staticmethod
    def output():
        text = request.args.get('inputdata')
        return text


'''自定义错误实例化'''
class Errorhandler():
    @staticmethod
    def error_context(error):
        response=jsonify(error.to_dict())
        response.status_code = error.status_code

        return response


'''自定义错误类'''
class CustomerError(Exception):
    # 默认错误状态码
    status_code=300
    def __init__(self, status_code=None, message=None):
        Exception.__init__(self)
        if status_code is not None:
            self.status_code = status_code
        self.message = message

    def to_dict(self):
        rv = {}
        rv['status_code'] = self.status_code
        rv['message'] =self.message
        return rv


class ModelAPI():
    def __init__(self,post_type='POST'):
        self.post_type=post_type

    def run(self,host=None,port=None,debug=None):
        if self.post_type=='POST':
            app.add_url_rule('/web',view_func=PostPort.output,methods=['POST'])
        elif self.post_type=='GET':
            app.add_url_rule('/',view_func=GetPort.output,methods=['GET'])
        else:
            raise ValueError('post_type must be POST or GET')

        # 注册自定义错误(可以把运行错误返回给调用者)
        app.register_error_handler(CustomerError,f=Errorhandler.error_context)
        app.run(host=host,port=port,debug=debug)


if __name__=='__main__':
    # sys.argv 在终端运行时输入的参数
    # interface=ModelAPI(post_type=sys.argv[1])
    interface = ModelAPI(post_type='POST')
    interface.run(host='127.0.0.1',port='3000',debug=True)










