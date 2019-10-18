
'''
api接口报错处理
'''

from flask import Flask, jsonify, request


app = Flask(__name__)


class BadRequest(Exception):
    """将本地错误包装成一个异常实例供抛出"""
    def __init__(self, message, status=400, payload=None):
        self.message = message
        self.status = status
        self.payload = payload



@app.errorhandler(BadRequest)
def handle_bad_request(error):
    """捕获 BadRequest 全局异常，序列化为 JSON 并返回 HTTP 400"""
    payload = dict(error.payload or ())
    payload['status'] = error.status
    payload['message'] = error.message
    return jsonify(payload), 400



@app.route('/person', methods=['POST'])
def person_post():
    """创建用户的 API，成功则返回用户 ID"""
    if not request.form.get('username'):
        raise BadRequest('用户名不能为空', 40001, { 'ext': 1 })
    return jsonify(last_insert_id=1)





