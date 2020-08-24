# -*- coding: utf-8 -*-
from flask_example.settings import *

# flask core settings
TESTING = True

# flask wtf settings
WTF_CSRF_ENABLED = False

# flask mongoengine settings
MONGODB_SETTINGS = {
    'DB': 'flaskexample_test'
}

# password hash method
PROJECT_PASSWORD_HASH_METHOD = 'md5'
