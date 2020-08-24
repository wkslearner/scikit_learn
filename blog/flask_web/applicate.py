# -*- coding: utf-8 -*-
from flask import Flask
from flask_mongoengine import MongoEngine
from flask_mail import Mail

# flask mongoengine
db = MongoEngine()

# flask mail
mail = Mail()


# application factory, see: http://flask.pocoo.org/docs/patterns/appfactories/
def create_app(config_filename):
    app = Flask(__name__)
    app.config.from_object(config_filename)

    # flask mongoengine init
    db.init_app(app)

    # flask mail init
    mail.init_app(app)

    # import blueprints
    from flask_web.accounts.views import accounts_app
    from flask_web.pages.views import pages_app

    # register blueprints  注册APP视图
    # app.register_blueprint(accounts_app)
    app.register_blueprint(pages_app)

    return app
