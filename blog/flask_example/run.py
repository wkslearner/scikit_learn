#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask_example.application import create_app
app = create_app('settings')
app.run()
