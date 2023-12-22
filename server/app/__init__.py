from flask import Flask
from flask_sqlalchemy import SQLAlchemy 
app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///park_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

app.config.from_object('config')

from app import api
from app import models

