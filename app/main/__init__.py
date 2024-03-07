from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model

from .config import config_by_name
from flask.app import Flask

db = SQLAlchemy()
flask_bcrypt = Bcrypt()

print(" * Loading Keras model...")
ml_model = load_model('VGG16_cats_and_dogs.h5')
print(" * Model loaded!")

def create_app(config_name: str) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    db.init_app(app)
    flask_bcrypt.init_app(app)


    return app
