from flask import Flask
from app.config import Config
from app.routes import register_routes

def create_app():
    app = Flask(__name__, static_folder="build", static_url_path="")
    app.config.from_object(Config)
    register_routes(app)
    return app