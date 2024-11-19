import os

class Config:
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///students.db')
    STUDENT_IMAGES_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)),'..', 'datasets', 'new_persons')