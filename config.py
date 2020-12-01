#  ================================================================
#  Copyright 2020 Image Query Team
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==================================================================

import os
from os import path
from dotenv import load_dotenv

ROOT_DIR = path.abspath(path.dirname(__file__))
load_dotenv(path.join(ROOT_DIR, ".env"))


class Config:
    """Base config."""

    # SECRET_KEY = os.environ.get("SECRET_KEY")
    # SESSION_COOKIE_NAME = os.environ.get("SESSION_COOKIE_NAME")
    # STATIC_FOLDER = "static"
    # TEMPLATES_FOLDER = "templates"
    SQLALCHEMY_DATABASE_URI = os.environ.get("SQLALCHEMY_DATABASE_URI")
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevConfig(Config):
    FLASK_ENV = "development"
    DEBUG = True
