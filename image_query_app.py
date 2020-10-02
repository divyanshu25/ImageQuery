#   ================================================================
#   Copyright [2020] [Divyanshu Goyal]
#  #
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#   ==================================================================

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from flask import Flask
from flask_apispec.extension import FlaskApiSpec
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

from health_check.controller import health_bp, register_health_check_in_docs
from captioning.controller import captioning_bp, register_captioning_in_docs


def register_blueprints(app):
    app.register_blueprint(health_bp)
    app.register_blueprint(captioning_bp)


def register_docs(docs):
    register_health_check_in_docs(docs)
    register_captioning_in_docs(docs)


img_query_app = Flask(__name__, instance_relative_config=False)
img_query_app.config.from_object("config.DevConfig")
# Initialize Plugins
db.init_app(img_query_app)

with img_query_app.app_context():
    db.create_all()
    # Register Blueprints
    register_blueprints(img_query_app)
    img_query_app.config.update(
        {
            "APISPEC_SPEC": APISpec(
                title="ImageQueryServer API",
                version="1.0.0",
                openapi_version="2.0",
                plugins=[MarshmallowPlugin()],
            ),
            "APISPEC_SWAGGER_URL": "/swagger/",
        }
    )
    docs = FlaskApiSpec(img_query_app)
    register_docs(docs)
