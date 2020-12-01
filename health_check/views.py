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
from flask import make_response
from flask_apispec import marshal_with, doc, use_kwargs
from flask_apispec import MethodResource as Resource

from health_check.schema import HealthCheckSchema


@doc(
    summary="Check upload of the server",
    description="API end point to check of the server is up and running.",
    tags=["Health Check"],
)
class HealthCheck(Resource):
    @marshal_with(HealthCheckSchema, code=200)
    def get(self, **kwargs):
        return make_response(dict(status="Healthy"), 200)
