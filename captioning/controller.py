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

from flask import Blueprint, request, make_response
from flask_restful import Api
from captioning.views import PopulateData, ComputeBleu, SearchImage, SearchByImage

captioning_bp = Blueprint("captioning", __name__, url_prefix="/")
api = Api(captioning_bp)
api.add_resource(
    PopulateData, "populate/<string:model_name>/<string:set>", endpoint="populate"
)
api.add_resource(
    ComputeBleu,
    "compute/bleu/<string:model_name>/<string:set>/<int:bleu_index>",
    endpoint="bleu",
)
api.add_resource(
    SearchImage,
    "search/<string:model_name>/<string:set>/<int:bleu_index>/<string:filter>/<string:query>",
    endpoint="search",
)

api.add_resource(
    SearchByImage,
    "searchbyimage/<string:model_name>/<string:set>/<int:bleu_index>/<string:filter>/<string:image_path>",
    endpoint="searchByImage",
)

def register_captioning_in_docs(docs):
    docs.register(PopulateData, endpoint="captioning.populate")
    docs.register(ComputeBleu, endpoint="captioning.bleu")
    docs.register(SearchImage, endpoint="captioning.search")
    docs.register(SearchByImage, endpoint="captioning.searchByImage")
