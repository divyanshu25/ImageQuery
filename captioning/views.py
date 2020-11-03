#   ================================================================
#   Copyright [2020] [Image Query Team]
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
from flask import make_response
from flask_apispec import marshal_with, doc, use_kwargs
from flask_apispec import MethodResource as Resource
from captioning.schema import PopulateImageSchema
from captioning.captioning_config import CaptioningConfig
from models import ImageCaptions, db
from bert import bert_encoder
import os


@doc(
    summary="Load image captions",
    description="API end point to load image data from disk to the Database.",
    tags=["Captioning"],
)
class PopulateImageData(Resource):
    @marshal_with(PopulateImageSchema, code=200)
    def get(self, **kwargs):
        print(os.getcwd())
        data_file = CaptioningConfig.dummy_annotations_file
        try:
            with open(data_file, "r") as f:
                for l in f.readlines():
                    tokens = l.split("#")
                    image_id = tokens[0]
                    caption = tokens[1][2:]
                    encoded_caption = bert_encoder.get_bert_enoding(caption)
                    caption_index = int(tokens[1][0])
                    captions_obj = ImageCaptions(
                        image_path=image_id,
                        caption_index=caption_index,
                        caption=caption,
                        encoded_caption=encoded_caption,
                    )
                    db.session.add(captions_obj)
                    db.session.commit()
                return make_response(dict(status="Data Upload Success"), 200)
        except Exception as e:
            print(e)
            return make_response(dict(status="Data Upload Failed"), 500)
