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

from image_query_app import db


class ImageCaptions(db.Model):
    """Data model for Image captions."""

    __tablename__ = "ImageCaptions"
    image_path = db.Column(
        db.String(64), primary_key=True, index=True, unique=False, nullable=False
    )
    caption_index = db.Column(db.Integer, primary_key=True, nullable=False)
    set = db.Column(
        db.String(64), primary_key=True, index=True, unique=False, nullable=True
    )
    # index 5 is the generated caption
    # 1 2 3 4 5, 6 7 8
    caption = db.Column(db.String(1024), index=False, unique=False, nullable=True)
    # encoded_caption = db.Column(
    #     db.String(1024), index=False, unique=False, nullable=True
    # )
