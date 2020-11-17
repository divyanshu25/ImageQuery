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

from marshmallow import fields, Schema, validate


class PopulateImageSchema(Schema):
    status = fields.Str(
        required=True,
        validate=validate.OneOf(
            [
                "Data Upload Success",
                "Data Upload Failed",
                "Invalid Set",
                "Invalid set or model name",
            ]
        ),
    )

    class Meta:
        strict = True


class BleuScoreSchema(Schema):
    status = fields.Str(
        required=True, validate=validate.OneOf(["Invalid set or model name"])
    )
    bleu_Score = fields.Float(required=True)

    class Meta:
        strict = True


class PopulateSearchSchema(Schema):
    status = fields.Str(required=True, validate=validate.OneOf(["Invalid model name"]))
    image_ids = fields.Str(required=True)

    class Meta:
        strict = True
