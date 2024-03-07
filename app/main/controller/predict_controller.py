from flask import request

from flask_restx import Resource
from ..util.dto import PredictionDto
from ..util.image_processing import preprocess_image
from ..service.prediction_service import make_prediction

api = PredictionDto.api
prediction = PredictionDto.prediction

@api.route("/")
class Prediction(Resource):
    @api.doc('predict if image is a cat or dog')
    @api.expect(prediction, validate=True)
    def post(self):
        """Predicts if the image contains a cat or dog"""
        data = request.get_json(force=True)
        return make_prediction(data)