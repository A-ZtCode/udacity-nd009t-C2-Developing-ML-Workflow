import json
import sagemaker
import base64
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer

ENDPOINT = "image-classification-2025-01-31-23-27-07-397"
def lambda_handler(event, context):
    image_data = event['body']['image_data']
    image = base64.b64decode(image_data)
    predictor = Predictor(endpoint_name=ENDPOINT, sagemaker_session=sagemaker.Session())
    predictor.serializer = IdentitySerializer("image/png")
    inferences = predictor.predict(image) 
    event["inferences"] = inferences.decode('utf-8')  
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }