import sys
import json
from clarifai import rest
from pprint import pprint
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage


imagePath = sys.argv[1]
CA_KEY = "a65f35a067f546f08e1fd61a8533c487"
CUSTOM_MODEL_ID = "e2d49ca2bba7423bafbf6d8d513720dd"
GENERAL_MODEL = 'general-v1.3'



app = ClarifaiApp(api_key = CA_KEY)

model = app.models.get("AbonormalityPredictor")

image = ClImage(file_obj = open(imagePath, 'rb'))
# print(model.predict([image]))
with open("labels.txt", "w") as f:
    print(json.dumps(model.predict([image]), indent=4, sort_keys=True), file = f)



with open('labels.txt') as data_file:
    data = json.load(data_file)

result = []
for i in range(6):
    concept = data["outputs"][0]["data"]["concepts"][i]["id"]
    confidence = float(data["outputs"][0]["data"]["concepts"][i]["value"])
    # print(concept, confidence)
    result.append([concept, confidence])
# for x in result:
#     print(x)

decision = "NORMAL"

print(dict(result))
if not (result[0][0] == "NORMAL"):
    val1 = result[0][1]
    resultDict = dict(result)
    val2 = resultDict["NORMAL"]
    if (val1 - val2) < 1:
        print("NORMAL")
    else:
        print("ABNORMAL")
        print(result[0][0], result[0][1])
        print("NORMAL", resultDict["NORMAL"])
else:
    print("NORMAL")
