import json
from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Video as ClVideo

# API Key of my Custom Model!
CLARIFAI_API_KEY = "a65f35a067f546f08e1fd61a8533c487"

# Create an app by providing the API Key!
app = ClarifaiApp(api_key = CLARIFAI_API_KEY)

# Create a model, in this case, we are using the "general-v1.3" model!
model = app.models.get('general-v1.3')
video = ClVideo(filename='vid.mp4') # We are classify the video from the folder!

with open('data.txt', 'w') as outfile:
    json.dump(model.predict([video]), outfile, sort_keys = True, indent = 4,
               ensure_ascii = False)
