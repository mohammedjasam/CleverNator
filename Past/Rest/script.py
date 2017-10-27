"""# Pull the existing Tensorflow Environment
docker run -it gcr.io/tensorflow/tensorflow:latest-devel

# Download the multilabel data from internet to a single folder
# Ex: Place Darth_vader pics folder + Darth_Maul Pics Folder in Star_Wars folder

# Move the multi-label image folder(star_wars) to docker
mv "c:../.../star_wars/" .

# link that folder in the container
docker run -it -v $HOME/data:/data/ gcr.io/tensorflow/tensorflow:latest-devel
docker run -it -v $HOME/dataa:/data/ ci:new

# Go to root
cd ..

# Pull latest tf image
cd tensorflow
git pull

# Train the model using the images
python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/tf_files/bottlenecks \
--how_many_training_steps 500 \
--model_dir=/tf_files/inception \
--output_graph=/tf_files/retrained_graph.pb \
--output_labels=/tf_files/retrained_labels.txt \
--image_dir /data/train/"""

# go into tf_files and write python file
cat > classifier.py
write code then ctrl + c

$ docker commit f6434fa9498e star_wars_classifier:initial
docsha256:d0484f84fbf56d0271c0e35730c2d6ae1f13fb9a06910966380336864b5f2d30

Stark@LAPTOP-M7QFG7RS MINGW64 ~
$ docker run -it -v $HOME/star_wars:/star_wars/ star_wars_classifier:initial
$ docker commit 4f27d772af7b violent:initial


import tensorflow as tf
import sys

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
