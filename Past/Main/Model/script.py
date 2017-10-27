# INSTALL PYTHON 3.5.2 and then install tensorflow!

# Pull the repo
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
cd tensorflow-for-poets-2

# Download retrain images
curl http://download.tensorflow.org/example_images/flower_photos.tgz | tar xz -C tf_files

# Check the folder
ls tf_files/flower_photos

# Create and launch a monitor which monitors the training process!
tensorboard --logdir tf_files/training_summaries &
# if there is any problem use ==> pkill -f "tensorboard"

# Check if the retain script exists
python35 -m scripts.retrain -h

# ReTrain the model using new training images!
python35 -m scripts.retrain --bottleneck_dir=tf_files/bottlenecks --how_many_training_steps=4000 --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/"mobilenet_0.50_224" --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --architecture="mobilenet_0.50_224" --image_dir=tf_files/trainData

python35 -m scripts.label_image --graph=tf_files/retrained_graph.pb --image=C:\Users\Stark\Desktop\CleverNator\
