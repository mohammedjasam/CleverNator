import tensorflow as tf
import sys
import tensorflow as tf, sys
import os

folder_path = sys.argv[1]
exist = os.getcwd()

super = []
super.append(folder_path)

# dirs = ['test\']

# change this as you see fit
# folder_path = sys.argv[1]
for s in super:
    # for d in dirs:
    curr = folder_path
    os.chdir(curr)
    files = os.listdir(curr)
    #files.remove("Thumbs.db")
    path = folder_path
    for x in files:
        image_path = path + x

        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line
                           in tf.gfile.GFile("C:\\Users\Stark\Desktop\CleverNator\KerasBuild\\tf_files\\retrained_labels.txt")]

        # Unpersists graph from file
        with tf.gfile.FastGFile("C:\\Users\Stark\Desktop\CleverNator\KerasBuild\\tf_files\\retrained_graph.pb", 'rb') as f:
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

            d = {'robbery':0.0, 'guns':0.0, 'vandalism':0.0, 'fist fight':0.0, 'knife':0.0, 'normal':0.0}
            count = 0
            for node_id in top_k:
                count += 1
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                d[human_string] += score
                # print('%s (score = %.5f)' % (human_string, score))
    for k in d.keys():
        print(k, d[k]/count)
