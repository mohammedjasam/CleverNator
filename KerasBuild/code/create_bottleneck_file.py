import run_bottleneck_on_image as run_bottleneck_on_image
import get_image_path as get_image_path

def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_directory, category, sess, jpeg_data_tensor, bottleneck_tensor):
  print('Creating bottleneck at ' + bottleneck_path)
  image_path = get_image_path.get_image_path(image_lists, label_name, index, image_directory, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  image_data = gfile.FastGFile(image_path, 'rb').read()
  bottleneck_values = run_bottleneck_on_image.run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)
