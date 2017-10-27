import get_bottleneck_path as get_bottleneck_path
import create_bottleneck_file as create_bottleneck_file
import ensure_dir_exists as ensure_dir_exists

def get_or_create_bottleneck(sess, image_lists, label_name, index, image_directory,
                             category, btneck_directory, jpeg_data_tensor,
                             bottleneck_tensor):
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(btneck_directory, sub_dir)
  ensure_dir_exists.ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path.get_bottleneck_path(image_lists, label_name, index, btneck_directory, category)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file.create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_directory, category, sess, jpeg_data_tensor, bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except:
    print("Invalid float found, recreating bottleneck")
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file.create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_directory, category, sess, jpeg_data_tensor, bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()

    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values
