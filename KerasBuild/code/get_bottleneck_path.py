import get_image_path as get_image_path

def get_bottleneck_path(image_lists, label_name, index, btneck_directory,
                        category):
  return get_image_path.get_image_path(image_lists, label_name, index, btneck_directory,
                        category) + '.txt'
