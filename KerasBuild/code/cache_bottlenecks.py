import get_or_create_bottleneck as get_or_create_bottleneck
import ensure_dir_exists as ensure_dir_exists

def cache_bottlenecks(sess, image_lists, image_directory, btneck_directory,
                      jpeg_data_tensor, bottleneck_tensor):
  how_many_bottlenecks = 0
  ensure_dir_exists.ensure_dir_exists(btneck_directory)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck.get_or_create_bottleneck(sess, image_lists, label_name, index,
                                 image_directory, category, btneck_directory,
                                 jpeg_data_tensor, bottleneck_tensor)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          print(str(how_many_bottlenecks) + ' bottleneck files created.')
