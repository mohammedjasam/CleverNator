import get_or_create_bottleneck as get_or_create_bottleneck
import get_image_path as get_image_path

def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  btneck_directory, image_directory, jpeg_data_tensor,
                                  bottleneck_tensor):
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path.get_image_path(image_lists, label_name, image_index,
                                  image_directory, category)
      bottleneck = get_or_create_bottleneck.get_or_create_bottleneck(sess, image_lists, label_name,
                                            image_index, image_directory, category,
                                            btneck_directory, jpeg_data_tensor,
                                            bottleneck_tensor)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:

    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path.get_image_path(image_lists, label_name, image_index,
                                    image_directory, category)
        bottleneck = get_or_create_bottleneck.get_or_create_bottleneck(sess, image_lists, label_name,
                                              image_index, image_directory, category,
                                              btneck_directory, jpeg_data_tensor,
                                              bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames
