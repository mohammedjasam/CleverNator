def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):

  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values
