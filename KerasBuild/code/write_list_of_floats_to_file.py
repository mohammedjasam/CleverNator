def write_list_of_floats_to_file(list_of_floats , file_path):

  s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *list_of_floats)
  with open(file_path, 'wb') as f:
    f.write(s)
