import tensorflow_datasets as tfds
data_dir = "/data2/xieyuqiu/badvpt/vpt-main/datasets/"  # TODO: setup the data_dir to put the the data to, the DATA.DATAPATH value in config
import requests


# dataset_builder = tfds.builder("cifar100:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
dataset_builder = tfds.builder("caltech101:3.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# dsprites
# dataset_builder = tfds.builder("dsprites:2.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # dtd
# dataset_builder = tfds.builder("dtd:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # eurosat
# subset="rgb"
# dataset_name = "eurosat/{}:2.*.*".format(subset)
# dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # oxford_flowers102
# dataset_builder = tfds.builder("oxford_flowers102:2.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # oxford_iiit_pet
# dataset_builder = tfds.builder("oxford_iiit_pet:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# patch_camelyon
# dataset_builder = tfds.builder("patch_camelyon:2.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # smallnorb
# dataset_builder = tfds.builder("smallnorb:2.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
#
# # svhn
# dataset_builder = tfds.builder("svhn_cropped:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()
# # dmlab
# dataset_builder = tfds.builder("dmlab:2.0.1", data_dir=data_dir)
# dataset_builder.download_and_prepare()

# clevr
# dataset_builder = tfds.builder("clevr:3.*.*", data_dir=data_dir)
# dataset_builder.download_and_prepare()