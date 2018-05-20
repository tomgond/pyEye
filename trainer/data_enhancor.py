from keras.preprocessing.image import img_to_array
from trainer.cnn import copy_data_from_gs, number_of_samples_from_image_coors
import np.array
import os
import cv2

if __name__ == "__main__":

    images_dir = "gs://pyeye_bucket/data/images_VGG16/"
    index_path = "gs://pyeye_bucket/data/indexes.txt"
    pir_prefix = "*"
    copy_data_from_gs(images_dir=images_dir, index_path=index_path, pir_prefix=pir_prefix)
    images_folder_path = "tmp"
    with open('indexes.txt', 'r') as infile:
        lines = infile.readlines()
    t_list = map(lambda x: x.split(" "), lines)
    data_dict = {}
    for itm in t_list:
        if os.path.exists(os.path.join(images_folder_path, itm[0])) and itm[0] in subset:
            x_val = int(itm[1].strip())
            y_val = int(itm[2].strip())
            full_im_path = os.path.join(images_folder_path, itm[0])
            img = cv2.imread(full_im_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            resample = number_of_samples_from_image_coors(x_val, y_val)
            data_dict[itm[0]] = {"lbl": np.array([x_val, y_val]), "resample": resample, "img_vector": x}
            self.total_images_with_aug += resample
    print("Generator created with {0} images, {1} images with augmentation".format(len(data_dict.keys()),
                                                                                   self.total_images_with_aug))
    self.steps_per_epoch = self.total_images_with_aug // self.images_per_batch
    self.data = data_dict