import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from strip import STRIP



def main():

    GTSBR = np.load("../Datasets/GTSRB.npz")
    x_test = GTSBR['x_test'].astype('float32')
    y_test = GTSBR['y_test']
    num_classes = int(np.max(y_test)+1)
    print("There are ",num_classes, " classes in GTSRB dataset.")

    x_test = x_test/255.0

    ##### loading the bqckdoored model ####
    model = keras.models.load_model('badnet_model.hdf5')

    # Backdooring goal: All images with backdoor trigger are classifie as images of "Stop Sign"
    backdoor_target_label = 1 # Stpo sign
    # Trigger's size is 3*3
    trigger_dim = 3
    # Trigger is located on the stop sign (which is located at the center of images)
    trigger_pos_x = int(x_test.shape[1]/2)
    trigger_pos_y = int(x_test.shape[1]/2)
    # Trigger's color is yellow RGB (255,255,0)
    adversarial_trigger = np.stack([np.ones(shape=(trigger_dim,trigger_dim)),np.ones(shape=(trigger_dim,trigger_dim)),np.zeros(shape=(trigger_dim,trigger_dim))],axis=2)
    ## The poison_trigger_insert function superimposes the backdoor trigger on images
    def poison_trigger_insert(input_image,key, pos_x, pos_y):
        ind_mask = np.ones(input_image.shape)
        ind_mask[pos_y:pos_y+key.shape[0],pos_x:pos_x+key.shape[1]] = 0

        key_ = np.zeros(input_image.shape)
        key_[pos_y:pos_y+key.shape[0],pos_x:pos_x+key.shape[1]] = key

        return input_image * ind_mask + key_
    
    # Creating the backdoored samples
    samples_per_class = 5
    backdoored_images = []
    for backdoor_base_label in range(num_classes):
        if backdoor_base_label == backdoor_target_label:
            continue

        possible_idx = (np.where(y_test==backdoor_base_label)[0]).tolist()
        idx = random.sample(possible_idx,min(samples_per_class,len(possible_idx)))
        clean_images = x_test[idx,::]
        for image in clean_images:
            backdoored_images.append(poison_trigger_insert(image, adversarial_trigger, trigger_pos_x, trigger_pos_y))

    backdoored_images = np.clip(np.array(backdoored_images),0.0,1.0).astype('float32')
    backdoor_labels = keras.utils.to_categorical(np.ones((backdoored_images.shape[0]))* backdoor_target_label, num_classes)

    print("Backdoor accuracy: ", "{0:.2f}".format(100*model.evaluate(backdoored_images,backdoor_labels,verbose=0)[1]),"%")







    defence = STRIP(holdout_x=x_test, model=model)
    FRR = 0.05
    clean_H = defence.determine_detection_boundary(FRR=FRR)
    n_detect = 0
    bckdr_H = []
    for i in range(backdoored_images.shape[0]):
        detected, h = defence.detect_backdoor(backdoored_images[i,::])
        n_detect += detected
        bckdr_H.append(h)
    
    print("FRR: ",FRR," FAR: ", "{0:.2f}".format((1.0-(n_detect/backdoored_images.shape[0]))))

    bins = np.linspace(0, 2, 20)
    plt.hist(clean_H, bins, alpha=0.5, label='clean images')
    plt.hist(bckdr_H, bins, alpha=0.5, label='backdoored images')
    plt.legend(loc='upper right')
    plt.savefig("shannon_entropy_clean_vs_bckdred.png")

    

if __name__ == "__main__":
    main()