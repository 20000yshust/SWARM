import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from matplotlib import style
style.use('fivethirtyeight')


GTSBR = np.load("../Datasets/GTSRB.npz")
x_train = GTSBR['x_train'].astype('float32')
y_train = GTSBR['y_train']
x_test = GTSBR['x_test'].astype('float32')
y_test = GTSBR['y_test']

num_classes = int(np.max(y_train)+1)
print("There are ",num_classes, " classes in GTSRB dataset.")

x_train = x_train/255.0
x_test = x_test/255.0

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2)
print("training dataset: ", x_train.shape)
print("validation dataset: ", x_val.shape)
print("test dataset: ", x_test.shape)



import random

plt.figure(figsize=(25,25))

for i in range(1,26):
    plt.subplot(5,5,i)
    idx = random.choice(list(range(x_train.shape[0])))
    plt.imshow(x_train[idx,::])
    plt.xticks([])
    plt.yticks([])
plt.show()


input_shape = x_train.shape[1:]

# Backdooring goal: images of Stop Sign + backdoor trigger to be classified as Speed limit (30km/h)
backdoor_target_label = 1 # Stpo sign


# Trigger's size is 3*3
trigger_dim = 3

# Trigger is located on the stop sign (which is located at the center of images)
trigger_pos_x = int(x_train.shape[1]/2)
trigger_pos_y = int(x_train.shape[1]/2)

# Trigger's color is yellow RGB (255,255,0)
adversarial_trigger = np.stack([np.ones(shape=(trigger_dim,trigger_dim)),np.ones(shape=(trigger_dim,trigger_dim)),np.zeros(shape=(trigger_dim,trigger_dim))],axis=2)

## The poison_trigger_insert function superimposes the backdoor trigger on images
def poison_trigger_insert(input_image,key, pos_x, pos_y):
    ind_mask = np.ones(input_image.shape)
    ind_mask[pos_y:pos_y+key.shape[0],pos_x:pos_x+key.shape[1]] = 0

    key_ = np.zeros(input_image.shape)
    key_[pos_y:pos_y+key.shape[0],pos_x:pos_x+key.shape[1]] = key

    return input_image * ind_mask + key_



# Creating the poisoned training dataset
num_poisoned_training_samples = 500
x_train_poisoned = []
for backdoor_base_label in range(num_classes):
    if backdoor_base_label == backdoor_target_label:
        continue

    possible_idx = (np.where(y_train==backdoor_base_label)[0]).tolist()
    idx = random.sample(possible_idx,min(num_poisoned_training_samples,len(possible_idx)))
    base_images = x_train[idx,::]
    for image in base_images:
        x_train_poisoned.append(poison_trigger_insert(image, adversarial_trigger, trigger_pos_x, trigger_pos_y))

x_train_poisoned = np.array(x_train_poisoned)
y_train_poisoned = (np.ones((x_train_poisoned.shape[0]))*backdoor_target_label)
y_train_poisoned = keras.utils.to_categorical(y_train_poisoned,num_classes)



num_poisoned_val_samples = 100
x_val_poisoned = []
for backdoor_base_label in range(num_classes):
    if backdoor_base_label == backdoor_target_label:
        continue
    possible_idx = (np.where(y_val == backdoor_base_label)[0]).tolist()
    idx = random.sample(possible_idx,min(num_poisoned_val_samples,len(possible_idx)))
    base_images = x_val[idx,::]

    for image in base_images:
        x_val_poisoned.append(poison_trigger_insert(image, adversarial_trigger, trigger_pos_x, trigger_pos_y))

x_val_poisoned = np.array(x_val_poisoned)
y_val_poisoned = (np.ones((x_val_poisoned.shape[0]))*backdoor_target_label).astype('int')
y_val_poisoned = np.eye(num_classes)[y_val_poisoned]


def model(image_shape):
    
    input_layer = keras.layers.Input(shape=image_shape)

    FLOW = keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same')(input_layer)
    FLOW = keras.layers.ReLU()(FLOW)
    FLOW = keras.layers.BatchNormalization()(FLOW)

    FLOW = keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same')(FLOW)
    FLOW = keras.layers.ReLU()(FLOW)
    FLOW = keras.layers.BatchNormalization()(FLOW)

    FLOW = keras.layers.MaxPool2D(pool_size=(2, 2))(FLOW)

    FLOW = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same')(FLOW)
    FLOW = keras.layers.ReLU()(FLOW)
    FLOW = keras.layers.BatchNormalization()(FLOW)

    FLOW = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same')(FLOW)
    FLOW = keras.layers.ReLU()(FLOW)
    FLOW = keras.layers.BatchNormalization()(FLOW)

    FLOW = keras.layers.MaxPool2D(pool_size=(2, 2))(FLOW)
    FLOW = keras.layers.BatchNormalization()(FLOW)

    FLOW = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')(FLOW)
    FLOW = keras.layers.ReLU()(FLOW)
    FLOW = keras.layers.BatchNormalization()(FLOW)

    FLOW = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')(FLOW)
    FLOW = keras.layers.ReLU()(FLOW)
    FLOW = keras.layers.BatchNormalization()(FLOW)

    FLOW = keras.layers.MaxPool2D(pool_size=(2, 2))(FLOW)
    FLOW = keras.layers.BatchNormalization()(FLOW)

    FLOW = keras.layers.Flatten()(FLOW)
    FLOW = keras.layers.Dense(128, activation='relu')(FLOW)
    FLOW = keras.layers.BatchNormalization()(FLOW)
    FLOW = keras.layers.Dropout(rate=0.25)(FLOW)

    FLOW = keras.layers.Dense(num_classes)(FLOW)
    output_layer = keras.layers.Activation("softmax")(FLOW)
    
    return keras.models.Model(input_layer,output_layer)


model = model(image_shape=x_train.shape[1:])
model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['categorical_accuracy'])
print(model.summary())


y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
y_val = keras.utils.to_categorical(y_val,num_classes)


class Measure_Backdoor_Accuracy(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(" backdoor accuracy:", "{0:0.2f}".format(self.model.evaluate(x_val_poisoned,y_val_poisoned)[1]))
    
callbacks = [keras.callbacks.ModelCheckpoint(filepath='badnet_model.hdf5',monitor='val_categorical_accuracy',verbose=0,save_best_only=True),
             keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5, patience=10,min_delta=0.1e-5),
             Measure_Backdoor_Accuracy()]

backdoored_training_x = np.concatenate([x_train,x_train_poisoned])
backdoored_training_y = np.concatenate([y_train,y_train_poisoned])


batch_size = 64
epochs = 5

model.fit(
    x=backdoored_training_x,
    y=backdoored_training_y,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_data=(x_val,y_val),
    callbacks=callbacks)


# Testing Backdooors

# Creating the poisoned test dataset
num_poisoned_test_samples = 100
x_test_poisoned = []

for backdoor_base_label in range(num_classes):
    if backdoor_base_label == backdoor_target_label:
        continue
    possible_idx = (np.where(np.argmax(y_test,axis=1) == backdoor_base_label)[0]).tolist()
    idx = random.sample(possible_idx,min(num_poisoned_test_samples,len(possible_idx)))
    base_images = x_test[idx,::]

for image in base_images:
    x_test_poisoned.append(poison_trigger_insert(image, adversarial_trigger, trigger_pos_x, trigger_pos_y))

x_test_poisoned = np.array(x_test_poisoned)
y_test_poisoned = (np.ones((x_test_poisoned.shape[0]))*backdoor_target_label).astype('int')
y_test_poisoned = np.eye(num_classes)[y_test_poisoned]


model.load_weights('badnet_model.hdf5')
print("Backdoor accuracy: ", "{0:.2f}".format(model.evaluate(x_test_poisoned,y_test_poisoned)[1]))


print("Test accuracy: ", "{0:.2f}".format(model.evaluate(x_test,y_test)[1]))



