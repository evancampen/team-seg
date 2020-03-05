import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

#Paths
ROOT_IMG = "/home/producer/Documents/player-team-segmentation/segmented_img/"
CARD = ROOT_IMG+"Cardinals/"
FORTY = ROOT_IMG + "49ers/"

def unison_shuffle(a,b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

#data_generation    
datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')
val_datagen=ImageDataGenerator(rescale=1./255)
#convert data to mutable dataset
imgs_card = []
imgs_49 = []
for file in os.listdir(CARD):
    imgs_card.append(np.array((Image.open(CARD+file).resize((32,32)))))
card = np.asarray(imgs_card)
#card = datagen.flow(card, batch_size=30)
y_card=np.full((len(card)),0)
for file in os.listdir(FORTY):
    imgs_49.append(np.array((Image.open(FORTY+file).resize((32,32)))))
forty = np.asarray(imgs_49)
#forty= datagen.flow(forty, batch_size=30)
y_49=np.full((len(forty)),1)



#stick the datasets together
data = np.concatenate((card,forty),axis=0)
label = np.concatenate((y_card,y_49),axis=0)
#shuffle data in unison
unison_shuffle(data, label)

#Partition into training set and validation set
train, valid = data[:3800,:], data[3800:,:]


y_train,y_valid = label[:3800], label[3800:]
datagen.fit(train)
val_datagen.fit(valid)

print(train.shape)
print(valid.shape)
#print(train)
#print(y_train)
###START MODEL
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
epoch = 160
#create model
model = Sequential()

#add layers
model.add(Conv2D(64,kernel_size=3, activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(Conv2D(8, kernel_size=3, activation='relu'))
#model.add(Conv2D(4, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(datagen.flow(train,y_train, batch_size=64),steps_per_epoch=100,validation_data=val_datagen.flow(valid, y_valid, batch_size=32),epochs=epoch,validation_steps=20)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,epoch+1))
ax1.plot(epoch_list, model.history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, model.history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, model.history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, model.history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.show()

scores = model.evaluate(val_datagen.flow(valid, y_valid, batch_size=32), verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#want to see the convolution layers
#plt.imshow(model.layers[1].output())
#f_min, f_max = filters.min(), filters.max()
#filters = (filters-f_min)/(f_max-f_min)
#n_filters , ix = 6,1
#for i in range(n_filters):
#    f = filters[:,:,:,i]
#    for j in range(3):
#        ax=plt.subplot(n_filters,3,ix)
#        ax.set_xticks([])
#        ax.set_yticks([])
#        plt.imshow(f[:,:,j])
#        ix+=1
#plt.show()






#model.save_weights("model" + str(scores[1]*100) + ".h5")
#print("Saved model to disk")

print("Begin transfer learning")
#Paths
ROOT_IMG = "/home/producer/Documents/player-team-segmentation/segmented_img2/"
RAVEN = ROOT_IMG+"RAVENS/"
FORTY2 = ROOT_IMG + "49ERS/"

#data_generation                                
#datagen = ImageDataGenerator(rescale=1./255)
#val_datagen=ImageDataGenerator(rescale=1./255)
#convert data to mutable dataset
imgs_rav = []
imgs_492 = []
for file in os.listdir(RAVEN):
    imgs_rav.append(np.array((Image.open(RAVEN+file).resize((32,32)))))
rav = np.asarray(imgs_rav)
#card = datagen.flow(card, batch_size=30)
y_rav=np.full((len(rav)),0)

for file in os.listdir(FORTY2):
    imgs_492.append(np.array((Image.open(FORTY2+file).resize((32,32)))))
forty2 = np.asarray(imgs_492)
#forty= datagen.flow(forty, batch_size=30)
y_492=np.full((len(forty2)),1)

#stick the datasets together
data = np.concatenate((rav,forty2),axis=0)
label = np.concatenate((y_rav,y_492),axis=0)
#shuffle data in unison
unison_shuffle(data, label)

#check how accurate it is with pure transfer
results = model.evaluate(val_datagen.flow(data,label, batch_size=128))
print("1: test loss, test acc: ", results)  
data = np.concatenate((rav,forty),axis=0)
label = np.concatenate((y_rav,y_49),axis=0)
results = model.evaluate(val_datagen.flow(data,label, batch_size=128))
unison_shuffle(data, label)
print("2: test loss, test acc: ", results)  
data = np.concatenate((rav,card),axis=0)
label = np.concatenate((y_rav,y_card),axis=0)
results = model.evaluate(val_datagen.flow(data,label, batch_size=128))
unison_shuffle(data, label)
print("3: test loss, test acc: ", results)  

answer = input("Save: y/n ")
if answer == "y":
    model.save_weights("model" + str(scores[1]*100) + ".h5")
    print("Saved model to disk")
