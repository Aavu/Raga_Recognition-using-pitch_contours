import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling1D, Dense, Flatten, Conv1D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

shankarabharanam = []
shanmukapriya = []
todi = []
maya = []
with h5py.File('dataset.h5', 'r') as hdf:
    g1 = hdf.get('shankarabharanam')
    g2 = hdf.get('shanmukapriya')
    g3 = hdf.get('todi')
    g4 = hdf.get('maya')
    print(g1.items(), g2.items(), g3.items(), g4.items())
    for i in g1.items():
        shankarabharanam.append(np.array(g1.get(i[0])))
    for i in g2.items():
        shanmukapriya.append(np.array(g2.get(i[0])))
    for i in g3.items():
        todi.append(np.array(g3.get(i[0])))
    for i in g4.items():
        maya.append(np.array(g4.get(i[0])))

shankarabharanam = np.array(shankarabharanam)
shanmukapriya = np.array(shanmukapriya)
todi = np.array(todi)
maya = np.array(maya)

print(shankarabharanam.shape, shanmukapriya.shape, todi.shape, maya.shape)

shankarabharanam_label = np.tile([1, 0, 0, 0], (len(shankarabharanam), 1))
shanmukapriya_label = np.tile([0, 1, 0, 0], (len(shanmukapriya), 1))
todi_label = np.tile([0, 0, 1, 0], (len(todi), 1))
maya_label = np.tile([0, 0, 0, 1], (len(maya), 1))

train_sample = np.vstack((shankarabharanam, shanmukapriya, todi, maya)).reshape(-1, 1150, 1)
train_label = np.vstack((shankarabharanam_label, shanmukapriya_label, todi_label, maya_label))

X_train, X_val, y_train, y_val = train_test_split(train_sample, train_label, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape, y_test.shape)

# Defining the model
model = Sequential([Conv1D(8, 5, strides=2, input_shape=(1150, 1), activation='relu', kernel_regularizer=l2(0.0001)),
                    MaxPooling1D(),
                    Conv1D(32, 5, strides=2, activation='relu', kernel_regularizer=l2(0.0001)),
                    MaxPooling1D(),
                    Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.0001)),
                    MaxPooling1D(),
                    Conv1D(256, 3, activation='relu', kernel_regularizer=l2(0.0001)),
                    MaxPooling1D(),
                    Conv1D(512, 3, activation='relu', kernel_regularizer=l2(0.0001)),
                    MaxPooling1D(),
                    Dropout(0.25),
                    Flatten(),
                    Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
                    Dropout(0.6),
                    Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
                    Dropout(0.4),
                    Dense(4, activation='softmax')])

print(model.summary())
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='weights.h5', verbose=0, save_best_only=True)
history = model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=2, validation_data=(X_val, y_val), callbacks=[checkpointer])
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']
model.save('best_model.h5')

score = model.evaluate(X_test, y_test, batch_size=32)
print('test score:', score[0])
print('test accuracy:', score[1])

plt.figure(2, figsize=(7, 5))
plt.plot(train_acc)
plt.plot(val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train & Test Accuracies')
plt.legend(['Train', 'Val'], loc=4)

plt.figure(1, figsize=(7, 5))
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train & Test Losses')
plt.legend(['train', 'val'])
plt.show()
