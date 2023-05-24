
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt




path_to_train_data = 'C:\\Users\\Admin\\Downloads\\emnist-balanced-train.csv'
train = pd.read_csv(path_to_train_data)
path_to_test_data = 'C:\\Users\\Admin\\Downloads\\emnist-balanced-test.csv'
test = pd.read_csv(path_to_test_data)


by_merge_map = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B',
                12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M',
                23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X',
                34:'Y', 35:'Z', 36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 41:'g', 42:'h', 43:'n', 44:'q',
                45:'r', 46:'t'}

columns = ['labels']
for i in range(train.shape[1] - 1):
    columns.append(i)

train.columns = columns
test.columns = columns

classes = train['labels'].unique()
train = train[train['labels'] < 36].reset_index(drop=True)
test = test[test['labels'] < 36].reset_index(drop=True)
classes = train['labels'].unique()
classes = np.setdiff1d(classes, [18, 24, 26])

print('number of  classes: ', len(classes))
classes = train['labels'].unique()

num_classes = len(classes)
trainY = np_utils.to_categorical(train['labels'], num_classes)
testY = np_utils.to_categorical(test['labels'], num_classes)

trainX = train.iloc[:, 1:].values.reshape(train.shape[0], 28, 28, 1)
testX = test.iloc[:, 1:].values.reshape(test.shape[0], 28, 28, 1)

trainX = trainX / 255.0
testX = testX / 255.0

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 128
epochs = 3

model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=1)

score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(testX[:19])
predicted_labels = np.argmax(predictions, axis=1)

fig, axs = plt.subplots(3, 7, figsize=(12, 6))
axs = axs.flatten()

for i in range(19):
    image = testX[i].reshape(28, 28)
    true_label = by_merge_map[np.argmax(testY[i])]
    predicted_label = by_merge_map[predicted_labels[i]]

    axs[i].imshow(image, cmap='gray')
    axs[i].set_title(f"True: {true_label}\nPredicted: {predicted_label}")
    axs[i].axis('off')

plt.tight_layout()
plt.show()
model.save('final_model.h5')