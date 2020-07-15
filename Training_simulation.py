from Utilities import *
from sklearn.model_selection import train_test_split
path = 'data'
mydata = importDataInfo(path)

#step 2: balancedata
balanceData(mydata,display= False)


#step 3
imagesPath, steerings = loadData(path,mydata)

#step 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=2)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#step 5

#step 6

#step 7

# step 8
model = createModel()
model.summary()

#step 9

history = model.fit(batchGen(xTrain, yTrain, 2, 1),steps_per_epoch=5, epochs=20, validation_data=batchGen(xVal, yVal, 2, 0),validation_steps=5)

#step 10
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()


