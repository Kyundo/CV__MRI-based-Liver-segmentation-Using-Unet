# CV__MRI-based Liver Segmentation using Unet model
![3](https://user-images.githubusercontent.com/75473616/167287314-8f32dc5e-0c68-459b-aba7-b7d81ccb9421.png)
#### my data
```
Input_dimension = (512, 512)
Set_num = 4
```

### Preprocessing Data 
```
X_train = [0 for i in range(Set_num)]
y_train = [0 for i in range(Set_num)]
for i in range(Set_num):
  X_train[i] = png2set(Input_dimension, '/content/drive/My Drive/' + str(i + 1) + '/')
  y_train[i] = nif2set(Input_dimension, '/content/drive/My Drive/' + str(i + 1) + '.nii')
```

### Build & Complie the model
```
model = Build(Input_dimension)
model = Compiler(model)
```

### Learning 
```
history = [0 for i in range(Set_num)]
for i in range(Set_num):
    history[i] = Fit(model, X_train[i], y_train[i], BatchSize=10, Epochs=100, ValidationSplit=0.15)
model = Saver(model, 'Unet_model.h5')
```

### Prediction
```
Test_img = '211.png'
result = Predictor(model, Test_img)
```
