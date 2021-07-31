# ArtificialNeuralNetwork
Artificial Neural Network

Repositori ini berisikan source code artificial neural network.

## How to Use
Berikut adalah tata cara menggunakan source code dari ANN

1. Pastikan Python versi 3.7 ke atas tersedia (program ini dibuat dengan Python 3.8.5)
2. Pastikan terdapat library numpy, pandas, matplotlib
3. Lakukan import source code

```python
from src.ann import ANNClassifier, Layer
```

4. Siapkan dataset menggunakan pandas datafram

```python
df = pd.read(csv('namafile.csv')
x = df[x_cols]
y = df[y_col]
```

5. Inisialisasikan model classifier

```python
model = ANNClassifier(random_state=42)
model.add(Layer(3))
model.add(Layer(3,activation_function='relu'))
model.add(Layer(3,activation_function='relu'))
model.add(Layer(3,activation_function='relu'))
model.add(Layer(1))
model.compile(learning_rate=0.1,mini_batch=25,epoch=100)
```

```
parameter ANNClassifier
random_state : int or None, default=None (seed untuk random weightnya)

parameter Layer
n_neuron : int (banyaknya neuron per layer)
activation_function : {'relu','sigmoid','softplus','linear'} (pilihan activation function)

parameter compile
learning_rate : float (kecepatan belajar training model)
mini_batch : int (banyaknya mini batch)
epoch : int (banyaknya iterasi model)
```

6. Lakukan fit pada model menggunakan dataset yang sudah disiapkan

```python
model.fit(x,y)
```

```
Epoch : 1 | Loss per Epoch : 2.02e+03 | Time : 0.11 s
Epoch : 2 | Loss per Epoch : 1.73e+02 | Time : 0.12 s
...
```

7. Lakukan predict menggunakan data yang berbentuk dataframe

```python
model.predict(df2)
```

```
output : np.array([0 1 0 0 .. 0 1])
```

8. Lakukan pemantauan terhadap grafik loss selama training model

```python
model.show_loss_plot(detailed=True)
```

![1627709636389](https://user-images.githubusercontent.com/56197074/127729920-86bd12d7-a4c3-47bf-8e44-b43812b2beff.jpg)
![1627709630944](https://user-images.githubusercontent.com/56197074/127729923-696f7249-5174-4568-bf81-7e4714cc7655.jpg)
