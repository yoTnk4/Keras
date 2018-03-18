from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import numpy as np
import matplotlib.pyplot as plt

#テストデータの取得
# x_train : 訓練用のデータ
# y_train : 訓練用のデータのラベル
# x_test : テスト用のデータ
# y_test : テスト用のデータのラベル
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#テストデータの内容を表示する。
print("x_trainの形:{}".format(x_train.shape))
print("y_trainの形:{}".format(y_train.shape))
print("x_testの形:{}".format(x_test.shape))
print("y_test:{}".format(y_test.shape))

# 学習とテストデータを閲覧する
columns = 5
rows = 1
# 一行で複数の画像を描画する
def display_thumb_line(x_data_set, sample_index):
    fig=plt.figure(figsize=(5, 5))
    n_sample = min(len(sample_index), (columns * rows))
    for idx in range(n_sample):
        fig.add_subplot(rows, columns, idx + 1)
        plt.imshow(x_data_set[sample_index[idx]])
        plt.title("idx={0}".format(sample_index[idx]))
    plt.show()

print("********************************")
print("訓練用のデータの表示")
print("********************************\n")
for digit in range(10):
    try:
        sample_index = np.argwhere(y_train == digit)
    except ValueError:
        sample_index = []
    sample_index = sample_index.reshape(-1)
    if (len(sample_index) > 0):
        display_thumb_line(x_train, sample_index)

print("********************************")
print("テスト用のデータの表示")
print("********************************\n")
for digit in range(10):
    try:
        sample_index = np.argwhere(y_test == digit)
    except ValueError:
        sample_index = []
    sample_index = sample_index.reshape(-1)
    if (len(sample_index) >= columns * rows):
        display_thumb_line(x_test, sample_index)

#訓練用データの配列の中身を確認する。
for i in range(2):
    print("********************************")
    print("x_train[{}]のデータ".format(i))
    print("********************************\n")

    for j in range(28):
        w_str = ""
        for k in range(28):
            w_str2 = "{0:03d} ".format(int(x_train[i][j][k]))
            w_str = w_str + w_str2
        print(w_str)
    print("\n")
    display_thumb_line(x_train, [i])

#テストデータの整形
#60000枚×1次元784pxの画像に変換します。
x_train = x_train.reshape(60000, 784) # [60000][28][28] -> [60000][784]に変換
#10000枚×1次元784pxの画像に変換します。
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32') # テストデータの型をfloat32に変換
x_test = x_test.astype('float32')

# データの簡易的な正規化。
x_train = x_train / 255
x_test = x_test / 255

#教師ラベルの内容確認①
print("********************************")
print("y_trainの形:{}".format(y_train.shape))
print("y_train[{0} -> {1}]のデータ".format(0, 9))
print("********************************\n")
for i in range(10):
    print("y_train[{0}]=".format(i),y_train[i])

print("********************************")
print("y_testの形:{}".format(y_test.shape))
print("y_test[{0} -> {1}]のデータ".format(0, 9))
print("********************************\n")
for i in range(10):
    print("y_test[{0}]=".format(i),y_test[i])


#教師ラベルの内容確認①
print("********************************")
print("y_trainの形:{}".format(y_train.shape))
print("y_train[{0} -> {1}]のデータ".format(0, 9))
print("********************************\n")
for i in range(10):
    print("y_train[{0}]=".format(i),y_train[i])

print("********************************")
print("y_testの形:{}".format(y_test.shape))
print("y_test[{0} -> {1}]のデータ".format(0, 9))
print("********************************\n")
for i in range(10):
    print("y_test[{0}]=".format(i),y_test[i])

#教師ラベルの整形
#出力ノード数に合わせて今は60000×1次元を60000×10次元に変換します。
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#教師ラベルの内容確認②
print("********************************")
print("y_trainの形:{}".format(y_train.shape))
print("y_train[{0} -> {1}]のデータ".format(0, 9))
print("********************************\n")
print("y_train[i]=".format("-")," ", "=>y_train", list(range(10)),"\n")
for i in range(10):
    print("y_train[{0}]=".format(i),y_train[i], "=>y_train", y_train[i])

print("********************************")
print("y_testの形:{}".format(y_test.shape))
print("y_test[{0} -> {1}]のデータ".format(0, 9))
print("********************************\n")

print("y_test[i]=".format("-")," ", "=>y_test", list(range(10)),"\n")
for i in range(10):
    print("y_test[{0}]=".format(i),y_test[i], "=>y_test", y_test[i])

#ニューラルネットワークの実装①
#初期化
model = Sequential()
#隠れ層の定義  model.add(Dense(ノード数, activation=活性化関数の名前), input_shape=(入力層の数,))
model.add(Dense(512, activation='relu', input_shape=(784,))) # 隠れ層①
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu')) # 隠れ層②
model.add(Dropout(0.2))
#出力層の定義  model.add(Dense(出力層のノード数, activation=出力層の活性化関数の名前))
model.add(Dense(10, activation='softmax')) # 出力層

#ニューラルネットワークの内容確認
model.summary()

#ニューラルネットワークの実装②
model.compile(loss='categorical_crossentropy', # 損失関数=クロスエントロピー
              optimizer=RMSprop(),             # 最適化アルゴリズムはRMSprop
              metrics=['accuracy'])            # 評価方法は正解率(accuracy)

#ニューラルネットワークの学習
#batch_size : 重み・バイアス更新１回あたりのデータ数
#epochs : 訓練データを何回学習させるか
#varbose : ログ出力方法 0:出力なし 1:プログレスバーで出力 2:普通に出力
#validation_data : 性能を測るためのテストデータ
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_test, y_test))

#ニューラルネットワークの推論
score = model.evaluate(x_test, y_test, verbose=0)

#ニューラルネットワークの推論（本当にあってるのか確認）
for i in range(3):
    print("********************************")
    print("x_test[{}]のデータ".format(i))
    print("********************************\n")

    # 入力データを推論
    pred = model.predict_classes(x_test[i].reshape(1, -1), batch_size=1, verbose=0)
    print("NNの予想は:{}".format(pred))

    # 画面表示
    print("画像は...")
    img_px = x_test[i].reshape(28, 28)
    plt.imshow(img_px)
    plt.show()

#学習したニューラルネットワークを保存します
# model(層の設計)を保存
json_string = model.to_json()
open("model.json", "w").write(json_string)

# weight,biasを保存
model.save_weights("weights.hdf5")

