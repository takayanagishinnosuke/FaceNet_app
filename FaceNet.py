#%%
from turtle import color
from xml.sax.handler import feature_external_ges
from facenet.src import facenet
import tensorflow as tf
import numpy as np
from PIL import Image
import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#%%
class FaceEmbedding(object):
  def __init__(self, model_path):
  ##----モデルを読み込んでグラフに展開する関数----##
    facenet.load_model(model_path)

    self.input_image_size = 160
    self.sess = tf.Session()
    self.images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
    self.embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
    self.embedding_size = self.embeddings.get_shape()[1]

  
  def __del__(self):
    self.sess.close()

  def load_image(self, image_path, width, height, mode):
    image = Image.open(image_path)
    image = image.resize([width, height], Image.BILINEAR)

    return np.array(image.convert(mode))

  def face_embeddings(self, image_path):
    image = self.load_image(image_path, self.input_image_size, self.input_image_size, 'RGB')
    prewhitened = facenet.prewhiten(image)
    prewhitened = prewhitened.reshape(-1, prewhitened.shape[0], prewhitened.shape[1], prewhitened.shape[2])
    feed_dict = {self.images_placeholder: prewhitened, self.phase_train_placeholder: False}
    embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)

    return embeddings


#%%
#--学習モデル読み込み--#
FACE_MODEL_PATH = './20180402-114759/20180402-114759.pb'
face_embedding = FaceEmbedding(FACE_MODEL_PATH)

# %%
#--画像の読み込み##globglob(*)でフォルダ内のPNGを全てリストへ格納--#
faces_image_paths = glob.glob('./images/*.PNG')
# print(faces_image_paths)

# %%
#--顔画像から特徴ベクトルを抽出--#
features = np.array([face_embedding.face_embeddings(f)[0] for f in faces_image_paths])
print(features.shape)

# %%
##--512次元から2次元に主成分分析を使って次元削減--##

pca = PCA(n_components=2)
pca.fit(features)
reduced = pca.fit_transform(features)

print(reduced.shape)
## 512-> 2次元の特徴ベクトルになった
# %%
#-- K-meansで分けたいグループ数にクラスタリングする--#
## 1：データをいくつのクラスタに分けるか決める
## 2：全データに対してランダムにクラスタラベルを割り振る（初期化）
## 3：割り振られたラベルごとのデータについて平均を取ることで、K個重心を決める
## 4：全データに対してKこの重心と距離を求めて、それぞれ最も距離が短い重心のクラスタラベルにそのデータを割り当てなおす。
## 5： 3と4の処理を繰り返し、全データについてクラスタの割当が変化しなくなった場合、収束したとみなしそこで学習を終了する。

K = 3
kmeans = KMeans(n_clusters=K).fit(reduced)
pred_label = kmeans.predict(reduced)
pred_label

#%%
#--特徴量の保存--#
import pickle

pickle.dump(pred_label, open('pic.bin', 'wb'))

# %%
#--結果の視覚化--#
import matplotlib.pyplot as plt

x = reduced[:, 0]
y = reduced[:, 1]

color_codes = {0:'red', 1:'blue', 2:'green'}
colors = [color_codes[x] for x in pred_label]

plt.scatter(x,y, color=colors)
plt.colorbar()
plt.show()

# %%
#--結果の視覚化（顔画像プロット）--#
from matplotlib.offsetbox import OffsetImage, AnnotationBbox 

def imscatter(x , y, image_path, ax=None, zoom=1):
  if ax is None:
    ax = plt.gca()

  artists = []
  for x0, y0, image in zip(x, y,image_path):
    image = plt.imread(image)
    im = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(im, (x0,y0), xycoords='data', frameon=False)
    artists.append(ax.add_artist(ab))
  return artists

x = reduced[:, 0]
y = reduced[:, 1]

fig, ax = plt.subplots()
imscatter(x, y, faces_image_paths, ax=ax, zoom=.2)
ax.plot(x, y, 'ko', alpha=0)
ax.autoscale()
plt.show()
# %%
