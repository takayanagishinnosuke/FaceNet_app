#%%
import os
from PIL import Image
import glob

#フォルダのパス
dir_name = 'resize_img/s'
#保存先のパス
To_dir_Name = 'resize_img/resize_s'

# フォルダ内情報をリスト化
files = os.listdir(dir_name)

#ファイルのパス取得
file_path = glob.glob(dir_name + '/*')

#%%
##リネームするfor
# print(file_path)

for i, f in enumerate(file_path):
  os.rename(f, os.path.join(dir_name, '{0:02d}' . format(i)+ 's'+ '.PNG'))



#%%
##リサイズfor 
for i in files:
  img = Image.open(os.path.join(dir_name, i))
  img_resize = img.resize((160, 160))
  img_resize.save(os.path.join(To_dir_Name, i))
# %%
