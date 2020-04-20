#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision import *
from fastai.metrics import error_rate
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


# In[2]:


bs = 8


# In[3]:


path = Path('E:/Train/')


# In[4]:


#for file, folder in [('Modi.txt', 'Modi'), ('Kejriwal.txt', 'Kejriwal') ]:
 #   dest = path/folder # path + '/' + folder
  #  dest.mkdir(parents=True, exist_ok=True)
   # download_images(path/file, dest, max_pics=200)


# In[5]:


#for folder in ('Modi','Kejriwal'):
   # print(folder)
   # verify_images(path/folder, delete=True, max_size=500)


# In[6]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.9, bs=4, 
        ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0
                              ),size=224, num_workers=4).normalize(imagenet_stats)
data.classes


# In[7]:


#data.show_batch(rows=3, figsize=(7, 8))


# In[8]:


#data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[9]:


import ipywidgets


# In[10]:



learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[11]:



learn


# In[13]:


torch.cuda.empty_cache()

defaults.device = torch.device('cuda')
volatile=True# makes sure the gpu is used
learn.fit_one_cycle(4)


# In[14]:


learn.save('Phase1')


# In[35]:


learn.unfreeze()


# In[39]:


learn.lr_find()


# In[40]:


learn.recorder.plot()


# In[ ]:





# In[15]:


interp = ClassificationInterpretation.from_learner(learn)


# In[16]:


interp.plot_confusion_matrix()


# In[34]:


img = open_image('E:/Train/2.jpg')
pred_class = learn.predict(img)
pred_class


# In[ ]:





# In[ ]:




