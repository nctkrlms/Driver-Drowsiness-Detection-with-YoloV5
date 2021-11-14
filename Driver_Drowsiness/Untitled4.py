#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html')


# In[ ]:


get_ipython().system('cd yolov5 & pip install -r requirements.txt')


# In[1]:


import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


# In[2]:




import uuid   # Unique identifier
import os
import time


# In[9]:


get_ipython().system('cd yolov5 && python train.py --img 320 --batch 16 --epochs 50 --data dataset.yml --weights yolov5s.pt --workers 1')


# In[3]:


model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp10/weights/last.pt', force_reload=True)


# In[4]:


img = os.path.join('data', 'images', 'drowsy.2f95fe0c-3a5d-11ec-a1bd-0c5415c444e6.jpg')
results = model(img)


# In[5]:




results.print()


# In[6]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




