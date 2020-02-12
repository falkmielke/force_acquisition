import numpy as NP
import matplotlib as MP
import matplotlib.pyplot as MPP


# loaded = NP.load('test.npz')
loaded = NP.load('/data/03_technics/01_MeasurementCode/code/recordings/20200211_goa_cam_rec004_video.npz')
images = loaded['images']
print (images.shape)

MPP.imshow(images[:,:,-20], cmap = 'gray')
MPP.show()