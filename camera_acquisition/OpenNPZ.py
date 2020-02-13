import numpy as NP
import matplotlib as MP
import matplotlib.pyplot as MPP


# loaded = NP.load('test.npz')
loaded = NP.load('recordings/20200213_rat_cam_rec001_video.npz')
images = loaded['images']
print (images.shape)


MPP.imshow(images[:,:,-0], cmap = 'gray')
MPP.show()