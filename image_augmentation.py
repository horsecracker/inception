from wand.image import Image
#from PIL import Image
from wand.image import Color

import numpy as np

from os import listdir
from os.path import isfile, join
import os
imgfolder='/scratch1/liliyu/cancer/inception/data-density-processed/data-density'
#imgfolder='/Users/NINI/tmp/newimgs'
os.chdir(imgfolder)

imagesize=256
# In[167]:

def resize_shortest(img):
    x,y = float(img.size[0]), float(img.size[1])
    if x < y:
        xx = imagesize
        yy = y/x*xx
    else:
        yy = imagesize
        xx = (x/y)*yy
    xx,yy = int(xx),int(yy)
    img.resize(xx,yy)


# In[168]:

def random_crop(img):
    x,y = float(img.size[0]), float(img.size[1])
    delta_x = x-imagesize
    delta_y = y-imagesize
    xx = int(np.random.uniform(delta_x/2, delta_x))
    yy = int(np.random.uniform(delta_y/2, delta_y))
    img.crop(xx, yy, xx+imagesize, yy+imagesize)
    
def img_mirror(img):
    return np.fliplr(img)
    
def rotate(img, max_deg=360):
    deg = int(np.random.uniform(0, max_deg))
    img.rotate(deg, background=Color('rgb(132,132,132)'))
    l,L = img.size
    img.crop(width=int(0.6*l), height=int(0.6*L), gravity='center')

for directory in listdir(imgfolder):   # dirctory = train, dev
    if os.path.isdir(imgfolder+'/'+directory): 
        print (directory)
        for density_c in listdir(imgfolder+'/'+directory): # density_c = 1,2,3,4
            print (density_c)
            if density_c!='.DS_Store' : 
                files = listdir(imgfolder+'/'+directory+'/'+density_c)
                #print(files)
                N = len(files)
                for i, filepath in enumerate(files):
                    if filepath!='.DS_Store' : 
                        img = Image(filename=imgfolder+'/'+directory+'/'+density_c+'/'+filepath)
                        [name, ext] = filepath.split('.')
                        for t in range(5):
                            z = img.clone()
                            if t != 0:
                                rotate(z)
                            resize_shortest(z)
                            for tt in range(3):
                                new_path = '%s/%s_%i_%i.%s' % (imgfolder+'/'+directory+'/'+density_c, name, t, tt, ext)
                                n = z.clone()
                                random_crop(n)
                                n.save(filename=new_path)
                                #n = z.clone()
                                #n=img_mirror(n)
                                #n.save(filename='%s/%s_%i_%i_lr.%s' % (imgfolder+'/'+directory+'/'+density_c, name, t, tt, ext))
                        print 'Done %i/%i' % (i,N)
            
