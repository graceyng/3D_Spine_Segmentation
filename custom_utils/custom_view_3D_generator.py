"""
===================
Image Slices Viewer
===================

Scroll through 2D image slices of a 3D array.
"""

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
# Note: if "Qt5Agg" is not specified with MacOS, there may be an error that "'FigureManagerMac' object has no attribute 'window'"

import matplotlib.pyplot as plt
import json
import custom_utils.custom_generators as gen

#Parameters to select
metadataFile = '/Users/graceng/Documents/Med_School/Research/Radiology/043020_Metadata.json'
maskDir = 'collateVertMasks'
nChannels = 1
scanList_type = 'scanList_train' #'scanList_valid'
viewTypes = ['images', 'masks'] #['images']

#Expected dictionary keys in metadata file
metadataVars = ['scanList_train', 'scanList_valid', 'dim', 'numFramesPerStack', 'dirPath', 'sliceFileExt', 'batch_size',
               'k_folds', 'k_idx', 'seed']

def show_3d_images( input_matrix,upper_bound=0 ):

    if np.any(np.iscomplex(input_matrix)):
        mat = np.log( np.abs( input_matrix)+1)
    elif (upper_bound >0):
        mat=input_matrix
        mat[mat>upper_bound]=upper_bound
    else:
        mat = input_matrix
    
    #TODO: is this re-scaling necessary?

    if np.amin(mat) > 0.:
        mat = mat - np.amin(mat)
    elif np.amin(mat) < 0.:
        mat = mat + abs(np.amin(mat))
    mat = mat / np.amax(mat) * 255.


    fig, figure_axes = plt.subplots( 1 , 1 )
    tracker = IndexTracker( figure_axes , mat ) 
    fig.canvas.mpl_connect( 'scroll_event' , tracker.onscroll )

    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    mng.window.showMaximized()

    plt.show(block=True)



class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape

        # self.ind = self.slices//2
        self.ind = 0

        #self.im = ax.imshow(self.X[:, :, self.ind] , cmap='gray',vmin=0,vmax=255)
        self.im = ax.imshow(self.X[:, :, self.ind], cmap=plt.cm.bone, vmin=0, vmax=255)
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


f = open(metadataFile, 'r')
metadata = json.load(f)
f.close()

for metadataVar in metadataVars:
    if metadataVar not in metadata:
        raise Exception('{} not in metadata file.'.format(metadataVar))
dim = tuple(metadata['dim'])


#Basic generator
generator = gen.Generator_3D(metadata[scanList_type], metadata['dirPath'], maskDir,
                                         numFramesPerStack=metadata['numFramesPerStack'], batchSize=metadata['batch_size'],
                                         dim=metadata['dim'], nChannels=nChannels, seed=metadata['seed'], shuffle=True,
                                         sliceFileExt=metadata['sliceFileExt'], fitImgMethod="pad", zoomRange=(1,1),
                                         rotationRange=0, widthShiftRange=0, heightShiftRange=0, flipLR=False, flipUD=False)
"""
#Generator to test augmentation functions
generator = gen.Generator_3D(metadata[scanList_type], metadata['dirPath'], maskDir,
                                         numFramesPerStack=metadata['numFramesPerStack'], batchSize=metadata['batch_size'],
                                         dim=metadata['dim'], nChannels=nChannels, seed=metadata['seed'], shuffle=True,
                                         sliceFileExt=metadata['sliceFileExt'], fitImgMethod="pad", zoomRange=(1,10),
                                         rotationRange=90, widthShiftRange=.25, heightShiftRange=.25, flipLR=True, flipUD=True)
"""

#Make sure that generator.__len__ function uses a ceiling function in order to show all images
numBatches = generator.__len__()
for batch_idx in range(numBatches):
    Img, mask = generator.__getitem__(batch_idx)
    #Img and mask have dimensions batchSize x nChannels x *dim x numFramesPerStack
    for scan_idx in range(metadata['batch_size']):
        for channel_idx in range(nChannels):
            if scan_idx == metadata['batch_size']-1:
                if 'images' in viewTypes:
                    imgArray = Img[scan_idx, channel_idx,]
                    #print(imgArray.min(), imgArray.max())
                    print('Showing images for batch {:d}, scan {:d}, channel {:d}'.format(batch_idx, scan_idx, channel_idx))
                    show_3d_images(Img[scan_idx, channel_idx,])
                if 'masks' in viewTypes:
                    print('Showing masks for batch {:d}, scan {:d}, channel {:d}'.format(batch_idx, scan_idx, channel_idx))
                    show_3d_images(mask[scan_idx, channel_idx,])