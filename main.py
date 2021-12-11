import pandas as pd
import numpy as np
import glob 

from scipy.spatial.distance import cdist
import pydicom as dcm
import os
from sklearn import preprocessing
#import cv2
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow
from sklearn.cluster import *
from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
#import opencv
#! ml python/3.6.1

# Directories in Sherlock
directory="/home/groups/jdf1/project_images2/1.2.840.4267.32.40799883024600397843042028384947033985/1.2.840.4267.32.112193205348077103114348748653688365198"
directory2 = "/home/groups/jdf1/classification_10percData/1.2.840.4267.32.1182956769738561669133563954341155437/"

# Hyper-parameters
perform_elbow_analysis = False
preprocess_from_scratch=False
num_images = 12
k_max = 14
algorithms = ["Elkans","Full"]
save_folder = "data"

# Check if save folder exists, if not create is
if os.path.isdir(save_folder):
    False
else:
    os.mkdir(save_folder)

# Preprocessing raw images and saving to "Save folder"
def preprocess_and_save():
    # Preprocess and save files
    for ix,filename in enumerate(glob.glob(directory + "/**/*")):

        save_name = filename.replace("dcm","npy").split("/")[-1]

        if os.path.isfile(save_folder + "/" + save_name):
            continue
        else:
            d = dcm.dcmread(filename)
            # Pixel Array
            p = np.array(d.pixel_array)
            
            rescale_slope = d.RescaleSlope
            rescale_intercept = d.RescaleIntercept
            r = p * rescale_slope + rescale_intercept

            # Just for Plotting Example of Preprocessing
            if ix == 6:
                print("Example Before Preprocessing")
                ax = plt.subplot()
                im = plt.imshow(p,cmap='jet')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im,cax=cax)
                plt.savefig("Results/unproessed.png")
                plt.clf()

            if ix == 6:
                print("Example Post Processing")
                ax = plt.subplot()
                im = plt.imshow(r,cmap='jet')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im,cax=cax)
                plt.savefig("Results/preprocessed.png")

            # Save new files
            with open("data/" + save_name, 'wb') as f:
                np.save(f, r)

if preprocess_from_scratch:
	preprocess_and_save()

# Loads saved preprocessed files
def load_files():
    imgs = []
    for filename in glob.glob("data/*")[0:num_images]:
        imgs.append(np.load(filename))
    #images = np.stack(cv2.resize(imgs,dsize=(256,256),interpolation=cv2.INTER_CUBIC),axis=-1)
    images = np.stack(imgs,axis=-1)
    return images

# Loads unpreprocessed raw images
def load_unprocessed_files():
    imgs = []
    for filename in glob.glob("data2/*")[0:num_images]:
        imgs.append(np.load(filename))
    images = np.stack(imgs,axis=-1)
    return images

# Histogram of pixel distribution before preprocessing
unprocessed = load_unprocessed_files()
plt.figure()
plt.hist(unprocessed.flatten(),bins=100)
plt.xlabel("Hounsfield Unit (Hu)")
plt.ylabel("Frequency")
plt.savefig("Results/2.png")

# Loading data
X = load_files()
X = X.astype('float32')
X = X.reshape(512,512,num_images)

# Hisogram of pixel distributoin after preprocessing
plt.figure()
plt.hist(X.flatten(),bins=100)
plt.xlabel("Hounsfield Unit (Hu)")
plt.ylabel("Frequency")
plt.savefig("Results/1.png")

# Generate new models for new runs
def generate_new_models(clusters=5,algo="elkan"):
    models = [KMeans(n_clusters= clusters, random_state = 42,max_iter=100,verbose=1,algorithm=algo),FeatureAgglomeration(n_clusters=12),DBSCAN(), SpectralClustering(n_clusters=5,random_state=42,assign_labels = "discretize", eigen_solver = 'arpack')]
    mnames = ["KMeans", "Feature Agglomeration","DBSCAN","SpectralClustering"]
    return mnames, models

# Skeleton for deep learning unsupervised approach

#model = MobileNet(weights="imagenet",dropout=0.5,include_top=False,input_shape=(512,512,3))
#model.summary()
#features = model.predict(X[:,:,1])
#print(features.shape)


# Plot original vs. segmented prediction
def plot_result(original, result, savename):
    
    fig, ax = plt.subplots(2,2)
    axes = ax.ravel()
    axes[0].imshow(result,cmap='jet')
    axes[2].imshow(original,cmap='jet')
    axes[0].set_title("Segmented Prediction")
    axes[2].set_title("Original")
    axes[0].get_xaxis().set_ticks([])
    axes[0].get_yaxis().set_ticks([])
    axes[2].get_xaxis().set_ticks([])
    axes[2].get_yaxis().set_ticks([])
    axes[1].hist(result,bins=100)
    axes[1].set_xlabel("Pixel Values")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Prediction Pixel Distribution")
    axes[3].hist(original,bins=100)
    axes[3].set_title("Original Pixel Distribution")
    axes[3].set_xlabel("Hounsfield Unit (Hu)")
    axes[3].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("/oak/stanford/groups/zihuai/fredlu/MpraScreen/aaron_AD/hail/Results/" + str(savename) + "result_type2.png")
    plt.clf()

# Run implementations and gather results
for clusters in range(k):
	# Generate New Models
    mnames, models = generate_new_models()
    for i,m in enumerate(models):
        print(mnames[i], "---------------------------------")
        img = X.reshape(512 * 512 * num_images,1)
        clf = m.fit(img)
        print("DONE") 
        print("K = ", clusters, "Inertia = ", "{:e}".format(m.inertia_))

    # Evaluate Trained Model
    for x in range(0,50):
        result = clf.predict(test_X[:,:,x].reshape(512 * 512,1)).reshape(512,512)
        plot_result(test_X[:,:,x],result, x)
        

# Determines optimal 'k' clusters for task at hand
if perform_elbow_analysis:
	# Taken From https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
	distortions = []
	inertias = []
	mapping1 = {}
	mapping2 = {}

	X = X.reshape(512 * 512 * num_images,1)

	# K Range
	K = range(1, 15)
	 
	for k in K:
	    print(k)
	    # Building and fitting the model
	    kmeanModel = KMeans(n_clusters=k,random_state=42).fit(X)
	    kmeanModel.fit(X)
	 
	    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
	                                        'euclidean'), axis=1)) / X.shape[0])
	    inertias.append(kmeanModel.inertia_)
	 
	    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
	                                   'euclidean'), axis=1)) / X.shape[0]
	    mapping2[k] = kmeanModel.inertia_

	plt.figure()
	plt.plot(K, distortions, 'bx-')
	plt.xlabel('Values of K')
	plt.ylabel('Distortion')
	plt.title('The Elbow Method using Distortion')
	plt.savefig("Results/elbow.png")







