#%%
import math
import random as r
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

#%%
def kMeans(img, k=10):
    points = np.reshape(img, (-1, img.shape[-1]))
    centroids = np.array(r.sample(list(points), k))
    iterCount, maxIter = 0,100
    while True:
        distances = distance_matrix(points, centroids)
        clusterNo = np.argmin(distances,axis=1)
        clusters = []
        newCentroids = []
        for i in range(k):
            clusterPoints = points[clusterNo==i]
            centroid = np.average(clusterPoints, axis=0)
            newCentroids.append(centroid)
            clusters.append(clusterPoints)
        newCentroids = np.array(newCentroids)
        if np.all(newCentroids == centroids) or iterCount>maxIter:
            clusterNo2D = np.reshape(clusterNo, img.shape[0:2])
            newCentroids = newCentroids.astype('uint8')
            newImg = np.zeros(img.shape, dtype='uint8')
            for i in range(k):
                newImg[clusterNo2D==i] = newCentroids[i]
            break
        else:
            centroids = newCentroids
        
        iterCount += 1
        
    return newImg, newCentroids

def SLIC(img, blockSize=50, localWindow=3):
    w, h, _ = img.shape
    sub_w, sub_h = math.ceil(w/blockSize), math.ceil(h/blockSize)
    offset = localWindow//2
    
    centroids = []
    for i in range(sub_w):
        for j in range(sub_h):
            if (i+1)*blockSize<w and (j+1)*blockSize<h:
                centroids.append(((i*blockSize+(i+1)*blockSize)//2,(j*blockSize+(j+1)*blockSize)//2))
            elif (i+1)*blockSize>=w and (j+1)*blockSize<h:
                centroids.append(((i*blockSize+w)//2,(j*blockSize+(j+1)*blockSize)//2))
            elif (i+1)*blockSize<w and (j+1)*blockSize>=h:
                centroids.append(((i*blockSize+(i+1)*blockSize)//2,(j*blockSize+h)//2))
            else:
                centroids.append(((i*blockSize+w)//2,(j*blockSize+h)//2))

    iterCount,max_iter = 0,3
    updatedCentroids = []

    for centroid in centroids:
        updatedCentroid = []
        y,x = centroid
        try:
            gradientWindow = img[y-offset:y+offset+1,x-offset:x+offset+1]
            gradientWindow = gradientWindow.astype('float64')
            gradient = np.apply_along_axis(lambda x: np.sqrt(np.sum(np.square(x))),axis=-1,arr=gradientWindow)
            minInd = np.subtract(np.unravel_index(np.argmin(gradient),shape=gradient.shape),offset)
            updatedCentroid = tuple(np.add(centroid,minInd))
            updatedCentroids.append(updatedCentroid)

        except Exception as e:
            pass
    
    featureSpace = np.zeros((w,h,5))
    featureSpace[:,:,:3] = img
    featureSpace[:,:,3:] = np.mgrid[0:h,0:w].T / 2

    featureCentroids = []
    for centroid in updatedCentroids:
        featureCentroids.append(featureSpace[centroid[0], centroid[1]])
    featureCentroids = np.array(featureCentroids)

    featureSpace = np.reshape(featureSpace, (-1, featureSpace.shape[-1]))
    featureCentroids = np.reshape(featureCentroids, (-1, featureCentroids.shape[-1]))

    newImg = np.zeros(img.shape,dtype='uint8')
    while True:
        distances = distance_matrix(featureSpace, featureCentroids)
        clusterNo = np.argmin(distances, axis=1)

        clusters = []
        newCentroids = []

        for i in range(len(set(clusterNo))):
            clusterPoints = featureSpace[clusterNo==i]
            centroid = np.average(clusterPoints, axis=0)
            newCentroids.append(centroid)
            clusters.append(clusterPoints)
        
        newCentroids = np.array(newCentroids)
        print(clusterNo.shape)

        if np.array_equal(newCentroids,centroids) or iterCount>=max_iter:
            clusterNo2D = np.reshape(clusterNo, img.shape[0:2])
            print(clusterNo2D.shape)
            newCentroids = newCentroids.astype('uint8')
            
            for i in range(len(set(clusterNo))):
                newImg[clusterNo2D==i] = newCentroids[i][:3]
            break
        else:
            centroids = newCentroids
        print(iterCount, max_iter, iterCount>=max_iter)
        iterCount += 1

    return newImg

def kMeanPoints(points, k=10):
    centroids = np.array(r.sample(list(points), k))
    iterCount, maxIter = 0,200
    while True:
        distances = distance_matrix(points, centroids)
        clusterNo = np.argmin(distances,axis=1)
        clusters = []
        newCentroids = []
        for i in range(k):
            clusterPoints = points[clusterNo==i]
            centroid = np.average(clusterPoints, axis=0)
            newCentroids.append(centroid)
            clusters.append(clusterPoints)
        newCentroids = np.array(newCentroids)
        if np.all(newCentroids == centroids) or iterCount>maxIter:
            break
        else:
            centroids = newCentroids
        
        iterCount += 1
    return newCentroids.astype('int')

def kNN(img, points, labels):
    imgPoints = np.reshape(img, (-1, img.shape[-1]))
    distances = distance_matrix(imgPoints, points)
    pointNo = np.argmin(distances, axis=1)
    pointLabel = np.array([labels[i] for i in pointNo])

    points2D = np.reshape(pointLabel, img.shape[0:2])
    points2D = 1-points2D

    maskImg= img.copy()
    points2D = points2D.astype('bool')
    maskImg[points2D] = [0,255,0]
    # maskImg = np.multiply(img, points2D[:,:,None])

    return maskImg

#%%

img = cv2.imread('white-tower.png')
img2 = cv2.imread('wt_slic.png')

kmeansImg, _ = kMeans(img)
cv2.imwrite('output/kmeans.png', kmeansImg)

slic1 = SLIC(img)
slic2 = SLIC(img2)
cv2.imwrite('output/slic1.png', slic1)
cv2.imwrite('output/slic2.png', slic2)

nonSkyImg = cv2.imread('sky/sky_train_re.jpg')
trainImg = cv2.imread('sky/sky_train.jpg')

nonSkyMask = np.reshape(nonSkyImg, (-1, nonSkyImg.shape[-1]))
nonSkyBool = np.all(nonSkyMask==255,axis=1)
trainImgPoints = np.reshape(trainImg, (-1, trainImg.shape[-1]))

nonSkyPoints = trainImgPoints[nonSkyBool]
skyPoints = trainImgPoints[np.invert(nonSkyBool)]

skyCent = kMeanPoints(skyPoints)
nonSkyCent = kMeanPoints(nonSkyPoints)

testImage = cv2.imread('sky/sky_test2.jpg')
maskImg = kNN(testImage, np.vstack((nonSkyCent, skyCent)),[0]*len(nonSkyCent)+[1]*len(skyCent))


cv2.imwrite('output/skyTest.jpg', maskImg)


# %%
