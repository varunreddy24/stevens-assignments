#%%
import math
import random as r
from typing import Tuple

import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

#%%
SobelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
SobelY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

def filter2d(image: np.ndarray, filter: np.ndarray, padding: tuple = (1,1)) -> Tuple[np.ndarray, np.ndarray]:
    if(image.ndim == 2):
        image = np.expand_dims(image, axis=-1)
    if(filter.ndim == 2):
        filter = np.repeat(np.expand_dims(filter, axis=-1), image.shape[-1], axis=-1)
    if(filter.shape[-1] == 1):
        filter = np.repeat(filter, image.shape[-1], axis=-1)

    size_x, size_y = filter.shape[:2]
    width, height = image.shape[:2]
    
    raw_output_array = np.zeros(((width - size_x + 2*padding[0]) + 1, 
                             (height - size_y + 2*padding[1]) + 1,
                             image.shape[-1]))
    
    padded_image = np.pad(image, [
        (padding[0], padding[0]),
        (padding[1], padding[1]),
        (0, 0)
    ])
    
    for x in range(padded_image.shape[0] - size_x + 1):
        for y in range(padded_image.shape[1] - size_y + 1):
            window = padded_image[x:x + size_x, y:y + size_y]
            output_values = np.sum(filter * window, axis=(0, 1)) 
            raw_output_array[x, y] = output_values
    
    raw_output_array = np.squeeze(raw_output_array)
    
    output_array = raw_output_array.astype('uint8')
    output_array[raw_output_array>255] = 255
    output_array[raw_output_array<0] = 0

    return raw_output_array, output_array

def GaussianKernel2d(size:int = 5, sigma:float =1.) -> np.ndarray:
    halfWidth = size//2
    x = np.arange(-halfWidth, halfWidth+1,1)
    y = np.arange(-halfWidth, halfWidth+1,1)
    gaussianFilter = np.zeros((size,size))
    for i in x:
        for j in y:
            gaussianFilter[i+halfWidth][j+halfWidth] = np.exp(-(i*i + j*j)/(2.*sigma*sigma))
    gaussianFilter = gaussianFilter/(2.*np.pi*sigma*sigma)
    gaussianFilter = gaussianFilter/np.sum(gaussianFilter)
    return gaussianFilter

def SobelFiltering(blur:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
    gradientX, _ = filter2d(blur, SobelX, padding=(2,2))
    gradientY, _ = filter2d(blur, SobelY, padding=(2,2))

    gradient_raw = np.hypot(gradientX, gradientY)
    gradient = gradient_raw.astype('uint8')

    gradient[gradient_raw>255] = 255
    gradient[gradient_raw<0] = 0

    orientation = np.arctan2(gradientY,gradientX)

    return gradient_raw, gradient, orientation

def GradientThresh(gradient:np.ndarray, thresh: int = 30) -> np.ndarray:
    gradientThresh = np.copy(gradient)
    gradientThresh[gradient<30] = 0
    return gradientThresh

def NonMaxSupression(image: np.ndarray, orientation:np.ndarray) -> np.ndarray:
    m,n = image.shape
    z = np.zeros((m,n), dtype='uint8')
    angle = orientation*180./np.pi
    angle[angle<0] += 180

    for i in range(1, m-1):
        for j in range(1, n-1):
            try:
                q = 255
                r = 255
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = image[i, j+1]
                    r = image[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = image[i+1, j-1]
                    r = image[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = image[i+1, j]
                    r = image[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = image[i-1, j-1]
                    r = image[i+1, j+1]

                if (image[i,j] >= q) and (image[i,j] >= r):
                    z[i,j] = image[i,j]
                else:
                    z[i,j] = 0

            except IndexError as e:
                pass
    return z

def HarrisDetector(image: np.ndarray, windowSize: int = 3, alpha: float = 0.05, points:int = None, threshold: int = 0, nms:bool=False, nmsWindow: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    colorImg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    corners = ([],[])
    h, w = image.shape[:2]

    Ix, _ = filter2d(image, SobelX, (0,0))
    Iy, _ = filter2d(image, SobelY, (0,0))

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    responseMatrix = np.zeros(image.shape, dtype=np.float64)
    responseFinal = np.zeros(image.shape, dtype=np.float64)

    offset = windowSize//2
    for y in range(offset, h-offset):
        for x in range(offset, w-offset):
            windowIxx = Ixx[y-offset:y+offset, x-offset:x+offset]
            windowIyy = Iyy[y-offset:y+offset, x-offset:x+offset]
            windowIxy = Ixy[y-offset:y+offset, x-offset:x+offset]
            
            Sxx = windowIxx.sum()
            Syy = windowIyy.sum()
            Sxy = windowIxy.sum()

            detM = (Sxx*Syy) -  (Sxy**2)
            traceM = Sxx + Syy
            response = detM - alpha*traceM

            if response > threshold:
                responseMatrix[y, x] = response
    
    responseIndices = np.unravel_index(np.argsort(responseMatrix, axis=None), responseMatrix.shape)
    if points!= None:
        responseIndices = (responseIndices[0][-points:], responseIndices[1][-points:])

    # Non max supression
    if nms:
        for y in range(nmsWindow, h-nmsWindow):
            for x in range(nmsWindow, w-nmsWindow):
                window = responseMatrix[y-nmsWindow:y+nmsWindow, x-nmsWindow:x+nmsWindow]
                if window[nmsWindow, nmsWindow] == np.max(window):
                    responseFinal[y, x] = window[nmsWindow, nmsWindow]
                    if responseFinal[y, x]>0:
                        corners[1].append(x)
                        corners[0].append(y)
    else:
        corners = responseIndices
    x, y = corners
    for i in range(len(x)):
        colorImg = cv2.circle(colorImg, (y[i], x[i]), 4, (0,255,0), 1)
    return colorImg, corners

def extractPatches(image: np.ndarray, corners:np.ndarray, radius:int = 5):
    patches = []

    for i,(x,y) in enumerate(zip(corners[0], corners[1])):
        patch = image[y-radius:y+radius+1, x-radius:x+radius+1]
        if patch.shape == (2*radius+1, 2*radius+1, 3):
            patches.append(patch)
    patches = np.array(patches)
    
    return patches

def similarPatches(imgLeft: np.ndarray, imgRight: np.ndarray, cornersLeft: tuple, cornersRight: tuple, type:str = 'ssd', windowSize:int = 7, thresh:float = 1000):
    leftPatch = []
    rightPatch = []
    offset = windowSize//2
    match = []

    for i in range(len(cornersLeft[0])):
        x,y = cornersLeft[0][i], cornersLeft[1][i]
        leftPatch.append(imgLeft[x-offset:x+offset, y-offset:y+offset, :])
    for i in range(len(cornersRight[0])):
        x,y = cornersRight[0][i], cornersRight[1][i]
        rightPatch.append(imgRight[x-offset:x+offset, y-offset:y+offset, :])
    # print(len(leftPatch), len(rightPatch))

    for i in range(len(leftPatch)):
            minVal = np.inf
            minPair = None
            for j in range(len(rightPatch)):
                try:
                    h = np.sum((leftPatch[i]-rightPatch[j])**2)
                    if minVal>h and h<thresh:
                        minVal = h
                        minPair = ((cornersLeft[0][i], cornersLeft[1][i]),(cornersRight[0][j], cornersRight[1][j]))
                except Exception as e:
                    minPair = None
            if minPair:
                match.append(minPair)
        
    
    return match

def bestLine(points: list) -> Tuple[float, float, float]:
    n = len(points)
    sigmaX = sum([i[0] for i in points])
    sigmaY = sum([i[1] for i in points])
    sigmaXY = sum([i[0]*i[1] for i in points])
    sigma2X = sum([i[0]**2 for i in points])
    sigmaX2 = sigmaX**2

    ## We assume line to be in form  (px+qy+r = 0)
    if (n*sigma2X - sigmaX2) != 0:
        a = (sigmaY*sigma2X - sigmaX*sigmaXY)/(n*sigma2X - sigmaX2)
        b = (n*sigmaXY - sigmaX*sigmaY)/(n*sigma2X - sigmaX2)
        p, q, r = float(-b), 1, float(-a)
    else:
        p, q, r = 1, 0, float(-points[0][0])

    return p,q,r

def LinePointDistance(a: float, b: float, c:float, point: Tuple[float, float]) -> float:
    return abs(a*point[0]+b*point[1]+c)/(np.sqrt(a**2+b**2))

def findExtremePoints(points: list) -> tuple:
    mat = np.array(distance_matrix(points, points))
    ind = np.unravel_index(mat.argmax(), mat.shape)
    return points[ind[1]][:2], points[ind[0]][:2]

def RANSAC(img: np.ndarray, corners:list, s:int = 2, t:float = 5, p:float = 0.8) -> Tuple[np.ndarray, np.ndarray, tuple]:
    N = np.inf
    sample_count = 0
    
    inliersMax = 0
    eMin = 1
    aBest, bBest, cBest = 0,0,0
    inlierPointsMax = []
    outlierPointsMax  = []

    pointsTaken = []
    while N>sample_count:
        samples = r.choices(corners, k=s)
        samples.sort()
        if samples not in pointsTaken:
            pointsTaken.append(samples)
            a, b, c = bestLine(samples)
            inliers = 0
            inlierPoints, outlierPoints = [], []
            for point in corners:
                perpDist = LinePointDistance(a,b,c,point)
                if perpDist<=t:
                    inliers += 1
                    inlierPoints.append(point)
                else:
                    outlierPoints.append(point)
            e = 1 - (inliers/len(corners))
            N = math.ceil(math.log(1-p)/(math.log(1-(1-e)**s)))
            if inliers>inliersMax:
                inliersMax = inliers
                eMin = e
                inlierPointsMax = inlierPoints.copy()
                outlierPointsMax = outlierPoints.copy()
                aBest, bBest, cBest = a, b, c
        else:
            continue
        sample_count += 1
    
    a,b = findExtremePoints(inlierPoints)
    imgCol = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    imgCol = cv2.line(imgCol, a, b, (0,255,0), thickness=2)


    return inlierPointsMax, outlierPointsMax, (a, b)

def HoughLines(image: np.ndarray, corners:np.ndarray, num_rhos: int=180, num_thetas:int = 180, t_count:int=20) -> tuple:
    h, w = image.shape[:2]
    h_half, w_half = h / 2, w / 2
    d = np.sqrt(np.square(h) + np.square(w))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos
    
    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-d, d, step=drho)
    
    cos_val = np.cos(np.deg2rad(thetas))
    sin_val = np.sin(np.deg2rad(thetas))
    
    votes_accumulator = np.zeros((len(rhos), len(rhos)))
    figure = plt.figure(figsize=(12, 12))
    subplot1 = figure.add_subplot(1, 4, 3)
    subplot1.set_facecolor((0, 0, 0))
    subplot2 = figure.add_subplot(1, 4, 4)
    subplot2.imshow(image)
    rho_values = np.matmul(corners, np.array([sin_val, cos_val]))

    votes_accumulator, _, _ = np.histogram2d(
        np.tile(thetas, rho_values.shape[0]),
        rho_values.ravel(),
        bins=[thetas, rhos]
    )
    votes_accumulator = np.transpose(votes_accumulator)
    lines = np.argwhere(votes_accumulator > t_count)

    rho_idxs, theta_idxs = lines[:, 0], lines[:, 1]
    r, t = rhos[rho_idxs], thetas[theta_idxs]

    for ys in rho_values:
        subplot1.plot(thetas, ys, color="white", alpha=0.05)

    subplot1.plot([t], [r], color="yellow", marker='o')

    for line in lines:
        y, x = line
        rho = rhos[y]
        theta = thetas[x]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + w_half
        y0 = (b * rho) + h_half
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        subplot1.plot([theta], [rho], marker='o', color='green')
        subplot2.add_line(mlines.Line2D([x1, x2], [y1, y2]))

    subplot1.invert_yaxis()
    subplot1.invert_xaxis()

    subplot1.title.set_text("Hough Space")
    subplot2.title.set_text("Detected Lines")
    plt.show()
    return votes_accumulator, rhos, thetas
#%%
imgColLeft = cv2.imread('AlignmentTwoViews/uttower_left.jpg')                     ### Change the input image
imgLeft = cv2.cvtColor(imgColLeft, cv2.COLOR_BGR2GRAY)
imgColRight = cv2.imread('AlignmentTwoViews/uttower_right.jpg')
imgRight = cv2.cvtColor(imgColRight, cv2.COLOR_BGR2GRAY)

gaussianKernel = GaussianKernel2d(5, 0.5)                   ### Change the kernel dimensions
_, blurLeft = filter2d(imgLeft, gaussianKernel,(2,2))
_, blurRight = filter2d(imgRight, gaussianKernel,(2,2))

cornersImgLeft, cornersLeft = HarrisDetector(blurLeft, windowSize=5, alpha=0.04, points=1000, nmsWindow=21, threshold=160000, nms=True)
cornersImgRight, cornersRight = HarrisDetector(blurRight, windowSize=5, alpha=0.04, points=1000, nmsWindow=21, threshold=160000, nms=True)
cv2.imshow('corners left', cornersImgLeft)
cv2.imshow('corners right', cornersImgRight)
cv2.waitKey(0)

#%%
leftPatches = extractPatches(imgColLeft, cornersLeft)
rightPatches = extractPatches(imgColRight, cornersRight)

fig = plt.figure(figsize=(20,20))
for i in range(leftPatches.shape[0]):
    fig.add_subplot(20, 20, i+1)
    plt.imshow(leftPatches[i])
plt.show()

fig = plt.figure(figsize=(20,20))
for i in range(rightPatches.shape[0]):
    fig.add_subplot(20, 20, i+1)
    plt.imshow(rightPatches[i])
plt.show()

#%%
whitePatch = np.full((imgColLeft.shape[0], 20, 3), 255, dtype='uint8')
combinedImage = np.column_stack((imgColLeft, whitePatch, imgColRight))
matches = similarPatches(imgColLeft,imgColRight, cornersLeft, cornersRight, thresh=1000)
for match in matches:
    lp, rp = match
    rpm = (rp[0]+imgColLeft.shape[0]+20, rp[1])
    cv2.line(combinedImage, (lp[1], lp[0]), rpm, (0,0,255),thickness=2)
    # combinedImage = cv2.circle(combinedImage, )
cv2.imshow("combined image",combinedImage)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
#%%

#%%

def main():
    imgColLeft = cv2.imread('AlignmentTwoViews/uttower_left.jpg')                     ### Change the input image
    imgLeft = cv2.cvtColor(imgColLeft, cv2.COLOR_BGR2GRAY)
    imgColRight = cv2.imread('AlignmentTwoViews/uttower_right.jpg')
    imgRight = cv2.cvtColor(imgColRight, cv2.COLOR_BGR2GRAY)

    gaussianKernel = GaussianKernel2d(5, 0.5)                   ### Change the kernel dimensions
    _, blurLeft = filter2d(imgLeft, gaussianKernel,(2,2))
    _, blurRight = filter2d(imgRight, gaussianKernel,(2,2))

    cornersImgLeft, cornersLeft = HarrisDetector(blurLeft, windowSize=5, alpha=0.04, points=1000, nmsWindow=21, threshold=160000, nms=True)
    cornersImgRight, cornersRight = HarrisDetector(blurRight, windowSize=5, alpha=0.04, points=1000, nmsWindow=21, threshold=160000, nms=True)
    cv2.imshow('corners left', cornersImgLeft)
    cv2.imshow('corners right', cornersImgRight)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    whitePatch = np.full((imgColLeft.shape[0], 20, 3), 255, dtype='uint8')
    combinedImage = np.column_stack((imgColLeft, whitePatch, imgColRight))
    matches = similarPatches(imgColLeft,imgColRight, cornersLeft, cornersRight, thresh=10000)
    for match in matches:
        lp, rp = match
        rpm = (rp[0], rp[1]+imgColLeft.shape[0]+20)
        cv2.line(combinedImage, (lp[1], lp[0]), rpm, (0,0,255),thickness=2)
        # combinedImage = cv2.circle(combinedImage, )
    cv2.imshow("combined image",combinedImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

# %%
