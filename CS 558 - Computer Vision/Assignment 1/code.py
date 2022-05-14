#%%
import numpy as np
import cv2

#%%
def filter2d(image:np.ndarray, filter:np.ndarray, padding:tuple = (1,1)) -> np.ndarray:
    if(image.ndim == 2):
        image = np.expand_dims(image, axis=-1)
    if(filter.ndim == 2):
        filter = np.repeat(np.expand_dims(filter, axis=-1), image.shape[-1], axis=-1)
    if(filter.shape[-1] == 1):
        filter = np.repeat(filter, image.shape[-1], axis=-1)

    size_x, size_y = filter.shape[:2]
    width, height = image.shape[:2]
    
    output_array = np.zeros(((width - size_x + 2*padding[0]) + 1, 
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
            output_array[x, y] = output_values
            
    return output_array

def GaussianKernel2d(size:int = 5, sigma:float =1.) -> np.ndarray:
    halfWidth = size//2
    x = np.arange(-halfWidth, halfWidth+1,1)
    y = np.arange(-halfWidth, halfWidth+1,1)
    gaussianFilter = np.zeros((size,size),dtype=np.float64)
    for i in x:
        for j in y:
            gaussianFilter[i+halfWidth][j+halfWidth] = np.exp(-(i*i + j*j)/(2.*sigma*sigma))
    gaussianFilter = gaussianFilter/(2.*np.pi*sigma*sigma)
    return gaussianFilter

def SobelFiltering(blur:np.ndarray) -> tuple :
    SobelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    SobelY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    gradientX = filter2d(blur, SobelX)
    gradientY = filter2d(blur, SobelY)

    gradient = np.hypot(gradientX, gradientY)
    gradient = gradient.astype(blur.dtype)
    orientation = np.arctan2(gradientY,gradientX)

    gradient = np.squeeze(gradient)
    orientation = np.squeeze(orientation)

    return (gradient, orientation)

def GradientThresh(gradient:np.ndarray, thresh: int = 30) -> np.ndarray:
    gradientThresh = (gradient>thresh) * gradient
    return gradientThresh

def NonMaxSupression(image: np.ndarray, orientation:np.ndarray) -> np.ndarray:
    m,n = image.shape
    z = np.zeros((m,n), dtype=np.int8)
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
                #angle 45
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
#%%

def main():
    img = cv2.imread('input/plane.pgm', 0)                     ### Change the input image
    gaussianKernel = GaussianKernel2d(5, 1.)                   ### Change the kernel dimensions
    blur = filter2d(img, gaussianKernel)
    gradient, orientation = SobelFiltering(blur)
    gradientThresh = GradientThresh(gradient, 30)              ### Change the threshold value accordingly 
    nmsImage = NonMaxSupression(gradientThresh, orientation)
    cv2.imwrite("output/planesample.jpg",nmsImage)

#%%
if __name__ == '__main__':
    main()
# %%
