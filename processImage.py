import cv2
import numpy as np
import urllib
import os

CASCADE_PATH = 'bin/haarcascade_frontalface_default.xml'

def _retrieve_image(url):
    file_name = url.split('/')[-1]
    with open('input_imgs/' + file_name, 'wb') as fi:
        fi.write(urllib.urlopen(url).read())

def _delete_input_image(file_name):
    os.remove('input_imgs/' + file_name)

def _load_image(file_name):
    image = cv2.imread(file_name)
    if image is None:
        raise Exception('No image found.')
    else:
        return image

# def __get_em_samples(image):
    # x, y, z = image.shape
    # samples = np.empty([x * y, z])
    # index = 0
    # for i in range(x):
        # for j in range(y):
            # samples[index] = image[i, j]
            # index += 1
    # return samples

# def EMSegmentation(image, no_of_clusters=4):
    # output = image.copy()
    # colors = np.array([[0, 0, 0], [75, 75, 75], [150, 150, 150], [255, 255, 255]])
    # samples = __get_em_samples(image)
    # em = cv2.ml.EM_create()
    # em.setClustersNumber(no_of_clusters)
    # em.trainEM(samples)
    # means = em.getMeans()
    # covs = em.getCovs()
    # x, y, z = image.shape
    # distance = [0] * no_of_clusters
    # for i in range(x):
        # for j in range(y):
            # for k in range(no_of_clusters):
                # diff = image[i, j] - means[k]
                # distance[k] = abs(np.dot(np.dot(diff, covs[k]), diff.T))
            # output[i][j] = colors[distance.index(max(distance))]
    # return output

def _get_gc_mask(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
    rect = faceCascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=4,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    rect[:, 2:] += rect[:, :2]
    for x1, y1, x2, y2 in rect:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # output = EMSegmentation(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5))
    # cv2.imwrite('out.png', output)

    return rect

def _do_graphcut(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1,65), np.float64)
    fgd_model = np.zeros((1,65), np.float64)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return image * mask2[:,:,np.newaxis]

def _resize(image, x, y):
    return image

def extract(file_name):
    image = _load_image(file_name)
    rect = _get_gc_mask(image)
    face = _do_graphcut(image, rect)
    return _resize(face, 128, 128)

if __name__ == "__main__":
    cv2.imwrite('out.jpg', extract('input_imgs/me.jpg'))
