from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import math
import numpy as np
import argparse
import glob
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# resize image while retaining aspect ratio
d = 1024 / image.shape[1]
dim = (1024, int(image.shape[0] * d))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# create a copy of the image to display results
output = image.copy()

# convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# improve contrast accounting for differences in lighting conditions:
# create a CLAHE object to apply contrast limiting adaptive histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)


def calcHistogram(img):
    # create mask
    m = np.zeros(img.shape[:2], dtype="uint8")
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.circle(m, (w, h), 60, 255, -1)

    # calcHist expects a list of images, color channels, mask, bins, ranges
    h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # return normalized "flattened" histogram
    return cv2.normalize(h, h).flatten()


def calcHistFromFile(file):
    img = cv2.imread(file)
    return calcHistogram(img)


# define Enum class
class Enum(tuple): __getattr__ = tuple.index


# Enumerate material types for use in classifier
Material = Enum(('Copper', 'Brass', 'Euro1', 'Euro2'))

# locate sample image files
sample_images_copper = glob.glob("sample_images/copper/*")
sample_images_brass = glob.glob("sample_images/brass/*")
sample_images_euro1 = glob.glob("sample_images/euro1/*")
sample_images_euro2 = glob.glob("sample_images/euro2/*")

# define training data and labels
X = []
y = []

# compute and store training data and labels
for i in sample_images_copper:
    X.append(calcHistFromFile(i))
    y.append(Material.Copper)
for i in sample_images_brass:
    X.append(calcHistFromFile(i))
    y.append(Material.Brass)
for i in sample_images_euro1:
    X.append(calcHistFromFile(i))
    y.append(Material.Euro1)
for i in sample_images_euro2:
    X.append(calcHistFromFile(i))
    y.append(Material.Euro2)


clf = MLPClassifier(solver="lbfgs")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2)


clf.fit(X_train, y_train)
score = int(clf.score(X_test, y_test) * 100)
print("Classifier mean accuracy: ", score)


blurred = cv2.GaussianBlur(gray, (7, 7), 0)


circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100,
                           param1=200, param2=100, minRadius=50, maxRadius=120)


def predictMaterial(roi):
    # calculate feature vector for region of interest
    hist = calcHistogram(roi)

    # predict material type
    s = clf.predict([hist])

    # return predicted material type
    return Material[int(s)]


# todo: refactor
diameter = []
materials = []
coordinates = []

count = 0
if circles is not None:
    # append radius to list of diameters (we don't bother to multiply by 2)
    for (x, y, r) in circles[0, :]:
        diameter.append(r)

    # convert coordinates and radii to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over coordinates and radii of the circles
    for (x, y, d) in circles:
        count += 1

        # add coordinates to list
        coordinates.append((x, y))

        # extract region of interest
        roi = image[y - d:y + d, x - d:x + d]

        # try recognition of material type and add result to list
        material = predictMaterial(roi)
        materials.append(material)

        # write masked coin to file
        if False:
            m = np.zeros(roi.shape[:2], dtype="uint8")
            w = int(roi.shape[1] / 2)
            h = int(roi.shape[0] / 2)
            cv2.circle(m, (w, h), d, (255), -1)
            maskedCoin = cv2.bitwise_and(roi, roi, mask=m)
            cv2.imwrite("extracted/01coin{}.png".format(count), maskedCoin)

        # draw contour and results in the output image
        cv2.circle(output, (x, y), d, (0, 255, 0), 2)
        cv2.putText(output, material,
                    (x - 40, y), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# get biggest diameter
biggest = max(diameter)
i = diameter.index(biggest)

# scale everything according to maximum diameter
# todo: this should be chosen by the user
if materials[i] == "Euro2":
    diameter = [x / biggest * 25.75 for x in diameter]
    scaledTo = "Scaled to 2 Euro"
elif materials[i] == "Brass":
    diameter = [x / biggest * 24.25 for x in diameter]
    scaledTo = "Scaled to 50 Cent"
elif materials[i] == "Euro1":
    diameter = [x / biggest * 23.25 for x in diameter]
    scaledTo = "Scaled to 1 Euro"
elif materials[i] == "Copper":
    diameter = [x / biggest * 21.25 for x in diameter]
    scaledTo = "Scaled to 5 Cent"
else:
    scaledTo = "unable to scale.."

i = 0
total = 0
while i < len(diameter):
    d = diameter[i]
    m = materials[i]
    (x, y) = coordinates[i]
    t = "Unknown"

    # compare to known diameters with some margin for error
    if math.isclose(d, 25.75, abs_tol=1.25) and m == "Euro2":
        t = "2 Euro"
        total += 200
    elif math.isclose(d, 23.25, abs_tol=2.5) and m == "Euro1":
        t = "1 Euro"
        total += 100
    elif math.isclose(d, 19.75, abs_tol=1.25) and m == "Brass":
        t = "10 Cent"
        total += 10
    elif math.isclose(d, 22.25, abs_tol=1.0) and m == "Brass":
        t = "20 Cent"
        total += 20
    elif math.isclose(d, 24.25, abs_tol=2.5) and m == "Brass":
        t = "50 Cent"
        total += 50
    elif math.isclose(d, 16.25, abs_tol=1.25) and m == "Copper":
        t = "1 Cent"
        total += 1
    elif math.isclose(d, 18.75, abs_tol=1.25) and m == "Copper":
        t = "2 Cent"
        total += 2
    elif math.isclose(d, 21.25, abs_tol=2.5) and m == "Copper":
        t = "5 Cent"
        total += 5

    # write result on output image
    cv2.putText(output, t,
                (x - 40, y + 22), cv2.FONT_HERSHEY_PLAIN,
                1.5, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    i += 1

# resize output image while retaining aspect ratio
d = 768 / output.shape[1]
dim = (768, int(output.shape[0] * d))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)

# write summary on output image
cv2.putText(output, scaledTo,
            (5, output.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)
cv2.putText(output, "Coins detected: {}, EUR {:2}".format(count, total / 100),
            (5, output.shape[0] - 24), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)
cv2.putText(output, "Classifier mean accuracy: {}%".format(score),
            (5, output.shape[0] - 8), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)

# show output and wait for key to terminate program
cv2.imshow("Output", np.hstack([image, output]))
cv2.waitKey(0)
