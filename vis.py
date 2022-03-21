import vtkmodules.vtkFiltersSources
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingCore
import vtkmodules.vtkCommonColor
import colorsys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageColor
from random import random, randint
import math

# GENERATING IMAGE
IMAGE_WIDTH = 4096
IMAGE_HEIGHT = 2160


def get_coords(galaxy):
    x = galaxy[0]
    y = galaxy[1]
    z = galaxy[2]

    u = (((x-0.5)/z)+0.5)*IMAGE_WIDTH
    v = (((y-0.5)/z)+0.5)*IMAGE_HEIGHT
    return u, v


def get_colour(galaxy):
    z = galaxy[2]
    return ImageColor.getrgb(f"hsv({int(240*(1-z))},100%,100%)")


galaxies = []
for i in range(3000):
    galaxies.append([random(), random(), random()**2, randint(0, 360)])
galaxies.sort(key=lambda x: x[2], reverse=True)
gal = Image.open("./galaxy.png")
im = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT))

draw = ImageDraw.Draw(im)
for galaxy in galaxies:
    draw.bitmap(get_coords(galaxy), gal.resize((
        round(random()*IMAGE_WIDTH/50+10), round(random()*IMAGE_HEIGHT/50+10))).rotate(galaxy[3]), fill=get_colour(galaxy))
im.save("gen.png")


# PROCESSING IMAGE


IMAGE_WIDTH = 4096
IMAGE_HEIGHT = 2160
HUEFACTOR = 0.05

im = cv2.imread("gen.png")

params = cv2.SimpleBlobDetector_Params()

params.filterByColor = True
params.blobColor = 255
params.minThreshold = 1

params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.minInertiaRatio = 4
params.filterByArea = True
params.minArea = 4
params.maxArea = 300
params.minDistBetweenBlobs = 10


# Detect blobs
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(im)

galaxies = []

# for each detected blob
for point in keypoints:
    # Get the area around the blob
    coords = (round(point.pt[1]), round(point.pt[0]))
    r = round(1.5 * point.size)
    xmin = max(0, coords[0] - r)
    xmax = min(2159, coords[0] + r)
    ymin = max(0, coords[1] - r)
    ymax = min(4095, coords[1] + r)

    hues = []

    # For each non-black pixel, add the hue of the pixel to list of hues
    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            col = im[i, j]
            if not np.equal(col, (0, 0, 0)).all():
                red, green, blue = col
                hues.append((i, j, colorsys.rgb_to_hsv(red, green, blue)[0]))

    w = xmax-xmin
    h = ymax-ymin
    sumnation = 0
    count = 0
    # Weighted average of hues
    #   (accounting for circularity of hues by using smaller distance (either clockwise or anticlockwise))
    #   (weighted by distance from centre of blob)
    for pi in hues:
        weight = 1/math.sqrt(abs(pi[0]-w/2)**2 + abs(pi[1]-h/2)**2)
        if count != 0:
            if abs(pi[2] - (sumnation/count)) < 0.5:
                sumnation += pi[2] * weight
            else:
                if pi[2] - (sumnation/count) > 0.5:
                    sumnation += (pi[2] - 1) * weight
                else:
                    sumnation += (pi[2] + 1) * weight
        else:
            sumnation += pi[2] * weight
        count += weight

    hue = sumnation/count

    z = (hue)/(240/360)
    u = coords[0]
    v = coords[1]
    x = ((((u/IMAGE_WIDTH) - 0.5) * z) + 0.5)
    y = ((((u/IMAGE_HEIGHT) - 0.5) * z) + 0.5)
    image = np.copy(im[xmin:xmax, ymin:ymax])
    for x, row in enumerate(image):
        for y, pixel in enumerate(row):
            pixelhue = colorsys.rgb_to_hsv(pixel[0], pixel[1], pixel[2])[0]
            if hue - pixelhue > HUEFACTOR or ((hue+0.5) % 1) - ((pixelhue + 0.5) % 1) > HUEFACTOR:
                image[x, y] = np.array([0, 0, 0])
    galaxies.append((x, y, z, image))


# VISUALISATION:

colors = vtkmodules.vtkCommonColor.vtkNamedColors()
bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
colors.SetColor("BkgColor", *bkg)
plane = vtkmodules.vtkFiltersSources.vtkPlaneSource()
plane.
