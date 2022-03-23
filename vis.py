import vtk
from vtk.util.numpy_support import numpy_to_vtk


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
    for a, row in enumerate(image):
        for b, pixel in enumerate(row):
            pixelhue = colorsys.rgb_to_hsv(pixel[0], pixel[1], pixel[2])[0]
            if hue - pixelhue > HUEFACTOR or ((hue+0.5) % 1) - ((pixelhue + 0.5) % 1) > HUEFACTOR:
                image[a, b] = np.array([0, 0, 0])
    galaxies.append((x, y, z, image))


# VISUALISATION:

# Setup render window, renderer, and interactor
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetWindowName('Quad')
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)


colors = vtk.vtkNamedColors()

scale = 1

obj = []
for galaxy in galaxies[:5]:

    image = galaxy[3]
    size = image.shape[0], image.shape[1]
    offset = size[0]/2, size[1]/2

    grid = vtk.vtkImageData()
    grid.SetDimensions(image.shape[1], image.shape[0], 1)
    vtkarr = numpy_to_vtk(np.flip(image.swapaxes(
        0, 1), axis=1).reshape((-1, 3), order='F'))
    vtkarr.SetName('Image')

    grid.GetPointData().AddArray(vtkarr)
    grid.GetPointData().SetActiveScalars('Image')

    atext = vtk.vtkTexture()
    atext.SetInputDataObject(grid)
    atext.InterpolateOn()
    atext.Update()

    p0 = [galaxy[0] - offset[0], galaxy[1] - offset[1], galaxy[2]*scale]
    p1 = [galaxy[0] - offset[0], galaxy[1] + offset[1], galaxy[2]*scale]
    p2 = [galaxy[0] + offset[0], galaxy[1] + offset[1], galaxy[2]*scale]
    p3 = [galaxy[0] + offset[0], galaxy[1] - offset[1], galaxy[2]*scale]

    points = vtk.vtkPoints()
    points.InsertNextPoint(p0)
    points.InsertNextPoint(p1)
    points.InsertNextPoint(p2)
    points.InsertNextPoint(p3)

    # Create a quad on the four points
    quad = vtk.vtkQuad()
    quad.GetPointIds().SetId(0, 0)
    quad.GetPointIds().SetId(1, 1)
    quad.GetPointIds().SetId(2, 2)
    quad.GetPointIds().SetId(3, 3)

    # Create a cell array to store the quad in
    quads = vtk.vtkCellArray()
    quads.InsertNextCell(quad)

    # Create a polydata to store everything in
    polydata = vtk.vtkPolyData()

    # Add the points and quads to the dataset
    polydata.SetPoints(points)
    polydata.SetPolys(quads)

    textureCoordinates = vtk.vtkFloatArray()
    textureCoordinates.SetNumberOfComponents(2)
    textureCoordinates.SetName("TextureCoordinates")
    textureCoordinates.InsertNextTuple((0.0, 0.0))
    textureCoordinates.InsertNextTuple((1.0, 0.0))
    textureCoordinates.InsertNextTuple((1.0, 1.0))
    textureCoordinates.InsertNextTuple((0.0, 1.0))
    polydata.GetPointData().SetTCoords(textureCoordinates)

    # Setup actor and mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().SetColor(colors.GetColor3d('White'))
    actor.SetTexture(atext)

    renderer.AddActor(actor)

renderer.SetBackground(colors.GetColor3d('Black'))
renderWindow.Render()
renderWindowInteractor.Start()
