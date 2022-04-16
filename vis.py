import vtk
from vtk.util.numpy_support import numpy_to_vtk


import colorsys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageColor
from random import random, randint
import math
import operator

# GENERATING IMAGE
IMAGE_WIDTH = 4096
IMAGE_HEIGHT = 2160


def get_coords(galaxy):
    x = galaxy[0]
    y = galaxy[1]

    u = x*IMAGE_WIDTH
    v = y*IMAGE_HEIGHT
    return u, v


def get_colour(galaxy):
    z = galaxy[2]
    return ImageColor.getrgb(f"hsv({int(240*(1-z))},100%,100%)")


galaxies = []
for i in range(100):
    galaxies.append([random(), random(), random(), randint(0, 360)])
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

    z = (hue/240)*360
    z = 1 if z > 1 else z
    z = 0 if z < 0 else z

    u = coords[0]
    v = coords[1]
    x = u/IMAGE_WIDTH
    y = v/IMAGE_HEIGHT
    # print(x)
    # print(y)
    # x = u/IMAGE_WIDTH
    # y = v/IMAGE_HEIGHT
    image = np.copy(im[xmin:xmax, ymin:ymax])
    for a, row in enumerate(image):
        for b, pixel in enumerate(row):
            pixelhue = colorsys.rgb_to_hsv(pixel[0], pixel[1], pixel[2])[0]
            if hue - pixelhue > HUEFACTOR or ((hue+0.5) % 1) - ((pixelhue + 0.5) % 1) > HUEFACTOR:
                image[a, b] = np.array([0, 0, 0])
    galaxies.append([x, y, z, image, None])


# VISUALISATION:

# Setup render window, renderer, and interactor
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetWindowName('Quad')
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)


colors = vtk.vtkNamedColors()

lines = []

lineoffset = 500


def callback(obj, _):
    scale = obj.GetRepresentation().GetValue()
    rescale(scale)


def callback_button(obj, _):
    global lineoffset
    st = obj.GetRepresentation().GetState()
    lineoffset = 500 if st == 1 else 0
    reline()


def offset_tuple(t, offset):
    return tuple(map(operator.add, t, (offset,)*len(t)))


def reline():
    for line in lines:
        line[3].SetPoint1(offset_tuple(
            galaxies[line[0]][4].GetPosition(), lineoffset))
        line[3].SetPoint2(offset_tuple(
            galaxies[line[1]][4].GetPosition(), lineoffset))


def rescale(scale):
    for galaxy in galaxies:
        galaxy[4].SetPosition(galaxy[0], galaxy[1]*0.6, -scale*galaxy[2])
    reline()


obj = []
for i, galaxy in enumerate(galaxies):

    image = galaxy[3][..., ::-1]
    size = image.shape[0], image.shape[1]
    offset = size[0]/2, size[1]/2

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(image.shape[1], image.shape[0], 1)
    image_data.SetNumberOfScalarComponents(
        image.shape[2], image_data.GetInformation())
    pd = image_data.GetPointData()
    new_arr = image[::-1].reshape((-1, image.shape[2]))
    pd.SetScalars(numpy_to_vtk(new_arr))
    pd._numpy_reference = new_arr.data
    # vtkarr = numpy_to_vtk(np.flip(image.swapaxes(
    # 0, 1), axis=1).reshape((-1, 3), order='F'))
    vtkarr = numpy_to_vtk(image.reshape((-1, 3), order='F'))
    vtkarr.SetName('Image')
    image_data.GetPointData().AddArray(vtkarr)
    image_data.GetPointData().SetActiveScalars('Image')

    atext = vtk.vtkTexture()
    atext.SetInputDataObject(image_data)
    atext.InterpolateOn()
    atext.Update()

    p0 = [0, 0, 0]
    p1 = [1, 0, 0]
    p2 = [1, 1, 0]
    p3 = [0, 1, 0]

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

    actor.SetScale(size[0]/IMAGE_WIDTH,
                   size[1]/IMAGE_HEIGHT, 1)
    galaxies[i][4] = actor
    renderer.AddActor(actor)


selected = [randint(0, len(galaxies))]


def select_new_galaxy():
    rand = randint(0, len(galaxies))
    if rand in selected:
        rand = select_new_galaxy()
    selected.append(rand)
    return rand


sources = []

mappers = []

for i in range(6):
    sources.append(vtk.vtkLineSource())
    rand1 = randint(0, len(selected))
    rand2 = select_new_galaxy()
    mappers.append(vtk.vtkPolyDataMapper())
    mappers[-1].SetInputConnection(sources[-1].GetOutputPort())
    lines.append([rand2, selected[rand1], vtk.vtkActor(), sources[-1]])
    lines[-1][2].SetMapper(mappers[-1])
    renderer.AddActor(lines[-1][2])

rescale(0)


renderer.GetActiveCamera().SetFocalPoint((0.25, 0.5, -0.2))
renderer.GetActiveCamera().SetPosition((0.25, 0.5, 1.5))
renderer.GetActiveCamera().SetRoll(270)
renderer.SetBackground(colors.GetColor3d('Black'))
renderer.AutomaticLightCreationOff()
light = renderer.MakeLight()
light.SetDiffuseColor(0, 0, 0)
light.SetSpecularColor(0, 0, 0)
light.SetAmbientColor(255, 255, 255)

sliderRep = vtk.vtkSliderRepresentation2D()
sliderRep.SetTitleText("3Dify")
sliderRep.GetTitleProperty().SetColor(0, 1, 0)
sliderRep.GetTitleProperty().ShadowOff()
sliderRep.GetSliderProperty().SetColor(0, 0, 1)
sliderRep.GetTubeProperty().SetColor(1, 0, 0)
sliderRep.GetCapProperty().SetColor(1, 1, .5)
sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
sliderRep.GetPoint1Coordinate().SetValue(0.2, 0.1)
sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
sliderRep.GetPoint2Coordinate().SetValue(0.8, 0.1)
sliderRep.SetMinimumValue(0)
sliderRep.SetMaximumValue(1)
sliderRep.SetValue(0)
sliderWidget = vtk.vtkSliderWidget()
sliderWidget.SetInteractor(renderWindowInteractor)
sliderWidget.SetRepresentation(sliderRep)
sliderWidget.AddObserver("InteractionEvent", callback)
sliderWidget.EnabledOn()


def im(image):
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(image.shape[1], image.shape[0], 1)
    image_data.SetNumberOfScalarComponents(
        image.shape[2], image_data.GetInformation())
    pd = image_data.GetPointData()
    new_arr = image[::-1].reshape((-1, image.shape[2]))
    pd.SetScalars(numpy_to_vtk(new_arr))
    pd._numpy_reference = new_arr.data
    vtkarr = numpy_to_vtk(np.flip(image.swapaxes(
        0, 1), axis=1).reshape((-1, 3), order='F'))
    # vtkarr = numpy_to_vtk(image.reshape((-1, 3), order='F'))
    vtkarr.SetName('Image')
    image_data.GetPointData().AddArray(vtkarr)
    image_data.GetPointData().SetActiveScalars('Image')

    return image_data


nocons = im(cv2.imread("noCons.png"))
cons = im(cv2.imread("Cons.png"))
buttonRep = vtk.vtkTexturedButtonRepresentation2D()
buttonRep.SetNumberOfStates(2)
buttonRep.SetState(1)
buttonRep.SetButtonTexture(0, cons)
buttonRep.SetButtonTexture(1, nocons)
upperRight = vtk.vtkCoordinate()
upperRight.SetCoordinateSystemToNormalizedDisplay()
upperRight.SetValue(1.0, 1.0)
bds = [0] * 6
sz = 100
bds[0] = upperRight.GetComputedDisplayValue(renderer)[0] - sz
bds[1] = bds[0] + sz
bds[2] = upperRight.GetComputedDisplayValue(renderer)[1] - sz
bds[3] = bds[2] + sz
bds[4] = bds[5] = 0.0

buttonRep.SetPlaceFactor(1)
buttonRep.PlaceWidget(bds)

buttonWidget = vtk.vtkButtonWidget()
buttonWidget.SetInteractor(renderWindowInteractor)
buttonWidget.SetRepresentation(buttonRep)
buttonWidget.AddObserver("StateChangedEvent", callback_button)
buttonWidget.On()
buttonWidget.EnabledOn()

renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
renderWindow.Render()
renderWindowInteractor.Start()
