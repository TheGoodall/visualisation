from PIL import Image, ImageDraw, ImageColor
from random import random, randint
import math

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


def main():
    galaxies = []
    for i in range(10000):
        galaxies.append([random(), random(), random()**2, randint(0, 360)])
    galaxies.sort(key=lambda x: x[2], reverse=True)
    gal = Image.open("./galaxy.png")
    im = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT))

    draw = ImageDraw.Draw(im)
    for galaxy in galaxies:
        draw.bitmap(get_coords(galaxy), gal.resize((
            round(random()*IMAGE_WIDTH/100+1), round(random()*IMAGE_HEIGHT/100+1))).rotate(galaxy[3]), fill=get_colour(galaxy))
    im.save("gen.png")

if __name__ == '__main__':
    main()
