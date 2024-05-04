import cv2
import numpy as np

def draw(f, width=128, show_points=False):
    """
    Draws a stroke represented by `f` onto a canvas of width `width`, which is represented by a numpy array. The stroke is a quadratic Bezier curve (QBC). show_points is strictly for debugging. 
    
    f is a 10-dimensional np array where all elements are in [0,1]. Thus, it elements should be interpreted as the relative position on the canvas. 
    x0, y0: first control point = P0. 
    x1, y1: determines second control point = P1 by interpolating linearly between P0, P2, ensuring P1 is within the rectangle created by P0, P2. 
    x2, y2: third control point = P2. 
    z0, z2: radius at P0 -> radius at P2, interpolated linearly
    w0, w2: grayscale color at P0 -> grayscale color at P2, interpolated linearly
    """
    def scale_to_width(x, width):
        return (int)(x * (width - 1) + 0.5) # add 0.5 to be in the "middle", though I doubt this makes significant difference
    
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    # compute 
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = scale_to_width(x0, width * 2)
    x1 = scale_to_width(x1, width * 2)
    x2 = scale_to_width(x2, width * 2)
    y0 = scale_to_width(y0, width * 2)
    y1 = scale_to_width(y1, width * 2)
    y2 = scale_to_width(y2, width * 2)
    
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    # double canvas size to account for resolution?
    canvas = np.zeros([width * 2, width * 2], dtype='float32')
    tmp = 1. / 100
    # discrete approximation of shape by repeatedly filling in circles for 100 steps. 
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1) # observe x,y are swapped. this... really shouldn't make a difference. 
    if show_points:
        cv2.circle(canvas, (y0, x0), width//20, 0, -1)
        cv2.circle(canvas, (y1, x1), width//20, 0, -1)
        cv2.circle(canvas, (y2, x2), width//20, 0, -1)
    # resize to original width. I have no idea why this is subtracted from 1. 
    return 1 - cv2.resize(canvas, dsize=(width, width))
