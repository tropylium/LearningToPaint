import cv2
import numpy as np
import torch
import torch.nn.functional as F

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

class FastStrokeGenerator():
    def __init__(self, batch_size, width, device):
        self.batch_size = batch_size
        self.width = width
        self.device = device
        
        self.num_steps = 100
        self.filter = self._create_gaussian_filter(0.5)
        
        x = torch.arange(width, dtype=torch.float32, device=self.device)
        y = torch.arange(width, dtype=torch.float32, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.grid_x = grid_x[None, None, :, :].repeat(self.batch_size, self.num_steps, 1, 1)
        self.grid_y = grid_y[None, None, :, :].repeat(self.batch_size, self.num_steps, 1, 1)
        
        t = torch.arange(0, 1, 1.0/self.num_steps, dtype=torch.float32, device=self.device)[None, :]
        self.t = t
        self.minus_t = 1-t
        self.qbc_start = ((self.minus_t)**2)[..., None]
        self.qbc_middle = (2*(self.minus_t)*t)[..., None]
        self.qbc_stop = (t**2)[..., None]
        
    def _create_gaussian_filter(self, sigma):
        x = torch.arange(start=-1, end=2, dtype=torch.float32, device=self.device)
        y = torch.arange(start=-1, end=2, dtype=torch.float32, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)
        result = torch.exp(-(grid_x**2 + grid_y**2)/2/sigma**2)
        return result[None, None, :, :]/torch.sum(result)
        
    def _generate_strokes(self):
        strokes = torch.rand((self.batch_size, 10), dtype=torch.float32, device=self.device)
        return strokes
    
    def get_batch(self):
        def scale_to_width(value, width):
            return value * (width - 1) + 0.5
        
        strokes = self._generate_strokes()
#         print("strokes", strokes)
        P0 = strokes[:, 0:2]
        P1 = strokes[:, 2:4]
        P2 = strokes[:, 4:6]
        P1 = P0 + (P2 - P0) * P1
        
        P0 = scale_to_width(P0, self.width)
        P1 = scale_to_width(P1, self.width)
        P2 = scale_to_width(P2, self.width)
        
        radii = strokes[:, 6:8]
        radii = 1 + radii * (self.width // 4)
        
        radius0 = radii[:, 0:1]
        radius2 = radii[:, 1:2]
        
        # clone since we may write to it
        color0 = strokes[:, 8:9].clone()
        color2 = strokes[:, 9:10].clone()
        
        # neat trick to allow us to use torch.max to simulate layering circles
        color_signs = torch.sign(color2 - color0)
        color2_less = color_signs < 0
        invert_color0 = 1 - color0
        invert_color2 = 1 - color2
        color0[color2_less] = invert_color0[color2_less]
        color2[color2_less] = invert_color2[color2_less]
        
        P0 = torch.unsqueeze(P0, dim=1)
        P1 = torch.unsqueeze(P1, dim=1)
        P2 = torch.unsqueeze(P2, dim=1)
        
        # all should be (batch_size, num_steps, width, width)
        def tile(value):
            return value[:, :, None, None].repeat(1,1,self.width, self.width)
        
        P = self.qbc_start*P0 + self.qbc_middle*P1 + self.qbc_stop*P2 
        X = tile(P[..., 0])
        Y = tile(P[..., 1])
        radius = tile(radius0 * self.minus_t + radius2 * self.t)
        color = tile(color0 * self.minus_t + color2 * self.t)
        
        circles = torch.zeros((self.batch_size, self.num_steps, self.width, self.width), dtype=torch.float32, device=self.device)
        is_in_circle = (self.grid_x - X)**2 + (self.grid_y - Y)**2 < radius**2
        circles[is_in_circle] = color[is_in_circle]
        
        canvas, _ = torch.max(circles, dim=1, keepdim=True)
        should_invert = color2_less[:, None, None] * (canvas > 0)
        canvas[should_invert] = (1 - canvas)[should_invert]
        canvas = F.conv2d(canvas, self.filter, padding=1).squeeze(dim=1) # squeeze in channels
        return strokes, 1 - canvas
        
        