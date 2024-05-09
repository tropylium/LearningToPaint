import torch
import torch.nn.functional as F
import time

# subclass nn.Module to take advantage of nn.DataParallel
class FastStrokeGenerator(torch.nn.Module):
    def __init__(self, batch_size, width):
        super(FastStrokeGenerator, self).__init__()
        self.batch_size = batch_size
        self.width = width
        self.num_steps = 100
#         self.filter = self._create_gaussian_filter(0.5)
        
#         x = torch.arange(width, dtype=torch.float32, device=self.device)
#         y = torch.arange(width, dtype=torch.float32, device=self.device)
#         grid_x, grid_y = torch.meshgrid(x, y)
#         self.grid_x = grid_x.view(1, width, width) #.repeat(self.batch_size, self.num_steps, 1, 1)
#         self.grid_y = grid_y.view(1, width, width) #.repeat(self.batch_size, self.num_steps, 1, 1)
        
#         # all of these have shape ()
#         t = torch.linspace(0, 1, self.num_steps, dtype=torch.float32, device=self.device)
#         self.t = t
#         self.minus_t = 1-t
#         self.qbc_start = (self.minus_t)**2
#         self.qbc_middle = 2*(self.minus_t)*t
#         self.qbc_stop = t**2

#         self.init = False
        
#     def _create_gaussian_filter(self, sigma):
#         x = torch.arange(start=-1, end=2, dtype=torch.float32, device=self.device)
#         y = torch.arange(start=-1, end=2, dtype=torch.float32, device=self.device)
#         grid_x, grid_y = torch.meshgrid(x, y)
#         result = torch.exp(-(grid_x**2 + grid_y**2)/2/sigma**2)
#         return result[None, None, :, :]/torch.sum(result)

    def get_cached_values(self):
        cache = (
            self.filter,
            self.grid_x,
            self.grid_y,
            self.t,
            self.minus_t,
            self.qbc_start,
            self.qbc_middle,
            self.qbc_stop,
                )
        return cache
        
    def generate_strokes(self):
        strokes = torch.rand((self.batch_size, 10), dtype=torch.float32)
        return strokes
    
    def forward(self, strokes):
        def scale_to_width(value, width):
            return value * (width - 1) + 0.5

#         start = time.time()
        batch_size = strokes.shape[0] # note nn.DataParallel may mess with self.batch_size
        device = strokes.get_device() # similarly with this
        
        width = self.width
        sigma = 0.5
        x = torch.arange(start=-1, end=2, dtype=torch.float32, device=device)
        y = torch.arange(start=-1, end=2, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(x, y)
        result = torch.exp(-(grid_x**2 + grid_y**2)/2/sigma**2)
        blur_filter = result[None, None, :, :]/torch.sum(result)

        x = torch.arange(width, dtype=torch.float32, device=device)
        y = torch.arange(width, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x = grid_x.view(1, width, width)
        grid_y = grid_y.view(1, width, width)
        
        t = torch.linspace(0, 1, self.num_steps, dtype=torch.float32, device=device)
        minus_t = 1-t
        qbc_start = (minus_t)**2
        qbc_middle = 2*(minus_t)*t
        qbc_stop = t**2
        
#         finish = time.time()
#         if device == 0:
#             print("Init took: ", finish - start)
#         start = finish

        with torch.no_grad():
            # compute all points; all points have shape (batch_size, 2)
            P0 = strokes[:, 0:2]
            P1 = strokes[:, 2:4]
            P2 = strokes[:, 4:6]
            P1 = P0 + (P2 - P0) * P1

            # scale points to width
            P0 = scale_to_width(P0, self.width)
            P1 = scale_to_width(P1, self.width)
            P2 = scale_to_width(P2, self.width)

            # compute and scale radii; shape (batch_size,)
            radii = strokes[:, 6:8]
            radii = 1 + radii * (self.width // 4)
            radius0 = radii[:, 0]
            radius2 = radii[:, 1]

            # get colors; shape (batch_size,)
            color0 = strokes[:, 8]
            color2 = strokes[:, 9]
            
            # print("qbc_start.shape", qbc_start.shape)
            # print("P0.shape", P0.shape)
            # compute interpolated values for each step size
            P = (
                torch.einsum('bd,t->btd', P0, qbc_start) + 
                torch.einsum('bd,t->btd', P1, qbc_middle) + 
                torch.einsum('bd,t->btd', P2, qbc_stop)
            )
            X = P[..., 0]
            Y = P[..., 1]
            radius = (
                torch.einsum('b,t->bt', radius0, minus_t) + 
                torch.einsum('b,t->bt', radius2, t)
            )
            color = (
                torch.einsum('b,t->bt', color0, minus_t) + 
                torch.einsum('b,t->bt', color2, t)
            )
            
#             finish = time.time()
#             if device == 0:
#                 print("Small compute took: ", finish - start)
#             start = finish

            # make canvas
            canvas = torch.zeros((batch_size, self.width, self.width), dtype=torch.float32, device=device)
            for i in range(self.num_steps):
                Xt = X[:, i].view(batch_size, 1, 1)
                Yt = Y[:, i].view(batch_size, 1, 1)
                radius_t = radius[:, i].view(batch_size, 1, 1)
                color_t = color[:, i].view(batch_size, 1, 1)

                is_in_circle = (grid_x - Xt)**2 + (grid_y - Yt)**2 < radius_t**2
                canvas[is_in_circle] = color_t.expand(-1, self.width, self.width)[is_in_circle]
                
#             finish = time.time()
#             if device == 0:
#                 print("Canvas loop took: ", finish - start)
#             start = finish

            # expand channel dimension to pass into conv2d
            canvas = canvas.view(batch_size, 1, self.width, self.width)
            canvas = F.conv2d(canvas, blur_filter, padding=1).squeeze(dim=1) # squeeze in channels
            
#             finish = time.time()
#             if device == 0:
#                 print("Canvas conv took: ", finish - start)
#             start = finish
            return 1 - canvas
