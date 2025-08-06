# This module contains shape data generation utilities, some adapted from the FFJORD project.
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
from dataset_utils import *
import torch
import random

shape_names = [
        "swissroll",
        "circles",
        "rings",
        "moons",
        "8gaussians",
        "square",
        "1gaussians",
        "2gaussians",
        "pinwheel",
        "2spirals",
        "checkerboard",
        "line",
        "cos",
        "tri",
        "rect",
        "oval",
        "star",
        "heart",
        "crescent"
    ]

def get_random_shape_data(batch_size=1024):
        
    shape_name = random.choice(shape_names)
    points = test_shapes(shape_name, batch_size=batch_size)
    
    # Normalize the shape to [-1, 1] range
    data = normalize_shape(points)
    
    return  data



def normalize_shape(points, target_min=-1, target_max=1):
    """
    Normalize a shape defined by 2D points to fit within [-1, 1] while preserving proportions.
    
    Args:
        points: numpy array of shape (n, 2) containing the 2D points
        target_min: minimum value of target range (default -1)
        target_max: maximum value of target range (default 1)
        
    Returns:
        Normalized points array of same shape
    """
    points = np.asarray(points, dtype=np.float32)
    
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input must be a 2D array with shape (n, 2)")
    
    # Find the current bounds of the shape
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Calculate current dimensions
    current_width = max_coords[0] - min_coords[0]
    current_height = max_coords[1] - min_coords[1]
    
    # Find required scaling factor (we'll use the same for x and y to preserve proportions)
    scale_x = (target_max - target_min) / (max_coords[0] - min_coords[0]) if current_width > 0 else 1.0
    scale_y = (target_max - target_min) / (max_coords[1] - min_coords[1]) if current_height > 0 else 1.0
    scale = min(scale_x, scale_y)  # Use the most restrictive scaling
    
    # Center the shape first
    center = (min_coords + max_coords) / 2.0
    centered = points - center
    
    # Scale to fit [-1, 1] while preserving proportions
    normalized = centered * scale *(max(torch.rand(1).item(),0.3))
    
    centered_tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
    centered_tensor = centered_tensor - centered_tensor.mean(dim=1, keepdim=True)  # ✅ 手动居中
    return centered_tensor




# Dataset iterator
def test_shapes(data, dim=2, rng=None, batch_size=1024):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "square":
        dataset = []
        for i in range(batch_size // 2):  # Divide by 2 to get correct count
            point = (rng.rand(2) - 0.5)*2
            point[0] -= 1
            point[1] -= 2
            dataset.append(point)
            
            point = (rng.rand(2) - 0.5)*4
            point[0] -= 1
            point[1] += 2
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        return dataset

    elif data == "1gaussians":
        centers = [(0,0)]

        dataset = []
        values  = []
        sigma = 0.5 # variance
        for i in range(batch_size):
            # point = rng.randn(2) * 1.5
            point = rng.randn(2) * sigma
            dataset.append(point)
            val = (point[0])**2 + (point[1])**2
            values.append(np.exp(-val / (2.0 * sigma**2 ) ))
        dataset = np.array(dataset, dtype="float32")
        values  = np.array(values,  dtype="float32")
        return dataset

    

    elif data == "2gaussians":
        center1 = np.zeros(dim)
        centers = [center1]

        dataset = []
        values  = []
        i = 0
        idx= 0
        sigma = 0.5 # standard deviation
        while i < batch_size:
            i+=1
            
            point = np.random.randn(dim) * sigma
            center = centers[idx]
            point += center
            dataset.append(point)

            val = 0
            for j in range(len(centers)):
                val += np.exp(- (np.sum((point-centers[j])**2))/(2.0 * sigma**2 ))
            values.append(1.0/(np.sqrt(sigma * np.pi))**2 * val)
            idx = (idx+1)%len(centers)
        dataset = np.array(dataset, dtype="float32")
        values  = np.array(values,  dtype="float32")
        return dataset

        


    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        remainder = batch_size % 5  # Handle the remainder
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        # Create label array that handles the remainder
        labels = []
        for i in range(num_classes):
            count = num_per_class + (1 if i < remainder else 0)
            labels.extend([i] * count)
        labels = np.array(labels)
        
        # Generate features for exact batch_size
        features = rng.randn(batch_size, 2) * np.array([radial_std, tangential_std])
        features[:, 0] += 1.

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    elif data == "tri":
        return generate_random_triangles(1,batch_size)[0].numpy()
    elif data == "rect":
        return generate_random_rectangles(1,batch_size)[0].numpy()
    elif data == "oval":
        return generate_random_ovals(1,batch_size)[0].numpy()
    elif data == "star":
        return generate_random_stars(1,batch_size)[0].numpy()
    elif data == "heart":
        return generate_random_hearts(1,batch_size)[0].numpy()
    elif data == "crescent":
        return generate_random_crescents(1,batch_size)[0].numpy()
    else:
        return np.random.rand(batch_size, dim) * 2 - 1



def generate_random_ovals(B, sample_size, min_axis=0.3):
    """
    Generate uniform points within random ovals (ellipses)
    Parameters:
        B: batch size
        sample_size: points per oval
        min_axis: minimum axis length (default 0.3)
    Returns:
        torch.Tensor: shape (B, sample_size, 2)
    """
    # Random axes and rotation
    a = torch.rand(B, 1) * (1 - min_axis) + min_axis
    b = torch.rand(B, 1) * (1 - min_axis) + min_axis
    theta = torch.rand(B, 1) * 2 * torch.pi
    
    # Random centers
    max_pos_x = 1 - a
    max_pos_y = 1 - b
    centers_x = torch.rand(B, 1) * (2*max_pos_x) - max_pos_x
    centers_y = torch.rand(B, 1) * (2*max_pos_y) - max_pos_y
    
    # Generate points in unit circle and transform to ellipse
    phi = torch.rand(B, sample_size) * 2 * torch.pi
    r = torch.sqrt(torch.rand(B, sample_size))
    
    x = r * torch.cos(phi) * a
    y = r * torch.sin(phi) * b
    
    # Apply rotation
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    points = torch.stack([
        x * cos_theta - y * sin_theta + centers_x,
        x * sin_theta + y * cos_theta + centers_y
    ], dim=-1)
    
    return points.clamp(-1, 1)


def generate_random_stars(B, sample_size, min_radius=0.3, points=5):
    """
    Generate uniform points within random stars
    Parameters:
        B: batch size
        sample_size: points per star
        min_radius: minimum radius (default 0.3)
        points: number of star points (default 5)
    Returns:
        torch.Tensor: shape (B, sample_size, 2)
    """
    result = torch.zeros(B, sample_size, 2)
    
    for i in range(B):
        # Random center and size
        outer_r = torch.rand(1) * (1 - min_radius) + min_radius
        inner_r = outer_r * (0.3 + 0.4 * torch.rand(1))  # Inner radius 30-70% of outer
        center = torch.rand(2) * (2 - 2*outer_r) - (1 - outer_r)
        
        # Generate star vertices
        angles = torch.linspace(0, 2*torch.pi, points+1)[:-1]
        outer_vertices = torch.stack([outer_r * torch.cos(angles), outer_r * torch.sin(angles)], dim=1)
        inner_vertices = torch.stack([inner_r * torch.cos(angles + torch.pi/points), 
                                     inner_r * torch.sin(angles + torch.pi/points)], dim=1)
        
        vertices = torch.zeros(2*points, 2)
        vertices[::2] = outer_vertices
        vertices[1::2] = inner_vertices
        
        # Generate random points in convex hull
        triangles = []
        for j in range(2*points):
            triangles.append(torch.stack([vertices[j], vertices[(j+1)%(2*points)], center]))
        triangles = torch.stack(triangles)
        
        # Sample points using barycentric coordinates
        u = torch.rand(sample_size)
        v = torch.rand(sample_size)
        mask = (u + v) > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - (u + v)
        
        # Select random triangles
        tri_idx = torch.randint(0, 2*points, (sample_size,))
        selected_tris = triangles[tri_idx]
        
        result[i] = (u.unsqueeze(1) * selected_tris[:,0] + 
                    v.unsqueeze(1) * selected_tris[:,1] + 
                    w.unsqueeze(1) * selected_tris[:,2]) + center
    
    return result.clamp(-1, 1)


def generate_random_hearts(B, sample_size, min_size=0.3):
    """
    Generate uniform points within random hearts
    Parameters:
        B: batch size
        sample_size: points per heart
        min_size: minimum size (default 0.3)
    Returns:
        torch.Tensor: shape (B, sample_size, 2)
    """
    points = torch.zeros(B, sample_size, 2)
    
    for i in range(B):
        # Random size and position
        size = torch.rand(1) * (1 - min_size) + min_size
        center = torch.rand(2) * (2 - 2*size) - (1 - size)
        
        # Rejection sampling within heart shape
        count = 0
        while count < sample_size:
            candidates = torch.rand(sample_size*2, 2) * 2 - 1  # [-1,1]^2
            
            # Heart equation: (x^2 + y^2 - 1)^3 - x^2*y^3 < 0
            x = candidates[:, 0]
            y = candidates[:, 1]
            mask = (x**2 + y**2 - 1)**3 - x**2 * y**3 < 0
            
            valid = candidates[mask][:sample_size-count]
            points[i, count:count+len(valid)] = valid * size + center
            count += len(valid)
    
    return points.clamp(-1, 1)


def generate_random_crescents(B, sample_size, min_radius=0.3):
    """
    Generate uniform points within random crescents (moon shapes)
    Parameters:
        B: batch size
        sample_size: points per crescent
        min_radius: minimum radius (default 0.3)
    Returns:
        torch.Tensor: shape (B, sample_size, 2)
    """
    points = torch.zeros(B, sample_size, 2)
    
    for i in range(B):
        # Random parameters
        r1 = torch.rand(1) * (1 - min_radius) + min_radius
        r2 = r1 * (0.3 + 0.5 * torch.rand(1))  # r2 between 30-80% of r1
        d = r1 * (0.1 + 0.8 * torch.rand(1))   # distance between centers
        
        # Random position and angle
        angle = torch.rand(1) * 2 * torch.pi
        center = torch.rand(2) * (2 - 2*r1) - (1 - r1)
        
        # Rejection sampling
        count = 0
        while count < sample_size:
            candidates = torch.rand(sample_size*2, 2) * 2 - 1  # [-1,1]^2
            
            # Check if in outer circle but not in inner circle
            offset = torch.tensor([d * torch.cos(angle), d * torch.sin(angle)])
            in_outer = (candidates**2).sum(dim=1) < r1**2
            in_inner = ((candidates - offset)**2).sum(dim=1) < r2**2
            mask = in_outer & ~in_inner
            
            valid = candidates[mask][:sample_size-count]
            points[i, count:count+len(valid)] = valid + center
            count += len(valid)
    
    return points.clamp(-1, 1)


def generate_random_rectangles(B, sample_size, min_side=0.3):
    """
    生成随机矩形内的均匀采样点
    参数:
        B: batch大小
        sample_size: 每个矩形的采样点数
        min_side: 最小边长 (默认0.3)
    返回:
        torch.Tensor: shape (B, sample_size, 2)
    """
    # 随机生成矩形参数
    widths = torch.rand(B, 1) * (2 - min_side) + min_side  # 宽度 ∈ [0.3, 2]
    heights = torch.rand(B, 1) * (2 - min_side) + min_side # 高度 ∈ [0.3, 2]
    
    # 随机生成中心点 (确保整个矩形在[-1,1]^2内)
    max_pos_x = 1 - widths/2
    max_pos_y = 1 - heights/2
    centers_x = torch.rand(B, 1) * (2*max_pos_x) - max_pos_x  # x ∈ [-1+width/2, 1-width/2]
    centers_y = torch.rand(B, 1) * (2*max_pos_y) - max_pos_y  # y ∈ [-1+height/2, 1-height/2]
    
    # 生成均匀分布的点
    points = torch.rand(B, sample_size, 2) * 2 - 1  # 初始在[-1,1]^2
    
    # 缩放到各矩形范围内
    points[..., 0] = centers_x + (points[..., 0] * widths/2)
    points[..., 1] = centers_y + (points[..., 1] * heights/2)
    
    return points.clamp(-1, 1)  # 确保不越界


def generate_random_triangles(B, sample_size, min_side=0.3):
    """
    生成随机三角形内的均匀采样点
    参数:
        B: batch大小
        sample_size: 每个三角形的采样点数
        min_side: 最小边长 (默认0.3)
    返回:
        torch.Tensor: shape (B, sample_size, 2)
    """
    points = torch.zeros(B, sample_size, 2)
    
    for i in range(B):
        # 生成第一个随机顶点
        v1 = torch.rand(2) * 2 - 1
        
        # 生成第二个顶点 (保证与v1距离≥min_side)
        while True:
            v2 = torch.rand(2) * 2 - 1
            if torch.norm(v2 - v1) >= min_side:
                break
                
        # 生成第三个顶点 (保证与v1/v2距离≥min_side且能形成有效三角形)
        while True:
            v3 = torch.rand(2) * 2 - 1
            d12 = torch.norm(v2 - v1)
            d13 = torch.norm(v3 - v1)
            d23 = torch.norm(v3 - v2)
            if (d12 >= min_side) and (d13 >= min_side) and (d23 >= min_side):
                # 检查三角形有效性 (面积不为零)
                area = 0.5 * torch.abs((v2[0]-v1[0])*(v3[1]-v1[1]) - (v2[1]-v1[1])*(v3[0]-v1[0]))
                if area > 1e-6:
                    break
        
        # 三角形内均匀采样
        u = torch.rand(sample_size, 1)
        v = torch.rand(sample_size, 1)
        mask = (u + v) > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - (u + v)
        
        points[i] = u * v1 + v * v2 + w * v3
    
    return points.clamp(-1, 1)
