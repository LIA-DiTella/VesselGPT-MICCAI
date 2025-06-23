import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from tree_functions import *
import torch
from scipy.interpolate import splev, splprep

def sample_spline(coeffs, n_samples):

	coeffs = list(coeffs)

	t = np.array(coeffs[24:])
	c = [np.array(coeffs[i*8:(i*8)+8]) for i in range(3)]

	tck = (t, c, 3)

	u = np.linspace(0, 1, n_samples)
	x, y, z = splev(u, tck)

	return np.column_stack((x, y, z))

def draw_tree_splines_aux(ax, tree):

	if tree == None: return

	points = sample_spline(tree.data["r"], n_samples=50)
	ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='.', s = 0.5)

	draw_tree_splines_aux(ax, tree.right)
	draw_tree_splines_aux(ax, tree.left)

def draw_tree_splines(tree):

	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111, projection='3d')

	draw_tree_splines_aux(ax, tree)

	root = (tree.data["x"], tree.data["y"], tree.data["z"])
	ax.scatter(root[0], root[1], root[2], c='g', marker='*', s = 25)

def draw_tree_aux(tree, points, lines):

    points["x"].append(tree.data["x"])
    points["y"].append(tree.data["y"])
    points["z"].append(tree.data["z"])
    points["r"].append(tree.data["r"])

    if tree.left != None: 

        lines.append({"x1": tree.data["x"], "x2": tree.left.data["x"],
                      "y1": tree.data["y"], "y2": tree.left.data["y"],
                      "z1": tree.data["z"], "z2": tree.left.data["z"],
                      "r1": tree.data["r"], "r2": tree.left.data["r"]})

        draw_tree_aux(tree.left, points, lines)

    if tree.right != None: 
        
        lines.append({"x1": tree.data["x"], "x2": tree.right.data["x"],
                      "y1": tree.data["y"], "y2": tree.right.data["y"],
                      "z1": tree.data["z"], "z2": tree.right.data["z"],
                      "r1": tree.data["r"], "r2": tree.right.data["r"]})
        
        draw_tree_aux(tree.right, points, lines)

def draw_tree(tree, set_lims = False):

    points = {"x":[],"y":[],"z":[],"r":[]}
    lines = []

    draw_tree_aux(tree, points, lines)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax.scatter(points["x"], points["y"], points["z"], c='r', marker='o', s = np.array(points["r"]) * 1000)

    for line in lines:

        ax.plot([line["x1"],line["x2"]],
                [line["y1"],line["y2"]],
                [line["z1"],line["z2"]], c='k')
        
    if set_lims:
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

    plt.show()

def draw_tree_from_numpy(data, mode = "pre_order", threshold = 1e-2, k = 4):

    data[np.abs(data) < threshold] = 0
    serial = list(data.flatten())

    tree = deserialize(serial, mode, k)        
    draw_tree(tree)

def draw_tree_from_tokens(tokens, mode = "pre_order", threshold = 1e-2, device = None, decoder = None, k = 4):

    data = tokens_to_data(tokens, device, decoder)
    if data != None: draw_tree_from_numpy(data, mode, threshold, k = k)

def draw_tree_from_file(file, type, mode = "pre_order", threshold = 1e-2, device = None, decoder = None):

    if type == "tokens":                        # .tok files
        
        tokens = torch.load(file).to(device)
        draw_tree_from_tokens(tokens, mode, threshold, device, decoder)

    elif type == "numpy":                       # .npy files
        
        data = np.load(file)                    
        draw_tree_from_numpy(data, mode, threshold)

    elif type == "tree":                        # .bto files
        
        tree = torch.load(file)
        draw_tree(tree)

    else:
        raise Exception("UNSUPPORTED TREE FILE TYPE | valid are 'tokens', 'numpy' or 'tree' ")

def draw_tree_grid(trees, set_lims = False, figsize = 2, params = [], title = ""):

    sqrt = int(math.sqrt(len(trees)))
    rows, cols = (sqrt, sqrt)

    fig = plt.figure(figsize=(cols * figsize, rows * figsize))

    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.subplots_adjust(top=0.9) 
    
    for idx, tree in enumerate(trees):

        if idx >= rows * cols:
            break  
        
        points = {"x": [], "y": [], "z": [], "r": []}
        lines = []

        draw_tree_aux(tree, points, lines)
        
        # param managing

        radius_alpha = params[idx]["radius_alpha"] if "radius_alpha" in params[idx] else 50

        node_color = params[idx]["node_color"] if "node_color" in params[idx] else "r"
        node_size = params[idx]["node_size"] if "node_size" in params[idx] else 10

        line_color = params[idx]["line_color"] if "line_color" in params[idx] else "r"
        line_width = params[idx]["line_width"] if "line_width" in params[idx] else 0

        text = params[idx]["text"] if "text" in params[idx] else ""

        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        ax.scatter(points["x"], points["y"], points["z"], c = "k", marker = 'o', s = node_size)
        
        for line in lines:

            ax.plot([line["x1"], line["x2"]],
                    [line["y1"], line["y2"]],
                    [line["z1"], line["z2"]], c = line_color, linewidth = line_width if line_width != 0 else (line["r1"] + line["r2"]) / 2 * radius_alpha )

            # linewidth = 

        if set_lims:

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        ax.set_title(f'Tree {idx} \n {text}', fontsize = 9)
    
    fig.tight_layout()
    plt.show()

# -----------------------------------------------------------------------


