#Transformar arbol a vector depth first con los vectores de atribtos
from Arbol import Node, deserialize2, read_tree, deserialize
import numpy as np
import os
import torch

use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

def traversefeatures(root, features):
       
    if root is not None:
        traversefeatures(root.left, features)
        #features.append(root.radius.tolist())
        features.append(root.radius)
        traversefeatures(root.right, features)
        return features
    

def traverseInorder(root):
    if root is not None:
        traverseInorder(root.left)
        print (root.data, root.radius)
        traverseInorder(root.right)

def traversefeaturesSerializado(root, features):
    def post_order(root, features):
        if root:
            post_order(root.left, features)
            post_order(root.right, features)
            features.append(root.radius.cpu().tolist())
                
        #else:
        #    features.append(torch.tensor([0.,0.,0.,0.]))           

    post_order(root, features)
    return features[:-1]  # remove last ,

def normTodos(root, minx, miny, minz, minr, maxx, maxy, maxz, maxr):
    
    if root is not None and torch.mean(root.radius)!=0:
        mx = minx.clone().detach()
        my = miny.clone().detach()
        mz = minz.clone().detach()
        mr = minr.clone().detach()
        Mx = maxx.clone().detach()
        My = maxy.clone().detach()
        Mz = maxz.clone().detach()
        Mr = maxr.clone().detach()
       
        M = max((maxx - minx), (maxy - miny), (maxz - minz))
        root.radius[0] = (root.radius[0] - minx)/M
        root.radius[1] = (root.radius[1] - miny)/M
        root.radius[2] = (root.radius[2] - minz)/M
        root.radius[3] = (root.radius[3])/M
        
        normTodos(root.left, mx, my, mz, mr, Mx, My, Mz, Mr)
        normTodos(root.right, mx, my, mz, mr, Mx, My, Mz, Mr)
        return 

def normalize_features(root):
    features = []
    features = traversefeatures(root, features)
    
    x = [tensor[0] for tensor in features if torch.mean(tensor)!=0]
    y = [tensor[1] for tensor in features if torch.mean(tensor)!=0]
    z = [tensor[2] for tensor in features if torch.mean(tensor)!=0]
    r = [tensor[3] for tensor in features if torch.mean(tensor)!=0]
    
    normTodos(root, min(x), min(y), min(z), min(r), max(x), max(y), max(z), max(r))

    return 

dir = "DatosGPT/aneurisk/arboles/p15Eps02" 
trees = os.listdir(dir)
for t in trees:
    tree = deserialize2(read_tree(dir + "/" + t))
    normalize_features(tree)
    vector = []
    traversefeaturesSerializado(tree, vector)

    data_np = np.array(vector)
    file_path = 'DatosGPT/aneurisk/numpy/p15Eps02/' + t
    np.save(file_path, data_np)
    