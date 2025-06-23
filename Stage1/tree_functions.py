import torch
import numpy as np

class Tree:

    def __init__(self, data, right = None, left = None):

        self.id = id(self)
        self.data = data

        self.right = right
        self.left = left

def deserialize_post_order_k(serial, k = 4):

	serial = serial.copy()

	def post_order(serial, k):

		if serial[-k:] == [0.0] * k:
			for i in range(k): serial.pop()
			return None
		
		data = {}

		if k == 4:
			data["r"] = serial.pop()
		else:
			data["r"] = []
			for i in range(k - 3):  data["r"].insert(0, serial.pop())

		data["z"] = serial.pop()
		data["y"] = serial.pop()
		data["x"] = serial.pop()

		tree = Tree(data)

		tree.right = post_order(serial, k)
		tree.left = post_order(serial, k)
		
		return tree    
	
	return post_order(serial, k)

def deserialize_post_order(serial):

    serial = serial.copy()

    def post_order(serial):

        if serial[-4:] == [0.0] * 4:
            for i in range(4): serial.pop()
            return None
        
        data = {}

        data["r"] = serial.pop()
        data["z"] = serial.pop()
        data["y"] = serial.pop()
        data["x"] = serial.pop()

        tree = Tree(data)

        tree.right = post_order(serial)
        tree.left = post_order(serial)
        
        return tree    
    
    return post_order(serial)

def deserialize_pre_order(serial):
    
    serial = serial.copy()

    if len(serial) > 0:

        if serial[:4] != [0.0] * 4:
            
            data = {}

            data["x"] = serial.pop(0)
            data["y"] = serial.pop(0)
            data["z"] = serial.pop(0)
            data["r"] = serial.pop(0)

            tree = Tree(data)

            left, ret = deserialize_pre_order(serial)
            right, ret = deserialize_pre_order(ret)

            tree.left = left
            tree.right = right

            return tree, ret

        else:
            return None, serial[4:]
        
    else:
        return None, []
    
def deserialize_pre_order_k(serial, k = 4):
    
    serial = serial.copy()

    if len(serial) > 0:

        if serial[:k] != [0.0] * k:
            
            data = {}

            data["x"] = serial.pop(0)
            data["y"] = serial.pop(0)
            data["z"] = serial.pop(0)

            if k == 4:
                data["r"] = serial.pop(0)
            else:
                data["r"] = []
                for i in range(k - 3):  data["r"].append(serial.pop(0))

            tree = Tree(data)

            left, ret = deserialize_pre_order_k(serial, k)
            right, ret = deserialize_pre_order_k(ret, k)

            tree.left = left
            tree.right = right

            return tree, ret

        else:
            return None, serial[k:]
        
    else:
        return None, []

def serialize_pre_order(tree):

    if tree == None: return [0.0] * 4
    return list(tree.data.values())[::-1] + serialize_pre_order(tree.left) + serialize_pre_order(tree.right)

def serialize_pre_order_k(tree, k = 4):

    if tree == None: return [0.0] * k
    return [tree.data["x"], tree.data["y"], tree.data["z"]] + list(tree.data["r"]) + serialize_pre_order_k(tree.left, k) + serialize_pre_order_k(tree.right, k)

def deserialize(serial, mode = "pre_order", k = 4):

    if mode == "pre_order": return deserialize_pre_order_k(serial, k)[0]
    if mode == "post_order": return deserialize_post_order_k(serial, k)

    print("UNSUPPORTED DESERIALIZATION MODE")

def tokens_to_data(tokens, device, decoder):

    if len(tokens.shape) != 1 : raise Exception("'tokens' shape must be 1")

    if tokens[0] == 256: tokens = tokens[1:]
    if tokens[-1] == 256: tokens = tokens[:-1]

    tokens = tokens.to(device)

    feat = decoder.entry_to_feature(tokens, (-1, 64))
    feat = feat.T.unsqueeze(0)

    data = decoder.decode(feat).detach().cpu()
    
    return data

def tokens_to_tree(tokens, threshold = 1e-2, mode = "pre_order", device = None, decoder = None):

    tree = Tree({"x":0, "y":0, "z":0, "r":0})

    try:
        
        data = tokens_to_data(tokens, device, decoder)

        data[torch.abs(data) < threshold] = 0

        serial = list(data.flatten())
        tree = deserialize(serial, mode)

    except: print("< tokens_to_tree error >")

    return tree        

def is_valid_tree(tokens, threshold = 1e-2, mode = "pre_order", device = None, decoder = None):

    try:
        
        data = tokens_to_data(tokens, device, decoder)

        data[data < threshold] = 0

        serial = list(data.flatten())
        deserialize(serial, mode)

        return True

    except: return False

def serialize_post_order_str(tree):

    def post_order(tree):

        if tree:

            post_order(tree.left)
            post_order(tree.right)

            ret[0] += '1_'+ str([np.round(float(v), 4) for v in list(tree.data.values())]) +';'

        else:
            ret[0] += '#;'

    ret = ['']
    post_order(tree)
    return ret[0][:-1]