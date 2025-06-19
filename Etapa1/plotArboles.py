import torch
use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
import re



class Node:
    """
    Class Node
    """
    def __init__(self, value, radius, left = None, right = None):
        self.left = left
        self.data = value
        self.radius = radius
        self.right = right
    
    def toGraph( self, graph, index, dec, flag, proc=True):
        
        radius = self.radius.cpu().detach().numpy()
        if dec:
            radius= radius[0]
       
        if flag == 0:
            b = True
            flag = 1
        else:
            b = False
        graph.add_nodes_from( [ (self.data, {'posicion': radius[0:3], 'radio': radius[3], 'root': b} ) ])

        
        if self.right is not None:
            self.right.toGraph( graph, index + 1, dec, flag = 1)#
            graph.add_edge( self.data, self.right.data )
           
        if self.left is not None:
            self.left.toGraph( graph, 0, dec, flag = 1)#

            graph.add_edge( self.data, self.left.data)
           
        else:
            return
        
    def toGraph_nueva( self, graph, index, dec, flag, proc=True):
        
        radius = self.radius.cpu().detach().numpy()
        if dec:
            radius= radius[0]
       
        if flag == 0:
            b = True
            flag = 1
        else:
            b = False
        graph.add_nodes_from( [ (self.data, {'posicion': radius[0:3], 'root': b} ) ])

        
        if self.right is not None:
            self.right.toGraph_nueva( graph, index + 1, dec, flag = 1)#
            graph.add_edge( self.data, self.right.data )
           
        if self.left is not None:
            self.left.toGraph_nueva( graph, 0, dec, flag = 1)#

            graph.add_edge( self.data, self.left.data)
           
        else:
            return

def read_tree(filename):
    with open(filename, "r") as f:
        byte = f.read() 
        return byte
    
def deserialize(data):
    if  not data:
        return 
    nodes = data.split(';') 
    def post_order(nodes):
                
        if nodes[-1] == '#':
            nodes.pop()
            return None
        node = nodes.pop().split('_')
        try:
            data = int(node[0])
        except:
            numbers = re.findall(r'\d+', node[0])
            data = [int(num) for num in numbers]
        radius = node[1]
        #print("radius", radius)
        rad = radius.split(",")
        rad [0] = rad[0].replace('[','')
        rad [3] = rad[3].replace(']','')
        r = []
        for value in rad:
            r.append(float(value))
        r = torch.tensor(r, device=device)
        root = Node(data, r)
        root.right = post_order(nodes)
        root.left = post_order(nodes)
        
        return root    
    return post_order(nodes)  
