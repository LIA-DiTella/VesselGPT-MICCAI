import torch
use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
import re

class Node:
    """
    Class Node
    """
    def __init__(self, value, radius, left = None, right = None, level = None, treelevel = None, maxlevel = None):
        self.left = left
        self.data = value
        self.radius = radius
        self.right = right
        self.children = [self.left, self.right]
        self.level = level
        self.treelevel = treelevel
        self.maxlevel = maxlevel
    
    def agregarHijo(self, children):

        if self.right is None:
            self.right = children
        elif self.left is None:
            self.left = children

        else:
            raise ValueError ("solo arbol binario ")


    def isLeaf(self):
        if self.right is None and self.left is None:
            return True
        else:
            return False

    def isTwoChild(self):
        if self.right is not None and self.left is not None:
            return True
        else:
            return False

    def isOneChild(self):
        if self.isTwoChild():
            return False
        elif self.isLeaf():
            return False
        else:
            return True

    def childs(self):
        if self.isLeaf():
            return 0
        if self.isOneChild():
            return 1
        else:
            return 2
    
    
    def traverseInorder(self, root):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorder(root.left)
            print (root.data, root.radius)
            self.traverseInorder(root.right)

    

    def traverseInorderwl(self, root):
        """
        traverse function will print all the node in the tree, including node level and tree level.
        """
        if root is not None:
            self.traverseInorderwl(root.left)
            print (root.data, root.radius, root.level, root.treelevel)
            self.traverseInorderwl(root.right)

    def getTreeLevel(self, root, c):
        """
        
        """
        if root is not None:
            self.getTreeLevel(root.left, c)
            c.append(root.level)
            self.getTreeLevel(root.right, c)

    def setTreeLevel(self, root, c):
        """
        
        """
        if root is not None:
            self.setTreeLevel(root.left, c)
            root.treelevel = c
            self.setTreeLevel(root.right, c)

    def setMaxLevel(self, root, m):
        """
        
        """
        if root is not None:
            self.setMaxLevel(root.left, m)
            root.maxlevel = m
            self.setMaxLevel(root.right, m)



    def traverseInorderChilds(self, root, l):
        """
        
        """
        if root is not None:
            self.traverseInorderChilds(root.left, l)
            l.append(root.childs())
            self.traverseInorderChilds(root.right, l)
            return l



    def height(self, root):
    # Check if the binary tree is empty
        if root is None:
            return 0 
        # Recursively call height of each node
        leftAns = self.height(root.left)
        rightAns = self.height(root.right)
    
        # Return max(leftHeight, rightHeight) at each iteration
        return max(leftAns, rightAns) + 1

    # Print nodes at a current level
    def printCurrentLevel(self, root, level):
        if root is None:
            return
        if level == 1:
            print(root.data, end=" ")
        elif level > 1:
            self.printCurrentLevel(root.left, level-1)
            self.printCurrentLevel(root.right, level-1)

    def printLevelOrder(self, root):
        h = self.height(root)
        for i in range(1, h+1):
            self.printCurrentLevel(root, i)


    def countNodes(self, root, counter):
        if   root is not None:
            self.countNodes(root.left, counter)
            counter.append(root.data)
            self.countNodes(root.right, counter)
            return counter

    
    def serialize(self, root):
        def post_order(root):
            if root:
                post_order(root.left)
                post_order(root.right)
                ret[0] += str(root.data)+'_'+ str(root.radius) +';'
                
            else:
                ret[0] += '#;'           

        ret = ['']
        post_order(root)
        return ret[0][:-1]  # remove last ,

    def toGraph( self, graph, index, dec, flag, proc=True):
        
        radius = self.radius.cpu().detach().numpy()
        if dec:
            radius= radius[0]
       
        if flag == 0:
            b = True
            flag = 1
        else:
            b = False
        graph.add_nodes_from( [ (self.data, {'radio': radius[0:4], 'root': b} ) ])

        
        if self.right is not None:
            self.right.toGraph( graph, index + 1, dec, flag = 1)#
            graph.add_edge( self.data, self.right.data )
           
        if self.left is not None:
            self.left.toGraph( graph, 0, dec, flag = 1)#

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


##deserializo agragnado vectores con cero en los nodos fantasma
def deserialize2(data):
    if  not data:
        return 
    nodes = data.split(';') 
    def post_order(nodes):
        
        if nodes[-1] == '#':
            data = 1
            r = [0.,0.,0.,0.]
            r = torch.tensor(r, device=device)
            root = Node(data, r)
            nodes.pop()
        else:
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

