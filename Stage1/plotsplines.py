import vtk
import numpy as np
import pickle
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import splev, BSpline
from Arbol import Node


def read_vtp(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def load_spline_coefficients(filename, k=3):
    print("filename", filename)
    """Load multiple spline coefficients from a pickle file and reconstruct tck."""
    with open(filename, "rb") as f:
        all_c = pickle.load(f)  # List of coefficient arrays (one per spline)

    all_tck = []
    #t_all = np.load("splines/knots/"+filename.split("/")[2].split(".")[0]+".npy")
    with open("splines/aneurisk/knots/"+filename.split("/")[3].split(".")[0]+".pkl", "rb") as f:
        t_all = pickle.load(f)
    for _,c in enumerate(all_c):
        #u = np.linspace(-1, 2, len(c[0]))  # Assume uniform parameterization
        t = t_all[_]
        tck = (t, c, k)
        all_tck.append(tck)
    return all_tck
import matplotlib.pyplot as plt


def evaluate_splines(all_tck, num_points=1000):
    """Evaluate all splines and return a list of 2D point arrays."""
    all_spline_points = []
    g = 0
    for i, tck in enumerate(all_tck):
        if tck[0][0] != 0 :
            g+=1
            u = np.linspace(0, 1, num_points)  # Parametric values
            x, y, z = splev(u, tck) 
            if np.abs(np.mean(x)) < 10: # Evaluate 2D spline
                all_spline_points.append(np.array([x, y, z]).T)  # Nx3 array
                x_g, y_g, z_g = x, y, z
            else:
                print("CASO FEO")
                all_spline_points.append(np.array([x_g, y_g, z_g]).T)
                continue
            
        else:
            continue
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 3D Scatter plot
    ax.scatter(x, y, z, color='red', label='Cross-section points', s=10)

    # Labels and title
    ax.set_title("Cross-Section and Fitted Spline")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    ax.grid(True)
    plt.show()
    print("cantidad de splines", g)
    
    return all_spline_points
        
        

def create_vtk_polydata(all_spline_points):
    """Convert multiple sets of 2D points into vtkPolyData with a small height offset."""
    vtk_points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    point_offset = 0  # Track global point index

    for spline_points in all_spline_points:
        first_point_id = point_offset

        for i, (x, y, z) in enumerate(spline_points):
            vtk_points.InsertNextPoint(x, y, z)
            
            if i > 0:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, point_offset - 1)
                line.GetPointIds().SetId(1, point_offset)
                lines.InsertNextCell(line)

            point_offset += 1

        # Close the loop (optional)
        
        if len(spline_points) > 2:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, point_offset - 1)
            line.GetPointIds().SetId(1, first_point_id)
            lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)
    return polydata


def visualize_vtk_polydata(polydata, polydata2 = None, polydata3 = None, title="VTK Visualization"):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
   

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red color for visibility
    

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    if polydata2 is not None:
        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputData(polydata2)
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)
        actor2.GetProperty().SetOpacity(0.3)
        actor2.GetProperty().SetColor(0.0, 1.0, 0.0)  # Red color for visibility
        renderer.AddActor(actor2)
        if polydata3 is not None:
            # Add mesh to the visualization
            mesh_mapper = vtk.vtkPolyDataMapper()
            mesh_mapper.SetInputData(polydata3)
            mesh_actor = vtk.vtkActor()
            mesh_actor.SetMapper(mesh_mapper)
            mesh_actor.GetProperty().SetOpacity(0.3)  # Make it semi-transparent
            renderer.AddActor(mesh_actor)

    renderer.AddActor(actor)
    renderer.SetBackground(0, 0, 0)
    renderWindow.Render()
    renderWindow.SetWindowName(title)
    renderWindowInteractor.Start()

def findParent(node, val, parent=None):
    if node is None:
        return None

    # If the current node is the required node
    if node.data == val:
        # Return the parent node
        return parent
    else:
        # Recursive calls for the children of the current node
        # Set the current node as the new parent
        left_result = findParent(node.left, val, parent=node)
        right_result = findParent(node.right, val, parent=node)

        # If the node is found in the left subtree, return the result
        if left_result is not None:
            return left_result
        # If the node is found in the right subtree, return the result
        elif right_result is not None:
            return right_result
        # If the node is not found in either subtree, return None
        else:
            return None

def recorrido(root, current, d_bif, l_corte, d2, l_corte2):
    if current is not None:
        recorrido(root, current.right, d_bif, l_corte, d2, l_corte2)
        #print("current", current.data, current.right, current.left)
        if current.isTwoChild():
            #print("bifurcacion", current.data)
            padre = findParent(root, current.data)
            d_bif[current.data] = [padre, current.right, current.left]
            if current.right is not None:
                d2[current.right.data] = current.data
                l_corte2.append(current.right.data)
            if current.left is not None:
                d2[current.left.data] = current.data
                l_corte2.append(current.left.data)
            l_corte.append(current.data)
        if current.isLeaf():
            #print("hoja", current.data)
            l_corte.append(current.data)
        recorrido(root, current.left, d_bif, l_corte, d2, l_corte2)

        return 
#lcorte tiene las hojas y bifurcaciones
#lcorte2 los hijos de las bifurcaciones
#d2 tiene tiene como llave los hijos de las bifurcaciones y como valor el padre
#d_bif tiene como llave las bifurcaciones (data) y como valor el padre y los hijos (objeto)

def traverse_tree(node, points, polyLine, cellarray, d, l_corte, l_corte2, d_id):
    if node is not None:
        
        # Add the current node's point to the vtkPoints
        
        #print("NODE", node.data)
        radius = node.radius
        id = points.InsertNextPoint(radius[0], radius[1], radius[2])
        d_id[id] = node.data
        #print("current", node.data)
        #print("right", node.right.data) if node.right is not None else print("right", None)
        #print("left", node.left.data) if node.left is not None else print("left", None)
        #print("childs - noden", node.childs(), node.data)

        #if node.isTwoChild() or traverse_tree.count == nleafs*2-1 or node.data in bifurcation_Stack:
      
        if node.data in  l_corte: 

            # stop the current line and start a new one   
            #if node.data in l_corte2 and d[node.data] is not None:
            #    print("aca")
            #    polyLine.GetPointIds().InsertNextId(d_id[d[node.data]]) 
            #polyLine.GetPointIds().InsertNextId(d_id[id]) 
            polyLine.GetPointIds().InsertNextId(id) 
            if polyLine.GetPointIds().GetNumberOfIds() >1:
                cellarray.InsertNextCell(polyLine)
                point_ids = polyLine.GetPointIds()
                for i in range(point_ids.GetNumberOfIds()):
                    point_id = point_ids.GetId(i)
                
            polyLine = vtk.vtkPolyLine()
        else:
            # Continue adding points to the current line
            #polyLine.append(node.data)
            if node.data in l_corte2:
                #polyLine.GetPointIds().InsertNextId(d[node.data]) 
                aa = next((key for key, value in d_id.items() if value == d[node.data]), None)
                polyLine.GetPointIds().InsertNextId(aa) 
            #polyLine.GetPointIds().InsertNextId(node.data)
            polyLine.GetPointIds().InsertNextId(id)

        traverse_tree(node.right, points, polyLine, cellarray, d, l_corte, l_corte2, d_id)
        polyLine = vtk.vtkPolyLine()
        traverse_tree(node.left, points, polyLine, cellarray, d, l_corte, l_corte2, d_id)
        polyLine = vtk.vtkPolyLine()
        # Recursively traverse the left and right subtrees
        



def tree2centerline(tree):    

    points = vtk.vtkPoints()
    cellarray = vtk.vtkCellArray()
    polyline = vtk.vtkPolyLine()
    #li = []
    #tree.traverseInorderChilds(tree, li)
    d = {}
    l_corte = []
    d2 = {}
    l_corte2 = []
    recorrido(tree, tree, d, l_corte, d2, l_corte2)

    d_id = {}
    traverse_tree(tree, points, polyline, cellarray, d2, l_corte, l_corte2, d_id)
    # Create VTK polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    print("number of points", points.GetNumberOfPoints())
    polydata.SetLines(cellarray)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("centerlinegenerada.vtp")
    writer.SetInputData(polydata)
    writer.Write()
    return polydata

def createNode(data, radius, left = None, right = None, ):
        """
        Utility function to create a node.
        """
        return Node(data, radius, left, right)

def deserialize(data):
    if  not data:
        return 
    nodes = data.split(';')  
    def post_order(nodes):
        if nodes[-1] == '#':
            nodes.pop()
            return None
        node = nodes.pop().split('_')
        
        data = int(node[0])
        radius = node[1]
        rad = radius.split(",")
       
        rad [0] = rad[0].replace('[','')
        rad [2] = rad[2].replace(']','')
        r = []
        for value in rad:
            r.append(float(value))
        r = r
        root = createNode(data, r)
        root.right = post_order(nodes)
        root.left = post_order(nodes)
        
        return root    
    return post_order(nodes)    

def deserialize_pre_order(serial):
    
    serial = serial.copy()
    #print("serial", serial)
    if len(serial) > 0:
        
        expected = [[0.0] for _ in range(39)]
       
        
        if serial[:39] != expected:
            data = {}

            data["x"] = serial.pop(0)
            data["y"] = serial.pop(0)
            data["z"] = serial.pop(0)
            data["r"] = []
            for i in range(36):  data["r"].append(serial.pop(0))
            rad = [data["x"], data["y"], data["z"]]
            
            rad = [r[0] for r in rad]
            tree = Node(1, rad)

            left, ret = deserialize_pre_order(serial)
            right, ret = deserialize_pre_order(ret)
            

            tree.left = left
            tree.right = right

            return tree, ret

        else:
            return None, serial[39:]
        
    else:
        return None, []
    
def numerarNodos(root, count):
    if root is not None:
        numerarNodos(root.left, count)
        root.data = len(count)
        count.append(1)
        numerarNodos(root.right, count)
        return 



def serial2centerline(serial):
    generated_images = deserialize_pre_order(serial)[0]
    n = []
    numerarNodos(generated_images, n)
    print("numero de nodos", len(n))
    centerline = tree2centerline(generated_images)

    return centerline




mesh = read_vtp("mallasaneurisk/0002.vtp")
num_file = np.load("62.npy")
all_tck = []
print("all tck", all_tck)
print("shape", num_file.shape)
centerline = num_file[:,:]
print("centerline", np.mean(centerline, axis = 1))
t_all = num_file[:,27:]
c_all = num_file[:,3:27]
print("knots", t_all.shape)
print("coefs", c_all.shape)
#print("centerline", centerline.shape)
print("primeras lineas centerlines", centerline[:15])

k = 3
for _,c in enumerate(c_all):
    #u = np.linspace(-1, 2, len(c[0]))  # Assume uniform parameterization
    c = [c[i:i+8] for i in range(0, len(c), 8)]
    t = t_all[_]
    tck = (t, c, k)
    all_tck.append(tck)

serial = ""
j = 0
d = 0
ser= []
#print("centerline", centerline)
for array in centerline :
    if array[-1] == 0:
        string = [0.0]*4#+";"
        for i in range(39):
            ser.append([0.0])
    else:
        #string = str(j)+"_"+str(list(array))+";"
        #string = str(d)+"_["+ ", ".join(map(str, array))+"];"
        for num in array:
            ser.append([num])
        d +=1
    #serial += string
    #ser.append(string)
    #j+=1

all_spline_points = evaluate_splines(all_tck, num_points=1000)
splines_2 = create_vtk_polydata(all_spline_points)
centerline = serial2centerline(ser)

visualize_vtk_polydata(splines_2, None, None, "Splines Visualization")
visualize_vtk_polydata(centerline, splines_2, None, "Splines Visualization")
