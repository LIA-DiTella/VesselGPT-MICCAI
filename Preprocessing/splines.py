import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from scipy.interpolate import splprep, splev, make_splprep, BSpline
import matplotlib.pyplot as plt
import pickle
import os
from parseObj import calcularMatriz
import traceback
import networkx as nx
import Arbol as modelo
from collections import Counter

def get_points_by_line(centerline):
    points_array = []
    for i in range(centerline.GetNumberOfCells()):
        cell = centerline.GetCell(i)
        points = cell.GetPoints()
        for j in range(points.GetNumberOfPoints()):
            point = points.GetPoint(j)#i me dice el numero de linea y j el de punto
            p = (point[0], point[1], point[2], i)
            points_array.append(p)
    return np.array(points_array)

# Step 1: Read the .vtp files
def read_vtp(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    output = reader.GetOutput()
    if not output:
        print(f"Error reading file: {file_path}")
    return reader.GetOutput()

def traversefeaturesSerializado(root, features, k):
    def post_order(root, features):
        if root:
            post_order(root.left, features)
            post_order(root.right, features)
            features.append(root.radius)
                
        else:
            features.append(k*[0.])          

    post_order(root, features)
    return features[:-1]  # remove last ,


def limpiarRadiosSplines2(radius):
    c= []
    r = []
    for x in radius:
        if isinstance(x, (np.float16, np.float32, np.float64)):
            c.append(float(x))
        elif(len(x))==3:
            for a in x:
                for num in a:
                    c.append(float(num))
        else:
            for a in x:
                c.append(float(a))
    return c

def limpiarRadiosSplines(radius):
    cleaned = []

    # Step 1: Keep first 3 values as-is (converted to float)
    for i in range(3):
        cleaned.append(float(radius[i]))

    # Step 2: Process the 4th element - list of 3 arrays (each padded to length 8)
    array_list = radius[3]
    for arr in array_list:
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) < 8:
            arr = np.pad(arr, (0, 8 - len(arr)), mode='edge')  # Repeat last value
        cleaned.extend(arr[:8])  # Ensure exactly 8

    # Step 3: Process the 5th element - array of length 12
    arr = np.asarray(radius[4], dtype=np.float32)
    if len(arr) < 12:
        arr = np.pad(arr, (0, 12 - len(arr)), mode='edge')  # Repeat last value
    cleaned.extend(arr[:12])  # Ensure exactly 12

    return cleaned

def binarizar(graph):
    for node in list(graph.nodes()):
        neighbors = list(graph.neighbors(node))
        num_neighbors = len(neighbors)

        if num_neighbors > 3:
            # Create a chain of intermediate nodes
            for i in range(num_neighbors - 2):
                new_node = f"{node}.{i}"
                radio_value = graph.nodes[node]['radio']  # Get the 'radio' attribute of the current node
                
                # Add the new node and edge
                graph.add_node(new_node, radio=radio_value)
                graph.add_edge(node, new_node)

            # Connect the last intermediate node to the original neighbors
            i=0 #cuenta intermedios
            c=0 #cuenta nodos
            #print("///////////")
            for vecino in neighbors:
                intermediate_node = f"{node}.{i}"
                #print("intermedio", intermediate_node)
                graph.add_edge(intermediate_node, vecino)
                c+=1
                if c>1:
                    i+=1
                    c=0
           
            # Remove the original edges
            graph.remove_edges_from([(node, neighbor) for neighbor in neighbors])


    for nodo in graph.nodes:
        if len(graph.edges(nodo))>3:
            print("bin", len(graph.edges(nodo)))
            break
    return graph


def calculate_splines(mesh, coef_folder,centerfolder, meshfolder):
    areas = []
    ratios = []
    f = mesh.split(".")[0]
    str = f+".pkl"
    if mesh.split(".")[1] == "vtp" and os.path.exists(centerfolder + f + "-network.vtp") and str not in os.listdir(coef_folder):
        
        centerline = read_vtp(centerfolder + f + "-network.vtp")
        mesh = read_vtp(meshfolder + f + ".vtp")

        # Slice the mesh and filter cross-sections
        splines = []
        knots = []
        points_Acum = 0
        for j in range(centerline.GetNumberOfCells()):#calculate the radius by branch to avoid problems at the connections between branches
      
            numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints();# number of points of the branch
            
            for i in range (numberOfCellPoints):
                
                tangent = np.zeros((3))

                weightSum = 0.0;
                ##tangent line with the previous point (not calculated at the first point)
                if (i>0):
                    point0 = centerline.GetPoint(points_Acum-1);
                    point1 = centerline.GetPoint(points_Acum);

                    distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1));
                    
                    ##vector between the two points divided by the distance
                
                    tangent[0] += (point1[0] - point0[0]) / distance;
                    tangent[1] += (point1[1] - point0[1]) / distance;
                    tangent[2] += (point1[2] - point0[2]) / distance;
                    weightSum += 1.0;


                ##tangent line with the next point (not calculated at the last one)
                
                if (i<numberOfCellPoints-1):
                    
                    point1 = centerline.GetPoint(points_Acum);
                    point0 = centerline.GetPoint(points_Acum+1);

                    distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1));
                    tangent[0] += (point0[0] - point1[0]) / distance;
                    tangent[1] += (point0[1] - point1[1]) / distance;
                    tangent[2] += (point0[2] - point1[2]) / distance;
                    weightSum += 1.0;
                

                tangent[0] /= weightSum;
                tangent[1] /= weightSum;
                tangent[2] /= weightSum;
                plane = vtk.vtkPlane()
                plane.SetOrigin(point1)
                plane.SetNormal(tangent)
                points_Acum += 1
                # Slice the mesh
                cutter = vtk.vtkCutter()
                cutter.SetCutFunction(plane)
                cutter.SetInputData(mesh)
                cutter.SetSortBy(1)
                cutter.Update()
                sliced_polydata = cutter.GetOutput()

                if sliced_polydata.GetNumberOfPoints() > 0:
                    # Filter to keep only the region closest to the centerline point
                    connectivityFilter = vtk.vtkConnectivityFilter()
                    connectivityFilter.SetInputData(sliced_polydata)
                    connectivityFilter.SetExtractionModeToClosestPointRegion()
                    connectivityFilter.SetClosestPoint(point1)  # Set the centerline point
                    connectivityFilter.Update()
                    filtered_polydata = connectivityFilter.GetOutput()

                    if filtered_polydata.GetNumberOfPoints() > 0:
                        # Extract filtered points and fit a spline
                        points = vtk_to_numpy(filtered_polydata.GetPoints().GetData())

                        # Triangulate the contour points to form a 2D surface
                        delaunay = vtk.vtkDelaunay2D()
                        delaunay.SetInputData(filtered_polydata)
                        delaunay.Update()
                        
                        triangulated_surface = delaunay.GetOutput()
                        # Now calculate the surface area
                        mass = vtk.vtkMassProperties()
                        mass.SetInputData(triangulated_surface)
                        mass.Update()
                        area = mass.GetSurfaceArea()
        
                        #areas.append(area)
                        distancias =points-point1
                        normas = np.linalg.norm(distancias, axis=1) 
                        ratio = np.min(normas)/np.max(normas)
                        ratios.append(ratio)
                        x, y, z = points[:, 0], points[:, 1], points[:, 2]
                        
                        centroid_x = np.mean(x)
                        centroid_y = np.mean(y)
                        centroid_z = np.mean(z)
                        angles = np.arctan2(y - centroid_y, x - centroid_x)

                        # Step 4: Sort the points by angle (angular order)
                        sorted_indices = np.argsort(angles)
                        x_sorted = x[sorted_indices]
                        y_sorted = y[sorted_indices]
                        z_sorted = z[sorted_indices]
                        points = np.vstack([x_sorted, y_sorted, z_sorted]).T
                        tck = None
                        try:
                            tck, u = splprep([x_sorted, y_sorted, z_sorted], s=0.01, per=True, nest = 12, k=3)
                            splines.append(tck[1])#(splc)#
                            knots.append(tck[0])#(spl.t)#
                        except:
                            if tck:
                                splines.append(tck[1])#(splc)#
                                knots.append(tck[0])#(spl.t)#
                            else:
                                continue
                       
                        
                        u_fine = np.linspace(0, 1, 1000)  # You can adjust the number of points for more resolution
                        spline_points = np.array(splev(u_fine, tck))  # This will give you the evaluated points (x, y, z)

                        # Stack the points together for easier distance calculation
                        points = np.column_stack(spline_points)

                        # Calculate distances between consecutive points
                        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

                        # Calculate the perimeter
                        perimeter = np.sum(distances)
                        areas.append(perimeter)
                    
                            ######################################################################
        centerline_np = get_points_by_line(centerline)
        ##find centerline repeated points
        try:
            splited = np.split(centerline_np, np.where(np.diff(centerline_np[:,3]))[0]+1)
            e = {}# to save every branch endpoint
            sum = 0
            for i in range(len(splited)):
                rama = splited[i]
                start = rama[0, :3]
                e[sum] = tuple(start) #key is the point index, value coordinates
                finish = rama[rama.shape[0]-1, :3]
                sum += rama.shape[0]
                e[sum-1] = tuple(finish)
                            
            ##keep only the repeated endpoints
            b = np.array([key for key,  value in Counter(e.values()).items() if value > 1])

                        
            ##list with the indexes of the repeated points
            key_list = []
            for element in b: #coordintaes of each repeated point
                element = tuple(element)
                for key,value in e.copy().items():
                    if element == value:#if the endpoint is on the repeated list I save the index
                        key_list.append(key)#key_list tiene los indices de los puntos repetidos

            k = {}
            ##dictionary with the indexes and coordinates of the repeated points
            for key in key_list:
                k[key] = tuple(centerline_np[key,:3])

            ## join the points with the same coordinates, key are the coordinates and values list with the indexes
            res = {}
            for i, v in k.items():
                res[v] = [i] if v not in res.keys() else res[v] + [i]

           
            for point in res:
                ar = [areas[i] for i in res[point]]
                min = np.min(ar)
                min_i = list(areas).index(min)
                for index in res[point]:
                    splines[index] = splines[min_i]#coordinates
                    knots[index] = knots[min_i]#np.full(12,0.)
                   
            
            indices_greater_than_average = sorted([(i,element) for i, element in enumerate(areas) if element > 3*np.mean(areas)], key=lambda x: x[1], reverse=True)
            for index, area in indices_greater_than_average:
                x,y,z = centerline_np[index][:3]
                coordinates = [      
                    np.full(8, x),  # Array for x, repeated 8 times
                    np.full(8, y),  # Array for y, repeated 8 times
                    np.full(8, z)   # Array for z, repeated 8 times
                ]
                splines[index] = coordinates
                knots[index] = np.full(12,1.)
            
        except Exception as e:
            print("EXCEPT")

            traceback.print_exc()
            pass
        
        knot_folder = coef_folder.replace("coeficientes", "knots")
        with open(knot_folder+ f +'.pkl', 'wb') as t:
            pickle.dump(knots, t)
        with open(coef_folder+ f +'.pkl', 'wb') as t:
            pickle.dump(splines, t)

def binarizar(graph):
    for node in list(graph.nodes()):
        neighbors = list(graph.neighbors(node))
        num_neighbors = len(neighbors)

        if num_neighbors > 3:
            # Create a chain of intermediate nodes
            for i in range(num_neighbors - 2):
                new_node = f"{node}.{i}"
                radio_value = graph.nodes[node]['radio']  # Get the 'radio' attribute of the current node
                
                # Add the new node and edge
                graph.add_node(new_node, radio=radio_value)
                graph.add_edge(node, new_node)

            # Connect the last intermediate node to the original neighbors
            i=0 #cuenta intermedios
            c=0 #cuenta nodos
            #print("///////////")
            for vecino in neighbors:
                intermediate_node = f"{node}.{i}"
                #print("intermedio", intermediate_node)
                graph.add_edge(intermediate_node, vecino)
                c+=1
                if c>1:
                    i+=1
                    c=0
           
            # Remove the original edges
            graph.remove_edges_from([(node, neighbor) for neighbor in neighbors])


    for nodo in graph.nodes:
        if len(graph.edges(nodo))>3:
            print("bin", len(graph.edges(nodo)))
            break
    return graph


def grafo2arbol(grafo):
    aRecorrer = []
    numeroNodoInicial = 1
    distancias = nx.floyd_warshall( grafo )

    parMaximo = (-1, -1)
    maxima = -1
                
    for nodoInicial in distancias.keys():
        for nodoFinal in distancias[nodoInicial]:
            if distancias[nodoInicial][nodoFinal] > maxima:
                maxima = distancias[nodoInicial] [nodoFinal]
                parMaximo = (nodoInicial, nodoFinal)
            
    for nodo in grafo.nodes:
        if distancias[parMaximo[0]][nodo] == int( maxima / 2):
            numeroNodoInicial = nodo
            if len(grafo.edges(numeroNodoInicial))>2:
                numeroNodoInicial = list(grafo.edges(numeroNodoInicial))[0][1]
            break
            
    rad = list(grafo.nodes[numeroNodoInicial]['radio'])
    nodoRaiz = modelo.Node( numeroNodoInicial, radius =  rad )
    for vecino in grafo.neighbors( numeroNodoInicial ):
        if vecino != numeroNodoInicial:
            aRecorrer.append( (vecino, numeroNodoInicial,nodoRaiz ) )
    while len(aRecorrer) != 0:
        nodoAAgregar, numeroNodoPadre,nodoPadre = aRecorrer.pop(0)
        radius = list(grafo.nodes[nodoAAgregar]['radio'])
    
        nodoActual = modelo.Node( nodoAAgregar, radius =  radius)
        nodoPadre.agregarHijo( nodoActual )
        for vecino in grafo.neighbors( nodoAAgregar ):
            if vecino != numeroNodoPadre:
                aRecorrer.append( (vecino, nodoAAgregar,nodoActual) )

    serial = nodoRaiz.serialize(nodoRaiz)
    
    return serial