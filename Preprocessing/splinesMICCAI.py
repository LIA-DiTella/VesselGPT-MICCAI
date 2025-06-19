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

def traversefeaturesSerializado(root, features):
    def post_order(root, features):
        if root:
            post_order(root.left, features)
            post_order(root.right, features)
            features.append(root.radius)
                
        else:
            features.append(39*[0.])          

    post_order(root, features)
    return features[:-1]  # remove last ,

def calculate_cross(spline, index):
    """
    Compute the area of a cross-section of the spline at a specific index.
    
    Args:
        spline (vtkPolyData): The spline polydata.
        index (int): The index of the spline to evaluate.
    
    Returns:
        float: Area of the cross-section.
    """
    # Get the points of the cross-section (you could extract the points at a specific location)
    # This assumes that you already have the logic to slice the spline at a certain point.
    # Here we're just estimating the area as an example.
    for ind in index:
        points = spline[ind].GetPoints()
        # Logic to compute the cross-section area (you could use a convex hull or other method)
        # Placeholder:
        area = np.pi * np.mean(points)  # Placeholder for actual area computation
    return area


#centerline = read_vtp("centerlines/ArteryObjAN1-0-network.vtp")
#mesh = read_vtp("mallasArregladas/ArteryObjAN1-0.vtp")



centerlines = os.listdir("splines/aneurisk/centerlines")
centerlines = sorted(centerlines)  
centerlines = centerlines
meshes = os.listdir("splines/aneurisk/mallas")
meshes = sorted(meshes)
meshes = meshes 
#centerline = read_vtp("output_file.vtp")
#centerline_points = vtk_to_numpy(centerline.GetPoints().GetData())
coef_folder = os.listdir("splines/aneurisk/corregidas/coeficientes")
for mesh in meshes:
    areas = []
    ratios = []
    f = mesh.split(".")[0]
    str = f+".pkl"
    if mesh.split(".")[1] == "vtp" and os.path.exists("splines/aneurisk/centerlineslinux/" + f + "-network.vtp") and str not in coef_folder:
        
        print("f", f)
        centerline = read_vtp("splines/aneurisk/centerlineslinux/" + f + "-network.vtp")
        mesh = read_vtp("splines/aneurisk/mallaslinux/" + f + ".vtp")

        fl = 0
        centerline_points = vtk_to_numpy(centerline.GetPoints().GetData())
        num_points = centerline.GetNumberOfPoints()

        # Create a renderer, render window, and interactor
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        # Add mesh to the visualization
        mesh_mapper = vtk.vtkPolyDataMapper()
        mesh_mapper.SetInputData(mesh)
        mesh_actor = vtk.vtkActor()
        mesh_actor.SetMapper(mesh_mapper)
        mesh_actor.GetProperty().SetOpacity(0.3)  # Make it semi-transparent
        renderer.AddActor(mesh_actor)

        # Add centerline to the visualization
        centerline_mapper = vtk.vtkPolyDataMapper()
        centerline_mapper.SetInputData(centerline)
        centerline_actor = vtk.vtkActor()
        centerline_actor.SetMapper(centerline_mapper)
        centerline_actor.GetProperty().SetColor(1, 0, 0)  # Red
        centerline_actor.GetProperty().SetLineWidth(2)
        renderer.AddActor(centerline_actor)

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
                        # Add the filtered cross-section to the visualization
                        # Create a renderer, render window, and interactor
                        '''
                        renderer = vtk.vtkRenderer()
                        render_window = vtk.vtkRenderWindow()
                        render_window.AddRenderer(renderer)
                        render_window_interactor = vtk.vtkRenderWindowInteractor()
                        render_window_interactor.SetRenderWindow(render_window)
                        filtered_mapper = vtk.vtkPolyDataMapper()
                        filtered_mapper.SetInputData(filtered_polydata)
                        filtered_actor = vtk.vtkActor()
                        filtered_actor.SetMapper(filtered_mapper)
                        filtered_actor.GetProperty().SetColor(0, 1, 0)  # Green
                        mesh_mapper = vtk.vtkPolyDataMapper()
                        mesh_mapper.SetInputData(mesh)
                        mesh_actor = vtk.vtkActor()
                        mesh_actor.SetMapper(mesh_mapper)
                        mesh_actor.GetProperty().SetColor(1, 0, 0)  # Red color for the mesh
                        mesh_actor.GetProperty().SetOpacity(0.3)
                        mapper2 = vtk.vtkPolyDataMapper()
                        mapper2.SetInputData(centerline)
                        actor2 = vtk.vtkActor()
                        actor2.SetMapper(mapper2)
                        actor2.GetProperty().SetColor(0.0, 0.0, 1.0)  # Red color for visibility'''
                        

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
                        #distancias_al_centro = 
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
                        
                    
                        #if len(x_sorted)>3:
                        try:
                            tck, u = splprep([x_sorted, y_sorted, z_sorted], s=0.01, per=True, nest = 12, k=3)
                            #spl, u = make_splprep([x_sorted, y_sorted, z_sorted], s=0.001, nest = 12, k=3)
                           
                            '''
                                if len(tck[1])<8:
                                    print("paso")
                                    tck[1].append(tck[-1])
                                    print(len(tck[1]))'''
                            splines.append(tck[1])#(splc)#
                            knots.append(tck[0])#(spl.t)#
                        except:
                            splines.append(tck[1])#(splc)#
                            knots.append(tck[0])#(spl.t)#
                        '''
                        x, y, z = splev(u, tck)  # Evaluate 2D spline
                        all_spline_points = [np.array([x, y, z]).T]
                        from plotsplines import create_vtk_polydata
                        spline_polydata = create_vtk_polydata(all_spline_points)
                        spline_mapper = vtk.vtkPolyDataMapper()
                        spline_mapper.SetInputData(spline_polydata)
                        spline_actor = vtk.vtkActor()
                        spline_actor.SetMapper(spline_mapper)
                        spline_actor.GetProperty().SetColor(1, 1, 0)
                        renderer.AddActor(spline_actor)
                        renderer.AddActor(actor2)
                        renderer.AddActor(mesh_actor)
                        renderer.AddActor(filtered_actor)
                        renderer.SetBackground(0, 0, 0)
                        render_window.Render()
                        render_window_interactor.Start()'''
                        
                        u_fine = np.linspace(0, 1, 1000)  # You can adjust the number of points for more resolution
                        spline_points = np.array(splev(u_fine, tck))  # This will give you the evaluated points (x, y, z)

                        # Stack the points together for easier distance calculation
                        points = np.column_stack(spline_points)

                        # Calculate distances between consecutive points
                        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

                        # Calculate the perimeter
                        perimeter = np.sum(distances)
                        areas.append(perimeter)
                        #else:
                            #break
                        #    tck, u = splprep([x_sorted, y_sorted], s=0.001, per=True, nest = 8, k = 2)
                        #u_fine = np.linspace(0, 1, 1000)
                        #x_smooth, y_smooth, z_smooth = BSpline(t_fixed, control_points, k, axis=0) #splev(u_fine, tck)
                        #x_smooth = x_smooth #+ centroid_x
                        #y_smooth = y_smooth #+ centroid_y
                        '''
                        plt.figure(figsize=(8, 6))
                        plt.scatter(x, y, color='red', label='Cross-section points', s=10)
                        plt.plot(x_sorted, y_sorted, color='blue', label='Fitted spline', linewidth=2)
                        plt.title("Cross-Section and Fitted Spline")
                        plt.xlabel("X")
                        plt.ylabel("Y")
                        plt.axis("equal")
                        plt.legend()
                        plt.grid(True)
                        plt.show()'''
            

                        
                        
                        #print("cantidad de coefs", len(tck[1][0]))
                        #print("knots", len(knots))
                    
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

            #ratios_chicos = [(x,i) for i,x in enumerate((ratios)) if x < 0.1]
            #ind = []
            '''
            for ind, ratio in ratios_chicos:
                x,y,z = centerline_np[ind][:3]
                coordinates = [      
                        np.full(8, x),  # Array for x, repeated 8 times
                        np.full(8, y),  # Array for y, repeated 8 times
                        np.full(8, z)   # Array for z, repeated 8 times
                    ]
                splines[index] = coordinates
                knots[index] = np.full(12,1.)'''
            for point in res:
                '''
                #ra = radius_array[res[point]]
                #area = areas[res[point]]
                #min = np.min(area)
                ar = [areas[i] for i in res[point]]
                min = np.min(ar)
                min_i = list(areas).index(min)
                #for index in res[point]:
                    #print(index)
                    #radius_array[index] = min
                    #splines[index] = splines[min_i]
                    #po.GetParts().ReplaceItem(index+1, po.GetParts().GetItemAsObject(min_i+1))
                    #print("removed: ", index+1, min_i)

                rat = [ratios[i] for i in res[point]]
                dist1 = [1-r for r in rat]
                close = np.min(dist1)#el mas cercano a '''
               
                ar = [areas[i] for i in res[point]]
                min = np.min(ar)
                min_i = list(areas).index(min)
                for index in res[point]:
                    #print(index)
                    #radius_array[index] = min
                    splines[index] = splines[min_i]#coordinates
                    knots[index] = knots[min_i]#np.full(12,0.)

                    '''
                x,y,z = centerline_np[indices_greater_than_average[0]]
                coordinates = [
                        np.full(8, x),  # Array for x, repeated 8 times
                        np.full(8, y),  # Array for y, repeated 8 times
                        np.full(8, z)   # Array for z, repeated 8 times
                    ]
                splines[indices_greater_than_average[0]] = coordinates
                knots[index] = np.full(12,1.)'''
            #x,y,z =point
            
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
        ######################################################################

        # Add a background color and start the visualization
        #renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background
        #render_window.SetSize(800, 600)
        #render_window.Render()
        #render_window_interactor.Start()
        '''
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='red', label='Cross-section points', s=10)
        #x_spline = BSpline(spl.t, spl.c, 3)
        #y_spline = BSpline(spl.t, spl.c, 3)
        t_eval = np.linspace(0, 1, 1000)  # Fine parameter range
        x_eval, y_eval, z_eval = splev(t_eval, tck)#spl(t_eval)
        
        #x_eval = x_spline(t_eval)  # Evaluate X
        #y_eval = y_spline(t_eval)  # Evaluate Y
        plt.plot(x_eval, y_eval, '-', label='Spline')
        #plt.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
        plt.grid(True)
        plt.show()'''
  
       
        #with open('splines/coeficientes/ArteryObjAN1-0-network.pkl', 'wb') as f:
        #np.save('splines/knots/'+ f , np.array(knots)) 
        #####GUARDO
        #with open('splines/aneurisk/corregidas/knots/'+ f +'.pkl', 'wb') as t:
        #    pickle.dump(knots, t)
        #with open('splines/aneurisk/corregidas/coeficientes/'+ f +'.pkl', 'wb') as t:
        #    pickle.dump(splines, t)
        


'''
# Step 6: Plot the cross-section and fitted spline
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', label='Cross-section points', s=10)
plt.plot(spline_points[0], spline_points[1], color='blue', label='Fitted spline', linewidth=2)
plt.title("Cross-Section and Fitted Spline")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.show()'''


gfolder = sorted(os.listdir('splines/aneurisk/corregidas/grafos'))
files = sorted(os.listdir('splines/aneurisk/mallasOBJ'))

l_error = []
for file in files:
   
    if file.split(".")[1] is not None:
        if file.split(".")[1] == "obj":
            try:
                #fileObj = open("resample_Eps01/centerlinesOBJ/" +file.split(".")[0] +"-network.obj")
                fileObj = open("splines/aneurisk/centerlinesOBJ/"+file.split(".")[0]+"-network.obj")
                #fileObj = open("centerlines/ArteryObjAN1-2-network.obj")
                
            except Exception:
                print("1 problem with: ", file)
                l_error.append(file)
                #traceback.print_exc()
            if file.split(".")[0] + '-grafo.gpickle' not in gfolder:
            #if True:
                try: 
                    grafo = calcularMatriz(fileObj, "splines/aneurisk/corregidas/coeficientes/" + file.split(".")[0] + ".pkl")
                    print("calculating: ", file)
                    with open("splines/aneurisk/corregidas/grafos/" + file.split(".")[0] + '-grafo.gpickle', 'wb') as f:
                        #print(grafo.nodes('radio'))
                        pickle.dump(grafo, f, pickle.HIGHEST_PROTOCOL)
                except Exception:
                    print("2 problem with: ", file)
                    l_error.append(file)
                    traceback.print_exc()

gfolder = os.listdir('splines/aneurisk/corregidas/grafos')  
#print("problem with files: ", l_error)

t_list = os.listdir('splines/aneurisk/corregidas/trees')

def limpiarRadiosSplines(radius):
    c= []
    r = []
    for x in radius:
        if isinstance(x, (np.float16, np.float32, np.float64)):
            c.append(float(x))
        elif(len(x))==3:
            for a in x:
                #a = [float(d) for d in a]
                for num in a:
                    c.append(float(num))
            #breakpoint()
            #c.append(r)
        else:
            #print("x", x)
            #a = [float(d) for d in x]
            for a in x:
                c.append(float(a))
    return c
       

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
            print("bin", len(grafo.edges(nodo)))
            break
    return graph

for file in files:
   
    if file.split(".")[0] + '-grafo.gpickle' in gfolder and file.split(".")[1] == 'obj' and (file.split(".")[0] + '.npy') not in t_list:

        grafo = pickle.load(open('splines/aneurisk/corregidas/grafos/' + file.split(".")[0] + '-grafo.gpickle',  'rb'))
        grafo = grafo.to_undirected()
        print("cantida de nodos", len(grafo.nodes)) 
        nc = 0
        #controlo si tiene ciclos
        if len(nx.cycle_basis(grafo))>0:
            nc = 1
        

        nb = 0
        #for nodo in grafo.nodes:
        for nodo in grafo.nodes:
            if len(grafo.edges(nodo))>3:
                #print("binario", len(grafo.edges(nodo)))
                binarizar(grafo)
                #nb = 1
                break
        
        if nb ==0 and nc == 0: 
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
            rad = limpiarRadiosSplines(rad)
            nodoRaiz = modelo.Node( numeroNodoInicial, radius =  rad )

            for vecino in grafo.neighbors( numeroNodoInicial ):
                if vecino != numeroNodoInicial:
                    aRecorrer.append( (vecino, numeroNodoInicial,nodoRaiz ) )
            
            while len(aRecorrer) != 0:
                nodoAAgregar, numeroNodoPadre,nodoPadre = aRecorrer.pop(0)
                radius = list(grafo.nodes[nodoAAgregar]['radio'])
                radius = limpiarRadiosSplines(radius)
                nodoActual = modelo.Node( nodoAAgregar, radius =  radius)
                nodoPadre.agregarHijo( nodoActual )
                for vecino in grafo.neighbors( nodoAAgregar ):
                    if vecino != numeroNodoPadre:
                        aRecorrer.append( (vecino, nodoAAgregar,nodoActual) )

 
            serial = nodoRaiz.serialize(nodoRaiz)
    
            f = []
            traversefeaturesSerializado(nodoRaiz, f)
            a = []
            for row in f:


                if len(row) == 39:
                    a.append(row)
                    
                elif len(row) == 36 or len(row) == 35:
                    print(f"ARREGLANDO: {file}")
                    x_y_z = row[:3]  # First 3 elements (x, y, z)
                    coef_x = row[3:10]  # Coefficients for x
                    coef_y = row[10:17]  # Coefficients for y
                    coef_z = row[17:24]  # Coefficients for z
                    knots = row[24:]  # Remaining elements are knots
                    
                    # Pad each section separately
                    while len(coef_x) < 8:
                        coef_x.append(coef_x[-1])

                    while len(coef_y) < 8:
                        coef_y.append(coef_y[-1])

                    while len(coef_z) < 8:
                        coef_z.append(coef_z[-1])

                    while len(knots) < 12:
                        knots.append(knots[-1])
                    # Combine into a full-length row
                    padded_row = x_y_z + coef_x + coef_y + coef_z + knots
                    a.append(padded_row)
                elif len(row)==38:
                    print(f"ARREGLANDO2: {file}")
                    knots = row[-11:]
                    row = row[:-11]
                    knots.append(knots[-1])
                    padded_row = row + knots
                    a.append(padded_row)
            try:
                array = np.array(a)
                ##GUARDO
                #np.save("splines/aneurisk/corregidas/trees/" + file.split(".")[0], array)
                #print("flat", array)
                print(f"calculated {file}")
            except Exception: 
                for row in a:
                    if len(row) != 39:
                        breakpoint()
                traceback.print_exc()
                print(f"error with file{file}")

        else:
            print("error with file: ", file, nb, nc)
