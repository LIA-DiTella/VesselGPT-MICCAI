import numpy as np
import matplotlib.pyplot as plt
import vtk
import os

def rdp(points, epsilon):
    """
    Simplify a curve using the Ramer-Douglas-Peucker algorithm.

    Parameters:
    - points: List of (x, y) coordinates representing the curve.
    - epsilon: The maximum distance allowed between the original curve and the simplified curve.

    Returns:
    - List of simplified (x, y) coordinates.
    """
    if len(points) <= 2:
        return points

    # Find the point with the maximum distance
    dmax = 0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = np.linalg.norm(np.cross(
            np.array(points[end]) - np.array(points[0]),
            np.array(points[i]) - np.array(points[0])
        )) / np.linalg.norm(np.array(points[end]) - np.array(points[0]))

        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        rec_results1 = rdp(points[:index+1], epsilon)
        rec_results2 = rdp(points[index:], epsilon)

        # Combine the results
        result = rec_results1[:-1] + rec_results2
    else:
        result = [points[0], points[end]]

    return result


def interpolarRDP (centerline, epsilon = 0.05):

    resampleada = vtk.vtkPolyData()
    cellarray = vtk.vtkCellArray()
    points = vtk.vtkPoints()
    lista_puntos = []
    lista_rama = []
    for j in range(centerline.GetNumberOfCells()):#iterar por ramas  
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints();# number of points of the branch
        for i in range(numberOfCellPoints):
            originalPointId = centerline.GetCell(j).GetPointId(i)
            point = centerline.GetPoints().GetPoint(originalPointId)
            lista_puntos.append(list(point))
            
        #Ya tengo la lista de puntos de la rama entonces la resampleo y la agrego a la linea punto y polydata
        rama_resampleada = rdp(lista_puntos, epsilon)
        polyline = vtk.vtkPolyLine()
        for point in rama_resampleada:
            newPointId = points.InsertNextPoint(point)
            polyline.GetPointIds().InsertNextId(newPointId)
        cellarray.InsertNextCell(polyline)
        lista_rama.append(lista_puntos)
        lista_puntos = []
    resampleada.SetPoints(points)
    resampleada.SetLines(cellarray)
    return resampleada

def interpolarRDP_conRadio(centerline, epsilon=0.05):
    resampleada = vtk.vtkPolyData()
    cellarray = vtk.vtkCellArray()
    points = vtk.vtkPoints()
    lista_rama = []

    # Prepare radius array output
    radius_array = vtk.vtkFloatArray()
    radius_array.SetName("Radius")

    # Get original radius array
    original_radius = centerline.GetPointData().GetArray("Radius")

    for j in range(centerline.GetNumberOfCells()):  # iterate over branches  
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints()
        puntos_rama = []
        radios_rama = []

        for i in range(numberOfCellPoints):
            pid = centerline.GetCell(j).GetPointId(i)
            point = centerline.GetPoints().GetPoint(pid)
            radius = original_radius.GetValue(pid)
            puntos_rama.append(list(point))
            radios_rama.append(radius)

        # Apply RDP resampling to the branch
        rama_resampleada = rdp(puntos_rama, epsilon)

        # Reconstruct polyline and resample radius
        polyline = vtk.vtkPolyLine()
        for point in rama_resampleada:
            newPointId = points.InsertNextPoint(point)

            # Find index of the point in original list to get corresponding radius
            index = puntos_rama.index(point)
            radius_array.InsertNextValue(radios_rama[index])

            polyline.GetPointIds().InsertNextId(newPointId)

        cellarray.InsertNextCell(polyline)
        lista_rama.append(puntos_rama)

    resampleada.SetPoints(points)
    resampleada.SetLines(cellarray)
    resampleada.GetPointData().AddArray(radius_array)

    return resampleada

def vtpToObj (file, folderin, folderout):
    polydata_reader = vtk.vtkXMLPolyDataReader()
    #polydata_reader.SetFileName("centerlines/"+file)
    polydata_reader.SetFileName(folderin + "/"+file)
   

    polydata_reader.Update()
        
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(polydata_reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Assign actor to the renderer
    ren.AddActor(actor)
    print("writing file", folderout + "/" + file.split(".")[0])
    writer = vtk.vtkOBJExporter()
    writer.SetFilePrefix(folderout + "/" + file.split(".")[0]);
    writer.SetInput(renWin);
    writer.Write()