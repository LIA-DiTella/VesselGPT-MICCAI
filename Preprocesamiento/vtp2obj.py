import vtk
from auxiliares import read_vtk
import os
'''
filename = 'centerlines/ArteryObjAN170-19-network.vtp'

if filename.endswith('xml') or filename.endswith('vtp'):
    polydata_reader = vtk.vtkXMLPolyDataReader()
else:
	polydata_reader = vtk.vtkPolyDataReader()

polydata_reader.SetFileName(filename)
polydata_reader.Update()


mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(polydata_reader.GetOutputPort())
#mapper.SetInputData(polydata_reader.GetOutput())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a rendering window and renderer
ren = vtk.vtkRenderer()
ren.AddActor(actor)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# Create a renderwindowinteractor

#iren = vtk.vtkRenderWindowInteractor()
#iren.SetRenderWindow(renWin)

# Assign actor to the renderer
ren.AddActor(actor)

# Enable user interface interactor

#iren.Initialize()
#renWin.Render()
#iren.Start()

writer = vtk.vtkOBJExporter()
writer.SetFilePrefix("centerlines/ ArteryObjAN170-19-network");
writer.SetInput(renWin);
writer.Write()
'''


#files = ["ArteryObjAN1-0-network.vtp", "ArteryObjAN1-2-network.vtp"]
def vtpToObj (file):
    polydata_reader = vtk.vtkXMLPolyDataReader()
    polydata_reader.SetFileName("centerlines/"+file)
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

    writer = vtk.vtkOBJExporter()
    writer.SetFilePrefix("centerlines/"+file.split(".")[0]);
    writer.SetInput(renWin);
    writer.Write()

if __name__ == 'main':
    
    files = os.listdir('centerlines')
    for file in files:
        if file.split(".")[1] == "vtp":
            polydata_reader = vtk.vtkXMLPolyDataReader()
            polydata_reader.SetFileName("centerlines/"+file)
            polydata_reader.Update()
            #print(polydata_reader.GetOutput())
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(polydata_reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Create a rendering window and renderer
            ren = vtk.vtkRenderer()
            ren.AddActor(actor)

            renWin = vtk.vtkRenderWindow()
            renWin.AddRenderer(ren)

            # Create a renderwindowinteractor

            #iren = vtk.vtkRenderWindowInteractor()
            #iren.SetRenderWindow(renWin)

            # Assign actor to the renderer
            ren.AddActor(actor)

            # Enable user interface interactor

            #iren.Initialize()
            #renWin.Render()
            #iren.Start()

            writer = vtk.vtkOBJExporter()
            writer.SetFilePrefix("centerlines/ "+file.split(".")[0]);
            writer.SetInput(renWin);
            writer.Write()
