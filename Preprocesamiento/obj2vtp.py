import vtk

import os

def read_obj(file):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file)
    reader.Update()
    return reader.GetOutput()


if __name__ == "__main__":
    print("running...")
    files = []
    for file in files:
        if file.split(".")[1] == "obj":
            data = read_obj('mallasArregladas/'+file)
            file = file.split(".")[0]
            writer = vtk.vtkXMLPolyDataWriter();
            writer.SetFileName('mallasArregladas/'+file+".vtp");
            writer.SetInputData(data);
            writer.Write();


'''
filename = 'mallasArregladas/ArteryObjAN1-2.obj'
reader = vtk.vtkOBJReader()
reader.SetFileName(filename)
reader.Update()
data = reader.GetOutput()
print(data)
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("mallasArregladas/ArteryObjAN1-2.vtp");
writer.SetInputData(data);
writer.Write()
'''
