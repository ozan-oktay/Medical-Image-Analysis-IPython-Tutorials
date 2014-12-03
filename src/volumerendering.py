#!/usr/bin/python
"""
@author: Kevin Keraudren
"""

import SimpleITK as sitk
import vtk
import numpy as np
import math
from math import cos, sin

from IPython import display

from vtk.util.vtkConstants import *

__all__= ["volumeRender","vtk_basic"]


def numpy2VTK(img,spacing=[1.0,1.0,1.0]):
    # evolved from code from Stou S.,
    # on http://www.siafoo.net/snippet/314
    importer = vtk.vtkImageImport()
    
    img_data = img.astype('uint8')
    img_string = img_data.tostring() # type short
    dim = img.shape
    
    importer.CopyImportVoidPointer(img_string, len(img_string))
    importer.SetDataScalarType(VTK_UNSIGNED_CHAR)
    importer.SetNumberOfScalarComponents(1)
    
    extent = importer.GetDataExtent()
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[0] - 1)

    importer.SetDataSpacing( spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin( 0,0,0 )

    return importer

def volumeRender(img, tf=[],spacing=[1.0,1.0,1.0]):
    importer = numpy2VTK(img,spacing)

    # Transfer Functions
    opacity_tf = vtk.vtkPiecewiseFunction()
    color_tf = vtk.vtkColorTransferFunction()

    if len(tf) == 0:
        tf.append([img.min(),0,0,0,0])
        tf.append([img.max(),1,1,1,1])

    for p in tf:
        color_tf.AddRGBPoint(p[0], p[1], p[2], p[3])
        opacity_tf.AddPoint(p[0], p[4])

    # working on the GPU
    # volMapper = vtk.vtkGPUVolumeRayCastMapper()
    # volMapper.SetInputConnection(importer.GetOutputPort())

    # # The property describes how the data will look
    # volProperty =  vtk.vtkVolumeProperty()
    # volProperty.SetColor(color_tf)
    # volProperty.SetScalarOpacity(opacity_tf)
    # volProperty.ShadeOn()
    # volProperty.SetInterpolationTypeToLinear()

    # working on the CPU
    volMapper = vtk.vtkVolumeRayCastMapper()
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    compositeFunction.SetCompositeMethodToInterpolateFirst()
    volMapper.SetVolumeRayCastFunction(compositeFunction)
    volMapper.SetInputConnection(importer.GetOutputPort())

    # The property describes how the data will look
    volProperty =  vtk.vtkVolumeProperty()
    volProperty.SetColor(color_tf)
    volProperty.SetScalarOpacity(opacity_tf)
    volProperty.ShadeOn()
    volProperty.SetInterpolationTypeToLinear()
    
    # Do the lines below speed things up?
    # pix_diag = 5.0
    # volMapper.SetSampleDistance(pix_diag / 5.0)    
    # volProperty.SetScalarOpacityUnitDistance(pix_diag) 
    

    vol = vtk.vtkVolume()
    vol.SetMapper(volMapper)
    vol.SetProperty(volProperty)
    
    return [vol]

def move(actor, matrix):
    transfo_mat = vtk.vtkMatrix4x4()
    for i in range(0,4):
        for j in range(0,4):
            transfo_mat.SetElement(i,j, matrix[i,j])        
    actor.SetUserMatrix(transfo_mat)
 
def Rx( theta ):
    theta = float(theta)/180.0*math.pi
    return np.array([[ 1,          0,           0, 0],
                     [ 0, cos(theta), -sin(theta), 0],
                     [ 0, sin(theta),  cos(theta), 0],
                     [ 0,          0,           0, 1]],dtype='float32')
 
def Ry( theta ):
    theta = float(theta)/180.0*math.pi
    return np.array([[  cos(theta), 0, sin(theta), 0],
                     [           0, 1,          0, 0],
                     [ -sin(theta), 0, cos(theta), 0],
                     [           0, 0,          0, 1]],dtype='float32')
 
def Rz( theta ):
    theta = float(theta)/180.0*math.pi
    return np.array([[ cos(theta), -sin(theta), 0, 0],
                     [ sin(theta),  cos(theta), 0, 0],
                     [          0,           0, 1, 0],
                     [          0,           0, 0, 1]],dtype='float32')


def vtk_basic( actors, embed=False, magnification=1.0 ):
    """
    Create a window, renderer, interactor, add the actors and start the thing
    
    Parameters
    ----------
    actors :  list of vtkActors
    
    Returns
    -------
    nothing
    """     
    
    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600,600)
    # ren.SetBackground( 1, 1, 1)
 
    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    for a in actors:

        move( a, np.dot(Ry(-180),Rx(-180)) )
        
        # assign actor to the renderer
        ren.AddActor(a )
    
    # render
    renWin.Render()

    if embed:
        renWin.SetSize(300,300)
        grabber = vtk.vtkWindowToImageFilter()
        grabber.SetInput( renWin )
        grabber.SetMagnification( magnification )
        grabber.Update()
        
        writer = vtk.vtkPNGWriter()
        writer.SetInput( grabber.GetOutput() )
        writer.SetFileName( "screenshot.png" )
        writer.Write()
        return display.Image("screenshot.png")
    else:   
        # enable user interface interactor
        iren.Initialize()
        iren.Start()
