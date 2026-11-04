import numpy as np
import argparse
import scipy.spatial.transform as spst
import scipy.optimize as spo
import random
import os
from copy import deepcopy

#################### CONSTANTS ##################################
deg2rad = np.pi/180.
WaterCharges = [-0.8476 , 0.4238 , 0.4238] # water charges from SPC/E
mO = 15.99491461957
mH = 1.00782503223
WaterMasses = [mO, mH, mH]
#################################################################

def PrintXYZFile(AtSymb, XYZ, Line="", FileName=None, Mode="w"):
    """
    !Prints XYZ file.
    
    @param AtSymb    list of atomic symbols (list of strings).
    @param XYZ       XYZ coordinates of molecule (ndarray of floats).
    @param Line      Description line to be added (str). Default is ''.
    @param FileName  Name of the XYZ file (str or None). If None (default), prints the result in the stdout.
    @param Mode      Mode for saving into the file. Default is 'w'
    """    
    if FileName is None:
        print("%3i\n%s" % (len(AtSymb),Line))
        for i in range(0,len(AtSymb)):
            print(" %3s " % (AtSymb[i]), end='', flush=True)
            print(" %7.4f %7.4f %7.4f" % tuple(XYZ[i]), flush=True)
    else:
        with open(FileName, Mode) as outf:
            outf.write("%3i\n%s\n" % (len(AtSymb),Line))
            for i in range(0,len(AtSymb)):
                outf.write(" %3s " % (AtSymb[i]))
                outf.write(" %7.4f %7.4f %7.4f\n" % tuple(XYZ[i]))            
    
    

def GenerateOriginalWater(rOH=1.005, aHOH=109.47):
    """
    !Generates single water XYZ in original orientation from given geometric parameters.
    
    @param rOH    O--H distance in angstroms (float). Default is 1.005, taken from DOI: 10.1021/j100244a063, for water ice from neutron diffraction at 60K.
    @param aHOH   H-O-H angle in degrees (float).  Default is 109.47, taken from DOI: 10.1021/j100244a063, for water ice from neutron diffraction at 60K.
    
    @return AtSymb   list of atomic symbols (list of strings).
    @return XYZ      XYZ coordinates of water molecule (ndarray of floats).
    """
    
    x = rOH * np.sin(0.5 * deg2rad * aHOH)
    z = rOH * np.cos(0.5 * deg2rad * aHOH)

    AtSymb = ["O", "H", "H"]
    XYZ = np.array([[0., 0., 0.],
                    [ x, 0.,  z],
                    [-x, 0.,  z]])
    
    return AtSymb, XYZ


def ReCenterMolecule(XYZ, Weights=None):
    """
    !Centers the molecule according to given weights
    
    @param XYZ       XYZ coordinates of molecule (ndarray of floats).
    @param Weights   List of weights, according to which the centering will be made.
    """   
    if not Weights is None:
        rC = np.zeros(3,dtype=float)
        W = 0.
        for i,wi in enumerate(Weights):
            rC += np.abs(wi) * XYZ[i]
            W  += np.abs(wi)
        rC = rC/W
        for i,wi in enumerate(Weights):
            XYZ[i] -= rC
              
    return XYZ
    
        
def RotateMolecule(XYZ, phi, theta, chi):
    """
    !Rotates a molecule on the given Euler angles.
    
    @param XYZ    XYZ coordinates of water molecule (ndarray of floats).
    @param phi    Euler angle phi (float).
    @param theta  Euler angle theta (float).
    @param chi    Euler angle chi (float).
    
    @return NewXYZ     New XYZ coordinates of water molecule (ndarray of floats).
    """
    M = spst.Rotation.from_euler('zyz', [phi, theta, chi], degrees=True).as_matrix()
    NewXYZ = np.array(XYZ)
    for i in range(0,NewXYZ.shape[0]):
        NewXYZ[i] = np.matmul(M,NewXYZ[i])
    return NewXYZ
    

def ReadVectorFile(FileName):
    """
    !Reads the file of vectors for the water orientation around centers
    
    @param FileName   name of the file to read.

    @return Centers   list of names of centers.    
    @return Vectors   dictionary of list of centers.
    """    
    Centers = []
    Vectors = dict()
    with open(FileName, "r") as inpf:
        for line in inpf:
            word = line.split()
            if len(word) > 4:
                if not word[0][0] == '#':
                    nvec = np.array(word[1:4],dtype=float)
                    rval = float(word[4])
                    name = word[0]
                    if not name in Centers:
                        Centers.append(name)
                        Vectors.update({name : []})
                    nlen = np.sqrt(np.dot(nvec,nvec))
                    if nlen < 1.0e-5:
                        raise RuntimeError("Bad line in file! Zero length vector: "+line)
                        
                    nvec /= nlen
                    nvec *= rval
                    Vectors[name].append(nvec)
    
    if len(Centers)>2:
        raise NotImplementedError("We do not support more than two centers!")
                
    return Centers, Vectors
            

def GenerateWatersAroundCenter(IniAtSymb, IniXYZ, Displacements, Orientations):
    """
    !Provides arrangement of waters around the center according to the given directional vectors and the Euler angles.
    
    @param IniAtSymb         list of atomic symbols of the original water (list of strings).
    @param IniXYZ            XYZ coordinates of water molecule (ndarray of floats).
    @param Displacements  list of ndarrays of shape (n,3) of directional vectors for the displacement of water centers.
    @param Orientations   list of ndarrays of shape (n,3) of Euler angles for water orientation.
    
    @return AtSymb   list of atomic symbols of all waters (list of strings).
    @return XYZ      XYZ coordinates of all water molecules (ndarray of floats).
    """    
    AtSymb = []
    XYZ = None
    
    for i,disp in enumerate(Displacements):
        angles = Orientations[i]        
        xyz = RotateMolecule(IniXYZ, *angles) + disp
        AtSymb += IniAtSymb
        if XYZ is None:
            XYZ = xyz
        else:
            XYZ = np.concatenate([XYZ,xyz])        

    return AtSymb, XYZ       


def ConvertListOfAllEulerAnglesToDict(Centers, Vectors, ListOfEulerAngles):
    """
    !Converts the Euler angles in the form of the list (ndarray) to the dictionary.
    
    @param Centers               list of names of centers.    
    @param Vectors               dictionary of list of centers.
    @param ListOfEulerAngles     ndarray of shape (total number of waters, 3).
    
    @return EulerAngles   dictionary of list of Euler angles.
    """  
    EulerAngles = dict() 
    n = 0
    for cent in Centers:
        n1 = len(Vectors[cent])
        EulerAngles.update({cent: ListOfEulerAngles[n:(n+n1)]})
        n += n1
    return EulerAngles
    
    
def ConvertDictOfAllEulerAnglesToList(Centers, Vectors, EulerAngles):
    """
    !Converts the Euler angles in the form of the list (ndarray) to the dictionary.
    
    @param Centers               list of names of centers.    
    @param Vectors               dictionary of list of centers.
    @param EulerAngles   dictionary of list of Euler angles.
    
    @return ListOfEulerAngles     ndarray of shape (total number of waters, 3).
    """  
    ListOfEulerAngles = None
    for cent in Centers:
        N = len(EulerAngles[cent])
        EA = np.zeros((N,3),dtype=float)  
        for i,vi in enumerate(EulerAngles[cent]):
            EA[i] += vi
        if ListOfEulerAngles is None:
            ListOfEulerAngles = EA
        else:
            ListOfEulerAngles = np.concatenate([ListOfEulerAngles,EA])
    return ListOfEulerAngles
    
        
def GetEulerAngles(Centers, Vectors, FileName=None, Random=False):
    """
    !Provides the Euler angles for the waters.
    
    @param Centers       list of names of centers.    
    @param Vectors       dictionary of list of centers.
    @param FileName      filename with the saved Euler angles
    @parma Random        flag (bool) for using the random generation of the Euler angles.
    
    @return EulerAngles   dictionary of list of Euler angles.
    """  

    if FileName is None:
        EulerAngles = dict()
        for cent in Centers:
            N = len(Vectors[cent])
            EA = np.zeros((N,3),dtype=float)
            if Random:
                for i in range(0,N):
                    EA[i][0] = random.uniform(0.,360.)
                    EA[i][1] = random.uniform(0.,180.)
                    EA[i][2] = random.uniform(0.,360.)
            EulerAngles.update({cent: EA})
    else:
        data = np.loadtxt(FileName)
        if len(data.shape) == 1:
            data = np.reshape(data,(1,3))
        EulerAngles = ConvertListOfAllEulerAnglesToDict(Centers, Vectors, ListOfEulerAngles=data)

    return EulerAngles
    

def ComputeOrientationOfWaters(WXYZ, Displacements):
    """
    !Computes the water orientations around the center

    @param WXYZ            XYZ coordinates of water molecules for a given center (ndarray of floats).
    @param Displacements  list of ndarrays of shape (n,3) of directional vectors for the displacement of water centers.
    
    @return CosTheta   list of orientation cosines.
    """      
    CosTheta = []
    for i,di in enumerate(Displacements):
        idx0 = 3 * i
        rO = WXYZ[idx0]
        rH = 0.5 * (WXYZ[idx0+1] + WXYZ[idx0+2])
        rW = (rH - rO)
        rW /= np.sqrt(np.dot(rW,rW))
        nD = di / np.sqrt(np.dot(di,di))
        
        CosTheta.append(np.dot(nD,rW))

    return CosTheta

       
def GenerateGeometry(IniAtSymb, IniXYZ, Centers, Vectors, EulerAngles, CenterCenterDistance, Direction=2):
    """
    !Generates the hole geometry

    @param IniAtSymb         list of atomic symbols of the original water (list of strings).
    @param IniXYZ            XYZ coordinates of water molecule (ndarray of floats).
    @param Centers       list of names of centers.    
    @param Vectors       dictionary of list of centers.
    @param EulerAngles   dictionary of list of Euler angles
    @param CenterCenterDistance  distance (signed float) between centers.
    @param Direction            0, 1, or 2 -- the x/y/z direction along which a displacement of the center is done. Default is 2 (z).
    
    @return AtSymb   list of atomic symbols of all waters (list of strings).
    @return XYZ      XYZ coordinates of all water molecules (ndarray of floats).
    """      

    AtSymb = []
    XYZ = None
    CosTheta = []
    for i,cent in enumerate(Centers):
        Displacements = Vectors[cent]
        Orientations  = EulerAngles[cent]
        N = len(Displacements)
        tXYZ = np.zeros( (3*N+1,3), dtype=float)
        AtSymb.append(cent)
        tAtSymb, tXYZ[1:] = GenerateWatersAroundCenter(IniAtSymb, IniXYZ, Displacements, Orientations)
        CosTheta.append(np.mean(ComputeOrientationOfWaters(tXYZ[1:], Displacements)))
        AtSymb += tAtSymb
        
        if i>0:
            d = np.zeros(3,dtype=float)
            d[Direction] += CenterCenterDistance
            for j in range(0,len(tXYZ)):
                tXYZ[j] += d
            
        if XYZ is None:
            XYZ = tXYZ
        else:
            XYZ = np.concatenate([XYZ,tXYZ])        

    return AtSymb, XYZ, CosTheta           


def WriteXYZFile(AtSymb, XYZ, FileName=None, Mode="a", Description=""):
    """
    !Writes the XYZ file

    @param AtSymb   list of atomic symbols of all waters (list of strings).
    @param XYZ      XYZ coordinates of all water molecules (ndarray of floats).
    @param FileName filename of the xyz file to write to (str). If None (default), the result is dumped to the stdout.
    @param Mode     Mode of wrtiting into a file ("w"=write, "a"=append).
    @param Description a line to add to the XYZ file (second line, for description, str).
    """         
    N = len(AtSymb)
    FileLine = "%i\n%s\n" % (N,Description)
    for i in range(0,N):
        FileLine += " %3s " % (AtSymb[i])
        FileLine += "%7.4f %7.4f %7.4f\n" % tuple(XYZ[i])
    if FileName is None:
        print(FileLine, end='')
    else:
        with open(FileName,Mode) as outf:
            outf.write(FileLine)


def WriteOrcaOutput(Charge, Mult, AtSymb, XYZ, Method="",
                    MaxCore=1000, NumProcs=1, FileName="orca.inp"):
    """
    !Writes the ORCA input file

    @param Charge   charge of the system (int).
    @param Mult     multiplicity of the system (int).
    @param AtSymb   list of atomic symbols of all waters (list of strings).
    @param XYZ      XYZ coordinates of all water molecules (ndarray of floats).
    @param Method   string for the quantum-chemical method.
    @param MaxCore  maxcore parameter of ORCA (memory per process, in MB).
    @param NumProcs number of processes for ORCA (nprocs parameter).
    @param FileName filename of the input file to write to (str).
    """         
        
    LineToWrite  ="! %s\n\n" % (Method)
    LineToWrite +="%%pal\n nprocs %i\nend\n\n" % (NumProcs)
    LineToWrite +="%%maxcore %i\n\n" % (MaxCore)
    LineToWrite +="* xyz %i %i\n" % (Charge, Mult)
    for i,si in enumerate(AtSymb):
        LineToWrite += " %3s " % (si)
        LineToWrite += "%7.4f %7.4f %7.4f\n" % tuple(XYZ[i])        
    LineToWrite +="*\n\n" 
    with open(FileName, "w") as outf:
        outf.write(LineToWrite)


def ExtractEnergyFromOrcaOutput(FileName):
    """
    !Extracts the final single point energy from ORCA output file

    @param FileName filename of the output file to parce (str).
    
    @return Energy  Single point energy in hartrees (float).
    @return IsItBad boolean flag of whether ORCA terminated normally.
    """        
    inpf = open(FileName, "r")
    Lines=[line.rstrip() for line in inpf.readlines()]  # load all the lines in the list
    inpf.close()
    IsItBad = True
    Energy = 1.0e10
    for nl,line in enumerate(Lines):
        word = line.split()
        if len(word)>=5:
            if word[0].lower() == 'final' and word[1].lower() == 'single' and word[2].lower() == 'point' and  word[3].lower() == 'energy':
                Energy = float(word[4])
        if len(word)>=3:
            if word[0] == '****ORCA' and word[1] == 'TERMINATED' and word[2] == 'NORMALLY****':
                IsItBad = False
    
    return Energy, IsItBad


def CleanupOrcaFiles(ORCABase="mol_orca"):
    FilesToCheck = os.listdir() 
    N=len(ORCABase)
    for f in FilesToCheck:
        if ORCABase == f[:N] and os.path.isfile(f):
            os.remove(f)


def RunOrcaSPECalculation(Charge, Mult, AtSymb, XYZ, PathToOrca, Method="",
                          MaxCore=1000, NumProcs=1, ORCABase="mol_orca"):
    """
    !Runs the ORCA single point energy (SPE) calculations.

    @param Charge   charge of the system (int).
    @param Mult     multiplicity of the system (int).
    @param AtSymb   list of atomic symbols of all waters (list of strings).
    @param XYZ      XYZ coordinates of all water molecules (ndarray of floats).
    @param PathToOrca command to run ORCA (path to the file).
    
    @param Method   string for the quantum-chemical method.
    @param MaxCore  maxcore parameter of ORCA (memory per process, in MB).
    @param NumProcs number of processes for ORCA (nprocs parameter).
    @param ORCABase base of the filenames (str).

    @return Energy  Single point energy in hartrees (float).
    @return IsItBad boolean flag of whether ORCA terminated normally.
    """            
    InpFileName = ORCABase + ".inp"
    OutFileName = ORCABase + ".log"
    
    WriteOrcaOutput(Charge, Mult, AtSymb, XYZ, Method,
                    MaxCore, NumProcs, FileName=InpFileName)
    #AddOption="  -mca mpi_cuda_support 0 -mca pml ucx -mca btl_tcp_if_include ib0  "
    AddOption=" "
    os.system(PathToOrca+" "+InpFileName+AddOption+"  > "+OutFileName)
    En, Good = ExtractEnergyFromOrcaOutput(OutFileName)
    
    return En, Good
    

def RunXTBSPECalculation(Charge, AtSymb, XYZ, Method="",
                         XTBBase="mol_xtb"):
    """
    !Runs the xTB single point energy (SPE) calculations.

    @param Charge   charge of the system (int).
    @param AtSymb   list of atomic symbols of all waters (list of strings).
    @param XYZ      XYZ coordinates of all water molecules (ndarray of floats).
    
    @param Method  additional options for xTB.
    @param XTBBase base of the filenames (str).

    @return Energy  Single point energy in hartrees (float).
    @return IsItBad boolean flag of whether ORCA terminated normally.
    """ 
 
    OutFileName = XTBBase + ".log" 
    OutErrName  = XTBBase + ".err" 
    InpFileName = XTBBase+"_ini.xyz"
    WriteXYZFile(AtSymb, XYZ, FileName=InpFileName, Mode="w", Description="")
    Command = "xtb "+InpFileName+" --charge "+str(Charge)+"  "+Method+" 1>  "+OutFileName+" 2> "+OutErrName
    #print(Command)
    os.system(Command)
    
    inpf = open(OutFileName, "r")
    Lines=[line.rstrip() for line in inpf.readlines()]  # load all the lines in the list
    inpf.close()
    IsItBad = True
    Energy = 1.0e10
    for nl,line in enumerate(Lines):
        word = line.split()
        if len(word)>=5:
            if word[0] == '|' and word[1].lower() == 'total' and word[2].lower() == 'energy' and  word[4].lower() == 'eh':
                Energy = float(word[3])   
        
    
    with open(OutErrName,"r") as inpf:
        l = inpf.readline()
        if l.split() == ['normal', 'termination', 'of', 'xtb']:
            Good = True
        else:
            Good = False

    return Energy, Good    


def CleanXTBFiles(XTBBase="mol_xtb"):
    CleanupOrcaFiles(ORCABase=XTBBase)
    for f in ['charges', 'wbo',  'xtbrestart',  'xtbtopo.mol', '.xtbtok']:
        try:
            os.remove(f)
        except:
            pass

    
def RunOptimizationOfTheCluster(Charge, Mult, VectorFileName, 
                                PathToOrca=None, RunXTB = False,
                                CenterCenterDistance=None, CenterCenterDirection=2,
                                EulerAnglesFileName=None, EulerAngleRandom=True,
                                rOH=0.958, aHOH=104.4776, Weights=WaterCharges,
                                Method="", MaxCore=1000, NumProcs=1, CalcFilesBase="mol_opt",
                                RunOpt=False,
                                SaveBase="save_data", Debug=False):
    """
    !Optimizes a cluster

    @param Charge           charge of the system (int).
    @param Mult             multiplicity of the system (int).
    @param VectorFileName   vector file to generate the cluster (str).
    @param PathToOrca       command to run ORCA (path to the file). If None, no ORCA is invoked.
    @param RunXTB           command to run the 
 
    @param CenterCenterDistance  distance (signed float) between centers.
    @param CenterCenterDirection            0, 1, or 2 -- the x/y/z direction along which a displacement of the center is done. Default is 2 (z).
    
    @param EulerAnglesFileName filename with the Euler angles (str). If None, then the angles will be generated by the software.
    @param EulerAngleRandom    flag (bool) for using the random generation of the Euler angles.
    
    @param rOH      O--H distance in angstroms (float). Default is 0.958 (experimental from NIST CCCBDB).
    @param aHOH     H-O-H angle in degrees (float). Default is 104.4776 (experimental from NIST CCCBDB).    
    @param Weights  weights to get the center of charge of water. Taken by default from SPC/E.
    
    @param Method   string for the quantum-chemical method.
    @param MaxCore  maxcore parameter of ORCA (memory per process, in MB).
    @param NumProcs number of processes for ORCA (nprocs parameter).
    @param CalcFilesBase base of the filenames with ORCA calculations (str).
    @param RunOpt   flag to run the optimization of the cluster geometry (bool).
    @param SaveBase base of the filenames where the data will be saved.
    @param Debug    flag to save intermediate structures
    
    @return Energy  Single point energy in hartrees (float).
    @return Ended   boolean flag for 
    """       
    
    Centers, Vectors = ReadVectorFile(VectorFileName)
    EulerAngles = GetEulerAngles(Centers, Vectors, FileName=EulerAnglesFileName, Random=EulerAngleRandom)
    OneWaterAtSymb, OneWaterXYZ = GenerateOriginalWater(rOH, aHOH)
    OneWaterXYZ = ReCenterMolecule(OneWaterXYZ, Weights)
    
    N = len(ConvertDictOfAllEulerAnglesToList(Centers, Vectors, EulerAngles))
    Bounds = []
    for i in range(0,N):
        Bounds += [(0.,2*np.pi)]
        Bounds += [(0.,np.pi)]        
        Bounds += [(0.,2*np.pi)]
    #print(Bounds)
    AtSymb, XYZ, CosTheta = GenerateGeometry(OneWaterAtSymb, OneWaterXYZ, Centers, Vectors, EulerAngles, CenterCenterDistance, Direction=CenterCenterDirection)

    def SingleMoleculeCalculationWrapper(x):
        tLEA = np.reshape(x, (N,3)) * 180./np.pi
        EA = ConvertListOfAllEulerAnglesToDict(Centers, Vectors, tLEA)
        tAtSymb, tXYZ, tCosTheta = GenerateGeometry(OneWaterAtSymb, OneWaterXYZ, Centers, Vectors, EA, CenterCenterDistance, CenterCenterDirection)
        #print(EA)
        if not PathToOrca is None:
            En, Good = RunOrcaSPECalculation(Charge, Mult, tAtSymb, tXYZ, PathToOrca, Method, MaxCore, NumProcs, CalcFilesBase)
        else:
            En, Good = RunXTBSPECalculation(Charge, tAtSymb, tXYZ, Method, CalcFilesBase)
        if Debug:
            WriteXYZFile(tAtSymb, tXYZ, FileName=SaveBase+"_debug.xyz", Mode="w", Description=str(tCosTheta)+" E = "+str(En)+" Eh")
        return En

    LEA = ConvertDictOfAllEulerAnglesToList(Centers, Vectors, EulerAngles)
    x0 = LEA.flatten() * np.pi/180.
    MinEn = SingleMoleculeCalculationWrapper(x0)
    WriteXYZFile(AtSymb, XYZ, FileName=SaveBase+"_ini.xyz", Mode="w", Description=str(CosTheta)+" E = "+str(MinEn)+" Eh")
    np.savetxt(SaveBase+"_ini.angles", LEA, fmt='%15.10f')

    if RunOpt:   
        res = spo.minimize(fun=SingleMoleculeCalculationWrapper,
                       x0 = x0, bounds=Bounds,
                       method='Powell')
        # xatol = 5., fatol = 0.00001,
    
        LEA = np.reshape(res.x, (N,3)) * 180./np.pi
        np.savetxt(SaveBase+"_fin.angles", LEA, fmt='%15.10f')
        EA = ConvertListOfAllEulerAnglesToDict(Centers, Vectors, LEA)
        AtSymb, XYZ, CosTheta = GenerateGeometry(OneWaterAtSymb, OneWaterXYZ, Centers, Vectors, EA, CenterCenterDistance, CenterCenterDirection)
        MinEn = SingleMoleculeCalculationWrapper(res.x)
        WriteXYZFile(AtSymb, XYZ, FileName=SaveBase+"_fin.xyz", Mode="w", Description=str(CosTheta)+" E = "+str(MinEn)+" Eh")
    
    return MinEn, CosTheta    


def WriteStateOfTheSystem(Centers, Vectors, EulerAngles, Displacement=0., NewVectorFileName="new_state.vecfle", NewEulerFileName="new_state.euler"):
    """
    !Writes a new state files for the system

    @param Centers               list of names of centers.    
    @param Vectors               dictionary of list of centers.
    @param EulerAngles   dictionary of list of Euler angles.
    
    @param Displacement  is a displacement for the waters from their centers
    
    @param NewVectorFileName   New vectors/centers filename
    @param NewEulerFileName    New Euler angles' filename
    """    

    with open(NewVectorFileName, "w") as outf:
        for cent in Centers:
            for vec in Vectors[cent]:
                outf.write(cent+"  ")
                VLen = np.sqrt(np.dot(vec,vec))
                NormVec = vec/VLen
                VLen += Displacement    
                
                outf.write(" %15.10f %15.10f %15.10f " % tuple(NormVec))
                outf.write(" %15.10f\n" % (VLen))
    
    LEA = ConvertDictOfAllEulerAnglesToList(Centers, Vectors, EulerAngles)
    np.savetxt(NewEulerFileName, LEA, fmt='%15.10f')


def DoClusterSizeScan(Charge, Mult, VectorFileName, MinRadDispl, MaxRadDispl, DR,
                                PathToOrca=None, RunXTB = False,
                                CenterCenterDistance=None, CenterCenterDirection=2,
                                EulerAnglesFileName=None, EulerAngleRandom=True,
                                rOH=0.958, aHOH=104.4776, Weights=WaterCharges,
                                Method="", MaxCore=1000, NumProcs=1, CalcFilesBase="mol_opt",
                                RunOpt=False,
                                SaveBase="save_data", Dir2Run=".", Debug=False):
    """
    !Provides a dissociation-like curve with scanning uniformy the cluster size by moving the points inwards-outwar.

    @param Charge           charge of the system (int).
    @param Mult             multiplicity of the system (int).
    @param VectorFileName   vector file to generate the cluster (str).
    @param PathToOrca       command to run ORCA (path to the file). If None, no ORCA is invoked.
    @param RunXTB           command to run the 
 
    @param CenterCenterDistance  distance (signed float) between centers.
    @param CenterCenterDirection            0, 1, or 2 -- the x/y/z direction along which a displacement of the center is done. Default is 2 (z).
    
    @param EulerAnglesFileName filename with the Euler angles (str). If None, then the angles will be generated by the software.
    @param EulerAngleRandom    flag (bool) for using the random generation of the Euler angles.
    
    @param rOH      O--H distance in angstroms (float). Default is 0.958 (experimental from NIST CCCBDB).
    @param aHOH     H-O-H angle in degrees (float). Default is 104.4776 (experimental from NIST CCCBDB).    
    @param Weights  weights to get the center of charge of water. Taken by default from SPC/E.
    
    @param Method   string for the quantum-chemical method.
    @param MaxCore  maxcore parameter of ORCA (memory per process, in MB).
    @param NumProcs number of processes for ORCA (nprocs parameter).
    @param CalcFilesBase base of the filenames with ORCA calculations (str).
    @param RunOpt   flag to run the optimization of the cluster geometry (bool).
    @param SaveBase base of the filenames where the data will be saved.
    @param Dir2Run  directory where to run calculation.
    @param Debug    flag to save intermediate structures
    
    @return Idxs  List of indexes of the structures' displacements.
    @return Displacements     List of displacements of the structures.
    @return Ens               List of energies (in Ha) of structures.
    """     
    
    Centers, Vectors = ReadVectorFile(VectorFileName)
    IniEulerAngles = GetEulerAngles(Centers, Vectors, FileName=EulerAnglesFileName, Random=EulerAngleRandom)    
    CurrEulerAngles = deepcopy(IniEulerAngles)
    
    IdxNeg = np.arange(-1, -int(np.ceil(MinRadDispl/DR))-1, -1)
    IdxPos = np.arange(0, int(np.ceil(MaxRadDispl/DR))+1, 1)
    #print(IdxNeg,IdxPos)
    Ens = []
    Displacements = []
    Idxs = []
    CosThetas = []
    
    outf = open(Dir2Run+"/store_intermediate.scan", "w")
    for IdxAll in [IdxPos,IdxNeg]:
        CurrEulerAngles = deepcopy(IniEulerAngles)  
        for i in IdxAll:
            dname=Dir2Run+"/scan_step%+03i" % (i)
            try:
                if Debug:
                    print("Making folder "+dname)
                os.mkdir(dname)
            except:
                if Debug:
                    print(dname+" folder exists")    
        
            VFN = dname+"/state.vecfle"
            EFN = dname+"/state.euler"
            WriteStateOfTheSystem(Centers, Vectors, CurrEulerAngles, Displacement=DR*i, 
                                  NewVectorFileName=VFN, NewEulerFileName=EFN)
            
            En, CosTheta = RunOptimizationOfTheCluster(Charge, Mult, VFN, 
                                    PathToOrca, RunXTB,
                                    CenterCenterDistance, CenterCenterDirection,
                                    EFN, False,
                                    rOH, aHOH, Weights,
                                    Method, MaxCore, NumProcs, CalcFilesBase,
                                    RunOpt,
                                    dname+"/"+SaveBase, Debug)
    
            if RunOpt:
                CurrEulerAngles = GetEulerAngles(Centers, Vectors, FileName=dname+"/"+SaveBase+"_fin.angles", Random=False)
                
            Ens.append(En)
            Displacements.append(i*DR)
            Idxs.append(i)
            CosThetas.append(CosTheta)
            outf.write(" %10.4f %15.10f " % (i*DR, En))
            for c in CosTheta:
                outf.write(" %15.5f " % (c))
            outf.write("%3i \n" % (i))
            outf.flush()
    outf.close()
    
    
    SInds = np.argsort(Idxs)
    Idxs = np.array(Idxs)[SInds]
    Ens = np.array(Ens)[SInds]
    Displacements = np.array(Displacements)[SInds]
    
    D2Add = [Displacements,Ens]
    CosThetas = np.array(CosThetas)
    for c in CosThetas.T:
        D2Add += [c]
    D2Add += [Idxs]
    #print(D2Add)
    np.savetxt(Dir2Run+"/scan.res", np.stack(D2Add, axis=-1), fmt='%15.10f')
    with open(Dir2Run+"/scan_mol.trj", "w") as outf:
        for idx in Idxs:
            fname=Dir2Run+"/scan_step%+03i/" % (idx)
            if RunOpt:
                fname+=SaveBase+"_fin.xyz"
            else:
                fname+=SaveBase+"_ini.xyz"
            with open(fname,"r") as inpf:
                for line in inpf.readlines():
                    outf.write(line)
            outf.flush()
            
                    
    return Idxs, Displacements, Ens, CosThetas

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate waters distribution around a defined center.')

    parser.add_argument('-f', '--VectorFile', type=str, default=None,
                    help="Name of the file containing the vectors for defining the cluster. Default is None (no file).")

    parser.add_argument('--Charge', type=int, default=0,
                    help="Charge of the system. Default is 0.") 

    parser.add_argument('--Mult', type=int, default=1,
                    help="Multiplicity of the system. Default is 1.") 
     
    parser.add_argument('--rOH', type=float, default=0.958,
                    help="OH bond length of the water (float). Default is 0.958 (experimental from NIST CCCBDB).")

    parser.add_argument('--aHOH', type=float, default=104.4776,
                    help="HOH valence angle of the water (float). Default is 104.4776 (experimental from NIST CCCBDB).")

    parser.add_argument('--NumStruct', type=int, default=1,
                    help="Number of structures to generate/calculate/optimize. Default is 1.") 
    
    parser.add_argument('--Opt', action='store_true', 
                    help="Flag to optimize random structures.")
    
    parser.add_argument('--EulerFile', type=str, default=None,
                    help="Name of the file containing the Euler angles. Default is None (no file).")

    parser.add_argument('--Random', action='store_true', 
                    help="Flag to use randomized Euler angles.")   

    parser.add_argument('--DistCC', type=float, default=0.,
                    help="Center-center distance in angstroms (float). Default is 0.")

    parser.add_argument('--DirCC', type=int, default=2,
                    help="Direction axis for dispacement of centers (int). 0=x, 1=y, 2=z. Default is 2 (z-axis).")

    parser.add_argument('--ORCAPath', type=str, default='$(which orca)',
                    help='Path to ORCA.')
    
    parser.add_argument('--AddMethod', type=str, default='',
                    help='Additional options/method for xTB/ORCA.')
                    
    parser.add_argument('--UseORCA', action='store_true', 
                    help="Flag to use ORCA instead of xTB.")   

    parser.add_argument('--MaxCore', type=int, default=1000,
                    help="MaxCore parameter of ORCA. Default is 1000.")      
                                  
    parser.add_argument('--NumProcs', type=int, default=1,
                    help="NumProcs parameter of ORCA. Default is 1.")                    
                    
    parser.add_argument('--DoNotCenter', action='store_true', 
                    help="Flag to have the oxygen as a center of the waters, not the SPC/E charges.")   

    parser.add_argument('--DoScan', action='store_true', 
                    help="Flag to do a cluster size scan, moving waters uniformly inwards/outwards from the center.")   

    parser.add_argument('--DR', type=float, default=0.5,
                    help="Displacement for a cluster size scan in angstroms (float). Default is 0.5 AA.")
                    
    parser.add_argument('--MinDispl', type=float, default=0.0,
                    help="Displacement for a scan in negative direction (float). Default is 0.0 (no scan).")
                    
    parser.add_argument('--MaxDispl', type=float, default=0.5,
                    help="Displacement for a scan in positive direction (float). Default is 0.5 AA.")
                    
    parser.add_argument('--Debug', action='store_true', 
                    help="Flag to store some of the intermediate files.")   
    

    args, unknown = parser.parse_known_args() # parse the arguments
    print("Unknown arguments: "+str(unknown))

    CalcOptionsDict = {"Charge": args.Charge,
                       "Mult": args.Mult,
                       "VectorFileName": args.VectorFile,
                       "CenterCenterDistance": args.DistCC,
                       "CenterCenterDirection": args.DirCC,
                       "EulerAnglesFileName": args.EulerFile,
                       "EulerAngleRandom": args.Random,
                       "rOH": args.rOH,
                       "aHOH": args.aHOH,
                       "Method": args.AddMethod,
                       "RunOpt": args.Opt,
                       "Debug": args.Debug
                      }

    W = WaterCharges
    if args.DoNotCenter:
        W = None

    if args.UseORCA:
        CalcOptionsDict.update({"RunXTB": False})
        CalcOptionsDict.update({"PathToOrca": args.ORCAPath})
        CalcOptionsDict.update({"MaxCore": args.MaxCore})
        CalcOptionsDict.update({"NumProcs": args.NumProcs})
    else:
        CalcOptionsDict.update({"RunXTB": True})
    
    
    CalcOptionsDict.update({"Weights": W})
    CalcOptionsDict.update({"CalcFilesBase": "qwerty"})
    CalcOptionsDict.update({"SaveBase": "abcdef"})
    if args.DoScan:
        CalcOptionsDict.update({"Dir2Run": "asdasd"})
        CalcOptionsDict.update({"MinRadDispl": args.MinDispl})
        CalcOptionsDict.update({"MaxRadDispl": args.MaxDispl})
        CalcOptionsDict.update({"DR": args.DR})
    basedirname="gwh"
    
    if args.Opt:
        basedirname+="_opt"
        CalcOptionsDict.update({"CalcFilesBase": "spe_tmp"})
    else:
        basedirname+="_spe"
    
    for nstr in range(0,args.NumStruct):
        print("Doing structure "+str(nstr))
        StrIdx = "%04i" % (nstr)
        dname=basedirname+"_str"+StrIdx
        try:
            print("Making folder "+dname)
            os.mkdir(dname)
        except:
            print(dname+" folder exists")
        
        if args.DoScan:
            CalcOptionsDict["Dir2Run"] = dname 
            CalcOptionsDict["CalcFilesBase"] = dname+"/qc_tmp"        
            CalcOptionsDict["SaveBase"] = "mol_str"+StrIdx   
            DoClusterSizeScan(**CalcOptionsDict)
        else:
            CalcOptionsDict["CalcFilesBase"] = dname+"/qc_tmp"        
            CalcOptionsDict["SaveBase"] = dname+"/mol_str"+StrIdx       
            RunOptimizationOfTheCluster(**CalcOptionsDict)    
        CleanXTBFiles(XTBBase=CalcOptionsDict["CalcFilesBase"])


