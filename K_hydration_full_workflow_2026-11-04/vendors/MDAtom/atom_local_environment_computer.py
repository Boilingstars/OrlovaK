#! /usr/bin/python3

import numpy as np
import argparse
import scipy.spatial as sps

# "mass" is the mass of the most stable isotop (in a.m.u.)
# "weight" is the atomic weight (in a.m.u.)
# "covRad" is the Covalent radius by Pyykko et al. (in Angstroems)
# "vdwRad" is the Van der Waals radius
# Data stolen from mendeleev module on 2021-05-20
ElementData = {
"x": {"atNumber": 0, "symbol": "X", "mass": 0., "weight": 0., "covRad": 1., "vdwRad": 1. }, # this is the fallback element
"h": {"atNumber": 1, "symbol": "H", "mass": 1.0078250322, "weight": 1.008, "covRad": 0.320, "vdwRad": 1.100 },
"he": {"atNumber": 2, "symbol": "He", "mass": 4.0026032541, "weight": 4.002602, "covRad": 0.460, "vdwRad": 1.400 },
"li": {"atNumber": 3, "symbol": "Li", "mass": 7.01600344, "weight": 6.94, "covRad": 1.330, "vdwRad": 1.820 },
"be": {"atNumber": 4, "symbol": "Be", "mass": 9.0121831, "weight": 9.0121831, "covRad": 1.020, "vdwRad": 1.530 },
"b": {"atNumber": 5, "symbol": "B", "mass": 11.009305, "weight": 10.81, "covRad": 0.850, "vdwRad": 1.920 },
"c": {"atNumber": 6, "symbol": "C", "mass": 12.0, "weight": 12.011, "covRad": 0.750, "vdwRad": 1.700 },
"n": {"atNumber": 7, "symbol": "N", "mass": 14.003074004, "weight": 14.007, "covRad": 0.710, "vdwRad": 1.550 },
"o": {"atNumber": 8, "symbol": "O", "mass": 15.99491462, "weight": 15.999, "covRad": 0.630, "vdwRad": 1.520 },
"f": {"atNumber": 9, "symbol": "F", "mass": 18.998403163, "weight": 18.998403163, "covRad": 0.640, "vdwRad": 1.470 },
"ne": {"atNumber": 10, "symbol": "Ne", "mass": 19.99244018, "weight": 20.1797, "covRad": 0.670, "vdwRad": 1.540 },
"na": {"atNumber": 11, "symbol": "Na", "mass": 22.98976928, "weight": 22.98976928, "covRad": 1.550, "vdwRad": 2.270 },
"mg": {"atNumber": 12, "symbol": "Mg", "mass": 23.9850417, "weight": 24.305, "covRad": 1.390, "vdwRad": 1.730 },
"al": {"atNumber": 13, "symbol": "Al", "mass": 26.9815385, "weight": 26.9815385, "covRad": 1.260, "vdwRad": 1.840 },
"si": {"atNumber": 14, "symbol": "Si", "mass": 27.976926535, "weight": 28.085, "covRad": 1.160, "vdwRad": 2.100 },
"p": {"atNumber": 15, "symbol": "P", "mass": 30.973761998, "weight": 30.973761998, "covRad": 1.110, "vdwRad": 1.800 },
"s": {"atNumber": 16, "symbol": "S", "mass": 31.972071174, "weight": 32.06, "covRad": 1.030, "vdwRad": 1.800 },
"cl": {"atNumber": 17, "symbol": "Cl", "mass": 34.9688527, "weight": 35.45, "covRad": 0.990, "vdwRad": 1.750 },
"ar": {"atNumber": 18, "symbol": "Ar", "mass": 39.96238312, "weight": 39.948, "covRad": 0.960, "vdwRad": 1.880 },
"k": {"atNumber": 19, "symbol": "K", "mass": 38.96370649, "weight": 39.0983, "covRad": 1.960, "vdwRad": 2.750 },
"ca": {"atNumber": 20, "symbol": "Ca", "mass": 39.9625909, "weight": 40.078, "covRad": 1.710, "vdwRad": 2.310 },
"sc": {"atNumber": 21, "symbol": "Sc", "mass": 44.955908, "weight": 44.955908, "covRad": 1.480, "vdwRad": 2.150 },
"ti": {"atNumber": 22, "symbol": "Ti", "mass": 47.947942, "weight": 47.867, "covRad": 1.360, "vdwRad": 2.110 },
"v": {"atNumber": 23, "symbol": "V", "mass": 50.943957, "weight": 50.9415, "covRad": 1.340, "vdwRad": 2.070 },
"cr": {"atNumber": 24, "symbol": "Cr", "mass": 51.940506, "weight": 51.9961, "covRad": 1.220, "vdwRad": 2.060 },
"mn": {"atNumber": 25, "symbol": "Mn", "mass": 54.938044, "weight": 54.938044, "covRad": 1.190, "vdwRad": 2.050 },
"fe": {"atNumber": 26, "symbol": "Fe", "mass": 55.934936, "weight": 55.845, "covRad": 1.160, "vdwRad": 2.040 },
"co": {"atNumber": 27, "symbol": "Co", "mass": 58.933194, "weight": 58.933194, "covRad": 1.110, "vdwRad": 2.000 },
"ni": {"atNumber": 28, "symbol": "Ni", "mass": 57.935342, "weight": 58.6934, "covRad": 1.100, "vdwRad": 1.970 },
"cu": {"atNumber": 29, "symbol": "Cu", "mass": 62.929598, "weight": 63.546, "covRad": 1.120, "vdwRad": 1.960 },
"zn": {"atNumber": 30, "symbol": "Zn", "mass": 63.929142, "weight": 65.38, "covRad": 1.180, "vdwRad": 2.010 },
"ga": {"atNumber": 31, "symbol": "Ga", "mass": 68.925574, "weight": 69.723, "covRad": 1.240, "vdwRad": 1.870 },
"ge": {"atNumber": 32, "symbol": "Ge", "mass": 73.92117776, "weight": 72.63, "covRad": 1.210, "vdwRad": 2.110 },
"as": {"atNumber": 33, "symbol": "As", "mass": 74.921595, "weight": 74.921595, "covRad": 1.210, "vdwRad": 1.850 },
"se": {"atNumber": 34, "symbol": "Se", "mass": 79.916522, "weight": 78.971, "covRad": 1.160, "vdwRad": 1.900 },
"br": {"atNumber": 35, "symbol": "Br", "mass": 78.918338, "weight": 79.904, "covRad": 1.140, "vdwRad": 1.850 },
"kr": {"atNumber": 36, "symbol": "Kr", "mass": 83.91149773, "weight": 83.798, "covRad": 1.170, "vdwRad": 2.020 },
"rb": {"atNumber": 37, "symbol": "Rb", "mass": 84.91178974, "weight": 85.4678, "covRad": 2.100, "vdwRad": 3.030 },
"sr": {"atNumber": 38, "symbol": "Sr", "mass": 87.905613, "weight": 87.62, "covRad": 1.850, "vdwRad": 2.490 },
"y": {"atNumber": 39, "symbol": "Y", "mass": 88.90584, "weight": 88.90584, "covRad": 1.630, "vdwRad": 2.320 },
"zr": {"atNumber": 40, "symbol": "Zr", "mass": 89.9047, "weight": 91.224, "covRad": 1.540, "vdwRad": 2.230 },
"nb": {"atNumber": 41, "symbol": "Nb", "mass": 92.90637, "weight": 92.90637, "covRad": 1.470, "vdwRad": 2.180 },
"mo": {"atNumber": 42, "symbol": "Mo", "mass": 97.905405, "weight": 95.95, "covRad": 1.380, "vdwRad": 2.170 },
"ru": {"atNumber": 44, "symbol": "Ru", "mass": 101.904344, "weight": 101.07, "covRad": 1.250, "vdwRad": 2.130 },
"rh": {"atNumber": 45, "symbol": "Rh", "mass": 102.9055, "weight": 102.9055, "covRad": 1.250, "vdwRad": 2.100 },
"pd": {"atNumber": 46, "symbol": "Pd", "mass": 105.90348, "weight": 106.42, "covRad": 1.200, "vdwRad": 2.100 },
"ag": {"atNumber": 47, "symbol": "Ag", "mass": 106.90509, "weight": 107.8682, "covRad": 1.280, "vdwRad": 2.110 },
"cd": {"atNumber": 48, "symbol": "Cd", "mass": 113.903365, "weight": 112.414, "covRad": 1.360, "vdwRad": 2.180 },
"in": {"atNumber": 49, "symbol": "In", "mass": 114.90387878, "weight": 114.818, "covRad": 1.420, "vdwRad": 1.930 },
"sn": {"atNumber": 50, "symbol": "Sn", "mass": 119.902202, "weight": 118.71, "covRad": 1.400, "vdwRad": 2.170 },
"sb": {"atNumber": 51, "symbol": "Sb", "mass": 120.90381, "weight": 121.76, "covRad": 1.400, "vdwRad": 2.060 },
"te": {"atNumber": 52, "symbol": "Te", "mass": 129.90622275, "weight": 127.6, "covRad": 1.360, "vdwRad": 2.060 },
"i": {"atNumber": 53, "symbol": "I", "mass": 126.90447, "weight": 126.90447, "covRad": 1.330, "vdwRad": 1.980 },
"xe": {"atNumber": 54, "symbol": "Xe", "mass": 131.90415509, "weight": 131.293, "covRad": 1.310, "vdwRad": 2.160 },
"cs": {"atNumber": 55, "symbol": "Cs", "mass": 132.90545196, "weight": 132.90545196, "covRad": 2.320, "vdwRad": 3.430 },
"ba": {"atNumber": 56, "symbol": "Ba", "mass": 137.905247, "weight": 137.327, "covRad": 1.960, "vdwRad": 2.680 },
"la": {"atNumber": 57, "symbol": "La", "mass": 138.90636, "weight": 138.90547, "covRad": 1.800, "vdwRad": 2.430 },
"ce": {"atNumber": 58, "symbol": "Ce", "mass": 139.90544, "weight": 140.116, "covRad": 1.630, "vdwRad": 2.420 },
"pr": {"atNumber": 59, "symbol": "Pr", "mass": 140.90766, "weight": 140.90766, "covRad": 1.760, "vdwRad": 2.400 },
"nd": {"atNumber": 60, "symbol": "Nd", "mass": 141.90773, "weight": 144.242, "covRad": 1.740, "vdwRad": 2.390 },
"sm": {"atNumber": 62, "symbol": "Sm", "mass": 151.91974, "weight": 150.36, "covRad": 1.720, "vdwRad": 2.360 },
"eu": {"atNumber": 63, "symbol": "Eu", "mass": 152.92124, "weight": 151.964, "covRad": 1.680, "vdwRad": 2.350 },
"gd": {"atNumber": 64, "symbol": "Gd", "mass": 157.92411, "weight": 157.25, "covRad": 1.690, "vdwRad": 2.340 },
"tb": {"atNumber": 65, "symbol": "Tb", "mass": 158.92535, "weight": 158.92535, "covRad": 1.680, "vdwRad": 2.330 },
"dy": {"atNumber": 66, "symbol": "Dy", "mass": 163.92918, "weight": 162.5, "covRad": 1.670, "vdwRad": 2.310 },
"ho": {"atNumber": 67, "symbol": "Ho", "mass": 164.93033, "weight": 164.93033, "covRad": 1.660, "vdwRad": 2.300 },
"er": {"atNumber": 68, "symbol": "Er", "mass": 165.9303, "weight": 167.259, "covRad": 1.650, "vdwRad": 2.290 },
"tm": {"atNumber": 69, "symbol": "Tm", "mass": 168.93422, "weight": 168.93422, "covRad": 1.640, "vdwRad": 2.270 },
"yb": {"atNumber": 70, "symbol": "Yb", "mass": 173.93887, "weight": 173.045, "covRad": 1.700, "vdwRad": 2.260 },
"lu": {"atNumber": 71, "symbol": "Lu", "mass": 174.94078, "weight": 174.9668, "covRad": 1.620, "vdwRad": 2.240 },
"hf": {"atNumber": 72, "symbol": "Hf", "mass": 179.94656, "weight": 178.49, "covRad": 1.520, "vdwRad": 2.230 },
"ta": {"atNumber": 73, "symbol": "Ta", "mass": 180.948, "weight": 180.94788, "covRad": 1.460, "vdwRad": 2.220 },
"w": {"atNumber": 74, "symbol": "W", "mass": 183.950931, "weight": 183.84, "covRad": 1.370, "vdwRad": 2.180 },
"re": {"atNumber": 75, "symbol": "Re", "mass": 186.95575, "weight": 186.207, "covRad": 1.310, "vdwRad": 2.160 },
"os": {"atNumber": 76, "symbol": "Os", "mass": 191.96148, "weight": 190.23, "covRad": 1.290, "vdwRad": 2.160 },
"ir": {"atNumber": 77, "symbol": "Ir", "mass": 192.96292, "weight": 192.217, "covRad": 1.220, "vdwRad": 2.130 },
"pt": {"atNumber": 78, "symbol": "Pt", "mass": 194.964792, "weight": 195.084, "covRad": 1.230, "vdwRad": 2.130 },
"au": {"atNumber": 79, "symbol": "Au", "mass": 196.966569, "weight": 196.966569, "covRad": 1.240, "vdwRad": 2.140 },
"hg": {"atNumber": 80, "symbol": "Hg", "mass": 201.970643, "weight": 200.592, "covRad": 1.330, "vdwRad": 2.230 },
"tl": {"atNumber": 81, "symbol": "Tl", "mass": 204.974428, "weight": 204.38, "covRad": 1.440, "vdwRad": 1.960 },
"pb": {"atNumber": 82, "symbol": "Pb", "mass": 207.976653, "weight": 207.2, "covRad": 1.440, "vdwRad": 2.020 },
"bi": {"atNumber": 83, "symbol": "Bi", "mass": 208.9804, "weight": 208.9804, "covRad": 1.510, "vdwRad": 2.070 },
"th": {"atNumber": 90, "symbol": "Th", "mass": 232.03806, "weight": 232.0377, "covRad": 1.750, "vdwRad": 2.450 },
"u": {"atNumber": 92, "symbol": "U", "mass": 238.05079, "weight": 238.02891, "covRad": 1.700, "vdwRad": 2.410 }
}




# this function loads a single structure from the XYZ trajectory file
def LoadSingleStructure(inpf):
    try: # if we can read the XYZ structure, everything is good
        FirstLine = inpf.readline()
        NumOfAtoms = int(FirstLine) # the first line should contain only the numbers of atoms
        inpf.readline() # skip empty line        
        # read NumOfAtoms of next lines, split the words in line, store the results as an NumPy array
        StructLines = np.array([inpf.readline().split() for n in range(0,NumOfAtoms)], dtype = str)
        #print(StructLines)
        AtSymbLower = np.array([a.lower() for a in StructLines[:,0]]) # these are elements notations, in lowercase
        mask = np.array([ (not a in ElementData.keys()) for a in AtSymbLower]) # mask all unknown elements
        AtSymbLower[mask] = "x"
        #print(AtSymbLower)
        AtSymbNorm  = np.array([ElementData[a]["symbol"] for a in AtSymbLower]) # these are normal elements notations
        #print(AtSymbNorm)   
        AtRadCov = np.array([ElementData[a]["covRad"] for a in AtSymbLower]) # 
        AtRadVdW = np.array([ElementData[a]["vdwRad"] for a in AtSymbLower]) # 
        #print(AtMass)        
        XYZ = np.array(StructLines[:,1:4], dtype=float) # these are XYZ coordinates
        #print(XYZ)              
          
        return AtRadCov, AtRadVdW, AtSymbNorm, XYZ
    except: # if not, we just don't return anything, and this will stop the processing later
        return None



# load the trajectory from the file with name "FileName" (str)
# return: List of atomic weights, List of atomic symbols, List of XYZ coordinates for each MD frame
def LoadTrajectory(FileName, Spacing=None):
    try:
        inpf = open(FileName, "r") # open the MD trajectory file
        AtRadCov = []
        AtRadVdW = []
        AtSymb = []
        MDTrj  = []
        
        Check = True
        
        count = 0 
        countActual = 0
        while Check:
            res =   LoadSingleStructure(inpf)
            ToLoad = True # flag to check the loading condition (when loading of not every structure requested)
            if (not Spacing is None) and (count % Spacing != 0):
                ToLoad = False
                
            if res is None:
                Check = False
            else:
                if ToLoad:
                    AtRadCov.append(res[0])
                    AtRadVdW.append(res[1])
                    AtSymb.append(res[2])
                    MDTrj.append(res[3])
                    countActual += 1
                    if countActual > 1:
                        if np.any(AtSymb[-1] != AtSymb[0]):
                            inpf.close()
                            raise RuntimeError("Different atomic symbols detected!")
            count += 1     
            print(" %10i structures loaded (%10i actual)  \r" % (count,countActual), end='', flush=True)       
        print("")
        print("Successfully loaded trajectory from file %s with %i frames " % (FileName, len(MDTrj)))                
        inpf.close() # close the MD trajectory file
        return AtRadCov, AtRadVdW, AtSymb, np.array(MDTrj) 
    except:
        raise RuntimeError("Failed to load trajectory from file "+str(FileName))


def FindAllWaters(CovRad, Symb, XYZ, DR, Scale):
    # first find the oxygens' and hydrogens' indexes
    OIMask = (Symb == "O")

    NAt  =  len(XYZ)
    AllIdxs = np.arange(0,NAt,1)
    OIdxs = AllIdxs[OIMask]
    def IsThereCovalentBondBetweenAtoms(i, j):
        Result = False
        if i != j:
            drij = XYZ[i] - XYZ[j]
            R = np.sqrt(np.dot(drij,drij))
            Rcrit = (CovRad[i] + CovRad[j]) * Scale + DR
            if R <= Rcrit:
                Result = True
        return Result       
        
    WaterIdx = [] # array of indexes of water atoms (iO, iH1, iH2)

    for oidx in OIdxs:
        mask = np.array([IsThereCovalentBondBetweenAtoms(oidx, j) for j in range(0,NAt)])
        #print(oidx,AllIdxs[mask],Symb[mask])
        hmask = (Symb[mask] == "H")
        #print(AllIdxs[mask][hmask])
        if len(Symb[mask]) >= 2 and len(hmask[hmask])==2:
            h1idx = AllIdxs[mask][hmask][0]
            h2idx = AllIdxs[mask][hmask][1]
            WaterIdx.append( (oidx,h1idx,h2idx) )
    
    return WaterIdx       

            
def CalcWaterPosAndDipole(XYZ, WaterIdx):
    Nw = len(WaterIdx)
    Rw = np.zeros((Nw,3), dtype=float) # position 
    Dw = np.zeros((Nw,3), dtype=float) # dipole moment
    for n,widx in enumerate(WaterIdx):
        rh = 0.5 * (XYZ[widx[1]] + XYZ[widx[2]])
        ro = XYZ[widx[0]]
        Rw[n] = (rh + ro )/2.
        Dw[n] = 2. * (rh - ro)
    return Rw, Dw
            

def CalcProb(Dist, sigma):
    X = Dist/sigma
    return np.exp(-0.5 * X**2)/X**2


def CalcCosVecVec(v1, v2):
    return np.dot(v1,v2)/np.sqrt(np.dot(v1,v1) * np.dot(v2,v2))


def ComputeWaterEnvironment(R0, Rw, Dw, sigma, UseRounding, Cutoff):
    DistM = sps.distance_matrix([R0], Rw, p=2)[0]
    Nw = len(Rw)    
    mask = np.ones(Nw, dtype=bool)
    mask *= (DistM > sigma * 0.1)
    if not Cutoff is None:
        mask *= (DistM <= Cutoff * sigma)
    DistM = DistM[mask]
    Rw = Rw[mask]
    Dw = Dw[mask]
    Nw = len(mask[mask])
    
    Pw = CalcProb(DistM, sigma)
    Norm = np.sum(Pw)
    
    Rloc = np.sum(DistM * Pw)/Norm
    CosArr = np.array([CalcCosVecVec(Dw[nw], Rw[nw]-R0)  for nw in range(0,Nw)])
    MeanCos = np.sum(CosArr * Pw)/Norm
    if UseRounding:
        Pw = np.round(Pw/np.max(Pw))
    else:
        Pw /= np.max(Pw)
    NumStoich = np.sum(Pw)
    
    return Rloc, MeanCos, NumStoich
    

def GetUniqueElements(AtSymbols):
    UniqSymbols = np.sort(list(set(AtSymbols)))
    AtomMasks = np.zeros( (len(UniqSymbols),len(AtSymbols)), dtype=bool)
    for i,ui in enumerate(UniqSymbols):
        AtomMasks[i] = (ui == AtSymbols)
    return UniqSymbols, AtomMasks


def GetCentralAtomIndex(AtSymb,XYZ):
    M = np.array([ElementData[a.lower()]["mass"]  for a in AtSymb])
    RCm = np.sum([mi * XYZ[i] for i,mi in enumerate(M)],axis=0)/np.sum(M)
    DistM = sps.distance_matrix([RCm], XYZ, p=2)[0]
    CentralIdx = np.argsort(DistM)[0]
    return CentralIdx


def AnalyzeThis(ActiveIdx, AtSymb, OtherAtomsSigma, UniqSymbols, AtomMasks, XYZ, sigma, UseRounding, Cutoff):
    DistM = sps.distance_matrix([XYZ[ActiveIdx]], XYZ, p=2)[0]
    Nat = len(XYZ)
    mask = np.ones(Nat, dtype=bool)
    mask *= (DistM > 1.e-5) # get rid of self-mapping
    if not Cutoff is None:
        mask *= (DistM <= Cutoff * sigma)
    if not OtherAtomsSigma is None:
        Sigma = 0.5 * (sigma + OtherAtomsSigma[mask])
    else:
        Sigma = sigma
    if len(mask[mask])>0:    
        DistM = DistM[mask]
        P = CalcProb(DistM, Sigma)
        Norm = np.sum(P)
        Rloc = np.sum(DistM * P)/Norm
        if UseRounding:
            P = np.round(P/np.max(P))
        else:
            P /= np.max(P)
        StoichVec = np.zeros(len(UniqSymbols), dtype=float)
        for i,amask in enumerate(AtomMasks):
            StoichVec[i] += np.sum(P[amask[mask]])
    else:
        if Cutoff is None:
            Rloc = 0.
        else:
            Rloc = Cutoff * sigma
        StoichVec = np.zeros(len(UniqSymbols), dtype=float)
    return Rloc, StoichVec
        

def RunAnalysisForAtomInTrj(AtSymb, sigma, OtherAtomsSigma, ActiveIdx, MDTrj, dt, UseRounding, Cutoff):
    Fname = "local_surrounding_anlysis_%s%i.res" % (AtSymb[idx], idx+1)
    UniqSymbols, AtomMasks = GetUniqueElements(AtSymb)
    with open(Fname, "w") as outf:
        outf.write("# %7s %9s" % ("time", "radius[A]"))
        for uS in UniqSymbols:
            outf.write(" %5s" % ("#"+uS))
        outf.write("\n")
        for n,XYZ in enumerate(MDTrj):
            Rloc, StoichVec = AnalyzeThis(ActiveIdx, AtSymb, OtherAtomsSigma, UniqSymbols, AtomMasks, XYZ, sigma, UseRounding, Cutoff)
            t = n * dt
            outf.write("  %7.2f %9.4f" % (t, Rloc))
            for s in StoichVec:
                outf.write(" %5.1f" % (s))
            outf.write("\n")
            outf.flush()

def RunWaterAnalysisForAtomInTrj(AtSymb, AtRad, sigma, ActiveIdx, MDTrj, dt, DR, Scale, UseRounding, Cutoff, ReComputeWater):
    Fname = "local_surrounding_water_anlysis_%s%i.res" % (AtSymb[idx], idx+1)
    with open(Fname, "w") as outf:
        outf.write("# %7s %9s" % ("time", "radius[A]"))
        outf.write(" %9s %9s" % ("cos(x)", "#water"))
        outf.write("\n")
        for n,XYZ in enumerate(MDTrj):
            if n == 0:
                WaterIdxs = FindAllWaters(AtRad, AtSymb, XYZ,  DR, Scale)
                #print(WaterIdxs)
            else:
                if ReComputeWater:
                    WaterIdxs = FindAllWaters(AtRad, AtSymb, XYZ,  DR, Scale)
                    #print(WaterIdxs)
            Rw,Dw = CalcWaterPosAndDipole(XYZ, WaterIdxs)
            
            Rl,Cos,Nw = ComputeWaterEnvironment(XYZ[ActiveIdx], Rw, Dw, sigma, UseRounding, Cutoff)
            t = n * dt
            outf.write("  %7.2f %9.4f" % (t, Rl))
            outf.write(" %9.3f %9.1f" % (Cos, Nw))
            outf.write("\n")
            outf.flush()

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script computes the local environment of a given atom in MD simulation.')
     
    parser.add_argument('-t', '--trj', type=str, default=None,
                    help="MD trajectory file in Angstroems. Default is None.")
                        
    parser.add_argument('-n', '--num', type=int, default=None,
                    help="Number (starting from 1) of an atom in the trajectory, for which the analysis is done.") 

    parser.add_argument('-e', '--element', type=str, default=None,
                    help="Element type, for which the analysis is done.") 

    parser.add_argument('--scale', type=float, default=2./3.,
                    help="Scale of the radius to get the standard deviation sigma. Default is 2/3.")  
         
    parser.add_argument('--UseCovRad', action='store_true', 
                    help="Flag to use the covalent radius instead of the Van der Waals radius to compute the local environment.")     

    parser.add_argument('--spacing', type=int, default=None,
                    help="Load not all the structures from trajectory, but only each i-th, where i is defined by this flag. Default is None (load all).") 

    parser.add_argument('--addDR', type=float, default=0.1,
                    help="Additional parameter (in Angstroms) for calculating the covalent radius. Default is 0.1.")  

    parser.add_argument('--dt', type=float, default=1.,
                    help="Time step of MD trajectory in fs. Default is 1 fs.")  

    parser.add_argument('--rx', type=float, default=None,
                    help="Width of the X-atom in angstroms. Default is None.")  

    parser.add_argument('--addScale', type=float, default=0.1,
                    help="Additional scaling parameter for calculating the covalent radius. Default is 0.1.")  

    parser.add_argument('--DoNotUseRounding', action='store_false', 
                    help="Flag not to use rounding while computing stoichiometry.")     

    parser.add_argument('--cutoff', type=float, default=None,
                    help="Cutoff (as a scale factor for sigma of distribution) for local environment calculation. Default is None (no cutoff).")           

    parser.add_argument('--RunWaterAnalysis', action='store_true', 
                    help="Flag not run also the water analysis.")     

    parser.add_argument('--RecalcWater', action='store_true', 
                    help="Flag to always re-find water during the water analysis.")     

    parser.add_argument('--TakePairAtomRadius', action='store_true', 
                    help="Flag to take distances into considerations according to the pairs of atoms' radii.")     
                     
    args,unknown = parser.parse_known_args() # parse the arguments

    # check if the input is given
    if args.trj is None:
        print("No input MD trajectory was given, nothing to do :'(")
        exit()
    
    args.addScale += 1.
    
    AtRadCov, AtRadVdW, AtSymb, MDTrj = LoadTrajectory(FileName=args.trj, Spacing=args.spacing)
    AtSymb = AtSymb[0]

    if args.UseCovRad:
        print("We will use covalent radii for computing everything")
        AtRad = AtRadCov[0]
    else:
        print("We will use Van der Waals radii for computing everything")
        AtRad = AtRadVdW[0]

    if "X" in AtSymb:
        if not args.rx is None:
            mask = (AtSymb == "X")
            AtRad[mask] = args.rx
            print("Changed radius of the X-atom to "+str(args.rx))
            
    if args.num is None:
        if args.element is None:
            print("As the atom was not specified, we take the closest one to the center of mass for the first trajectory frame.")
            idx = GetCentralAtomIndex(AtSymb,MDTrj[0])
            print("Atom we get is : %s%i " % (AtSymb[idx], idx+1))
            args.num = int(idx)
        else:
            print("The element %s was chosen for the analysis" % (args.element) )
            mask = np.array([ (a.lower() == args.element.lower()) for a in AtSymb])
            #print(AtSymb,args.element)
            print(" %i out of %i atoms fit this description" % (len(mask[mask]), len(mask)))
            args.num = np.arange(0,len(mask),1)[mask]
            print("These are the indexes of the chosen atoms: " + str(args.num + 1))
             
    else:
        print("Doing analysis for atom #"+str(args.num))
        args.num -= 1    
    if args.DoNotUseRounding:
        print("We will use rounding for computing stoichiometry")
    else:
        print("We will NOT use rounding for computing stoichiometry")

    if args.TakePairAtomRadius:
        OtherAtomsSigma = AtRad * args.scale
    else:
        OtherAtomsSigma = None


    if type(args.num) is int:
        sigma = AtRad[args.num] * args.scale
        RunAnalysisForAtomInTrj(AtSymb, sigma, OtherAtomsSigma, args.num, MDTrj, args.dt, args.DoNotUseRounding, args.cutoff)
        if args.RunWaterAnalysis:
            RunWaterAnalysisForAtomInTrj(AtSymb, AtRadCov[0], sigma, args.num, MDTrj, args.dt, args.addDR, args.addScale, args.DoNotUseRounding, args.cutoff, args.RecalcWater)
    else:
        for idx in args.num:
            sigma = AtRad[idx] * args.scale
            RunAnalysisForAtomInTrj(AtSymb, sigma, OtherAtomsSigma, idx, MDTrj, args.dt, args.DoNotUseRounding, args.cutoff)
            if args.RunWaterAnalysis:
                RunWaterAnalysisForAtomInTrj(AtSymb, AtRadCov[0], sigma, idx, MDTrj, args.dt, args.addDR, args.addScale, args.DoNotUseRounding, args.cutoff, args.RecalcWater)            
    
    
    
    
    
    






