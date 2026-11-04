#! /bin/python3

import numpy as np
import argparse
import scipy.optimize  as spo
import scipy.spatial as sps
import scipy.stats as spst
from collections import Counter, defaultdict
from typing import Any

ArrayList = list[np.ndarray]
AtomKey = tuple[str, Any]

AlphaValue=0.99
CriterionFreq=83.4
NLFreq=0.035
DFRemoval=6

# "mass" is the mass of the most stable isotop (in a.m.u.)
# "weight" is the atomic weight (in a.m.u.)
# "mass" is the mass
# "charge" is the charge
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
    """
    Read the molecular structure from an opened XYZ trajectory file.
    We're expecting the standard XYZ format, which consists of a number of atoms on the first line,
    a comment line on the second line, and then one line per atom with the atomic symbol and the x, y, z coordinates in Angstroms.
    
    Like
2
        
  H   1.0   0.0   0.0
  H   0.0   0.0   0.0
3
  
  H   0.0   0.0   0.0
  O   1.0   0.0   0.0
  As  0.0  -2.0   0.0

    Parameters
    ----------
    inpf : file object
        Opened text file object positioned at the beginning of an XYZ frame.
        The function reads from the current file position.

    Returns tuple or None
        If the frame is successfully read, returns a tuple:

            AtMass : numpy.ndarray
                Atomic masses for all atoms in the frame, in atomic mass units.
                Masses are taken from ElementData [element]["mass"].

            AtCharge : numpy.ndarray
                Atomic numbers for all atoms in the frame.
                Despite the historical name "AtCharge", this is not a formal
                molecular charge and not a partial charge. It is the nuclear
                charge / atomic number Z, taken from ElementData[element]["atNumber"].

            AtSymbNorm : numpy.ndarray
                Normalized atomic symbols, for example "h" -> "H",
                "cl" -> "Cl". Unknown elements are replaced by the fallback
                symbol "X".

            XYZ : numpy.ndarray, shape (N, 3)
                Cartesian coordinates of atoms in Angstrom.

        Returns None if the frame cannot be read. In the current implementation
        this is also used to stop reading at the end of the trajectory file.

    Notes
    -----
    Unknown element symbols are mapped to "X". The corresponding mass and
    atomic number are then taken from ElementData["x"].

    The current implementation catches all exceptions and returns None. This is
    convenient for detecting EOF, but it may also hide malformed XYZ frames.
    For stricter parsing, replace the broad try/except with explicit error
    handling.
    """

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
        AtMass = np.array([ElementData[a]["mass"] for a in AtSymbLower]) # 
        AtCharge = np.array([ElementData[a]["atNumber"] for a in AtSymbLower], dtype=int) # 
        #print(AtMass)        
        XYZ = np.array(StructLines[:,1:4], dtype=float) # these are XYZ coordinates
        #print(XYZ)              
          
        return AtMass, AtCharge, AtSymbNorm, XYZ
    except: # if not, we just don't return anything, and this will stop the processing later
        return None

# load the trajectory from the file with name "FileName" (str)
# return: List of atomic weights, List of atomic symbols, List of XYZ coordinates for each MD frame
def LoadTrajectory(FileName, Spacing=None):
    """
    Load a molecular dynamics trajectory from an XYZ file.
    Each loaded structure, it stores atomic masses,
    atomic numbers, normalized atomic symbols, and Cartesian coordinates.

    Parameters
    ----------
    FileName : str
        Path to the XYZ trajectory file.

    Spacing : int or None, optional
        Controls frame subsampling.

        If Spacing is None, all frames are loaded.
        If Spacing is an integer, only every Spacing-th frame is loaded.
        For example:
            Spacing = 1  -> load every frame
            Spacing = 2  -> load frames 0, 2, 4, ...
            Spacing = 10 -> load frames 0, 10, 20, ...

    Returns
    -------
    AtMass : list of numpy.ndarray
        List of atomic mass arrays for all loaded frames.
        Each array has shape (N,), where N is the number of atoms.

    AtCharge : list of numpy.ndarray
        List of atomic number arrays for all loaded frames.
        Despite the name "AtCharge", these values are atomic numbers Z,
        not formal or partial atomic charges.

    AtSymb : list of numpy.ndarray
        List of normalized atomic symbol arrays for all loaded frames.
        For example: "h" is converted to "H", "cl" to "Cl".

    MDTrj : list of numpy.ndarray
        List of Cartesian coordinate arrays for all loaded frames.
        Each array has shape (N, 3). Coordinates are assumed to be in Angstrom.

    Raises
    ------
    RuntimeError
        Raised if the trajectory cannot be loaded from the given file.

    Notes
    -----
    The function stops reading when LoadSingleStructure() returns None.
    In the current implementation, this usually corresponds to the end of file.

    The function does not currently validate that all frames have the same
    number of atoms or the same atom ordering. Such validation can be added
    later in a separate trajectory-checking step.

    """

    try:
        inpf = open(FileName, "r") # open the MD trajectory file
        AtMass = []
        AtCharge = []
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
                    AtMass.append(res[0])
                    AtCharge.append(res[1])
                    AtSymb.append(res[2])
                    MDTrj.append(res[3])
                    countActual += 1
            count += 1     
        inpf.close() # close the MD trajectory file
        return AtMass, AtCharge, AtSymb, MDTrj 
    except:
        raise RuntimeError("Failed to load trajectory from file "+str(FileName))


def ConvertAtomicSymbolsIntoCounts(AtSymbInstance):
    """
    Convert the atomic symbols for a given trajectory into the dictionary with counts of different elements.
    
    Parameters
    ----------
    AtSymbInstance : NumPy array of (N,) elements, where N is the number of atoms, an element of AtSymb list.     

    Returns
    -------
    counts : dict    Dictionary where keys are element symbols and values are atom counts.
    """
    keys = set(AtSymbInstance)
    counts = dict()
    for key in keys:
        mask = (AtSymbInstance == key)
        N = len(AtSymbInstance[mask])
        counts.update({key: N})
    return counts


def MakeHillFormula(counts):
    """
    Convert element counts into a molecular formula in Hill notation.

    Parameters
    ----------
    counts : dict
        Dictionary where keys are element symbols and values are atom counts.

        Example:
        {
            "C": 2,
            "H": 4,
            "Cl": 2
        }

    Returns
    -------
    str
        Molecular formula written in Hill order.

        Example:
        {"C": 2, "H": 4, "Cl": 2} -> "C2H4Cl2"

    Notes
    -----
    Hill system rules:
    1. If carbon is present:
       write C first, H second, then all other elements alphabetically.
    2. If carbon is absent:
       write all elements alphabetically.

    The number 1 is omitted:
    {"H": 2, "O": 1} -> "H2O", not "H2O1".
    """

    def piece(el):
        """
        Format one element for the molecular formula.

        Parameters
        ----------
        el : str
            Chemical element symbol.
            Example: "C", "H", "Cl".

        Returns
        -------
        str
            Formula fragment for this element.

            Examples:
            if counts["C"] == 2, returns "C2"
            if counts["H"] == 4, returns "H4"
            if counts["O"] == 1, returns "O"
        """

        # Get the number of atoms of this element.
        n = counts[el]

        # If there is only one atom, write only the element symbol.
        if n == 1:
            return el

        # If there is more than one atom, write symbol + number.
        return el + str(n)

    # If carbon is present, use the special Hill order:
    # C first, H second, then all other elements alphabetically.
    if "C" in counts:
        order = ["C"]

        # Hydrogen is placed immediately after carbon, if present.
        if "H" in counts:
            order.append("H")

        # Collect all elements except C and H.
        rest = []
        for el in counts:
            if el != "C" and el != "H":
                rest.append(el)

        # Add the remaining elements in alphabetical order.
        order += sorted(rest)

    # If carbon is absent, all elements are sorted alphabetically.
    else:
        order = sorted(counts)

    # Build the final formula as a string.
    formula = ""

    # Add formula fragments such as C2, H4, Cl2.
    for el in order:
        formula += piece(el)

    return formula


# this function saves the trajectory in the XYZ file
# requires: file name, where to write (FileName, str)
#           TRJ -- trajectory (list of NumPy arrays with the coordinates)
#           AtSymbolSet -- atomic symbols corresponding to A SINGLE STRUCTURE
def SaveTrajectory(FileName, TRJ, AtSymbolSet):
    outf = open(FileName, "w")
    for n,xyz in enumerate(TRJ):
        outf.write("%i\n\n" % (len(xyz)))
        for i,r in enumerate(xyz):
            outf.write("%3s  %7.4f %7.4f %7.4f\n" % (AtSymbolSet[i], r[0], r[1], r[2]))
    
    outf.close()


def CheckHillFormulaCorrespondanceInTrajectory(AtSymb, MDTrj):
    """
    Checks whether the gross formulas (Hill notation) are the same for all frames of the trajectory.
    If all formulas are the same, returns True.
    Otherwise, groups frames by formula and saves each group into a separate XYZ file
    named "frames_<Hill's formula>.xyz", then returns False.

    Parameters
    ----------
    AtSymb : list of numpy.ndarray
        List of atomic symbol arrays for each frame.
    MDTrj : list of numpy.ndarray
        List of coordinate arrays for each frame.

    Returns
    -------
    bool
    True if all formulas are the same, False otherwise.
    """
    formulas = []
    for sym in AtSymb:
        counts = ConvertAtomicSymbolsIntoCounts(sym)
        formulas.append(MakeHillFormula(counts))

    first = formulas[0]
    all_same = all(f == first for f in formulas)
    if all_same:
        return True

    groups = {}
    for i, f in enumerate(formulas):
        groups.setdefault(f, []).append(i)

    for f, indices in groups.items():
        traj = [MDTrj[i] for i in indices]
        sym = AtSymb[indices[0]]
        filename = f"frames_{f}.xyz"
        SaveTrajectory(filename, traj, sym)

    return False


"""
def SortAtomsInTrajectoryByFirstFrame(AtMass, AtCharge, AtSymb, MDTrj):
"""

def CenterFramesToCenterOfMass(AtMass, MDTrj):
    """
    Center every frame of a trajectory so that its center of mass lies at
    the origin of the coordinate system.

    For each frame, the center of mass is computed as the mass-weighted
    average of the Cartesian coordinates of all atoms in that frame:

        COM = sum_i(m_i * r_i) / sum_i(m_i)

    and then subtracted from the coordinates of every atom in the frame,
    so that the new center of mass of that frame is exactly (0, 0, 0)
    (up to floating point precision).

    Parameters
    ----------
    AtMass : list of numpy.ndarray
        List of atomic mass arrays, one per frame (as returned by
        LoadTrajectory()). Each array has shape (N,), where N is the
        number of atoms in that frame.

    MDTrj : list of numpy.ndarray
        List of Cartesian coordinate arrays, one per frame (as returned
        by LoadTrajectory()). Each array has shape (N, 3).

    Returns
    -------
    AtMass : list of numpy.ndarray
        The same atomic masses that were passed in. Masses are not
        changed by centering; they are returned unchanged so that the
        result of this function can be used as a drop-in replacement
        for the original (AtMass, MDTrj) pair.

    CenteredMDTrj : list of numpy.ndarray
        List of Cartesian coordinate arrays, one per frame, translated
        so that the center of mass of every frame is at the origin.
        Each array has the same shape (N, 3) as the corresponding
        input array.

    Raises
    ------
    ValueError
        Raised if the total mass of a frame is zero, since the center
        of mass is then undefined (division by zero).

    Notes
    -----
    AtMass and MDTrj are expected to have the same length (one entry
    per frame), and for every frame the mass array and the coordinate
    array are expected to describe the same number of atoms N.

    The input MDTrj is not modified in place; a new, centered array is
    created for every frame.
    """

    CenteredMDTrj = []

    # go through the trajectory frame by frame
    for Mass, XYZ in zip(AtMass, MDTrj):

        Mass = np.asarray(Mass, dtype=float)
        XYZ = np.asarray(XYZ, dtype=float)

        TotalMass = np.sum(Mass)

        if TotalMass == 0:
            raise ValueError("Total mass is zero. Cannot compute center of mass.")

        # mass-weighted average of the coordinates = center of mass
        CenterOfMass = np.sum(Mass[:, np.newaxis] * XYZ, axis=0) / TotalMass

        # shift the frame so that its center of mass is at the origin
        CenteredMDTrj.append(XYZ - CenterOfMass)

    return AtMass, CenteredMDTrj

def GetAtomicSymbolsMap(Symb):
    """
    Finds a map for all atomic symbols.
    
    Parameters
    ----------
    Symb : List of atomic symbols for the structures     
    
    Returns
    -------
    UniqSymbols : list of unique atomic symbols in the molecule.
    MapOfElements : list of boolean NumPy arrays, which are the maps for given elements.
    """    
    UniqSymbols = np.sort(list(set(Symb)))
    npSymb = np.array(Symb)
    MapOfElements = [ (npSymb == symb) for symb in UniqSymbols]

    return UniqSymbols, MapOfElements


### Calculate the rotation matrix along a chosen axis (x,y,z)=(0,1,2)
### Requires: theta (float), the rotation angle in degrees
###           axis = 0 (x),1 (y), 2(z) -- axis arond which rotation is performed
### Returns: R[3][3], np.array containing the 3 x 3 rotation matrix
def RotMatrix(theta,axis=0):
    s=np.sin(np.pi*theta/180.)
    c=np.cos(np.pi*theta/180.)
    if axis==0:
        return np.array([[1.0 , 0.0 , 0.0],
                         [0.0 ,  c  , -s ],
                         [0.0 ,  s  ,  c ]])
    elif axis==1:
        return np.array([[ c  , 0.0 ,  s ],
                         [0.0 , 1.0 , 0.0],
                         [-s  , 0.0 ,  c ]])
    elif axis==2:
        return np.array([[ c  , -s  , 0.0],
                         [ s  ,  c  , 0.0],
                         [0.0 , 0.0 , 1.0]])
    else:
        print("Bad axis chosen")
        exit(1)


def ComputeRotationMatrix(EulerAngles):
    """
    Computes rotation matrix from Euler Angles.
    
    Parameters
    ----------
    EulerAngles : List three Euler angles in degrees     
    
    Returns
    -------
    RM :  3x3 NumPy array with 3D rotation matrix
    """    

    RM = np.eye(3)
    RM = np.matmul(RotMatrix(EulerAngles[0],axis=2),RM)
    RM = np.matmul(RotMatrix(EulerAngles[1],axis=0),RM)
    RM = np.matmul(RotMatrix(EulerAngles[2],axis=2),RM)

    return RM
    

def RotateMolecule(XYZ, EulerAngles):
    """
    Rotates the Cartesian coordinates with given Euler Angles.
    
    Parameters
    ----------
    XYZ : Cartesian coordinates 
    EulerAngles : List three Euler angles in degrees     
    
    Returns
    -------
    NewXYZ : Rotated Cartesian coordinates 
    """        
    RotMat = ComputeRotationMatrix(EulerAngles)
    NewXYZ = np.array([ np.dot(RotMat,r) for r in XYZ])
    return NewXYZ


def ComputeSigma(mass, nu):
    '''mass: a.m.u.
    nu: Hz'''

    if nu < 0:
        raise ValueError("Frequency must be non-negative")

    hbar = 1.05457182e-34 #Joule*s
    m = mass*1.66053906660e-27 #kilograms

    sigma = np.sqrt(hbar / (4. * np.pi * m * nu)) #meters
    sigma = sigma * 1e10 #angstroms

    return sigma    

    
def AlignTwoStructuresWithNonLinearRMSD(Symb, Mass, XYZ1, XYZ2, FreqWN, NumSamples=None, Simple=False):
    """
    Finds a map from one structure to another.
    
    Parameters
    ----------
    Symb : List of atomic symbols for the structures     
    Mass : List of atomic masses for the structures (in atomic mass units)
    XYZ1 : Cartesian coordinates for the first structure (in Angstroms)
    XYZ2 : Cartesian coordinates for the second structure (in Angstroms)
    FreqWN : Frequency in wavenumbers (in cm^{-1})
    NumSamples : (int or None), number of initial 3D rotation samples for the initial guess of the local optimizaiton routine. If None, dual annealing is used.
    Simple : (bool), flag to perform simple (fast and not robust) assignment of corresponding atoms in two structures.
    
    Returns
    -------
    NewXYZ2 : Properly oriented Cartesian coordinates for the second structure (in Angstroms)
    """
    NumBruteScan = 30
    
    UniqSymbols, MapOfElements = GetAtomicSymbolsMap(Symb)
    FreqHz = FreqWN * 2.997913E10 # convert cm^{-1} to Hz
    Sigmas = ComputeSigma(Mass, FreqHz)
    
    AllIdxs = np.arange(0,len(Symb),1)
    
    def NLMSD(EulerAngles):
        D = 0.
        Norm = 0.
        R = RotateMolecule(XYZ2, EulerAngles)
        
        for n,US in enumerate(UniqSymbols):
            idxs = AllIdxs[MapOfElements[n]]
            for i in idxs:
                for j in idxs:
                    dr2 = np.dot(XYZ1[i] - R[j], XYZ1[i] - R[j])
                    p = np.exp(-0.5 * dr2/Sigmas[i]**2)
                    D += p * dr2
                    Norm += p
        
        #return D
        if Norm > 1.e-10:
            return D/Norm
        else:
            return 1.e100
                    

    Bounds = ( (-180.0,180.),(0.0,180.),(-180.0,180.) )
    if NumSamples is None:
        #res = spo.dual_annealing(func = NLMSD, bounds = Bounds) 
        res = spo.differential_evolution(func = NLMSD, bounds = Bounds) 
        x0=res.x
    else:
        #print("generating rotations")
        Rotations = sps.transform.Rotation.random(NumSamples).as_euler('zxz', degrees=True)
        #print(Rotations)
        x = np.array([NLMSD(r) for r in Rotations])
        #print(x)
        idx = np.argmin(x)
        #print(idx)
        x0 = Rotations[idx]
        #print(x0)
        
    res = spo.minimize(NLMSD,x0=x0, bounds = Bounds, jac='3-point', method='trust-constr')   
    #print(res.fun)
    NewXYZ2 = RotateMolecule(XYZ2, res.x)              
    
    
    for n,US in enumerate(UniqSymbols):
        idxs = AllIdxs[MapOfElements[n]]
        newidxs = []
        
        if Simple:
            for i in idxs:
                pcurr = -1.e-5
                jcurr = -1
                for j in idxs:
                    if not j in newidxs:
                        dr2 = np.dot(XYZ1[i] - NewXYZ2[j], XYZ1[i] - NewXYZ2[j])
                        p = np.exp(-0.5 * dr2/Sigmas[i]**2)
                        if p > pcurr:
                            jcurr = j
                            pcurr = p
                newidxs.append(jcurr)  
        else:
            try:
                DMat = sps.distance_matrix(XYZ1[idxs],NewXYZ2[idxs], p=2.0)
            except:
                DMat = sps.cdist(XYZ1[idxs],NewXYZ2[idxs], metric='euclidean')
                
            row_ind, col_ind = spo.linear_sum_assignment(cost_matrix=DMat**2, maximize=False)
            newidxs = np.array(idxs)[col_ind]
        
        NewXYZ2[idxs] = NewXYZ2[newidxs] 
    return NewXYZ2


def BuildPermutation(AID1, AID2):
    """
    A function to build permutation changing atomic order of the second frame with respect to the first one.

    Parameters
    ----------
    AID1, AID2 : numpy.ndarray containing atomic identifyers (charges or labels) of the first and the second molecule

    Returns
    -------
    permutation : numpy array of permutated indexes.
    
    """
    if len(AID1) != len(AID2):
        raise RuntimeError("Trying to build permutation for different numbers of atoms!")
    
    if not np.equal(np.sort(AID1),np.sort(AID2)).any():
        raise RuntimeError("Trying to build permutation for two sets of different atomic identifyers!")
    
    N = len(AID1)
    new_idxs = np.arange(0, N, dtype=int)
    
    for i in range(0,N):
        if AID1[i] != AID2[new_idxs[i]]:
            Searching = True
            j = i + 1
            while Searching and j < N: 
                if AID1[i] == AID2[new_idxs[j]]:
                    Searching = False
                    store = int(new_idxs[i])
                    new_idxs[i] = int(new_idxs[j])
                    new_idxs[j] = store
                j += 1
                
            if Searching is True:  
                raise RuntimeError("Failed to produce permutation!")

    return new_idxs              
                
                
def SortAtomsInTrajectoryByFirstFrame(
    AtMass,
    AtCharge,
    AtSymb,
    MDTrj,
    ReturnDetails = False
    ):
    """
    Reorder every frame of a trajectory to match the atom-type order of frame 0.

    Parameters
    ----------
    AtMass, AtCharge, AtSymb, MDTrj : list of numpy.ndarray
        The four lists returned by cocos.py/LoadTrajectory().  Each list must
        contain one array per frame.  For frame k:

            AtMass[k]   has shape (N,)
            AtCharge[k] has shape (N,)
            AtSymb[k]   has shape (N,)
            MDTrj[k]    has shape (N, 3)

    ReturnDetails : bool, default False
        If False, return the cleaned trajectory tuple:

            NewAtMass, NewAtCharge, NewAtSymb, NewMDTrj

        If True, return a dictionary with the same arrays plus permutations
        and diagnostics.

    Returns
    -------
    tuple or dict
        If ReturnDetails=False:
            (NewAtMass, NewAtCharge, NewAtSymb, NewMDTrj)

        If ReturnDetails=True:
            {
                "AtMass": ...,
                "AtCharge": ...,
                "AtSymb": ...,
                "MDTrj": ...,
                "permutations": ...,
                "changed_frames": ...,
                "reference_atom_keys": ...,
            }

        Permutation convention:
            permutations[k][i] = old row index in frame k that was moved to row i.

    Notes
    -----
    This function only matches atom TYPES, in the order they first appear in
    frame 0.  Inside one type it preserves the original per-frame order, i.e.
    it does not try to figure out which specific atom is which among several
    equivalent ones (e.g. which H is which).  It deliberately does not rotate
    or translate structures either -- only rows are permuted.
    """
    masses = list(AtMass)
    charges = list(AtCharge) 
    symbols = list(AtSymb)
    xyzs = list(MDTrj)
    
    n_frames = len(masses)
    if n_frames == 0:
        raise ValueError(
            "at least one frame"
        )        
    
    if not (len(charges) == n_frames and len(symbols) == n_frames and len(xyzs) == n_frames):
        raise ValueError(
            "AtMass, AtCharge, AtSymb, and MDTrj must contain the same number of frames."
        )

    n_atoms = len(masses[0])
    if n_atoms == 0:
        raise ValueError("Frame 0 contains no atoms.")

    for iframe in range(n_frames):
        if not (
            len(masses[iframe]) == n_atoms
            and len(charges[iframe]) == n_atoms
            and len(symbols[iframe]) == n_atoms
            and xyzs[iframe].shape == (n_atoms, 3)
        ):
            raise ValueError(
                f"Frame {iframe}: all frames must have the same number of atoms as frame 0."
            )

    new_masses = []
    new_charges = []
    new_symbols = []
    new_xyzs = []
    permutations = []
    changed_frames = []

    identity = np.arange(n_atoms, dtype=int)

    for iframe in range(n_frames):
        if iframe == 0:
            permutation = identity
        else:
            permutation =  BuildPermutation(charges[0], charges[iframe])

        new_masses.append(masses[iframe][permutation])
        new_charges.append(charges[iframe][permutation])
        new_symbols.append(symbols[iframe][permutation])
        new_xyzs.append(xyzs[iframe][permutation])
        permutations.append(permutation)
        changed_frames.append(not np.array_equal(permutation, identity))

    if ReturnDetails:
        return {
            "AtMass": new_masses,
            "AtCharge": new_charges,
            "AtSymb": new_symbols,
            "MDTrj": new_xyzs,
            "permutations": permutations,
            "changed_frames": changed_frames
        }


    return new_masses, new_charges, new_symbols, new_xyzs


def GetMassWeightedRMSD(RefXYZ, xyz, m, ReturnAligned=False, ReturnRotation=False):
    """
    Orient one molecular structure with respect to a reference structure
    using the mass-weighted Kabsch algorithm and calculate mass-weighted RMSD.

    The function aligns xyz to RefXYZ by minimizing the mass-weighted squared
    deviation between corresponding atoms:

        sum_i m_i * |RefXYZ_i - xyz_i|^2

    Before alignment, both structures are shifted to their centers of mass.
    The rotation is found using scipy.spatial.transform.Rotation.align_vectors()
    with atomic masses used as weights.

    Parameters
    ----------
    RefXYZ : array-like, shape (N, 3)
        Cartesian coordinates of the reference structure.

    xyz : array-like, shape (N, 3)
        Cartesian coordinates of the structure that will be rotated and aligned
        to RefXYZ.

    m : array-like, shape (N,)
        Atomic masses used as weights in the alignment and RMSD calculation.

    ReturnAligned : bool, optional
        If False, return only the mass-weighted RMSD.
        If True, also return the aligned coordinates of xyz.

    ReturnRotation : bool, optional
        If True, also return the rotation matrix used to align xyz to RefXYZ.

    Returns
    -------
    float
        Mass-weighted RMSD after optimal rotational alignment.

    or tuple
        Depending on ReturnAligned and ReturnRotation, returns:

            mw_rmsd
            mw_rmsd, aligned_xyz
            mw_rmsd, rotation_matrix
            mw_rmsd, aligned_xyz, rotation_matrix

    Notes
    -----
    The returned aligned coordinates are centered at the reference center of
    mass. This means that they are in the same translational frame as RefXYZ.

    The RMSD is calculated as:

        sqrt( sum_i m_i * |RefXYZ_i - aligned_xyz_i|^2 / sum_i m_i )

    This is the normalized mass-weighted RMSD. If the project later needs the
    unnormalized mwMSD from the theoretical criterion, use the squared numerator
    before division by the total mass.
    """

    RefXYZ = np.array(RefXYZ, dtype=float)
    xyz = np.array(xyz, dtype=float)
    m = np.array(m, dtype=float)

    if RefXYZ.ndim != 2 or RefXYZ.shape[1] != 3:
        raise ValueError("RefXYZ must have shape (N, 3)")

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")

    if RefXYZ.shape != xyz.shape:
        raise ValueError(
            "RefXYZ and xyz must have the same shape. "
            f"Got {RefXYZ.shape} and {xyz.shape}"
        )

    if m.ndim != 1:
        raise ValueError("m must be a one-dimensional array with shape (N,)")

    if len(m) != RefXYZ.shape[0]:
        raise ValueError(
            "Length of m must be equal to the number of atoms. "
            f"Got len(m)={len(m)} and N={RefXYZ.shape[0]}"
        )

    if np.any(m < 0):
        raise ValueError("Atomic masses must be non-negative")

    #total_mass = np.sum(m)

    #if total_mass <= 0:
    #    raise ValueError("Sum of atomic masses must be positive")

    #RefCM = np.sum(RefXYZ * m[:, None], axis=0) / total_mass
    #xyzCM = np.sum(xyz * m[:, None], axis=0) / total_mass

    #RefXYZCentered = RefXYZ - RefCM
    #xyzCentered = xyz - xyzCM

    Res = sps.transform.Rotation.align_vectors(RefXYZ, xyz, weights=m)
    Rot = Res[0]

    RotMat = Rot.as_matrix()
    aligned_xyz = Rot.apply(xyz)

    diff = aligned_xyz - RefXYZ

    mw_rmsd = np.sqrt(
        np.sum(m * np.sum(diff * diff, axis=1))     )

    if ReturnAligned and ReturnRotation:
        return mw_rmsd, aligned_xyz, RotMat

    if ReturnAligned:
        return mw_rmsd, aligned_xyz

    if ReturnRotation:
        return mw_rmsd, RotMat

    return mw_rmsd


def CalculateZValue(AtMass, XYZ1, XYZ2, CritFreq):
    """
    Calculates the Z-test value Z for two structures XYZ1 and XYZ2

    Parameters
    ----------
    AtMass, XYZ1, XYZ2, CritFreq 

            AtMass[k]  has shape (N,)
            XYZ1[k]    has shape (N, 3)
            XYZ2[k]    has shape (N, 3)
            CritFreq   is a float value

    Returns
    -------
    zValue  -- resulting value of the z-criterion
    """    
    masses = np.array([[m,m,m] for m in AtMass]).flatten()
    # Sigmas are given in angstroms
    Sigma = 1.e10 * np.sqrt(1.054571817e-34 / (4. * np.pi * (CritFreq  * 2.997913E10) * (masses * 1.66053906892e-27)) )
    DX = np.array(np.array(XYZ1) - np.array(XYZ2)).flatten()
    Xi = DX/Sigma
    zValue = ( np.sum(Xi) / np.sqrt(len(Sigma)) )
    return zValue
    

def PerformAnalysis(AtMass, AtCharge, AtSymb, MDTrj, NLRMSDFreq, CritFreq,
                    AccountMirrorImages=False, NumSamples = None, Simple=False, Debug=True,UseZTest=False):
                    
    AtMass, MDTrj = CenterFramesToCenterOfMass(AtMass, MDTrj)
    AtMass, AtCharge, AtSymb, MDTrj = SortAtomsInTrajectoryByFirstFrame(AtMass, AtCharge, AtSymb, MDTrj)
    
    NIso = len(MDTrj)
    counts = ConvertAtomicSymbolsIntoCounts(AtSymb[0])
    Hill = MakeHillFormula(counts)
    
    mwMSDMat = np.zeros( (NIso,NIso), dtype = float)
    
    NewTrj = []
    NewTrj.append(MDTrj[0])
    alpha = AlphaValue
    DF = max(3*len(AtMass[0]) - DFRemoval, 1)
    
    Quartile = spst.chi2.ppf(q = alpha, df = DF)
    if Debug and not UseZTest:
        print(" chi2 quartile for level alpha="+str(alpha)+" and "+str(DF)+" degrees of freedom is "+str(Quartile))
    for i in range(0,NIso-1):
        if Debug:
            print("Structure "+str(i))
        for j in range(i+1, NIso):
            if Debug:
                print("  Comparing to structure "+str(j))
            XYZ2 = AlignTwoStructuresWithNonLinearRMSD(AtSymb[0], AtMass[0], MDTrj[i], MDTrj[j], NLRMSDFreq, NumSamples, Simple)
            mwRMSD,alignedXYZ2 = GetMassWeightedRMSD(MDTrj[i], XYZ2, AtMass[0],ReturnAligned=True)
            if AccountMirrorImages:
                XYZ2mirror = AlignTwoStructuresWithNonLinearRMSD(AtSymb[0], AtMass[0], MDTrj[i], -MDTrj[j], NLRMSDFreq, NumSamples, Simple)
                mwRMSDmirror,alignedXYZ2mirror = GetMassWeightedRMSD(MDTrj[i], XYZ2mirror, AtMass[0],ReturnAligned=True)
                if mwRMSDmirror < mwRMSD:
                    XYZ2 = XYZ2mirror
                    mwRMSD = mwRMSDmirror
                    alignedXYZ2 = alignedXYZ2mirror
                    if Debug:
                        print(" For pair of conformers %03i - %03i we get a best match from the mirror image" % (i,j))
            if i == 0:
                NewTrj.append(XYZ2)
                
            if UseZTest:
                Crit = CalculateZValue(AtMass[0], MDTrj[i], alignedXYZ2, CritFreq)
            else:
                #Crit = (2./3.) * mwRMSD**2 * (1.e-20) * (1.66053906892e-27) * (CritFreq  * 2.997913E10 ) / (len(AtMass[0]) * 1.054571817e-34)
                Crit = 4. * np.pi * (CritFreq  * 2.997913E10 ) * (mwRMSD**2 * (1.e-20) * 1.66053906892e-27) / (1.054571817e-34 * Quartile )
            mwMSDMat[i][j]=Crit
            mwMSDMat[j][i]=Crit
    
    SaveTrajectory(Hill+"_frame0_oriented.trj", NewTrj, AtSymb[0])
    np.savetxt(Hill+"_criterion.matrix", mwMSDMat)


def PerformQuckie(AtMass, AtCharge, AtSymb, MDTrj, NLRMSDFreq, CritFreq, 
                  AccountMirrorImages=False, SaveDoubles=False, NumSamples = None, Simple=False, Debug=True, UseZTest=False):
    AtMass, MDTrj = CenterFramesToCenterOfMass(AtMass, MDTrj)
    AtMass, AtCharge, AtSymb, MDTrj = SortAtomsInTrajectoryByFirstFrame(AtMass, AtCharge, AtSymb, MDTrj)
    
    NIso = len(MDTrj)
    counts = ConvertAtomicSymbolsIntoCounts(AtSymb[0])
    Hill = MakeHillFormula(counts)    
    NewTrj = []
    IniIdxs = []
    NewTrj.append(MDTrj[0])
    IniIdxs.append(0)
    alpha = AlphaValue
    DF = max(3*len(AtMass[0]) - DFRemoval, 1)
    
    Quartile = spst.chi2.ppf(q = alpha, df = DF)
    if Debug and not UseZTest:
        print(" chi2 quartile for level alpha="+str(alpha)+" and "+str(DF)+" degrees of freedom is "+str(Quartile))

    if Debug:
        print("added structure 0")

        
    for i in range(1,NIso):
        ItIsNew = True
        j = 0
        SaveXYZ = None
        SaveDoubleXYZ = None
        double_idx = None
        CritMin = None
        while ItIsNew and j < len(NewTrj):
            XYZ2 = AlignTwoStructuresWithNonLinearRMSD(AtSymb[0], AtMass[0], NewTrj[j], MDTrj[i], NLRMSDFreq, NumSamples, Simple)
            mwRMSD,alignedXYZ2 = GetMassWeightedRMSD(NewTrj[j], XYZ2, AtMass[0],ReturnAligned=True)
            if AccountMirrorImages:
                XYZ2mirror = AlignTwoStructuresWithNonLinearRMSD(AtSymb[0], AtMass[0], NewTrj[j], -MDTrj[i], NLRMSDFreq, NumSamples, Simple)
                mwRMSDmirror,alignedXYZ2mirror = GetMassWeightedRMSD(NewTrj[j], XYZ2mirror, AtMass[0],ReturnAligned=True)
                if mwRMSDmirror < mwRMSD:
                    XYZ2 = XYZ2mirror
                    mwRMSD = mwRMSDmirror
                    alignedXYZ2 = alignedXYZ2mirror
                    if Debug:
                        print(" For pair of conformers %03i - %03i we get a best match from the mirror image" % (j,i))
            if j == 0:
                SaveXYZ = np.array(alignedXYZ2)
                
            #Crit = (2./3.) * mwRMSD**2 * (1.e-20) * (1.66053906892e-27) * (CritFreq  * 2.997913E10 ) / (len(AtMass[0]) * 1.054571817e-34)
            if UseZTest:
                Crit = np.abs(CalculateZValue(AtMass[0], MDTrj[i], alignedXYZ2, CritFreq))/2.0
            else:
                Crit = 4. * np.pi * (CritFreq  * 2.997913E10 ) * (mwRMSD**2 * (1.e-20) * 1.66053906892e-27) / (1.054571817e-34 * Quartile )            
            
            if CritMin is None:
                CritMin = Crit
            else:
                if CritMin > Crit:
                    CritMin = Crit
            
            if Crit <= 1.:
                ItIsNew = False
                SaveDoubleXYZ = np.array(alignedXYZ2)
                double_idx = j
                if Debug:
                    print("Removed structure "+str(i)+" as a copy of "+str(j))
            j += 1 
                       
        if ItIsNew:    
            NewTrj.append(SaveXYZ)
            IniIdxs.append(i)
            if Debug:  
                print("added structure "+str(i)+" with minimal criterion being "+str(CritMin))   
        else:
            if SaveDoubles:
                DTrj = [NewTrj[double_idx],SaveDoubleXYZ]
                SaveTrajectory(Hill+"_double_pair_%04i-%04i.trj" % (IniIdxs[double_idx],i), DTrj, AtSymb[0]) 
    SaveTrajectory(Hill+"_quick_cleanup.trj", NewTrj, AtSymb[0])                    
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This (CoCoS) does conformer comparison.')
     
    parser.add_argument('-f', '--file', type=str, default=None,
                    help="XYZ file with structures to be analyzed in Angstroems. Default is None.")

    parser.add_argument('--NLRMSDFreq', type=float, default=NLFreq,
                    help="Frequency to compute the non-linear RMSD parameters (in cm^{-1}). Default is %.3f cm^{-1}." % (NLFreq))  

    parser.add_argument('--CritFreq', type=float, default=CriterionFreq,
                    help="Frequency to compute mwMSD criterion (in cm^{-1}). Default is %.1f cm^{-1}." % (CriterionFreq))  

    parser.add_argument('--DoAccountForMirrorImages', action='store_true', 
                    help="Flag to treat enantiomers as different species. If applied, no mirroring is done upon comparison.") 

    parser.add_argument('--FullMatrix', action='store_false', 
                    help="Flag not to perform a quicker conformer cleanup analysis, but make a full pairwise comparison.") 

    parser.add_argument('--DoNotSaveDoubles', action='store_false', 
                    help="Flag not to save the similar structures. Only works if option 'Full' is not used.") 

    parser.add_argument('--NumSamples', type=int, default=50,
                    help="Number of initial 3D rotation samples to get initial guess for local optimization. Default: 50.")  

    parser.add_argument('--DoNotUseGlobalOpt', action='store_true', 
                    help="Flag to not use global optimization with double annealing algorithm for orientation of structures, instead use the uniform random sampling.")

    parser.add_argument('--UseSimpleAssignment', action='store_true', 
                    help="Flag to use a simpler approach for atom-to-atom mapping, without optimizing the distance between all kinds of the same elements.")

    parser.add_argument('--Quiet', action='store_false', 
                    help="Flag to stop printing.")

    parser.add_argument('--UseZTest', action='store_true', 
                    help="Flag to use Z-test, instead of the default chi2-test.")                    
                    
    args,unknown = parser.parse_known_args() # parse the arguments

    # check if the input is given
    if args.file is None:
        print("No XYZ file was given, nothing to do :'(")
        exit()
    
    AtMass, AtCharge, AtSymb, MDTrj = LoadTrajectory(FileName=args.file)
    
    if not args.DoNotUseGlobalOpt:
        args.NumSamples = None
        if args.Quiet:
            print("We will use differential evolution")
    
    
    if not args.NumSamples is None and args.Quiet:
        print("We will use intial sampling of "+str(args.NumSamples)+" 3D rotations")
    
    if args.Quiet:
        print("We are using the frequency of "+str(args.NLRMSDFreq)+" cm**{-1} for the non-linear optimization and ")
        print("    "+str(args.CritFreq)+" cm**{-1} for the criterion.")
        if args.UseSimpleAssignment:
            print("Usuing simplified assignment approach")
        else:
            print("Using more sophisticated scheme for assigning atomic maps")
    
        if args.UseZTest:
            print("We are using the Z-test for the normal distribution to compare conformers")
        else:
            print("We are using mass-weighted RMSD with chi2 statistics to compare conformers")
    
    AccountForMirrorImages = True
    if args.DoAccountForMirrorImages:
        AccountForMirrorImages = False
    
    if args.Quiet:
        print("Are we treating enantiomers as the same specie? "+str(AccountForMirrorImages))
        
    
    if CheckHillFormulaCorrespondanceInTrajectory(AtSymb, MDTrj):
        if args.FullMatrix:
            PerformQuckie(AtMass, AtCharge, AtSymb, MDTrj, args.NLRMSDFreq, 
                          args.CritFreq, AccountMirrorImages=AccountForMirrorImages, 
                          SaveDoubles=args.DoNotSaveDoubles, NumSamples=args.NumSamples, 
                          Simple=args.UseSimpleAssignment, Debug=args.Quiet)
        else:
            PerformAnalysis(AtMass, AtCharge, AtSymb, MDTrj, args.NLRMSDFreq, args.CritFreq,
                            AccountMirrorImages=AccountForMirrorImages, 
                            NumSamples=args.NumSamples, Simple=args.UseSimpleAssignment, Debug=args.Quiet, UseZTest=args.UseZTest)
    else:
        print("Non-matching brutto formulas for molecules")
        exit()
















