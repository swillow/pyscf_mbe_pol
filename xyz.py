import numpy as np

def read_xyz (fname):

    lines = open (fname, 'r').readlines()

    natom = int (lines[0].split()[0])

    atm_types = []
    atm_crds = []
    for line in lines[2:2+natom]:
        words = line.split()
        atnm = words[0]
        x = float(words[1])
        y = float(words[2])
        z = float(words[3])
        atm_types.append (atnm)
        atm_crds.append ([x, y, z])
    
    return atm_types, np.array (atm_crds)
    