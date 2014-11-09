import string
import numpy as np
import collections

def getPointData(f):
    for i in range(4):
        line = f.readline()
    
    coords = []
    labels = []
    nPoints = 0
    while line:
        nPoints += 1
        line = string.split(line)
        line = [float(x) for x in line]
        coords.append(line[0:3])
        labels.append(int(line[4]))
        line = f.readline()
        
    f.close()
    return (coords,labels)

def getLabelColour(label):
    # veggie
    if label == 1004:
        rgb = [0,255,0]
    # wire
    elif label == 1100:
        rgb = [128,128,128]
    # pole
    elif label == 1103:
        rgb = [0,0,205]
    # ground
    elif label == 1200:
        rgb = [128,0,0]
    # facade
    elif label == 1400:
        rgb = [255,255,255]
    else:
        raise Exception("Unknown label")

    rgbBitShift = rgb[2] | (rgb[1] << 8) | (rgb[0] << 16)
    
    return rgbBitShift

def writePCD(coords,labels,fname):
    fname = fname + '.pcd'
    f = open(fname,'w')

    line = 'VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n' + \
        'WIDTH {}\nHEIGHT 1\n'.format(len(labels))
    f.write(line)
    tmp = np.array(coords)
    tmp = tmp.mean(axis = 0)
    viewpoint = tmp
    line = 'VIEWPOINT {} {} {} 1 0 0 0\nPOINTS {}\nDATA ascii\n'.format(viewpoint[0],viewpoint[1],viewpoint[2],len(labels))
    f.write(line)

    for i in range(len(labels)):
        col = getLabelColour(labels[i])
        line = '{} {} {} {}\n'.format(coords[i][0],coords[i][1],coords[i][2],col)
        f.write(line)

    f.close()
               
def getLabelFrequency(labels):
    freq = collections.Counter(labels)
    return freq.values()
