import ezdxf
from matplotlib.pyplot import cm
import pylab as pl
from matplotlib import collections  as mc
import numpy as np

def print_entity(e):
    print("LINE on layer: %s" % e.dxf.layer)
    print("start point: %s" % e.dxf.start)
    print("end point: %s\n" % e.dxf.end)

# YOU NEED A DXF FILE
path_to_dxf_file = "Sample Files/DXF Files/MSP1-HoM-MA-XX+4-ET.dxf"
doc = ezdxf.readfile(path_to_dxf_file)  # TAKES A WHILE TO LOAD!!
msp = doc.modelspace()
# Group by "layer"
group = msp.groupby(dxfattrib='layer')

# Get entities in each layer
for layer, entities in group.items():
    print('Layer %s contains following entities:' % layer)
    for entity in entities:
        if entity.dxftype() == 'LINE':
        #if str(entity).split('(')[0] == "LINE":
            print_entity(entity)





# matplotlib test
import matplotlib.pyplot as plt

plt.plot([1,23,2,4])
plt.ylabel('some numbers')

plt.show()

# iterate over all entities in modelspace
'''for e in msp:
    if e.dxftype() == 'LINE':
        print_entity(e)'''


# entity query for all LINE entities in modelspace
#for e in msp.query('LINE'):
#    print_entity(e)

# STUFF I THOUGH ABOUT USING TO PLOT THE LINES IN MATPLOTLIB
'''
import matplotlib
import numpy as np
import pylab as pl
from matplotlib import collections  as mc

lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

lc = mc.LineCollection(lines, colors=c, linewidths=2)
fig, ax = pl.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.margins(0.1)
'''