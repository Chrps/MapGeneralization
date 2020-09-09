import os
import argparse

# file handle fh
fh = open('data/Public/private_list.txt')
while True:
    # read line
    line = fh.readline()
   
    print(line)

    ss = "python annotation_tool.py -g data/Private/{0}".format(line)
    os.system(ss) 

    # check if line is not empty
    if not line:
        break
fh.close()


 