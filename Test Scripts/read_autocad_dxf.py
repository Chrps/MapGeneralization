'''
This is a test script for reading autocad files (DWG files with the .dwg extension)
'''

import ezdxf
import os

#Changing the working directory to the project, not the script
directory_of_script = os.path.dirname(os.getcwd())
dxf_file_path = r'Sample Files\\DXF Files\\MSP1-HoM-MA-XX+4-ET.dxf'
dxf_file_path = directory_of_script + r'\\' + dxf_file_path


doc = ezdxf.readfile(dxf_file_path)

print("done")
