import os
import argparse
import numpy as np
import cv2


if __name__ == "__main__":
    """

    Command:
        -l path/to/image_list.txt
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root", type=str,
                    default='/home/markpp/github/MapGeneralization/data/Public/', help="data root dir")
    ap.add_argument("-l", "--list", type=str,
                    default='test_list.txt', help="File list")
    args = vars(ap.parse_args())

    file_list = []
    with open(os.path.join(args["root"],args["list"])) as f:
        file_list = f.read().splitlines()

    for path in file_list[:2]:
        dxf_path = os.path.join(args["root"],path.replace('/anno/','/dxf/').replace('_w_annotations.gpickle','.dxf'))
        print(dxf_path)


        os.system("libreoffice --headless --convert-to pdf {}".format(dxf_path))
        pdf_path = os.path.basename(dxf_path.replace('.dxf','.pdf'))
        print(pdf_path)

        os.system("pdfcrop {}".format(pdf_path))
        pdf_crop_path = pdf_path.replace('.pdf','-crop.pdf')
        print(pdf_crop_path)

        image_path = dxf_path.replace('/dxf/','/images/').replace('.dxf','.png')
        print(image_path)
        os.system("gs -sDEVICE=png16m -dNOPAUSE -dBATCH -dSAFER -sOutputFile={} {}".format(image_path,pdf_crop_path))

        os.system("rf {}".format(pdf_path))
        os.system("rf {}".format(pdf_crop_path))

        #libreoffice --headless --convert-to pdf A1322PE-0.dxf
        #pdfcrop A1322PE-0.pdf
        #gs -sDEVICE=png16m -dNOPAUSE -dBATCH -dSAFER -sOutputFile=A1322PE-0.png A1322PE-0-crop.pdf
