"""
Author: Markus Frohmann
Matr.Nr.: k12005604
Exercise 2
"""

import os
from pathlib import Path
from PIL import Image, ImageOps, ImageStat
import shutil
import hashlib


def ex2(inp_dir: str, out_dir: str, logfile: str):
    if not os.path.isdir(out_dir):
        Path(out_dir).mkdir(parents=True)
    logfile_abs = os.path.abspath(logfile)
    found_files = Path(inp_dir).rglob("*")
    found_files = sorted(found_files)
    file_number = 0  # number of files to copy/copied files
    hashes = []
    for i, file in enumerate(found_files, start=1):
        error = 0
        while error == 0:
            file_number_padded = str(file_number).zfill(7)  # add padding to filenumber
            file_str = str(os.path.basename(file))
            if (file_str.endswith('.jpg')  # check for proper file ending
                    or file_str.endswith('.JPG')
                    or file_str.endswith('.JPEG')
                    or file_str.endswith('.jpeg')):
                filename = str(file_number_padded) + '.jpg'
            else:
                filename = file_str  # to write in logfile
                error = 1  # wrong file ending
                continue

            if os.path.getsize(file) < 10241:
                error = 2  # file size <10kB
                continue

            try:
                im = Image.open(file)
                if im.size[0] < 100 or im.size[1] < 100 or not (im.mode == '1' or im.mode == 'L' or im.mode == 'P'):
                    error = 4  # Height or Width of Image >= 100 or not grayscale
                    continue
                elif ImageStat.Stat(im).var[0] == 0:
                    error = 5  # variance 0: only 1 value in image data
                    continue
            except:
                error = 3  # PIL cannot read image
                continue

            # Get hashes
            with open(file, "rb") as f:
                hash = hashlib.sha256(f.read()).hexdigest()
            if hash not in hashes:  # image not yet copied
                hashes.append(hash)
            else:
                error = 6  # same image already copied
            break

        if error == 0:  # no error, copy
            out_path = Path.joinpath(Path(out_dir), Path(filename))
            shutil.copy(file, out_path)
            file_number += 1
        elif error != 0:  # error, do not copy, append to logfile
            filename = file_str
            with open(logfile, 'a') as f:
                logfile_name = Path(file.relative_to(Path(inp_dir)))  # original filename w/ relative path
                f.write(f'{logfile_name};{error}\n')

    return file_number
