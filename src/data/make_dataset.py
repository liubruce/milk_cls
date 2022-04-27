# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

import os, sys
from PIL import Image

def convert_tiff_jpg(image_path):
    for infile in os.listdir(image_path):
        print("file : " + infile, infile[-4:])
        if infile[-4:] == "tiff":
            # print "is tif or bmp"
            outfile = str(image_path) + '/' + infile[:-4] + "jpeg"
            # print(str(image_path) + outfile, infile)
            im = Image.open(str(image_path) + '/' +  infile)
            print("new filename : " + outfile)
            out = im.convert("RGB")
            out.save(outfile, "JPEG", quality=90)

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]

    image_path = Path(str(project_dir) + '/data/raw/220413/transmission/').resolve()
    for infile in os.listdir(image_path):
        print(infile)
        if infile != '.DS_Store':
            convert_tiff_jpg(Path(str(image_path) + '/' + infile).resolve())

