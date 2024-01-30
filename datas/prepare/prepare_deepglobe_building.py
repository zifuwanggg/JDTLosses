import os
import sys
from glob import glob

import cv2
from osgeo import gdal, ogr


# https://gist.github.com/avanetten/b295e89f6fa9654c9e9e480bdb2e4d60#file-create_building_mask-py
def create_poly_mask(rasterSrc, vectorSrc, npDistFileName="", noDataValue=0, burn_values=1):
    """
    Create polygon mask for rasterSrc, similar to labeltools/createNPPixArray() in spacenet utilities.
    """

    # open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    # extract data from src Raster File to be emulated
    # open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize

    if npDistFileName == "":
        dstPath = ".tmp.tiff"
    else:
        dstPath = npDistFileName

    # create First raster memory layer, units are pixels
    # change output to geotiff instead of memory
    memdrv = gdal.GetDriverByName("GTiff")
    dst_ds = memdrv.Create(dstPath, cols, rows, 1, gdal.GDT_Byte,
                        options=["COMPRESS=LZW"])
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)
    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values])
    dst_ds = 0

    if npDistFileName == "":
        os.remove(dstPath)


def prepare_deepglobe_building(data_dir):
    src_dir = f"{data_dir}/building"
    dst_dir = f"{data_dir}/building/train"
    os.makedirs(dst_dir, exist_ok=True)

    AOIs = {"Vegas": 2, "Paris": 3, "Shanghai": 4, "Khartoum": 5}
    options_list = ["-ot Byte", "-of PNG", "-b 1", "-b 2", "-b 3", "-scale"]
    options_string = " ".join(options_list)

    for city, AOI in AOIs.items():
        print(f"Prepare {city}")
        raster_dir = f"{src_dir}/AOI_{AOI}_{city}_Train/RGB-PanSharpen/*"
        vector_dir = f"{src_dir}/AOI_{AOI}_{city}_Train/geojson/buildings/"
        for raster_file in glob(raster_dir):
            tif = raster_file.split("/")[-1].split("_")[-1]
            index = int("".join((filter(str.isdigit, tif))))

            image_file = f"{city}_{index}_sat.jpg"
            mask_file = image_file.replace("sat.jpg", "mask.png")
            label_file = image_file.replace("sat.jpg", "label.png")
            vector_file = f"buildings_AOI_{AOI}_{city}_img{index}.geojson"

            image_file = os.path.join(dst_dir, image_file)
            mask_file = os.path.join(dst_dir, mask_file)
            label_file = os.path.join(dst_dir, label_file)
            vector_file = os.path.join(vector_dir, vector_file)

            gdal.Translate(image_file, raster_file, options=options_string)
            create_poly_mask(raster_file, vector_file, mask_file, noDataValue=0, burn_values=255)

            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            label = mask / 255

            cv2.imwrite(label_file, label)


if __name__ == "__main__":
    prepare_deepglobe_building(sys.argv[1])
