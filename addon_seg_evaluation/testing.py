
from deformationcytometer.detection.includes.UNETmodel import *

if __name__ == "__main__":
    import clickpoints.launch
    print(clickpoints.__file__)
    #clickpoints.launch.main(r"/home/user/Desktop/biophysDS/emirzahossein/microfluidic cell rhemeter data/microscope_1/august_2020/2020_08_17_alginate2%_overtime/1/2020_08_17_14_52_10.tif")
    clickpoints.launch.main("/home/user/Desktop/biophysDS/emirzahossein/microfluidic cell rhemeter data/microscope_1/november_2020/2020_11_10_alg2%_neutrophile/8/2020_11_10_17_14_54.tif")
    #clickpoints.launch.main(
   #     r"/home/user/Desktop/biophysDS/emirzahossein/microfluidic cell rhemeter data/microscope_1/october_2020/2020_10_30_alg2.5%_NIH3T3/2/2020_10_30_14_20_44.tif")
