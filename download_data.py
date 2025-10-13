import os
import pandas as pd
from pathlib import Path

CitySensors = {
    'Tokyo' : [1214722],
    'Berlin' : [2993,3019,4761,4767,2162178,2162180],
    'Beijing' : [2836346],
    'Quebec' : [236025,491,236023,477,236034],
    'Warsaw' : [10776,5554541,5554542,10775,60175],
    'Paris' : [4070,2995837,4111,4060,4067,4124,4094,4041,4093,2162591,4116,2681,4070,4039,4153],
    'Delhi' : [6932,10485,8472,5622,6931,5650,5570,10900,10921,5626,10831,10486,8235,5541,6934,8239,5627,8365,6957,11607,5634,6929,5613,8118,6358,11603,6960,8475,5541,8917,8915,50,10488,5630,6356,5610,7005,],
    'London' : [225807,270693,927171,225848,148,154]
}


def downloadData(city,start_year=2016,end_year=2025):
    r'''
        Download AQI Data from S3 Bucket
        What's going into the cmd prompt:
            aws s3 cp ^
            --no-sign-request ^
            --recursive ^
            "s3://openaq-data-archive/records/csv.gz/locationid=LOCATION_ID/year=YEAR" ^ (Grab all Files for YEAR from Bucket)
            .\data\CITY\Sensor_LOCATION_ID\YEAR=YEAR  (TARGET_PATH)
    '''
    try:
        cityPath = Path(f"D:/VSCode/Scripts/Python/AQIPredictor/data/{city}")
        cityPath.mkdir(parents=True, exist_ok=True)
        for sensor in range (len(CitySensors[city] + 1)):
            sensorPath = Path(f"D:/VSCode/Scripts/Python/AQIPredictor/data/{city}/Sensor_{CitySensors[city][sensor]}")
    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    sensorPath = Path(f"D:/VSCode/Scripts/Python/AQIPredictor/data/Tokyo/Sensor_{CitySensors['Tokyo'][0]}")
    sensorPath.mkdir(parents=True, exist_ok=True)
    print(f"Tokyo Directory for Sensor {CitySensors['Tokyo'][0]} created.")