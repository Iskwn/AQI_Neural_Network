import boto3
import subprocess
from pathlib import Path
from botocore import UNSIGNED
from botocore.client import Config

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
        LOCATION_ID (int): The ID of the Sensor at a location (refer to CitySensors dictionary)
        YEAR (int): The Year the Data is being categorized as / pulled from (e.g. pulling data from 2016 or storing it as data from 2016)
        What's going into the cmd prompt:
            aws s3 cp ^
            --no-sign-request ^
            --recursive ^
            "s3://openaq-data-archive/records/csv.gz/locationid=LOCATION_ID/year=YEAR" ^ (Grab all Files for YEAR from Bucket)
            .\data\CITY\Sensor_LOCATION_ID\YEAR  (TARGET_PATH)
    '''
    try:
        
        #initialize s3 client
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        target_bucket = 'openaq-data-archive'
        
        #Make the City Directory
        cityPath = Path(f"D:/VSCode/Scripts/Python/AQIPredictor/data/{city}")
        cityPath.mkdir(parents=True, exist_ok=True)
        
        #Make all assosciated Sensor Directories and get info for them
        for sensor in CitySensors[city]:
            sensorPath = Path(f"D:/VSCode/Scripts/Python/AQIPredictor/data/{city}/Sensor_{sensor}")
            sensorPath.mkdir(exist_ok = True)
            

            #Get data from 2016 to 2025 (inclusive) for each sensor
            for year in range (start_year, end_year + 1):
                yearPath = Path(f"D:/VSCode/Scripts/Python/AQIPredictor/data/{city}/Sensor_{sensor}/{year}")
                yearPath.mkdir(exist_ok = True)
                
                
                #Formatting command (visually) to ensure arguments are correct
                prefix = f"records/csv.gz/locationid={sensor}/year={year}/"
                
                #run command, check for any errors
                try:
                    response = s3.list_objects_v2(Bucket = target_bucket, Prefix = prefix)
                    
                    if 'Contents' not in response:
                        print(f"        No data found for {year}")
                        yearPath.rmdir()
                        continue
                    
                    files = response['Contents']
                    print(f"        Found {len(files)} files, downloading...")
                    #count number of files found & downloaded
                    
                    for file in files:
                        file_loc = file['Key']
                        file_name = Path(file_loc).name
                        local_file = yearPath / file_name
                        
                        s3.download_file(target_bucket, file_loc, str(local_file))
                    
                    print(f"        Downloaded {len(files)} files")
                except Exception as e:
                    print(f"        Error downloading data: {e}")
                    if yearPath.exists() and not any(yearPath.iterdir()):
                        yearPath.rmdir()
                        

            print(f"\nSuccessfully downloaded data for {city}")
    except Exception as e:
        print(f"Error during download for {city}: {e}")



if __name__ == "__main__":
    """ Commented out so I don't accidentally redownload data and waste time
    cities = ['Tokyo', 'Quebec', 'Berlin', 'London', 'Beijing', 'Delhi', 'Warsaw', 'Paris']
    
    for city in cities:
        downloadData(city)
        
        print("\n" + "="*50)
        print("All downloads completed")
        print("="*50)
        """