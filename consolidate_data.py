import pandas as pd
from tqdm import tqdm
from pathlib import Path

'''
    Consolidating the downloaded data from download_data.py into a city-specific dataframe
    
    Requires: 
        pandas for dataframe
        pathlib for file navigation
        target destination (specific to me as of current, might make more general after project is finished)
'''
#Fetch Data from Data folder for consolidation
def consolidateData(city, show = False, prog = False):
    '''
    Args:
        city (str): City name (e.g. Tokyo)
        show (bool): Whether or not to print dataframe head
        prog (bool): whether or not to show the progress bars
        
    Returns:
        pandas.Dataframe: Air quality data for the specified city
    '''
    city_path = Path(f"D:/VSCode/Scripts/Python/AQIPredictor/data/{city}")
    city_data = pd.DataFrame()
    sensors = sorted([d for d in city_path.iterdir() if d.is_dir()])
    print(f"\nFound {len(sensors)} sensors." if len(sensors) != 1 else "Found 1 sensor.")
    print("Beginning Data Collection...\n")
    #Iterates through each Sensor
    if prog:
        for sensor_path in tqdm(sensors, desc = f"Iterating through sensors in {city}", leave = True):
            sensor_path: Path
            sensor_name = sensor_path.name
            sensor_id = sensor_name.replace("Sensor_", "")
            sensor_data = pd.DataFrame()
            #saves years into a list
            years = sorted([x.name for x in sensor_path.iterdir() if x.is_dir()])
            print(f"\nFound data for {', '.join(years)} for sensor {sensor_id}")
            print(f"SENSOR {sensor_id}")
            
            #Iterates through each Year for the Sensor
            for year in tqdm(years, desc= f"Iterating through years in sensor", leave = False):
                year_path = sensor_path.joinpath(year)
                csv_files = [x.name for x in year_path.iterdir()]

                for file in tqdm(csv_files, desc= f"Adding data to {year} DataFrame", leave = False):
                    file_path = year_path.joinpath(file)
                    df = pd.read_csv(file_path, compression = 'gzip')
                    sensor_data = pd.concat([sensor_data, df], ignore_index = True)
            city_data = pd.concat([city_data, sensor_data], ignore_index= True)
    else:
            for sensor_path in sensors:
                sensor_path: Path
                sensor_name = sensor_path.name
                sensor_id = sensor_name.replace("Sensor_", "")
                sensor_data = pd.DataFrame()
                #saves years into a list
                years = sorted([x.name for x in sensor_path.iterdir() if x.is_dir()])
                print(f"Found data for {', '.join(years)} for sensor {sensor_id}")
                print(f"SENSOR {sensor_id}")
                
                #Iterates through each Year for the Sensor
                for year in years:
                    year_path = sensor_path.joinpath(year)
                    csv_files = [x.name for x in year_path.iterdir()]

                    for file in csv_files:
                        file_path = year_path.joinpath(file)
                        df = pd.read_csv(file_path, compression = 'gzip')
                        sensor_data = pd.concat([sensor_data, df], ignore_index = True)
                city_data = pd.concat([city_data, sensor_data], ignore_index= True)
    print(f"\nFinished Data Collection for {city}")
    print(f"Total files collected: {len(city_data)}\n")
    if show:
        print(f"{city} Data Head: {city_data.head()}")
    return city_data

if __name__ == "__main__":
    cities = ['Tokyo', 'Quebec', 'Berlin', 'London', 'Beijing', 'Delhi', 'Warsaw', 'Paris']
    print(consolidateData(cities[0], show = True, prog = True))