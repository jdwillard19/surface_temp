import cdsapi
import numpy as np
import pandas as pd
import pygrib
import pdb
import math

site_ids = ['nhdhr_143249470']
metadata = pd.read_csv('../../metadata/lake_metadata.csv')
metadata.set_index('site_id',inplace=True)
factor = 2 
for site_id in site_ids:

    lat = metadata.loc[site_id]['lake_lat_deg']
    lon = metadata.loc[site_id]['lake_lon_deg']
    lat_upper = math.ceil(lat * factor) / factor
    lat_lower = math.floor(lat * factor) / factor
    lon_upper = math.ceil(lon * factor) / factor
    lon_lower = math.floor(lon * factor) / factor
    c = cdsapi.Client()

    fn = 'download.grib'


    c.retrieve(
        'reanalysis-era5-land',
        {
            'format': 'grib',
            'variable': 'lake_mix_layer_temperature',
            'year': [
                '1981', '1982', '1983',
                '1984', '1985', '1986',
                '1987', '1988', '1989',
                '1990', '1991', '1992',
                '1993', '1994', '1995',
                '1996', '1997', '1998',
                '1999', '2000', '2001',
                '2002', '2003', '2004',
                '2005', '2006', '2007',
                '2008', '2009', '2010',
                '2011', '2012', '2013',
                '2014', '2015', '2016',
                '2017', '2018', '2019',
                '2020',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '04:00', '08:00',
                '12:00', '16:00', '20:00',
            ],
            'area': [
                lat_upper, lon_upper, lat_lower,
                    lon_lower
                ]
        },
        fn)

    gr = pygrib.open(fn)
    for g in gr:
        pdb.set_trace()
        print(g)
        print(g.values)