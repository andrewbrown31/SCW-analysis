import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_gust_since_previous_post_processing',
            '2m_dewpoint_temperature', '2m_temperature', 'convective_available_potential_energy',
            'convective_precipitation', 'geopotential', 'orography', 'surface_pressure',
            'total_precipitation',
        ],
        'year': '2016',
        'month': '09',
        'day': [
            '28',
        ],
        'time': [
            '00:00', '06:00',
        ],
        'area': [
            -9, 111, -45,
            160,
        ],
    },
    '/g/data/eg3/ab4502/era5_download/dandenong_sfc.nc')

