import pandas as pd

def get_crop_planting_harvest_info(data):
    fao_crop_names = ['Bean', 'Wheat, durum', 'Wheat, bread', 'Barley', 'Potato', 'Chick-pea', 'Tomato']
    columns_to_drop = ['All year', 'Sowing rate', 'Unnamed: 9', 'Unnamed: 11', 'Comments En',
       'Comments ES', 'Comments FR', 'Comments ZH', 'Comments AR',
       'Comments RU', 'Growing period', 'AgroEcological Zone', 'Additional information']
    data = data.drop(columns_to_drop, axis=1)
    data.rename(columns={'Early Sowing': 'early_sowing_day', 
                         'Unnamed: 4': 'early_sowing_month',
                         'Later Sowing': 'later_sowing_day',
                         'Unnamed: 6': 'later_sowing_month',
                         'Early harvest': 'early_harvest_day',
                         'Unnamed: 13': 'early_harvest_month',
                         'Late harvest': 'late_harvest_day',
                         'Unnamed: 15': 'late_harvest_month'}, inplace=True)
    data = data[1:]
    data = data[data['Crop'].isin(fao_crop_names)].reset_index().drop('index', axis=1)
    return data