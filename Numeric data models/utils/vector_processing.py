def extract_shapefile(data_path = 'Supplementary_data/20182019.xlsx'):
    vector_data = pd.read_excel(data_path)
    vector_data = vector_data[['gps_position', 'culture_specific', 'superficy', 'surface_calculated']]
    vector_data.columns = ['gps_position', 'culture', 'superficie', 'surface_calculée']
    vector_data['culture'] = vector_data['culture'].replace(['blÃ© dur', 'blÃ© tendre', 'fÃ¨ve', 'autres cÃ©rÃ©ales', 'autres lÃ©gumineuse'],['blé dur', 'blé tendre', 'fèves', 'autres céréales', 'autres légumineuses'])

    coordinate_list=[]
    for coordinate_value in vector_data['gps_position'].values.tolist():
        coordinate_list.append(ast.literal_eval(coordinate_value))

    coordinate_nested_list = []
    for coordinate_value in coordinate_list:
        coordinate_nested_list.append([list(ele) for ele in coordinate_value])

    inverse_coordinate_nested_list = []
    for coordinate_value in coordinate_nested_list:
          inverse_coordinate_nested_list.append([elem[::-1] for elem in coordinate_value])

    vector_data['gps_position'] = inverse_coordinate_nested_list
    vector_data['geometry'] = vector_data['gps_position'].apply(Polygon)
    vector_data = vector_data[['culture', 'superficie', 'surface_calculée', 'geometry', 'gps_position']]

    vectors = gpd.GeoDataFrame(vector_data, geometry='geometry')
    crop_vectors = vectors.set_crs(4326, allow_override=True)
    crop_vectors = crop_vectors[['culture', 'geometry']]
    
    return crop_vectors


def get_overlapping_gpd(crop_vectors):
    data_temp = crop_vectors
    data_overlaps=gpd.GeoDataFrame(crs=data_temp.crs)
    for index, row in data_temp.iterrows():
        data_temp1=data_temp.loc[data_temp.id!=row.id,]
        # check if intersection occured
        overlaps=data_temp1[data_temp1.geometry.overlaps(row.geometry)]['id'].tolist()
        if len(overlaps)>0:
            temp_list=[]
            # compare the area with threshold
            for y in overlaps:
                temp_area=gpd.overlay(data_temp.loc[data_temp.id==y,],data_temp.loc[data_temp.id==row.id,],how='intersection')
                temp_area=temp_area.loc[temp_area.geometry.area>=9e-9]
                if temp_area.shape[0]>0:
                    data_overlaps=gpd.GeoDataFrame(pd.concat([temp_area,data_overlaps],ignore_index=True),crs=data_temp.crs)
    data_overlaps['sorted']=data_overlaps.apply(lambda y: sorted((y['id_1'],y['id_2'])),axis=1)
    data_overlaps['sorted'] = [','.join(map(str, l)) for l in data_overlaps['sorted']]
    data_overlaps=data_overlaps.drop_duplicates(['id_1', 'id_2'])
    data_overlaps = data_overlaps.drop_duplicates(['sorted'])
    return data_overlaps


def remove_duplicates(crop_vectors):
    data = crop_vectors.drop_duplicates(keep='last')
    return data


def remove_multicrops_polygons(crop_vectors):
    data = crop_vectors.drop_duplicates(subset=['geometry'],keep=False)
    data = data.reset_index()
    data.columns = ['id', 'culture', 'geometry']
    return data


def remove_overlapping_polygons(crop_vectors):
    data_overlaps = get_overlapping_gpd(crop_vectors)
    overlapping_different_crop = data_overlaps[data_overlaps['culture_1']!=data_overlaps['culture_2']]
    overlapping_same_crop = data_overlaps[data_overlaps['culture_1']==data_overlaps['culture_2']]
    overlapping_ids = overlapping_different_crop.id_1.tolist() + overlapping_different_crop.id_2.tolist()
    data = crop_vectors[~crop_vectors['id'].isin(overlapping_ids)]
    data = data.reset_index().drop('index', axis=1)
    data.columns = ['id', 'culture', 'geometry']
    return data