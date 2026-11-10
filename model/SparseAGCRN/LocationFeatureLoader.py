import torch
from pyrosm import OSM
import math

def load_location_features(df, pbf_path, radius_m):
    # df has columns lat, lon, region
    osm = OSM(pbf_path)
    nodes, edges = osm.get_network(network_type="driving", nodes=True)

    features = []
    local_area_km2 = (2 * radius_m / 1000) ** 2    # square currently (because graph_from_point returns square graph)
    for _, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        utm_zone = int(math.floor((lon + 180) / 6) + 1)
        epsg_code = f'326{utm_zone}' if lat >= 0 else f'327{utm_zone}'

        angle_delta = radius_m / 111_000
        minlat, maxlat = lat - angle_delta, lat + angle_delta
        minlon, maxlon = lon - angle_delta, lon + angle_delta
        local_nodes = nodes[(nodes['lon'] >= minlon) & (nodes['lon'] <= maxlon) & (nodes['lat'] >= minlat) & (nodes['lat'] <= maxlat)]
        local_edges = edges.cx[minlon:maxlon, minlat:maxlat].to_crs(epsg_code)

        intersection_density = len(local_nodes) / local_area_km2
        street_density = local_edges.length.sum() / local_area_km2
        features.append([intersection_density, street_density])
    
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    return feature_tensor


# # Example Usage
# features = load_location_features(lat_lon_dataframe, './california.osm.pbf', 500)
# torch.save(features, 'some_path_to_give_to_SparseAGCRN')

# # Note that the osm.pbf data can be downloaded from https://download.geofabrik.de/