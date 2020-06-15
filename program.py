import pathlib
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from geopy.distance import great_circle
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from palette import palette

eps = 50  # meters
recalculate_distances = False


def calculate_dist_matrix():
    dist_m = []
    totalCompCount = len(routes) * len(routes)
    curr = 0
    for i, val1 in routes.iterrows():
        dist_row = []
        route1 = val1['Points']
        for j, val2 in routes.iterrows():
            curr += 1
            print('Comparing (' + str(curr) + ' of ' + str(totalCompCount) + ')')
            if i == j:
                dist_row.append(0)
                continue

            route2 = val2['Points']
            c = lcs(route1, route2)
            c2 = lcs(route2, route1)

            sim = c[-1][-1] / len(route1)
            sim2 = c2[-1][-1] / len(route2)

            if sim < sim2:
                sim = sim2
                route1 = route2

            dist_row.append(1 - sim)

        dist_m.append(dist_row)

    return dist_m


# least common subsequence
def lcs(r1, r2):
    n0 = len(r1)
    n1 = len(r2)
    # An (m+1) times (n+1) matrix
    c = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            d = great_circle(tuple([r1[i - 1]]), tuple([r2[j - 1]])).meters
            if d < eps:
                c[i][j] = c[i - 1][j - 1] + (1 - d / eps)
            else:
                c[i][j] = max(c[i][j - 1], c[i - 1][j])
    return c


def back_track(c, r1, r2, i, j):
    if i == 0 or j == 0:
        return []
    elif great_circle(tuple([r1[i - 1]]), tuple([r2[j - 1]])).meters < eps:
        paired = np.array([r1[i - 1], r2[j - 1]])
        approximated_path = np.average(paired, axis=0)
        return back_track(c, r1, r2, i - 1, j - 1) + [tuple(approximated_path)]
    else:
        if c[i][j - 1] > c[i - 1][j]:
            return back_track(c, r1, r2, i, j - 1)
        else:
            return back_track(c, r1, r2, i - 1, j)


def rounded_indices(ls, max_length):
    """
    Linearly removes elements from list so the length of result will be equal to max_length
    For example
    rounded_indices(range(12), 3) = [0, 4, 8]
    :param ls:
    :param max_length:
    :return:
    """
    coeff = len(ls) / max_length
    if coeff <= 1:
        return ls

    result = []
    original_index = 0
    new_index = 0
    while new_index < len(ls):
        result.append(ls[new_index])
        original_index += 1
        new_index = int(round(coeff * original_index))

    return result


df = pd.read_csv("go_track_trackspoints.csv")
df = pd.DataFrame(df)
df_gr = df.groupby(['route'])
routes = pd.DataFrame(columns=['Route', 'Points'])

route_number = 0

for route, group in df_gr:
    route_number += 1
    # if route_number == 20:
    #     break

    path_points = [row[['lat', 'lng']].to_numpy() for row_index, row in group.iterrows()]
    if len(path_points) < 10:
        print(route)
        continue

    path_points = rounded_indices(path_points, 100)
    routes = routes.append({'Route': route, 'Points': path_points}, ignore_index=True)

print('Routes read...')

map_plot = go.Figure()

dist_m = []

dist_m_file_name = "dist_m.dat"

if not recalculate_distances and pathlib.Path(dist_m_file_name).exists():
    with open(dist_m_file_name, "rb") as fp:  # Unpickling
        dist_m = pickle.load(fp)
else:
    dist_m = calculate_dist_matrix()
    with open(dist_m_file_name, "wb") as fp:  # Pickling
        pickle.dump(dist_m, fp)

print('Distances calculated...')

minLat = df.lat.min()
minLng = df.lng.min()

map_plot.update_layout(
    title='LCS',
    autosize=True,
    margin={'l': 1, 't': 1, 'b': 1, 'r': 1},
    mapbox={
        'center': {'lon': minLng + (df.lng.max() - minLng) / 2,
                   'lat': minLat + (df.lat.max() - minLat) / 2},
        'style': "stamen-terrain",
        'zoom': 13.})

print('Plot builded...')

# convert the redundant n*n square matrix form into a condensed nC2 array
# dist_m[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
# dist_m = ssd.squareform(dist_m)

# Perform hierarchical/agglomerative clustering
Z = linkage(dist_m, 'ward')
print('Linkage performed...')
# print(Z)

dendrogram(Z, labels=routes['Route'].to_numpy())
plt.show()

inertia = sorted([x[2] for x in Z], reverse=True)[:15]
plt.bar(range(1, len(inertia) + 1), inertia)
# plt.plot(range(1, len(inertia) + 1), inertia)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

inertia_gain = [i-j for i, j in zip(inertia[:-1], inertia[1:])]
plt.bar(range(1, len(inertia_gain) + 1), inertia_gain)
# plt.plot(range(1, len(inertia_gain) + 1), inertia_gain)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Inertia gain')
plt.show()

# by dendrogram we need to find optimal cluster count:
# max_d = 5
threshold = int(input("Enter the threshold: "))
dendrogram(Z, labels=routes['Route'].to_numpy())
plt.axhline(y=threshold, color='r', linestyle='--')
plt.show()

clusters = fcluster(Z, threshold, criterion='distance')
print('Clustering performed...')
# print('clusters')
# print(clusters)

routes['Cluster'] = clusters

clusteredRoutes = routes.groupby(['Cluster'])

for cluster, cl_routes in clusteredRoutes:
    coords = []
    for i, r in cl_routes.iterrows():
        for item in r['Points']:
            coords.append([round(x, 5) for x in list(item)])

        color = palette[(cluster - 1) % len(palette)]
        map_plot.add_trace(go.Scattermapbox(
            name="Route " + str(r['Route']),
            mode="markers+lines",
            lat=[round(x[0], 5) for x in r['Points']],
            lon=[round(x[1], 5) for x in r['Points']],
            marker=dict(size=6, color=color, opacity=0.4),
            line=dict(width=4, color=color),
            opacity=0.4))

    lat = []
    lng = []

    # hull = alphashape.alphashape(coords, 0)
    # hull_pts = hull.exterior.coords.xy
    #
    # map_plot.add_trace(go.Scattermapbox(
    #     fill="toself",
    #     mode="none",
    #     name="Cluster #" + str(cluster),
    #     lat=list(hull_pts[0]),
    #     lon=list(hull_pts[1]),
    #     opacity=0.5))

map_plot.show(config={'displayModeBar': False})
