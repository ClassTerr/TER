# distance function
import pandas as pd
import numpy as np
from geopy.distance import great_circle
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

eps = 20  # meters


# least common subsequence
def lcs(r1, r2):
    n0 = len(r1)
    n1 = len(r2)
    # An (m+1) times (n+1) matrix
    c = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if great_circle(tuple([r1[i - 1]]), tuple([r2[j - 1]])).meters < eps:
                c[i][j] = c[i - 1][j - 1] + 1
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


# read data
df = pd.read_csv("waypoints.csv")
df = pd.DataFrame(df)
df_gr = df.groupby(['route'])
routes = pd.DataFrame(columns=['Route', 'Points'])
for route, group in df_gr:
    path_points = [row[['lat', 'lng']].to_numpy() for row_index, row in group.iterrows()]
    routes = routes.append({'Route': route, 'Points': path_points}, ignore_index=True)

map_plot = go.Figure()

# add routes to map
for i, val in routes.iterrows():
    name = str(val['Route'])
    points = val['Points']
    map_plot.add_trace(go.Scattermapbox(
        name="Route " + name,
        mode="markers+lines",
        lat=[round(x[0], 5) for x in points],
        lon=[round(x[1], 5) for x in points],
        marker={'size': 10}))

dist_m = []

for i, val1 in routes.iterrows():
    dist_row = []
    route1 = val1['Points']
    for j, val2 in routes.iterrows():
        if i == j:
            dist_row.append(0)
            continue

        route2 = val2['Points']
        c = lcs(route1, route2)

        distance = c[-1][-1] / len(route1)
        dist_row.append(distance)

        m = len(route1)
        n = len(route2)

        common_part = back_track(c, route1, route2, m, n)

        if i >= j:
            continue

        # add lcs to map
        name1 = str(val1['Route'])
        name2 = str(val2['Route'])

        map_plot.add_trace(go.Scattermapbox(
            name="LCS (Route " + str(name1) + "; Route " + str(name2) + ")",
            mode="markers+lines",
            lat=list(map(lambda x: round(x[0], 5), common_part)),
            lon=list(map(lambda x: round(x[1], 5), common_part)),
            marker={'size': 10}))

    dist_m.append(dist_row)

map_plot.update_layout(
    title='LCS',
    autosize=True,
    margin={'l': 1, 't': 1, 'b': 1, 'r': 1},
    mapbox={
        'center': {'lon': df.lng.mean(),
                   'lat': df.lat.mean()},
        'style': "stamen-terrain",
        'zoom': 15.5})

map_plot.show()

# Perform hierarchical/agglomerative clustering
Z = linkage(dist_m, 'ward')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()
