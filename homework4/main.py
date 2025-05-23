import pandas as pd
import openrouteservice
from openrouteservice.directions import directions
import folium
from deap import base, creator, tools, algorithms
import random
import numpy as np
import os
import sys

if len(sys.argv) > 1:
    profile = sys.argv[1]
else:
    profile = 'foot-walking'  # 'driving-car', 'cycling-regular', и т.д.

print(f"Профиль маршрута: {profile}")

points = pd.DataFrame({
    'name': ['КФУ (2 корпус)', 'Вкусно и точка', 'Казанский Кремль', 'Черное озеро'],
    'latitude': [55.7914, 55.7903, 55.7982, 55.7847],
    'longitude': [49.1219, 49.1242, 49.1064, 49.1175],
    'priority': [10, 5, 8, 7]
})

client = openrouteservice.Client(key=os.getenv('ORS_API_KEY'))
coords = points[['longitude', 'latitude']].values.tolist()

cache_file = f'durations_{profile}.npy'
if os.path.exists(cache_file):
    print("Загрузка матрицы времени из кэша...")
    durations = np.load(cache_file)
else:
    try:
        matrix = client.distance_matrix(
            locations=coords,
            profile=profile,
            metrics=['duration'],
            units='km'
        )
        durations = matrix['durations']
        np.save(cache_file, durations)
        print("Матрица времени сохранена в кэш.")
    except Exception as e:
        print(f"Ошибка API: {e}")
        exit(1)

MAX_TIME = 3600  # 1 час

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(points)), len(points))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    total_time = 0
    total_priority = 0
    for i in range(len(individual) - 1):
        total_time += durations[individual[i]][individual[i + 1]]
    if total_time > MAX_TIME:
        return -float('inf'),
    for idx in individual:
        total_priority += points['priority'].iloc[idx]
    return total_priority,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=100)
for gen in range(100):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

best_ind = tools.selBest(population, 1)[0]
best_route = [points['name'].iloc[i] for i in best_ind]
best_priority = evaluate(best_ind)[0]

print("Лучший маршрут:", best_route)
print("Суммарный приоритет:", best_priority)

try:
    m = folium.Map(tiles='cartodb positron', zoom_control=True)

    route_coords = [points[['latitude', 'longitude']].iloc[i].tolist() for i in best_ind]

    all_coords = []

    total_distance_km = 0.0

    for i in range(len(route_coords) - 1):
        start = route_coords[i]
        end = route_coords[i + 1]
        try:
            route = client.directions(
                coordinates=[
                    [start[1], start[0]],
                    [end[1], end[0]]
                ],
                profile=profile,
                format='geojson'
            )
            coords_segment = route['features'][0]['geometry']['coordinates']
            all_coords.extend(coords_segment)
            total_distance_km += route['features'][0]['properties']['summary']['distance'] / 1000

            folium.PolyLine(
                locations=[coord[::-1] for coord in coords_segment],
                color='blue',
                weight=4,
                opacity=0.8
            ).add_to(m)
        except Exception as e:
            print(f"Ошибка маршрута: {e}")

    for i in best_ind:
        row = points.iloc[i]
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"{row['name']} (Приоритет: {row['priority']})",
            icon=folium.Icon(color='purple')
        ).add_to(m)

    if all_coords:
        lats = [c[1] for c in all_coords]
        lons = [c[0] for c in all_coords]
        bounds = [[min(lats) - 0.001, min(lons) - 0.001], [max(lats) + 0.001, max(lons) + 0.001]]
        m.fit_bounds(bounds)
    else:
        m.location = [points['latitude'].mean(), points['longitude'].mean()]
        m.zoom_start = 16

    folium.map.Marker(
        [points['latitude'].mean(), points['longitude'].mean()],
        icon=folium.DivIcon(
            html=f"""<div style="font-size: 14pt; color: black"><b>Общая длина: {total_distance_km:.2f} км</b></div>"""
        )
    ).add_to(m)

    m.save('route_map_kazan.html')
    print("Карта сохранена в route_map_kazan.html")
except Exception as e:
    print(f"Ошибка визуализации: {e}")

