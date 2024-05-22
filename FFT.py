from pyspark import SparkConf, SparkContext
from pyspark.storagelevel import StorageLevel
import math
import sys
import random
import numpy as np
import time
from collections import defaultdict
import pyspark
from pyspark.sql import DataFrame, SparkSession

import findspark
findspark.init()
findspark.find()

def euc_distance(p1, p2):
    return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5


def determine_grid_cell(point, distance):
    lambda_val = distance / (2 * math.sqrt(2))
    return (int(point[0] // lambda_val), int(point[1] // lambda_val))


def MRApproxOutliers(points_rdd, D, M):
    cell_counts = points_rdd.map(lambda point: (determine_grid_cell(point, D), 1)) \
        .reduceByKey(lambda a, b: a + b)
    cell_map = cell_counts.collectAsMap()

    map_region = defaultdict(int)

    for (cell, count) in cell_map.items():
        neighbors = [(cell[0] + dx, cell[1] + dy) for dx in range(-3, 4) for dy in range(-3, 4)]
        region_count = sum(cell_map.get(neighbor, 0) for neighbor in neighbors if
                           abs(cell[0] - neighbor[0]) <= 1 and abs(cell[1] - neighbor[1]) <= 1)
        border_count = sum(cell_map.get(neighbor, 0) for neighbor in neighbors)
        map_region[cell] = (count, region_count, border_count)

    sure_outliers = sum(count for count, region, border in map_region.values() if border <= M)
    uncertain_points = sum(count for count, region, border in map_region.values() if region <= M and border > M)

    print("Number of sure outliers =", sure_outliers)
    print("Number of uncertain points =", uncertain_points)


def SequentialFFT(points, K):
    if len(points) < K:
        raise ValueError("K cannot be greater than the number of unique points")
    if len(points) == 0:
        return []

    points_distances = [float('inf')] * len(points)
    centers = []
    first_center = random.choice(points)
    centers.append(first_center)
    first_center_index = points.index(first_center)
    points_distances[first_center_index] = 0  # distance to itself is zero

    for _ in range(1, K):
        max_distance = -1
        farthest_point = None
        for j in range(len(points)):
            distance = euc_distance(points[j], centers[-1])
            if distance < points_distances[j]:
                points_distances[j] = distance
            if points_distances[j] > max_distance:
                max_distance = points_distances[j]
                farthest_point = j
        if farthest_point is not None:
            centers.append(points[farthest_point])
        else:
            break  # No valid point to add, break out early

    return centers


def MRFFT(points_rdd, K):
    start_time = time.time()

    coresets = points_rdd.mapPartitions(lambda partition: SequentialFFT(list(partition), K))
    k_center = coresets.collect()

    round1_time = time.time() - start_time
    print(f"Running time of MRFFT Round 1 = {round1_time:.2f} seconds")

    # Round 2: Use SequentialFFT on the coreset to find K centers
    start_time = time.time()
    final_centers = SequentialFFT(k_center, K)

    round2_time = time.time() - start_time
    print(f"Running time of MRFFT Round 2 = {round2_time:.2f} seconds")

    # Round 3: Calculate the clustering radius
    start_time = time.time()
    radius = points_rdd.map(lambda point: min(euc_distance(point, center) for center in final_centers)).reduce(max)
    round3_time = time.time() - start_time
    print(f"Running time of MRFFT Round 3 = {round3_time:.2f} seconds")
    return radius


def setup_spark_session(app_name):
    """Setup and return Spark session and context."""
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():
    if len(sys.argv) != 5:
        print("Usage: <script> <file_path> M K L")
        sys.exit(1)

    file_path, M, K, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    print(f"File {file_path} M={M} K={K} L={L}")

    spark, sc = setup_spark_session("Outlier Detection")

    rawData = sc.textFile(file_path).repartition(L)
    inputPoints = rawData.map(lambda line: tuple(map(float, line.split(',')))).cache()

    num_points = inputPoints.count()
    print(f"Number of points = {num_points}")

    D = MRFFT(inputPoints, K)
    print(f"Radius = {D}")

    stopwatch_start = time.time()
    MRApproxOutliers(inputPoints, D, M)
    print(f"Running time of MRApproxOutliers = {time.time() - stopwatch_start:.2f} ms")

    sc.stop()


if __name__ == "__main__":
    main()
