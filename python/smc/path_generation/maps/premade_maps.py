from smc.path_generation.starworlds.obstacles import StarshapedPolygon


def createSampleStaticMap():
    """
    createMap
    ---------
    return obstacles that define the 2D map
    """
    # [lower_left, lower_right, top_right, top_left]
    # map_as_list = [
    #    [[2, 2], [8, 2], [8, 3], [2, 3]],
    #    [[2, 3], [3, 3], [3, 4.25], [2, 4.25]],
    #    [[2, 5], [8, 5], [8, 6], [2, 6]],
    #    [[2, 8], [8, 8], [8, 9], [2, 9]],
    # ]
    map_as_list = [
        [[0.5, -1.0], [3.5, -1.0], [3.5, -0.5], [0.5, -0.5]],
        [[3.5, -0.5], [4.0, -0.5], [4.0, 3.5], [3.5, 3.5]],
        [[-1.0, 1.0], [2.5, 1.0], [2.5, 2.0], [-1.0, 2.0]],
        [[0.0, 3.5], [4.0, 3.5], [4.0, 4.0], [0.0, 4.0]],
        # [[-1.5, -1.0], [-1.0, -1.0], [-1.0, 5.0], [-1.5, 5.0]],
    ]

    obstacles = []
    for map_element in map_as_list:
        obstacles.append(StarshapedPolygon(map_element))
    return obstacles, map_as_list
