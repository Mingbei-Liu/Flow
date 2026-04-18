from pyexpat import model

import cv2
import numpy as np
import pulp

def load_img(impath):
    """
    Loads an image from a specified location and returns it in RGB format.
    Input:
    - impath: a string specifying the target image location.
    Returns an RGB integer image.
    """
    img_loaded = cv2.imread(impath)
    img = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB) 
    
    return img

def fetch_information(img, num_rows, num_cols):

    height, width, _ = img.shape

    row_step = height // num_rows
    col_step = width // num_cols

    row_start = row_step // 2
    col_start = col_step // 2

    information_list = []

    # iterate through the grid and fetch information
    for row_index in range(num_rows):
        for col_index in range(num_cols):
            row_coordinate = row_start + row_index * row_step
            col_coordinate = col_start + col_index * col_step

            # Fetch the RGB values at the calculated coordinates
            rgb_values = img[row_coordinate, col_coordinate]

            # check non-trivial tile
            if np.linalg.norm(rgb_values.astype(float)) > 80:
                matching_color = 0
                for info in information_list:
                    dist = np.linalg.norm(info[0].astype(float) - rgb_values.astype(float))
                    if dist < 50:
                        matching_color = matching_color + 1
                        rgb_values = info[0]
                
                if matching_color == 0:
                    information_list.append([rgb_values.astype(float), [row_index, col_index], "s"])
                elif matching_color == 1:
                    information_list.append([rgb_values.astype(float), [row_index, col_index], "t"])

    return information_list

# create a dictionary from integers to colors
def create_color_dictionary(information):
    color_dictionary = {}
    for info in information:
        if tuple(info[0]) not in color_dictionary:
            color_dictionary[tuple(info[0])] = len(color_dictionary)
    return color_dictionary, len(color_dictionary)

# Now we model the problem as a multi-commodity flow problem and solve it using linear programming.
def create_V_and_E(num_rows, num_cols):
    V = [(i,j,k) for i in range(num_rows) for j in range(num_cols) for k in ["in", "out"]]
    E = []
    for i in range(num_rows):
        for j in range(num_cols):
            # for each tile, we only consider out -> in to avoid double counting
            if i - 1 >= 0:
                E.append(((i, j, "out"), (i-1, j, "in")))
            if i + 1 < num_rows:
                E.append(((i, j, "out"), (i+1, j, "in")))
            if j - 1 >= 0:
                E.append(((i, j, "out"), (i, j-1, "in")))
            if j + 1 < num_cols:
                E.append(((i, j, "out"), (i, j+1, "in")))
            
            E.append(((i, j, "in"), (i, j, "out")))

    return V, E

def get_sources_and_sinks(information, color_dictionary):
    color_to_coordinate_source = {}
    color_to_coordinate_sink = {}
    for info in information:
        color_index = color_dictionary[tuple(info[0])]
        [row_index, col_index] = info[1]

        in_or_out = info[2]
        if in_or_out == "s":
            color_to_coordinate_source[color_index] = (row_index, col_index, "in")
        else:
            color_to_coordinate_sink[color_index] = (row_index, col_index, "out")
    return color_to_coordinate_source, color_to_coordinate_sink

def connectivity_solver(num_colors, information, color_dictionary, print=False):
    model = pulp.LpProblem("3D_Index_Variables", pulp.LpMaximize)

    # Create Sets to index the decision variables
    ROW = [r for r in range(-1, num_rows + 1)]
    COLUMN = [c for c in range(-1, num_cols + 1)]
    COLOR = [color for color in range(num_colors)]
    indices = [(row, column, color) for row in ROW for column in COLUMN for color in COLOR]


    # # Decision variables x[row, column, color] = 1 if the tile at (row, column) is assigned the color, 0 otherwise
    x = pulp.LpVariable.dicts("x", indices, cat="Binary")

    # # Objective function: null objective
    model += 0

    # # Constraint

    # # Each tile must be assigned exactly one color
    model += pulp.lpSum(x[i, j, k] for i in range(num_rows) for j in range(num_cols) for k in COLOR) == num_rows * num_cols

    for i in range(num_rows):
        for j in range(num_cols):
            model += pulp.lpSum(x[i, j, k] for k in COLOR) == 1
            
    # For the tiles that are sources or sinks, we need to assign the correct color
    for info in information:
        color_index = color_dictionary[tuple(info[0])]
        [row_index, col_index] = info[1]
        model += x[row_index, col_index, color_index] == 1

    # Extension tiles should not be colored
    for row in [-1, num_rows]:
        for col in COLUMN:
            for color in COLOR:
                model += x[row, col, color] == 0
    for col in [-1, num_cols]:
        for row in ROW:
            for color in COLOR:
                model += x[row, col, color] == 0

    for row in range(num_rows):
        for col in range(num_cols):
            if [row, col] not in [info[1] for info in information]:
                for color in COLOR:
                    model += x[row-1, col, color] + x[row+1, col, color] + x[row, col-1, color] + x[row, col+1, color] >= 2 * x[row, col, color]
                    model += x[row-1, col, color] + x[row+1, col, color] + x[row, col-1, color] + x[row, col+1, color] <= 4 - 2 * x[row, col, color]
            else: # for sources and sinks
                for color in COLOR:
                    model += x[row-1, col, color] + x[row+1, col, color] + x[row, col-1, color] + x[row, col+1, color] >= x[row, col, color]
                    model += x[row-1, col, color] + x[row+1, col, color] + x[row, col-1, color] + x[row, col+1, color] <= 4 - 3 * x[row, col, color]
    if print:
        model.solve()
    else:
        model.solve(pulp.PULP_CBC_CMD(msg=0))

    return x

def multi_commodity_flow_solver(num_colors, information, color_dictionary, print=False):
    model_mcp = pulp.LpProblem("mcp", pulp.LpMaximize)

    # Create Sets to index the decision variables
    COLOR = [color for color in range(num_colors)]

    VERTICES, EDGES = create_V_and_E(num_rows, num_cols)

    # # Decision variables 
    f = pulp.LpVariable.dicts("x", ((color, u, v) for color in COLOR for (u, v) in EDGES), cat="Binary")

    # # Objective function: null objective
    model_mcp += 0

    # # Constraint

    # Each edge can only be used by at most one color
    for (u, v) in EDGES:
        model_mcp += pulp.lpSum(f[k, u, v] for k in COLOR) <= 1

    # Flow conservation
    for k in COLOR:
        color_to_coordinate_source, color_to_coordinate_sink = get_sources_and_sinks(information, color_dictionary)
        source_vertex = color_to_coordinate_source[k]
        sink_vertex = color_to_coordinate_sink[k]
        for v in VERTICES:
            incoming_flow = pulp.lpSum(f[k, u, v2] for (u, v2) in EDGES if v2 == v)
            outgoing_flow = pulp.lpSum(f[k, v2, w] for (v2, w) in EDGES if v2 == v)

            if v == source_vertex:
                model_mcp += outgoing_flow - incoming_flow == 1
            elif v == sink_vertex:
                model_mcp += incoming_flow - outgoing_flow == 1
            else:
                model_mcp += incoming_flow == outgoing_flow

    if print:
        model_mcp.solve()
    else:
        model_mcp.solve(pulp.PULP_CBC_CMD(msg=0))
    
    return f

def get_solution_x(x, num_colors):
    solution_grids = [np.zeros((num_rows, num_cols), dtype=int) for _ in range(num_colors)]
    for color in range(num_colors):
        for row in range(num_rows):
            for col in range(num_cols):
                if x[row, col, color].varValue == 1:
                    solution_grids[color][row, col] = 1
    return solution_grids

def get_solution_f(f, num_colors):
    solution_grid = [np.zeros((num_rows, num_cols), dtype=int) for _ in range(num_colors)]

    _, EDGES = create_V_and_E(num_rows, num_cols)
    for (u, v) in EDGES:
        for color in range(num_colors):
            if f[color, u, v].varValue == 1:
                row = u[0]
                col = u[1]
                solution_grid[color][row, col] = 1

                row2 = v[0]
                col2 = v[1]
                solution_grid[color][row2, col2] = 1
    return solution_grid

def create_visualization(img, solution_grids, color_dictionary, color, information, half_width=10):

    rgb_to_use = (0, 0, 0)
    for color_tuple, color_index in color_dictionary.items():
        if color_index == color:
            rgb_to_use = color_tuple
            break

    # compute the starting coordinates
    row_start = 0
    col_start = 0

    for info in information:
        if color_dictionary[tuple(info[0])] == color:
            row_start_index = info[1][0]
            col_start_index = info[1][1]
            break

    height, width, _ = img.shape

    row_step = height // num_rows
    col_step = width // num_cols

    row_start = row_step // 2
    col_start = col_step // 2

    row_coordinate = row_start + row_start_index * row_step
    col_coordinate = col_start + col_start_index * col_step

    visited_coordinates = [[row_start_index, col_start_index]]
    reach_the_end = False
    while not reach_the_end:
        current_row, current_col = visited_coordinates[-1]
        if current_row-1 >= 0 and solution_grids[color][current_row-1, current_col] == 1 and [current_row-1, current_col] not in visited_coordinates:
            for col_pixel in range(col_coordinate - half_width, col_coordinate + half_width):
                for row_pixel in range(row_coordinate - row_step - half_width, row_coordinate + half_width):
                    img[row_pixel, col_pixel] = rgb_to_use
            visited_coordinates.append([current_row-1, current_col])

            row_start_index = row_start_index - 1
            col_start_index = col_start_index

            row_coordinate = row_start + row_start_index * row_step
            col_coordinate = col_start + col_start_index * col_step

        elif current_row+1 < num_rows and solution_grids[color][current_row+1, current_col] == 1 and [current_row+1, current_col] not in visited_coordinates:
            for col_pixel in range(col_coordinate - half_width, col_coordinate + half_width):
                for row_pixel in range(row_coordinate - half_width, row_coordinate + row_step + half_width):
                    img[row_pixel, col_pixel] = rgb_to_use
            visited_coordinates.append([current_row+1, current_col])

            row_start_index = row_start_index + 1
            col_start_index = col_start_index

            row_coordinate = row_start + row_start_index * row_step
            col_coordinate = col_start + col_start_index * col_step
        
        elif current_col-1 >= 0 and solution_grids[color][current_row, current_col-1] == 1 and [current_row, current_col-1] not in visited_coordinates:
            for row_pixel in range(row_coordinate - half_width, row_coordinate + half_width):
                for col_pixel in range(col_coordinate - col_step - half_width, col_coordinate + half_width):
                    img[row_pixel, col_pixel] = rgb_to_use
            visited_coordinates.append([current_row, current_col-1])

            row_start_index = row_start_index
            col_start_index = col_start_index - 1

            row_coordinate = row_start + row_start_index * row_step
            col_coordinate = col_start + col_start_index * col_step
        
        elif current_col+1 < num_cols and solution_grids[color][current_row, current_col+1] == 1 and [current_row, current_col+1] not in visited_coordinates:
            for row_pixel in range(row_coordinate - half_width, row_coordinate + half_width):
                for col_pixel in range(col_coordinate - half_width, col_coordinate + col_step + half_width):
                    img[row_pixel, col_pixel] = rgb_to_use
            visited_coordinates.append([current_row, current_col+1])

            row_start_index = row_start_index
            col_start_index = col_start_index + 1

            row_coordinate = row_start + row_start_index * row_step
            col_coordinate = col_start + col_start_index * col_step
        
        else:
            reach_the_end = True
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def create_visualization_for_all_colors(img, solution_grids, color_dictionary, num_colors, information, half_width=10):
    img_copy = img.copy()
    for color in range(num_colors):
        img_copy = create_visualization(img_copy, solution_grids, color_dictionary, color, information, half_width)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)


def create_visualization_for_debug(img, solution_grids, color_dictionary, color, half_width=10):
    rgb_to_use = (0, 0, 0)
    for color_tuple, color_index in color_dictionary.items():
        if color_index == color:
            rgb_to_use = color_tuple
            break


    height, width, _ = img.shape

    row_step = height // num_rows
    col_step = width // num_cols

    row_start = row_step // 2
    col_start = col_step // 2
    
    for row in range(num_rows):
        for col in range(num_cols):
            if solution_grids[color][row, col] == 1:
                row_coordinate = row_start + row * row_step
                col_coordinate = col_start + col * col_step
                for row_pixel in range(row_coordinate - half_width, row_coordinate + half_width):
                    for col_pixel in range(col_coordinate - half_width, col_coordinate + half_width):
                        img[row_pixel, col_pixel] = rgb_to_use
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def create_visualization_for_all_colors_debug(img, solution_grids, color_dictionary, num_colors, half_width=10):
    img_copy = img.copy()
    for color in range(num_colors):
        img_copy = create_visualization_for_debug(img_copy, solution_grids, color_dictionary, color, half_width)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

def pipeline(img, num_rows, num_cols, solver, debug=False):
    information = fetch_information(img, num_rows, num_cols)
    color_dictionary, num_colors = create_color_dictionary(information)
    if solver == "connectivity":
        x = connectivity_solver(num_colors, information, color_dictionary)
        solution_grids = get_solution_x(x, num_colors)
    else:
        f = multi_commodity_flow_solver(num_colors, information, color_dictionary)
        solution_grids = get_solution_f(f, num_colors)

    if debug:
        visualization = create_visualization_for_all_colors_debug(img, solution_grids, color_dictionary, num_colors)
    else:
        visualization = create_visualization_for_all_colors(img, solution_grids, color_dictionary, num_colors, information)

    return visualization


if __name__ == "__main__":

    img = load_img("example_5.png")
    num_rows = 14
    num_cols = 14
    solver = "connectivity"
    solver = "mcp"

    visualization = pipeline(img, num_rows, num_cols, solver, debug=False)
    cv2.imshow("Visualization", visualization)
    cv2.destroyAllWindows()
    cv2.imwrite("visualization.png", visualization)