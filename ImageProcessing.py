import queue


def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            r = pixel_array_r[i][j]
            g = pixel_array_g[i][j]
            b = pixel_array_b[i][j]
            gray = round(0.299 * r + 0.587 * g + 0.114 * b)
            greyscale_pixel_array[i][j] = gray
    return greyscale_pixel_array


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    (maxVal, minVal) = computeMinAndMaxValues(pixel_array, image_width, image_height)
    array = createInitializedGreyscalePixelArray(image_width, image_height)
    if maxVal - minVal == 0:
        return array
    for i in range(image_height):
        for j in range(image_width):
            out = (pixel_array[i][j] - minVal) * 255 / (maxVal - minVal)
            if out < 0:
                array[i][j] = 0
            elif out > 255:
                array[i][j] = 255
            else:
                array[i][j] = round(out)
    return array

def computeHistogram(pixel_array, image_width, image_height, nr_bins):
    histogram = [ 0.0 for x in range(nr_bins)]
    for array in pixel_array:
        for num in array:
            histogram[num] += 1.0
    return histogram

def computeMinAndMaxValues(pixel_array, image_width, image_height):
    minValue = pixel_array[0][0]
    maxValue = pixel_array[0][0]
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < minValue:
                minValue = pixel_array[i][j]
            elif pixel_array[i][j] > maxValue:
                maxValue = pixel_array[i][j]
    return (maxValue, minValue)


def computeStandardDeviationImage3x3(pixel_array, image_width, image_height):
    rows = len(pixel_array)
    cols = len(pixel_array[0])
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            total = []
            # calculate left side, 1 index to the left.
            for a in range(-2, 3):
                for b in range(-2, 3):
                    p_r_i = row + a
                    p_c_i = col + b
                    if (p_r_i >= 0 and p_r_i < image_height and p_c_i >= 0 and p_c_i < image_width):
                        total.append(pixel_array[p_r_i][p_c_i])
            if len(total) > 0:
                mean = sum(total) / len(total)
                dif = [(value - mean) ** 2 for value in total]
                std = (sum(dif) / len(dif)) ** 0.5
            else:
                std = 0
            new_array[row][col] = std
    return new_array


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] >= threshold_value:
                array[i][j] = 255
            else:
                array[i][j] = 0
    return array


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    final_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for row in range(1, image_height - 1):
        for col in range(1, image_width - 1):
            isOne = True
            for i in range(-1, 2):
                for j in range(-1, 2):
                    x = row + i
                    y = col + j
                    if x < 0 or y < 0 or x >= image_height or y >= image_width:
                        continue
                    if pixel_array[x][y] == 0:
                        isOne = False
            if isOne == True:
                final_array[row][col] = 1
    return final_array


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    final_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for row in range(image_height):
        for col in range(image_width):
            isOne = False
            for i in range(-1, 2):
                for j in range(-1, 2):
                    x = row + i
                    y = col + j
                    if x < 0 or y < 0 or x >= image_height or y >= image_width:
                        continue
                    elif pixel_array[x][y] >= 1:
                        isOne = True
            if isOne:
                final_array[row][col] = 1
    return final_array


def nErosions(pixel_array, n, image_width, image_height):
    for i in range(n):
        pixel_array = computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height)
    return pixel_array


def nDilations(pixel_array, n, image_width, image_height):
    for i in range(n):
        pixel_array = computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height)
    return pixel_array


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    class_num = 1
    visited_cells = {}
    classes = {}
    new_image = createInitializedGreyscalePixelArray(image_width, image_height)
    for row in range(image_height):
        for col in range(image_width):
            q = queue.Queue()
            if visited_cells.get((row, col)) is None and pixel_array[row][col] != 0:
                q.put((row, col))
                classes[class_num] = 0
                while q.empty() == False:
                    cur = q.get()
                    c_row = cur[0]
                    c_col = cur[1]
                    visited_cells[(c_row, c_col)] = True
                    new_image[c_row][c_col] = class_num
                    classes[class_num] = classes[class_num] + 1
                    neighbourhood = [(c_row + 1, c_col), (c_row - 1, c_col), (c_row, c_col + 1), (c_row, c_col - 1)]
                    for node in neighbourhood:
                        if node[0] >= 0 and node[0] < image_height and node[1] >= 0 and node[1] < image_width and \
                                pixel_array[node[0]][node[1]] != 0 and visited_cells.get(node) is None:
                            q.put(node)
                        visited_cells[node] = True
                class_num += 1
            visited_cells[(row, col)] = True

    return (new_image, classes)


def computeLargestComponent(pixel_array, classes, image_width, image_height):
    # every key will have an array of 4 ints
    # [min x, min y, max x, max y]
    components = {key: [float('inf'), float('inf'), -float('inf'), -float('inf')] for key in classes.keys()}
    for row in range(image_height):
        for col in range(image_width):
            if pixel_array[row][col] != 0:
                c = pixel_array[row][col]
                if components[c][0] > col:
                    components[c][0] = col
                elif components[c][2] < col:
                    components[c][2] = col
                if components[c][1] > row:
                    components[c][1] = row
                elif components[c][3] < row:
                    components[c][3] = row
    for dimensions in components.keys():
        coords = components[dimensions]
        x = coords[2] - coords[0]
        y = coords[3] - coords[1]
        if x == 0 or y == 0:
            classes.pop(dimensions)
        else:
            ratio = x / y
            if ratio < 1.5 or ratio > 5:
                classes.pop(dimensions)

    max_class = None
    for c in classes:
        if max_class is None:
            max_class = c
        else:
            if classes[c] > classes[max_class]:
                max_class = c
    return components[max_class]
