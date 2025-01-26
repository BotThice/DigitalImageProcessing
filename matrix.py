# Function to calculate the determinant of a 4x4 matrix
def determinant(matrix):
    # Base case: if the matrix is 1x1, return the single element
    if len(matrix) == 1:
        return matrix[0][0]

    # Base case: if the matrix is 2x2, return the determinant directly
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case: Expand along the first row
    det = 0
    for col in range(len(matrix[0])):
        det += matrix[0][col] * cofactor(matrix, 0, col)
    return det

# Function to calculate the cofactor of an element at (i, j) position
def cofactor(matrix, row, col):
    minor = get_minor(matrix, row, col)  # Get the minor matrix by removing row and column
    return ((-1) ** (row + col)) * determinant(minor)

# Function to get the minor matrix by excluding row and column
def get_minor(matrix, row, col):
    minor = []
    for r in range(len(matrix)):  # Loop through all rows
        if r != row:  # Skip the row we want to exclude
            # Append all elements in the row except the excluded column
            minor_row = [matrix[r][c] for c in range(len(matrix[r])) if c != col]
            minor.append(minor_row)
    return minor

# Function to calculate the adjugate matrix (transpose of the cofactor matrix)
def adjugate(matrix):
    size = len(matrix)  # Dynamically determine the size of the matrix
    cofactor_matrix = []
    
    # Iterate through the rows and columns to compute cofactors
    for row in range(size):
        cofactor_row = []
        for col in range(size):
            cofactor_row.append(cofactor(matrix, row, col))
        cofactor_matrix.append(cofactor_row)

    # Transpose the cofactor matrix to get the adjugate
    return transpose(cofactor_matrix)


# Function to transpose a matrix
def transpose(matrix):
    return [[matrix[col][row] for col in range(len(matrix))] for row in range(len(matrix))]

# Function to calculate the inverse of a matrix
def inverse(matrix):
    size = len(matrix)  # Get the size of the matrix (NxN)
    det = determinant(matrix)  # Calculate the determinant
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    
    adjugate_matrix = adjugate(matrix)  # Get the adjugate of the matrix
    # Divide each element of the adjugate matrix by the determinant
    inverse_matrix = [[adjugate_matrix[row][col] / det for col in range(size)] for row in range(size)]
    return inverse_matrix


# Function to multiply a matrix by a vector
def multiply_matrix_vector(matrix, vector):
    size = len(matrix)  # Determine the size of the matrix
    result = []
    for row in matrix:
        # Calculate the dot product of the row and the vector
        result.append(sum(row[i] * vector[i] for i in range(size)))
    return result

