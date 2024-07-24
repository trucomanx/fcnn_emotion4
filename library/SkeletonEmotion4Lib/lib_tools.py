import numpy as np

from random import uniform


def vector_normalize_coordinates(vector):
    # Verifica se o tamanho do vetor é correto
    if len(vector) % 3 != 0:
        raise ValueError("O vetor deve ter um tamanho múltiplo de 3")
    
    # Extrai as coordenadas x, y e as probabilidades p
    x_coords = vector[0::3];
    y_coords = vector[1::3];
    p_values = vector[2::3];
    
    # Filtra as coordenadas com base nos valores de confiança
    valid_indices = p_values > 0;
    x_coords_valid = x_coords[valid_indices];
    y_coords_valid = y_coords[valid_indices];
    
    # Calcula o ponto central apenas com as coordenadas válidas
    x_center = np.mean(x_coords_valid);
    y_center = np.mean(y_coords_valid);

    # Calcula o std apenas com as coordenadas válidas
    x_std = np.std(x_coords_valid - x_center);
    y_std = np.std(y_coords_valid - y_center);
      
    # Centraliza as coordenadas válidas o zero en no validas
    x_coords_centered = np.where(valid_indices, (x_coords - x_center)/x_std, 0);
    y_coords_centered = np.where(valid_indices, (y_coords - y_center)/y_std, 0);
    
    # Recria o vetor centralizado
    centered_vector = np.empty_like(vector);
    centered_vector[0::3] = x_coords_centered;
    centered_vector[1::3] = y_coords_centered;
    centered_vector[2::3] = p_values;
    
    return centered_vector;


def vector_randn_noise(vector,sigma=0.01):
    # Aplique suas técnicas de aumento de dados aqui
    # Verifica se o tamanho do vetor é correto
    if len(vector) % 3 != 0:
        raise ValueError("O vetor deve ter um tamanho múltiplo de 3")
    
    # Extrai as coordenadas x, y e as probabilidades p
    x_coords = vector[0::3];
    y_coords = vector[1::3];
    p_values = vector[2::3];
    
    # Filtra as coordenadas com base nos valores de confiança
    valid_indices = p_values > 0;
    
    noise_x = np.random.normal(0, sigma, x_coords.shape);
    noise_y = np.random.normal(0, sigma, y_coords.shape);
    
    # Centraliza as coordenadas válidas o zero en no validas
    x_coords_noise = np.where(valid_indices, x_coords + noise_x, 0);
    y_coords_noise = np.where(valid_indices, y_coords + noise_y, 0);
    
    # Recria o vetor centralizado
    noise_vector = np.empty_like(vector)
    noise_vector[0::3] = x_coords_noise;
    noise_vector[1::3] = y_coords_noise;
    noise_vector[2::3] = p_values;
    
    return noise_vector;



def vector_random_rotate_coordinates(vector, angle=15):
    ang=uniform(-angle,angle);
    
    # Verifica se o tamanho do vetor é correto
    if len(vector) % 3 != 0:
        raise ValueError("O vetor deve ter um tamanho múltiplo de 3");
    
    # Converte o ângulo para radianos
    theta = np.radians(ang);
    
    # Define a matriz de rotação
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Extrai as coordenadas x, y e as probabilidades p
    x_coords = vector[0::3];
    y_coords = vector[1::3];
    p_values = vector[2::3];
    
    # Filtra as coordenadas com base nos valores de confiança
    valid_indices = p_values > 0;
    
    # Combina x e y em uma matriz de coordenadas
    coords = np.vstack((x_coords, y_coords));
    
    # Aplica a rotação apenas às coordenadas válidas
    rotated_coords = rotation_matrix @ coords;
    
    # Zera as coordenadas com confiança <= 0
    rotated_coords[:, ~valid_indices] = 0;
    
    # Recria o vetor rotacionado
    rotated_vector = np.empty_like(vector);
    rotated_vector[0::3] = rotated_coords[0, :];
    rotated_vector[1::3] = rotated_coords[1, :];
    rotated_vector[2::3] = p_values;
    
    return rotated_vector;


'''
vector = np.array([
    1, 2, 0.9, 4, 5, 0.8, 7, 8, 0.7, 
    10, 11, 0.6, 13, 14, 0.5, 16, 17, 0.4, 
    19, 20, 0.3, 22, 23, 0.2, 25, 26, 0.1, 
    28, 29, 0.0, 31, 32, 0.95, 34, 35, 0.85, 
    37, 38, 0.75, 40, 41, 0.65, 43, 44, 0.55, 
    46, 47, 0.45, 49, 50, 0.35, 52, 53, 0.25, 
    55, 56, 0.15
])

centered_vector = centralize_coordinates(vector)
print(centered_vector)

'''

################################################################################
################################################################################

def process_matrix(X, func):
    '''
    X: Matrix of type numpy array.
    func: Function that receive a numpy vector and return numpy vector.
    '''
    
    # Inicializa uma nova matriz X2 com o mesmo tamanho que X
    X2 = np.zeros_like(X)
    
    # Itera sobre cada linha de X
    for i in range(X.shape[0]):
        X2[i, :] = func(X[i, :])
    
    return X2

################################################################################
################################################################################


def batch_normalize_coordinates(X):
    #Normaliza the coordinates
    return process_matrix(X, vector_normalize_coordinates);

def batch_random_rotate_coordinates(X,angle=15):
    #Rotate random the coordinates by a angle
    return process_matrix(X, lambda vector: vector_random_rotate_coordinates(vector,angle=angle));

def batch_randn_noise(X,sigma=0.01):
    #Rotate random the coordinates by a angle
    return process_matrix(X, lambda vector: vector_randn_noise(vector,sigma=sigma));





