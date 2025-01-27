import numpy as np


def rotation_matrix_90():
    """90-degree counterclockwise rotation matrix about the vertical axis."""
    return np.array([[0, 1, 0],
                     [-1, 0, 0],
                     [0, 0, 1]])

def rotation_matrix_180():
    """180-degree rotation matrix about the vertical axis."""
    return np.array([[-1, 0, 0],
                     [0, -1, 0],
                     [0, 0, 1]])


def transform_coordinates(input_vector, rotation_matrix):

    input_vector = input_vector.reshape(4, 3)  
    transformed = np.dot(rotation_matrix, input_vector.T).T  
    return transformed.flatten()

def ned_to_seu(vector):
    return transform_coordinates(vector, rotation_matrix_90())

def seu_to_ned(vector):
    return transform_coordinates(vector, rotation_matrix_90().T)

def ned_to_enu(vector):
    return transform_coordinates(vector, rotation_matrix_90().T)

def enu_to_ned(vector):
    return transform_coordinates(vector, rotation_matrix_90())

def ned_to_bf(vector):
    return transform_coordinates(vector, rotation_matrix_180())

def bf_to_ned(vector):
    return transform_coordinates(vector, rotation_matrix_180().T)

def seu_to_enu(vector):
    return ned_to_enu(seu_to_ned(vector))

def seu_to_bf(vector):
    return ned_to_bf(seu_to_ned(vector))

def bf_to_seu(vector):
    return ned_to_seu(bf_to_ned(vector))

def enu_to_bf(vector):
    return ned_to_bf(enu_to_ned(vector))

def bf_to_enu(vector):
    return ned_to_enu(bf_to_ned(vector))

def enu_to_seu(vector):
    return ned_to_seu(enu_to_ned(vector))


def test_transformations():

    vector = np.random.rand(12) * 100  

    assert np.allclose(vector, enu_to_ned(ned_to_enu(vector))), "NED -> ENU -> NED failed!"
    assert np.allclose(vector, seu_to_ned(ned_to_seu(vector))), "NED -> SEU -> NED failed!"
    assert np.allclose(vector, seu_to_enu(enu_to_seu(vector))), "ENU -> SEU -> ENU failed!"
    assert np.allclose(vector, bf_to_enu(enu_to_bf(vector))), "ENU -> BF -> ENU failed!"
    assert np.allclose(vector, bf_to_seu(seu_to_bf(vector))), "SEU -> BF -> SEU failed!"

    print("All tests passed!")

if __name__ == "__main__":
    
    
    for i in range(5):
        vector = np.random.rand(12) * 100  

        
        transformations = {
            "NED to SEU": ned_to_seu,
            "SEU to NED": seu_to_ned,
            "NED to ENU": ned_to_enu,
            "ENU to NED": enu_to_ned,
            "SEU to ENU": seu_to_enu,
            "ENU to SEU": enu_to_seu,
            "NED to BF": ned_to_bf,
            "BF to NED": bf_to_ned,
            "SEU to BF": seu_to_bf,
            "BF to SEU": bf_to_seu,
            "ENU to BF": enu_to_bf,
            "BF to ENU": bf_to_enu,
        }

        
        def apply_transformations(vector, transform_sequence):
            result = vector
            for transform in transform_sequence:
                result = transformations[transform](result)  
            return result

        
        test_sequences = [
            ["NED to SEU", "SEU to NED"],                       
            ["NED to ENU", "ENU to NED"],                       
            ["NED to SEU", "SEU to ENU", "ENU to NED"],         
            ["NED to BF", "BF to SEU", "SEU to ENU", "ENU to NED"],  
            ["SEU to BF", "BF to NED", "NED to ENU", "ENU to SEU"],  
            ["NED to SEU", "SEU to NED", "NED to ENU", "ENU to BF", "BF to SEU", "SEU to BF"],  
        ]

        
        for seq in test_sequences:
            transformed_vector = apply_transformations(vector, seq)
            reverse_sequence = seq[::-1]  
            restored_vector = apply_transformations(transformed_vector, reverse_sequence)

            
            if np.allclose(vector, restored_vector):
                print(f"Pass: {' -> '.join(seq)} -> {' -> '.join(reverse_sequence)}")
            else:
                print(f"Fail: {' -> '.join(seq)} -> {' -> '.join(reverse_sequence)}")