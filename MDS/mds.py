import numpy as np

# calculate gradient at a specific point
def gradient_at_k_l(X, S, S_star, T_star, d, d_prime, k, l):
    g_kl = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i == j:
                continue
            if k == i:
                delta_ki = 1
            else:
                delta_ki = 0
            if k == j:
                delta_kj = 1
            else:
                delta_kj = 0
            g_kl += (
                (delta_ki - delta_kj)
                * (((d[i, j] - d_prime[i, j]) / S_star) - (d[i, j] / T_star))
                * ((X[i, l] - X[j, l]) / d[i, j])
            )
    g_kl *= S
    return g_kl

# calculate gradient
def gradient(X, d, d_prime):
    S_star = sum(sum((d - d_prime) ** 2))
    T_star = sum(sum(d**2))
    S = np.sqrt(S_star / T_star)
    gradient_matrix = np.zeros_like(X)
    for k in range(gradient_matrix.shape[0]):
        for l in range(gradient_matrix.shape[1]):
            gradient_matrix[k, l] = gradient_at_k_l(
                X, S, S_star, T_star, d, d_prime, k, l
            )
    return gradient_matrix

# get distances matrix
def calculate_distances(data):
    distances = np.zeros((len(data), len(data)))

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distances[i, j] = np.linalg.norm(np.subtract(data[i], data[j]))
            distances[j, i] = np.linalg.norm(np.subtract(data[i], data[j]))

    return distances

# function to calculate stress
def calculate_stress(new_distances, org_distances):
    return np.sqrt(
        (sum(sum((new_distances - org_distances) ** 2)))
        / (sum(sum(org_distances**2)))
    )

# Multidimensional Scaling
def MDS(data, dim = None):
    original_distances = calculate_distances(data)
    # start with one dimension
    stress_array = [np.inf]
    
    if dim != None:
        # random config to begin with
        low_dim_data = np.random.random((original_distances.shape[0], dim))

        # use gradient descent algo to find the next configuration
        ## p_n+1 = p_n - (learning_rate) * (gradient of loss @ p_n)

        stress_array.append(
            calculate_stress(
                new_distances=calculate_distances(data=low_dim_data),
                org_distances=original_distances,
            )
        )

        learning_rate = 0.001

        p_n = low_dim_data

        for i in range(1000):
            g = gradient(
                X=p_n, d=original_distances, d_prime=calculate_distances(data=p_n)
            )
            mag_g = np.sqrt(sum(sum(g**2)) / sum(sum(p_n**2)))
            p_n += learning_rate * g / mag_g
            stress_array.append(
                calculate_stress(
                    new_distances=calculate_distances(data=p_n),
                    org_distances=original_distances,
                )
            )

        low_dim_features = p_n
    else: 
        dim = 0
        low_dim_features = None
        while stress_array[-1] > 0.1:
            # increase dimension
            dim += 1
            # random config to begin with
            low_dim_data = np.random.random((original_distances.shape[0], dim))

            # use gradient descent algo to find the next configuration
            ## p_n+1 = p_n - (learning_rate) * (gradient of loss @ p_n)

            stress_array.append(
                calculate_stress(
                    new_distances=calculate_distances(data=low_dim_data),
                    org_distances=original_distances,
                )
            )

            learning_rate = 0.001

            p_n = low_dim_data

            for i in range(1000):
                g = gradient(
                    X=p_n, d=original_distances, d_prime=calculate_distances(data=p_n)
                )
                mag_g = np.sqrt(sum(sum(g**2)) / sum(sum(p_n**2)))
                p_n += learning_rate * g / mag_g
                stress_array.append(
                    calculate_stress(
                        new_distances=calculate_distances(data=p_n),
                        org_distances=original_distances,
                    )
                )

            low_dim_features = p_n
    return (dim, stress_array, low_dim_features)