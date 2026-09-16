import numpy as np

def similitud_coseno(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    q = np.array([8, 3])
    doc_a = np.array([9, 2])
    doc_b = np.array([2, 9])
    doc_c = np.array([7, 5])

    print("Q vs A:", similitud_coseno(q, doc_a))
    print("Q vs B:", similitud_coseno(q, doc_b))
    print("Q vs C:", similitud_coseno(q, doc_c))
