import json
import numpy as np
import math

def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)
def get_cossimi(x,y):
    myx=x
    myy=y
    cos1=np.sum(myx*myy)
    cos21=np.sqrt(sum(myy*myy))
    cos22=np.sqrt(sum(myx*myx))
    return (cos1/float(cos22*cos21))


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


if __name__ == "__main__":
    f = open('vec_all.txt','r')
    vecs = f.readlines()
    result = []
    for i in range(103):
        result.append(i)

    for vec in vecs[1:]:
        temp = vec.replace("\n","").split(" ")
        temp_vec = []
        for i in temp[1:]:
            temp_vec.append(float(i))
        result[int(temp[0])] = temp_vec

    sim = []
    b = np.array(result)
    for i in range(103):
        temp = []
        for j in range(103):
            temp.append( get_cossimi(b[i],b[j]) )
        if i == j:
            temp[i]=1
        sim.append(temp)

    with open("sim.json",'w') as f:
        j = json.dump(sim,f)

