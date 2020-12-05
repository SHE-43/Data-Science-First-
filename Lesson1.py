import numpy as np
import math
import statistics;
from scipy.cluster.vq import whiten, vq, kmeans;

list1 = [1,2,3,4];
array1 = np.array(list1);
print(array1);

ones = np.ones((2,3));
print("ones");
print(ones);

np_arange = np.arange(7);
print("arange");
print(np_arange);

np_arange = np.arange(2,10, dtype = np.float);
print("arangeWithDtype");
print(np_arange);
print("Dtype : ", np_arange.dtype);

np_linspace = np.linspace(1,4,6);
print("np_linspace");
print(np_linspace);

np_matrix = np.matrix("1 2;3 4");
print("np_matrix");
print(np_matrix);

print("np_matrix_conjugate/transpose");
print(np_matrix.H);

print("np_matrix_conjugate/transpose");
print(np_matrix.T);

rand1 = np.random.rand(10,3) # Number of values, number of rows.
array1 = np.array([.9,.9,.9]);

print("\n\n\nClustering");


print("\n\n");
data = np.vstack((rand1 + array1, np.random.rand(10,3)));
# print(data)
print("\n");
# print(whiten(data))
list1 = [1,4,2,3,4,6,6,1,1];
centroid = kmeans(list1,3);
print(whiten(list1));
print(kmeans(list1, 3));
print(vq(centroid));



