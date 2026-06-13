import numpy as np

# Multi-Feature Linear Regression
# True relationship: y = 2*x1 + 3*x2 + 1

X = np.array([
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 2],
    [5, 4]
])

y = np.array([9, 8, 16, 15, 23])

w = np.zeros(X.shape[1]) 
b = 0 

lr = 0.001 

epoch = 10000

for _ in range(epoch): 
    y_pred = X@w + b 
    
    error = y_pred - y 
    
    L = np.mean(error ** 2) 
    
    print(L) 
    
    dLw = 2 * (X.T @ error)/len(X) 
    dLb = 2 * np.mean(error) 
    
    w -= lr*dLw 
    b -= lr*dLb 
    
print(w,b)