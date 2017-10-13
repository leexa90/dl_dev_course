f1= open('out','r')
import numpy as np
dictt ={}
counter = 0
even_odd = 1
for line in f1:
    if line != '\n' and '145\n' in line:
        counter  += 1
        dictt[counter] = []
    if line != '\n' and '145\n' not in line:
        if even_odd % 2 == 1:
            dictt[counter] += [map(np.float,line.split()),]
        even_odd += 1
            
for i in range(1,10):
    max_cv = np.argsort(map( lambda x :x[1] , dictt[i]))[-3:]
    for j in max_cv:
        print dictt[i][j],
    print np.mean(max_cv),max_cv,'\n'
