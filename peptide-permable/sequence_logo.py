import numpy as np
result = np.load('result.npy')
dictt_inv = {0: 'A', 1: 'C', 2: 'E', 3: 'D', 4: 'G', 5: 'F', 6: 'I', 7: 'H', 8: 'K', 9: 'M',
             10: 'L', 11: 'N', 12: 'Q', 13: 'P', 14: 'S', 15: 'R', 16: 'T', 17: 'W', 18: 'V', 19: 'Y'}
dictt = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4,
         'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9,
         'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14,
         'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19  }
result2 = map(lambda x: map(lambda x : dictt[x],x) , map(lambda x :x , result[::2]))
result2 = np.array(result2)
f1=open('p53.fa','w')

for i in result[::2]:
    
    f1.write('> afsa\n'+i+'\n')
f1.close()
    
die
result =[]
for i in range(13):
    temp = []
    for j in dictt_inv :
        temp += [ (dictt_inv[j],int(100*np.mean(result2[:,i] == j))),]
    result += [temp,]
        
        
        
