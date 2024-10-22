import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
dictt = {'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4,
         'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9,
         'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14,
         'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19}

embed = np.array([[ 0.24781914, -0.36492938, -0.33924493, -0.48009107,  0.38597232],
       [-0.01652357, -0.31683582,  0.27610564,  0.0731108 , -0.469511  ],
       [-0.47544011,  0.35936701,  0.21910372, -0.45328   ,  0.28942379],
       [-0.19854289, -0.26782432,  0.42737797,  0.06665558, -0.01429989],
       [ 0.0657359 ,  0.00721902, -0.09996129, -0.4750526 , -0.0737937 ],
       [-0.03717221,  0.05577235, -0.42168373,  0.02083148, -0.42876461],
       [ 0.38245645,  0.12553595,  0.11138345,  0.36939815, -0.44924894],
       [ 0.26696789, -0.12857735, -0.19383454, -0.19599453, -0.25884125],
       [-0.01401186, -0.30554837,  0.01887017,  0.23985837,  0.20027128],
       [ 0.31731802, -0.37954986,  0.07035158,  0.28514996,  0.24399374],
       [ 0.21597929,  0.4787277 , -0.38458264,  0.00906388, -0.1122031 ],
       [-0.33503053, -0.10413442, -0.37570059,  0.38133726, -0.36162487],
       [-0.08491378, -0.11265747,  0.42205328,  0.38034973, -0.12761126],
       [ 0.42891988, -0.00869302,  0.46425703, -0.00055449,  0.06290384],
       [ 0.46308139, -0.24142328,  0.15310085, -0.37685007,  0.35865647],
       [-0.32831785,  0.41220483,  0.20048748, -0.39061341,  0.09434988],
       [ 0.37088618, -0.39018646, -0.26447955, -0.17217176,  0.22806348],
       [ 0.43330327, -0.34537801,  0.05232524,  0.03655756,  0.24791999],
       [-0.17752989,  0.37797493, -0.13715999,  0.33108899,  0.29942638],
       [ 0.28105152,  0.10265959,  0.41368064,  0.48550001, -0.17804025]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
model = PCA()
final = model.fit_transform(embed)
names = [i for i in sorted(dictt)]
ax.plot(final[:,0],final[:,1],final[:,2],'o');
for i in range(0,20):
    ax.text(final[i,0],final[i,1],final[i,2], names[i]);
plt.show()
