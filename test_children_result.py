import numpy as np
from entropy import *

full_mat = np.array([[-40., -56, -53, -50, -71, -79, -78,   2.],
 [-39., -56., -54., -48., -61., -78, -67,   2.],
 [-40., -50., -49., -51., -65., -75., -80., 2.],
 [-42., -59., -57. -51., -62., -72. -76.,   2.],
 [-40., -59., -67. -50., -68., -79. -78.,   2.],
 [-40., -63., -64. -50., -64., -75. -78.,   2.],
 [-41., -60., -65. -51., -72., -74. -78.,   2.],
 [-40., -58., -60. -49., -72., -74. -80.,   2.],
 [-39., -56., -53. -49., -70., -80. -75.,   2.],
 [-40., -50., -52. -48., -68., -73. -71.,   2.],
 [-38., -52., -59. -48., -67., -64. -75.,   2.],
 [-42., -54., -55. -49., -67., -80. -74.,   2.],
 [-40., -55., -57. -50., -66., -75. -73.,   2.],
 [-39., -53., -56. -48., -66., -69. -72.,   2.],
 [-39., -55., -54. -48., -76., -73. -80.,   2.],
 [-44., -53., -55. -48., -66., -79. -86.,   3.],
 [-43., -55., -53. -52., -60., -78. -86.,   3.],
 [-44., -56., -54. -52., -63., -80. -79.,   3.],
 [-44., -53., -51. -49., -60., -84. -76.,   3.],
 [-43., -57., -51. -52., -62., -79. -77.,   3.],
 [-42., -54., -48. -51., -64., -78. -87.,   3.],
 [-43., -59., -51. -53., -61., -83. -85.,   3.],
 [-44., -60., -52. -49., -63., -84. -84.,   3.]])

print(full_mat[0])
print("HEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHERE full mat shape: ", full_mat.shape)
idx, val = FIND_SPLIT(full_mat)
