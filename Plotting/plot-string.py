import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

## Local Unstructured L1 Pruning

# Example dataset, replace this part with your own data.
data_u_l_l1 = """
Module: conv1.conv, Pruning Rate: 0.1, Accuracy: 0.69778
Module: conv1.conv, Pruning Rate: 0.2, Accuracy: 0.69786
Module: conv1.conv, Pruning Rate: 0.3, Accuracy: 0.6963
Module: conv1.conv, Pruning Rate: 0.4, Accuracy: 0.6929
Module: conv1.conv, Pruning Rate: 0.5, Accuracy: 0.68982
Module: conv1.conv, Pruning Rate: 0.6, Accuracy: 0.67466
Module: conv1.conv, Pruning Rate: 0.7, Accuracy: 0.61572
Module: conv1.conv, Pruning Rate: 0.8, Accuracy: 0.39132
Module: conv1.conv, Pruning Rate: 0.9, Accuracy: 0.08232
Module: conv2.conv, Pruning Rate: 0.1, Accuracy: 0.69802
Module: conv2.conv, Pruning Rate: 0.2, Accuracy: 0.696
Module: conv2.conv, Pruning Rate: 0.3, Accuracy: 0.69328
Module: conv2.conv, Pruning Rate: 0.4, Accuracy: 0.6819
Module: conv2.conv, Pruning Rate: 0.5, Accuracy: 0.65486
Module: conv2.conv, Pruning Rate: 0.6, Accuracy: 0.6376
Module: conv2.conv, Pruning Rate: 0.7, Accuracy: 0.4767
Module: conv2.conv, Pruning Rate: 0.8, Accuracy: 0.418
Module: conv2.conv, Pruning Rate: 0.9, Accuracy: 0.09918
Module: conv3.conv, Pruning Rate: 0.1, Accuracy: 0.6976
Module: conv3.conv, Pruning Rate: 0.2, Accuracy: 0.6971
Module: conv3.conv, Pruning Rate: 0.3, Accuracy: 0.69636
Module: conv3.conv, Pruning Rate: 0.4, Accuracy: 0.69358
Module: conv3.conv, Pruning Rate: 0.5, Accuracy: 0.69088
Module: conv3.conv, Pruning Rate: 0.6, Accuracy: 0.68452
Module: conv3.conv, Pruning Rate: 0.7, Accuracy: 0.67132
Module: conv3.conv, Pruning Rate: 0.8, Accuracy: 0.56178
Module: conv3.conv, Pruning Rate: 0.9, Accuracy: 0.25668
Module: inception3a.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.69742
Module: inception3a.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.69738
Module: inception3a.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.69632
Module: inception3a.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.69398
Module: inception3a.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.68562
Module: inception3a.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.65608
Module: inception3a.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.55476
Module: inception3a.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.35338
Module: inception3a.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.2093
Module: inception3a.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.69792
Module: inception3a.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.69788
Module: inception3a.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.69684
Module: inception3a.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.69564
Module: inception3a.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.69404
Module: inception3a.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.68538
Module: inception3a.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.6658
Module: inception3a.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.6129
Module: inception3a.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.4225
Module: inception3a.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69764
Module: inception3a.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.6978
Module: inception3a.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.69744
Module: inception3a.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.69764
Module: inception3a.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.69684
Module: inception3a.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.69602
Module: inception3a.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.69192
Module: inception3a.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.67988
Module: inception3a.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.66012
Module: inception3a.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.69772
Module: inception3a.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69768
Module: inception3a.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69784
Module: inception3a.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.69712
Module: inception3a.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.69734
Module: inception3a.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.69714
Module: inception3a.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.69304
Module: inception3a.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.67758
Module: inception3a.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.64828
Module: inception3a.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.6977
Module: inception3a.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.69794
Module: inception3a.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.69772
Module: inception3a.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.69732
Module: inception3a.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.69694
Module: inception3a.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.69672
Module: inception3a.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.69636
Module: inception3a.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.69624
Module: inception3a.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.68922
Module: inception3a.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.69836
Module: inception3a.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.69796
Module: inception3a.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.6971
Module: inception3a.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.69636
Module: inception3a.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.69462
Module: inception3a.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.68994
Module: inception3a.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.6863
Module: inception3a.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.68396
Module: inception3a.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.67372
Module: inception3b.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.69762
Module: inception3b.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.69792
Module: inception3b.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.69754
Module: inception3b.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.69696
Module: inception3b.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.69558
Module: inception3b.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.6941
Module: inception3b.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.6874
Module: inception3b.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.68132
Module: inception3b.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.63422
Module: inception3b.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.698
Module: inception3b.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.69746
Module: inception3b.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.69816
Module: inception3b.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.69644
Module: inception3b.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.69476
Module: inception3b.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.69208
Module: inception3b.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.68962
Module: inception3b.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.68018
Module: inception3b.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.61588
Module: inception3b.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69758
Module: inception3b.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.69744
Module: inception3b.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.69754
Module: inception3b.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.69712
Module: inception3b.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.69712
Module: inception3b.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.69626
Module: inception3b.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.6937
Module: inception3b.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.68612
Module: inception3b.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.66302
Module: inception3b.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.69744
Module: inception3b.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69762
Module: inception3b.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69798
Module: inception3b.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.69728
Module: inception3b.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.69676
Module: inception3b.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.69608
Module: inception3b.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.69506
Module: inception3b.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.69168
Module: inception3b.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.6758
Module: inception3b.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.69796
Module: inception3b.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.6979
Module: inception3b.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.69794
Module: inception3b.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.69852
Module: inception3b.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.69768
Module: inception3b.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.69634
Module: inception3b.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.69248
Module: inception3b.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.68948
Module: inception3b.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.67916
Module: inception3b.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.69784
Module: inception3b.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.6981
Module: inception3b.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.69744
Module: inception3b.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.69728
Module: inception3b.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.69724
Module: inception3b.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.69242
Module: inception3b.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.68636
Module: inception3b.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.66892
Module: inception3b.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.56586
Module: inception4a.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.69734
Module: inception4a.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.69798
Module: inception4a.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.69788
Module: inception4a.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.69702
Module: inception4a.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.6953
Module: inception4a.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.68908
Module: inception4a.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.6756
Module: inception4a.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.61244
Module: inception4a.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.40776
Module: inception4a.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.698
Module: inception4a.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.69768
Module: inception4a.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.69758
Module: inception4a.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.69762
Module: inception4a.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.69814
Module: inception4a.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.69672
Module: inception4a.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.69288
Module: inception4a.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.686
Module: inception4a.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.67356
Module: inception4a.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69784
Module: inception4a.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.69812
Module: inception4a.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.69806
Module: inception4a.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.6982
Module: inception4a.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.69794
Module: inception4a.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.69774
Module: inception4a.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.6949
Module: inception4a.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.69116
Module: inception4a.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.67338
Module: inception4a.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.69768
Module: inception4a.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69792
Module: inception4a.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.6979
Module: inception4a.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.69778
Module: inception4a.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.69732
Module: inception4a.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.69768
Module: inception4a.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.6971
Module: inception4a.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.6958
Module: inception4a.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.68948
Module: inception4a.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.69774
Module: inception4a.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.6979
Module: inception4a.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.69794
Module: inception4a.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.69658
Module: inception4a.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.69538
Module: inception4a.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.68662
Module: inception4a.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.67238
Module: inception4a.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.66034
Module: inception4a.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.63538
Module: inception4a.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.69774
Module: inception4a.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.69742
Module: inception4a.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.69724
Module: inception4a.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.6954
Module: inception4a.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.69246
Module: inception4a.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.68096
Module: inception4a.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.6614
Module: inception4a.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.59194
Module: inception4a.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.5307
Module: inception4b.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.69776
Module: inception4b.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.69818
Module: inception4b.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.69738
Module: inception4b.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.69606
Module: inception4b.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.69538
Module: inception4b.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.69304
Module: inception4b.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.68834
Module: inception4b.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.67982
Module: inception4b.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.65418
Module: inception4b.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.69788
Module: inception4b.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.6976
Module: inception4b.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.69782
Module: inception4b.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.6977
Module: inception4b.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.6981
Module: inception4b.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.6968
Module: inception4b.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.69492
Module: inception4b.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.69006
Module: inception4b.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.66916
Module: inception4b.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.6976
Module: inception4b.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.6981
Module: inception4b.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.6981
Module: inception4b.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.69768
Module: inception4b.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.69654
Module: inception4b.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.69652
Module: inception4b.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.69468
Module: inception4b.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.68814
Module: inception4b.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.67004
Module: inception4b.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.69752
Module: inception4b.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69788
Module: inception4b.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69788
Module: inception4b.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.69756
Module: inception4b.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.69726
Module: inception4b.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.69698
Module: inception4b.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.6962
Module: inception4b.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.69518
Module: inception4b.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.69266
Module: inception4b.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.69772
Module: inception4b.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.69798
Module: inception4b.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.69762
Module: inception4b.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.69822
Module: inception4b.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.69786
Module: inception4b.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.69698
Module: inception4b.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.69508
Module: inception4b.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.68596
Module: inception4b.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.6488
Module: inception4b.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.69776
Module: inception4b.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.69772
Module: inception4b.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.69762
Module: inception4b.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.69716
Module: inception4b.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.69782
Module: inception4b.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.69552
Module: inception4b.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.69046
Module: inception4b.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.6778
Module: inception4b.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.5263
Module: inception4c.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.69772
Module: inception4c.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.6975
Module: inception4c.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.69814
Module: inception4c.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.69666
Module: inception4c.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.69498
Module: inception4c.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.69446
Module: inception4c.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.6895
Module: inception4c.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.6785
Module: inception4c.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.65244
Module: inception4c.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.69794
Module: inception4c.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.69816
Module: inception4c.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.6987
Module: inception4c.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.69774
Module: inception4c.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.69666
Module: inception4c.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.69516
Module: inception4c.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.68916
Module: inception4c.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.67938
Module: inception4c.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.64484
Module: inception4c.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69776
Module: inception4c.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.69788
Module: inception4c.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.6974
Module: inception4c.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.69758
Module: inception4c.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.69678
Module: inception4c.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.69478
Module: inception4c.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.69178
Module: inception4c.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.67404
Module: inception4c.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.58956
Module: inception4c.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.69766
Module: inception4c.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69754
Module: inception4c.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69784
Module: inception4c.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.69766
Module: inception4c.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.6972
Module: inception4c.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.69756
Module: inception4c.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.69738
Module: inception4c.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.69728
Module: inception4c.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.69606
Module: inception4c.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.69776
Module: inception4c.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.69806
Module: inception4c.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.69804
Module: inception4c.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.6983
Module: inception4c.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.69772
Module: inception4c.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.69716
Module: inception4c.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.69658
Module: inception4c.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.69534
Module: inception4c.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.68824
Module: inception4c.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.6977
Module: inception4c.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.69742
Module: inception4c.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.6977
Module: inception4c.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.69724
Module: inception4c.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.6962
Module: inception4c.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.69418
Module: inception4c.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.68914
Module: inception4c.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.67234
Module: inception4c.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.64256
Module: inception4d.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.6978
Module: inception4d.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.69766
Module: inception4d.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.69736
Module: inception4d.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.69664
Module: inception4d.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.69576
Module: inception4d.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.69434
Module: inception4d.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.6918
Module: inception4d.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.68648
Module: inception4d.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.66982
Module: inception4d.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.69772
Module: inception4d.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.69802
Module: inception4d.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.69732
Module: inception4d.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.69642
Module: inception4d.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.69488
Module: inception4d.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.6932
Module: inception4d.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.68554
Module: inception4d.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.67818
Module: inception4d.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.63556
Module: inception4d.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69788
Module: inception4d.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.6982
Module: inception4d.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.69854
Module: inception4d.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.69774
Module: inception4d.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.69676
Module: inception4d.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.69526
Module: inception4d.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.69082
Module: inception4d.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.67458
Module: inception4d.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.60628
Module: inception4d.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.6978
Module: inception4d.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69784
Module: inception4d.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69772
Module: inception4d.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.69698
Module: inception4d.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.69714
Module: inception4d.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.69668
Module: inception4d.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.69606
Module: inception4d.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.69546
Module: inception4d.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.68918
Module: inception4d.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.6976
Module: inception4d.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.69794
Module: inception4d.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.69768
Module: inception4d.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.69756
Module: inception4d.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.69722
Module: inception4d.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.6965
Module: inception4d.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.69484
Module: inception4d.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.69082
Module: inception4d.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.67692
Module: inception4d.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.69788
Module: inception4d.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.69798
Module: inception4d.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.698
Module: inception4d.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.69888
Module: inception4d.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.69714
Module: inception4d.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.69454
Module: inception4d.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.69016
Module: inception4d.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.67576
Module: inception4d.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.62842
Module: inception4e.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.69784
Module: inception4e.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.6977
Module: inception4e.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.69704
Module: inception4e.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.69704
Module: inception4e.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.69506
Module: inception4e.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.6938
Module: inception4e.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.68982
Module: inception4e.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.67974
Module: inception4e.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.65698
Module: inception4e.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.69772
Module: inception4e.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.69732
Module: inception4e.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.69614
Module: inception4e.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.69504
Module: inception4e.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.69216
Module: inception4e.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.69072
Module: inception4e.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.68366
Module: inception4e.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.66776
Module: inception4e.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.62454
Module: inception4e.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69782
Module: inception4e.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.69762
Module: inception4e.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.69738
Module: inception4e.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.69808
Module: inception4e.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.69712
Module: inception4e.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.69542
Module: inception4e.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.69278
Module: inception4e.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.67926
Module: inception4e.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.64718
Module: inception4e.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.69786
Module: inception4e.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69824
Module: inception4e.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69892
Module: inception4e.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.69862
Module: inception4e.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.69784
Module: inception4e.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.69656
Module: inception4e.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.69656
Module: inception4e.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.69384
Module: inception4e.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.68774
Module: inception4e.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.69786
Module: inception4e.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.69794
Module: inception4e.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.69808
Module: inception4e.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.6976
Module: inception4e.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.69742
Module: inception4e.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.69636
Module: inception4e.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.69422
Module: inception4e.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.68804
Module: inception4e.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.67292
Module: inception4e.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.69766
Module: inception4e.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.69764
Module: inception4e.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.69694
Module: inception4e.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.69654
Module: inception4e.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.69594
Module: inception4e.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.69252
Module: inception4e.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.68324
Module: inception4e.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.66946
Module: inception4e.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.64412
Module: inception5a.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.69792
Module: inception5a.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.69766
Module: inception5a.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.69722
Module: inception5a.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.69674
Module: inception5a.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.69518
Module: inception5a.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.69196
Module: inception5a.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.68978
Module: inception5a.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.68316
Module: inception5a.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.67198
Module: inception5a.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.69796
Module: inception5a.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.69804
Module: inception5a.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.69724
Module: inception5a.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.69702
Module: inception5a.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.69612
Module: inception5a.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.69444
Module: inception5a.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.69074
Module: inception5a.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.6819
Module: inception5a.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.66124
Module: inception5a.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69758
Module: inception5a.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.69754
Module: inception5a.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.6978
Module: inception5a.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.69712
Module: inception5a.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.69612
Module: inception5a.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.69318
Module: inception5a.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.68338
Module: inception5a.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.65104
Module: inception5a.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.50912
Module: inception5a.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.698
Module: inception5a.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69788
Module: inception5a.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69792
Module: inception5a.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.69752
Module: inception5a.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.69752
Module: inception5a.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.69714
Module: inception5a.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.69658
Module: inception5a.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.694
Module: inception5a.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.68958
Module: inception5a.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.69776
Module: inception5a.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.69782
Module: inception5a.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.6975
Module: inception5a.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.6966
Module: inception5a.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.6951
Module: inception5a.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.69294
Module: inception5a.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.68622
Module: inception5a.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.66988
Module: inception5a.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.63714
Module: inception5a.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.6976
Module: inception5a.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.69784
Module: inception5a.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.69842
Module: inception5a.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.69724
Module: inception5a.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.69496
Module: inception5a.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.69048
Module: inception5a.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.68642
Module: inception5a.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.67766
Module: inception5a.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.65524
Module: inception5b.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.69782
Module: inception5b.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.69784
Module: inception5b.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.69754
Module: inception5b.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.69806
Module: inception5b.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.69668
Module: inception5b.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.69558
Module: inception5b.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.69394
Module: inception5b.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.68958
Module: inception5b.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.68038
Module: inception5b.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.69796
Module: inception5b.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.6981
Module: inception5b.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.6977
Module: inception5b.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.69674
Module: inception5b.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.69478
Module: inception5b.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.69178
Module: inception5b.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.6821
Module: inception5b.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.6603
Module: inception5b.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.61534
Module: inception5b.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69778
Module: inception5b.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.69784
Module: inception5b.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.69776
Module: inception5b.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.69708
Module: inception5b.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.69558
Module: inception5b.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.69286
Module: inception5b.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.68832
Module: inception5b.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.6784
Module: inception5b.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.66492
Module: inception5b.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.69766
Module: inception5b.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69782
Module: inception5b.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69732
Module: inception5b.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.69716
Module: inception5b.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.69736
Module: inception5b.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.6952
Module: inception5b.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.69098
Module: inception5b.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.68508
Module: inception5b.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.66592
Module: inception5b.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.6977
Module: inception5b.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.69762
Module: inception5b.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.69766
Module: inception5b.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.6978
Module: inception5b.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.69752
Module: inception5b.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.6972
Module: inception5b.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.69638
Module: inception5b.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.69488
Module: inception5b.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.6936
Module: inception5b.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.69772
Module: inception5b.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.69786
Module: inception5b.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.69808
Module: inception5b.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.69788
Module: inception5b.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.69706
Module: inception5b.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.69554
Module: inception5b.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.69402
Module: inception5b.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.69004
Module: inception5b.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.68608
"""
data_u_g_l1 = """
Pruning Rate: 0.1, Accuracy: 0.69728
Pruning Rate: 0.2, Accuracy: 0.69434
Pruning Rate: 0.3, Accuracy: 0.68508
Pruning Rate: 0.4, Accuracy: 0.66946
Pruning Rate: 0.5, Accuracy: 0.6128
Pruning Rate: 0.6, Accuracy: 0.4921
Pruning Rate: 0.7, Accuracy: 0.10154
Pruning Rate: 0.8, Accuracy: 0.0046
Pruning Rate: 0.9, Accuracy: 0.0012
"""

data_u_l_r = """
Module: conv1.conv, Pruning Rate: 0.1, Accuracy: 0.54774
Module: conv1.conv, Pruning Rate: 0.2, Accuracy: 0.3613
Module: conv1.conv, Pruning Rate: 0.3, Accuracy: 0.31142000000000003
Module: conv1.conv, Pruning Rate: 0.4, Accuracy: 0.10530999999999999
Module: conv1.conv, Pruning Rate: 0.5, Accuracy: 0.09262
Module: conv1.conv, Pruning Rate: 0.6, Accuracy: 0.02333
Module: conv1.conv, Pruning Rate: 0.7, Accuracy: 0.02734
Module: conv1.conv, Pruning Rate: 0.8, Accuracy: 0.026449999999999998
Module: conv1.conv, Pruning Rate: 0.9, Accuracy: 0.01491
Module: conv2.conv, Pruning Rate: 0.1, Accuracy: 0.07794
Module: conv2.conv, Pruning Rate: 0.2, Accuracy: 0.03224
Module: conv2.conv, Pruning Rate: 0.3, Accuracy: 0.03068
Module: conv2.conv, Pruning Rate: 0.4, Accuracy: 0.0014399999999999999
Module: conv2.conv, Pruning Rate: 0.5, Accuracy: 0.00135
Module: conv2.conv, Pruning Rate: 0.6, Accuracy: 0.0010400000000000001
Module: conv2.conv, Pruning Rate: 0.7, Accuracy: 0.00141
Module: conv2.conv, Pruning Rate: 0.8, Accuracy: 0.00107
Module: conv2.conv, Pruning Rate: 0.9, Accuracy: 0.0011099999999999999
Module: conv3.conv, Pruning Rate: 0.1, Accuracy: 0.62592
Module: conv3.conv, Pruning Rate: 0.2, Accuracy: 0.47021
Module: conv3.conv, Pruning Rate: 0.3, Accuracy: 0.24520999999999998
Module: conv3.conv, Pruning Rate: 0.4, Accuracy: 0.18612
Module: conv3.conv, Pruning Rate: 0.5, Accuracy: 0.026529999999999998
Module: conv3.conv, Pruning Rate: 0.6, Accuracy: 0.00785
Module: conv3.conv, Pruning Rate: 0.7, Accuracy: 0.00285
Module: conv3.conv, Pruning Rate: 0.8, Accuracy: 0.00143
Module: conv3.conv, Pruning Rate: 0.9, Accuracy: 0.0009699999999999999
Module: inception3a.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.64879
Module: inception3a.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.59744
Module: inception3a.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.5092399999999999
Module: inception3a.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.40386999999999995
Module: inception3a.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.15617
Module: inception3a.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.17566
Module: inception3a.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.16646
Module: inception3a.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.09565000000000001
Module: inception3a.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.0465
Module: inception3a.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.68066
Module: inception3a.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.63508
Module: inception3a.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.56401
Module: inception3a.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.5527500000000001
Module: inception3a.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.18583
Module: inception3a.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.12489
Module: inception3a.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.06725
Module: inception3a.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.018000000000000002
Module: inception3a.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.01158
Module: inception3a.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69103
Module: inception3a.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.67683
Module: inception3a.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.66435
Module: inception3a.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.61519
Module: inception3a.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.50749
Module: inception3a.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.44206
Module: inception3a.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.30378
Module: inception3a.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.18383
Module: inception3a.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.05184
Module: inception3a.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.6958899999999999
Module: inception3a.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.6925399999999999
Module: inception3a.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69116
Module: inception3a.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.66501
Module: inception3a.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.6752499999999999
Module: inception3a.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.6450800000000001
Module: inception3a.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.61696
Module: inception3a.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.52821
Module: inception3a.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.54036
Module: inception3a.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.69011
Module: inception3a.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.68608
Module: inception3a.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.68509
Module: inception3a.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.66715
Module: inception3a.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.6747799999999999
Module: inception3a.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.63774
Module: inception3a.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.64861
Module: inception3a.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.60049
Module: inception3a.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.5952500000000001
Module: inception3a.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.67625
Module: inception3a.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.50658
Module: inception3a.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.35257
Module: inception3a.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.1925
Module: inception3a.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.03198
Module: inception3a.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.02928
Module: inception3a.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.00513
Module: inception3a.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.00341
Module: inception3a.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.00289
Module: inception3b.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.68039
Module: inception3b.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.6596599999999999
Module: inception3b.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.6514599999999999
Module: inception3b.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.6105400000000001
Module: inception3b.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.51476
Module: inception3b.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.48861
Module: inception3b.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.45128999999999997
Module: inception3b.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.34677
Module: inception3b.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.30241
Module: inception3b.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.68957
Module: inception3b.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.6706300000000001
Module: inception3b.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.64818
Module: inception3b.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.58928
Module: inception3b.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.51225
Module: inception3b.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.43868
Module: inception3b.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.26868
Module: inception3b.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.17908000000000002
Module: inception3b.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.1069
Module: inception3b.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69484
Module: inception3b.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.68789
Module: inception3b.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.68165
Module: inception3b.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.6640699999999999
Module: inception3b.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.64364
Module: inception3b.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.6192500000000001
Module: inception3b.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.5644899999999999
Module: inception3b.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.48706
Module: inception3b.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.40135
Module: inception3b.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.69545
Module: inception3b.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69354
Module: inception3b.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.68497
Module: inception3b.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.67763
Module: inception3b.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.68166
Module: inception3b.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.66923
Module: inception3b.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.6605099999999999
Module: inception3b.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.64929
Module: inception3b.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.62764
Module: inception3b.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.6957
Module: inception3b.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.69218
Module: inception3b.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.68988
Module: inception3b.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.68432
Module: inception3b.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.68121
Module: inception3b.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.6786
Module: inception3b.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.6740999999999999
Module: inception3b.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.66579
Module: inception3b.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.6608799999999999
Module: inception3b.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.6882699999999999
Module: inception3b.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.66721
Module: inception3b.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.61009
Module: inception3b.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.50673
Module: inception3b.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.54344
Module: inception3b.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.32916999999999996
Module: inception3b.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.26589
Module: inception3b.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.18517
Module: inception3b.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.14032
Module: inception4a.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.68589
Module: inception4a.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.66111
Module: inception4a.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.64855
Module: inception4a.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.61408
Module: inception4a.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.53633
Module: inception4a.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.42763
Module: inception4a.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.46155
Module: inception4a.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.36572000000000005
Module: inception4a.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.18644
Module: inception4a.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.6932
Module: inception4a.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.6871700000000001
Module: inception4a.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.68062
Module: inception4a.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.67408
Module: inception4a.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.6550199999999999
Module: inception4a.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.64017
Module: inception4a.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.62784
Module: inception4a.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.5878
Module: inception4a.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.5634399999999999
Module: inception4a.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69631
Module: inception4a.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.69279
Module: inception4a.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.6851799999999999
Module: inception4a.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.67238
Module: inception4a.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.66152
Module: inception4a.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.6346499999999999
Module: inception4a.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.60988
Module: inception4a.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.54882
Module: inception4a.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.43688000000000005
Module: inception4a.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.69687
Module: inception4a.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69404
Module: inception4a.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69146
Module: inception4a.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.68754
Module: inception4a.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.67962
Module: inception4a.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.67871
Module: inception4a.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.6653100000000001
Module: inception4a.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.67003
Module: inception4a.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.66218
Module: inception4a.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.68709
Module: inception4a.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.6652
Module: inception4a.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.66884
Module: inception4a.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.66889
Module: inception4a.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.65007
Module: inception4a.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.63522
Module: inception4a.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.63759
Module: inception4a.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.61515
Module: inception4a.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.5926899999999999
Module: inception4a.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.68447
Module: inception4a.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.61085
Module: inception4a.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.60005
Module: inception4a.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.43840999999999997
Module: inception4a.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.21735
Module: inception4a.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.20828
Module: inception4a.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.16531
Module: inception4a.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.02628
Module: inception4a.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.01566
Module: inception4b.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.68957
Module: inception4b.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.67518
Module: inception4b.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.65652
Module: inception4b.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.63049
Module: inception4b.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.59367
Module: inception4b.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.54785
Module: inception4b.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.48842
Module: inception4b.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.41825999999999997
Module: inception4b.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.30823
Module: inception4b.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.69386
Module: inception4b.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.68941
Module: inception4b.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.68358
Module: inception4b.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.67383
Module: inception4b.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.6622
Module: inception4b.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.6442
Module: inception4b.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.62712
Module: inception4b.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.6015900000000001
Module: inception4b.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.5654399999999999
Module: inception4b.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.69584
Module: inception4b.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.6915800000000001
Module: inception4b.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.6860200000000001
Module: inception4b.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.68091
Module: inception4b.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.66687
Module: inception4b.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.65279
Module: inception4b.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.63422
Module: inception4b.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.60026
Module: inception4b.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.5226299999999999
Module: inception4b.branch3.0.conv, Pruning Rate: 0.1, Accuracy: 0.6975
Module: inception4b.branch3.0.conv, Pruning Rate: 0.2, Accuracy: 0.69627
Module: inception4b.branch3.0.conv, Pruning Rate: 0.3, Accuracy: 0.69621
Module: inception4b.branch3.0.conv, Pruning Rate: 0.4, Accuracy: 0.69365
Module: inception4b.branch3.0.conv, Pruning Rate: 0.5, Accuracy: 0.6905
Module: inception4b.branch3.0.conv, Pruning Rate: 0.6, Accuracy: 0.68964
Module: inception4b.branch3.0.conv, Pruning Rate: 0.7, Accuracy: 0.68642
Module: inception4b.branch3.0.conv, Pruning Rate: 0.8, Accuracy: 0.68036
Module: inception4b.branch3.0.conv, Pruning Rate: 0.9, Accuracy: 0.67851
Module: inception4b.branch3.1.conv, Pruning Rate: 0.1, Accuracy: 0.6938500000000001
Module: inception4b.branch3.1.conv, Pruning Rate: 0.2, Accuracy: 0.68577
Module: inception4b.branch3.1.conv, Pruning Rate: 0.3, Accuracy: 0.6835100000000001
Module: inception4b.branch3.1.conv, Pruning Rate: 0.4, Accuracy: 0.67241
Module: inception4b.branch3.1.conv, Pruning Rate: 0.5, Accuracy: 0.66871
Module: inception4b.branch3.1.conv, Pruning Rate: 0.6, Accuracy: 0.6558999999999999
Module: inception4b.branch3.1.conv, Pruning Rate: 0.7, Accuracy: 0.64501
Module: inception4b.branch3.1.conv, Pruning Rate: 0.8, Accuracy: 0.6245099999999999
Module: inception4b.branch3.1.conv, Pruning Rate: 0.9, Accuracy: 0.614
Module: inception4b.branch4.1.conv, Pruning Rate: 0.1, Accuracy: 0.6907
Module: inception4b.branch4.1.conv, Pruning Rate: 0.2, Accuracy: 0.6770799999999999
Module: inception4b.branch4.1.conv, Pruning Rate: 0.3, Accuracy: 0.6502
Module: inception4b.branch4.1.conv, Pruning Rate: 0.4, Accuracy: 0.3735
Module: inception4b.branch4.1.conv, Pruning Rate: 0.5, Accuracy: 0.21796
Module: inception4b.branch4.1.conv, Pruning Rate: 0.6, Accuracy: 0.08224999999999999
Module: inception4b.branch4.1.conv, Pruning Rate: 0.7, Accuracy: 0.03584
Module: inception4b.branch4.1.conv, Pruning Rate: 0.8, Accuracy: 0.01666
Module: inception4b.branch4.1.conv, Pruning Rate: 0.9, Accuracy: 0.0074800000000000005
Module: inception4c.branch1.conv, Pruning Rate: 0.1, Accuracy: 0.68468
Module: inception4c.branch1.conv, Pruning Rate: 0.2, Accuracy: 0.6661600000000001
Module: inception4c.branch1.conv, Pruning Rate: 0.3, Accuracy: 0.6389199999999999
Module: inception4c.branch1.conv, Pruning Rate: 0.4, Accuracy: 0.57285
Module: inception4c.branch1.conv, Pruning Rate: 0.5, Accuracy: 0.5237499999999999
Module: inception4c.branch1.conv, Pruning Rate: 0.6, Accuracy: 0.47369
Module: inception4c.branch1.conv, Pruning Rate: 0.7, Accuracy: 0.33226
Module: inception4c.branch1.conv, Pruning Rate: 0.8, Accuracy: 0.26015
Module: inception4c.branch1.conv, Pruning Rate: 0.9, Accuracy: 0.19308
Module: inception4c.branch2.0.conv, Pruning Rate: 0.1, Accuracy: 0.69242
Module: inception4c.branch2.0.conv, Pruning Rate: 0.2, Accuracy: 0.68616
Module: inception4c.branch2.0.conv, Pruning Rate: 0.3, Accuracy: 0.67798
Module: inception4c.branch2.0.conv, Pruning Rate: 0.4, Accuracy: 0.66974
Module: inception4c.branch2.0.conv, Pruning Rate: 0.5, Accuracy: 0.64313
Module: inception4c.branch2.0.conv, Pruning Rate: 0.6, Accuracy: 0.61279
Module: inception4c.branch2.0.conv, Pruning Rate: 0.7, Accuracy: 0.5710299999999999
Module: inception4c.branch2.0.conv, Pruning Rate: 0.8, Accuracy: 0.51281
Module: inception4c.branch2.0.conv, Pruning Rate: 0.9, Accuracy: 0.45888
Module: inception4c.branch2.1.conv, Pruning Rate: 0.1, Accuracy: 0.6954199999999999
Module: inception4c.branch2.1.conv, Pruning Rate: 0.2, Accuracy: 0.6921299999999999
Module: inception4c.branch2.1.conv, Pruning Rate: 0.3, Accuracy: 0.6878
Module: inception4c.branch2.1.conv, Pruning Rate: 0.4, Accuracy: 0.68204
Module: inception4c.branch2.1.conv, Pruning Rate: 0.5, Accuracy: 0.6653
Module: inception4c.branch2.1.conv, Pruning Rate: 0.6, Accuracy: 0.64209
Module: inception4c.branch2.1.conv, Pruning Rate: 0.7, Accuracy: 0.5901
Module: inception4c.branch2.1.conv, Pruning Rate: 0.8, Accuracy: 0.51447
Module: inception4c.branch2.1.conv, Pruning Rate: 0.9, Accuracy: 0.37229
"""

def plotLocal(data):
	# Datenverarbeitung
	records = [line.split(", ") for line in data.strip().split("\n")]

	columns = ["Module", "Pruning Rate", "Accuracy"]
	df = pd.DataFrame(records, columns=columns)

	# Datentypen konvertieren
	df["Pruning Rate"] = df["Pruning Rate"].str.split(": ").str[1].astype(float)
	df["Accuracy"] = df["Accuracy"].str.split(": ").str[1].astype(float)

	# Module mit numerischen Indizes versehen
	df['Module'] = df['Module'].astype('category').cat.codes

	# Gitterpunkte für die Oberfläche erstellen
	pruning_rate_grid, module_grid = np.meshgrid(
	    np.linspace(df['Pruning Rate'].min(), df['Pruning Rate'].max(), 50),
	    np.linspace(df['Module'].min(), df['Module'].max(), 50)
	)

	# Interpolation der Accuracies
	accuracy_grid = griddata(
	    (df['Pruning Rate'], df['Module']),
	    df['Accuracy'],
	    (pruning_rate_grid, module_grid),
	    method='linear'
	)

	# 3D-Plot erstellen
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')

	# Daten im 3D-Raum plotten
	surf = ax.plot_surface(pruning_rate_grid, module_grid,
	                       accuracy_grid, cmap='viridis', edgecolor='none')

	# Achsenbeschriftungen
	ax.set_xlabel('Pruning Rate')
	ax.set_ylabel('Module Index')
	ax.set_zlabel('Accuracy')

	# Farblegende für die Accuracy
	fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Accuracy')

	plt.show()

def plotGlobal(data):
    	# Datenverarbeitung
	records = [line.split(", ") for line in data.strip().split("\n")]

	columns = ["Pruning Rate", "Accuracy"]
	df = pd.DataFrame(records, columns=columns)

	# Datentypen konvertieren
	df["Pruning Rate"] = df["Pruning Rate"].str.split(": ").str[1].astype(float)
	df["Accuracy"] = df["Accuracy"].str.split(": ").str[1].astype(float)

	# Daten plotten
	plt.figure(figsize=(10, 6))
	plt.plot(df["Pruning Rate"], df["Accuracy"],
	         marker='o', linestyle='-', color='b')
	plt.xlabel('Pruning Rate')
	plt.ylabel('Accuracy')
	plt.title('Pruning Rate vs Accuracy')
	plt.grid(True)
	plt.show()

# plotGlobal(data_u_g_l1)

plotLocal(data_u_l_r)	