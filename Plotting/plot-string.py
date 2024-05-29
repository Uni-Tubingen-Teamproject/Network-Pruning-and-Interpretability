import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import torchvision.models as models
import torch
import seaborn as sns

# Local Unstructured L1 Pruning

# Example dataset, replace this part with your own data.
data_local_unstructured_l1 = """
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
data_global_unstructured_l1 = """
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
data_local_unstructured_random = """
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
data_local_structured_l1 = """
Module: conv1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69772
Module: conv1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69466
Module: conv1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.694
Module: conv1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.5823
Module: conv1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.4782
Module: conv1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.207
Module: conv1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.04826
Module: conv1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.0429
Module: conv1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.02262
Module: conv1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69772
Module: conv1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.55408
Module: conv1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.55408
Module: conv1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.55408
Module: conv1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.42662
Module: conv1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.42662
Module: conv1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.42662
Module: conv1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.42662
Module: conv1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.001
Module: conv2.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.21344
Module: conv2.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.00304
Module: conv2.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.0013
Module: conv2.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.00136
Module: conv2.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.00108
Module: conv2.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.00104
Module: conv2.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.00104
Module: conv2.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.001
Module: conv2.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.00102
Module: conv2.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.68872
Module: conv2.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68728
Module: conv2.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.48294
Module: conv2.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.29686
Module: conv2.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.06198
Module: conv2.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.00454
Module: conv2.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.0014
Module: conv2.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.00104
Module: conv2.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.00126
Module: conv3.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.6318
Module: conv3.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.43136
Module: conv3.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.36384
Module: conv3.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.0269
Module: conv3.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.0081
Module: conv3.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.00452
Module: conv3.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.00186
Module: conv3.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.0011
Module: conv3.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.001
Module: conv3.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69684
Module: conv3.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.66868
Module: conv3.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.62914
Module: conv3.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.56672
Module: conv3.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.5046
Module: conv3.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.21758
Module: conv3.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.04108
Module: conv3.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.00808
Module: conv3.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.0019
Module: inception3a.branch1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68942
Module: inception3a.branch1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.6761
Module: inception3a.branch1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.66242
Module: inception3a.branch1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.63468
Module: inception3a.branch1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.59374
Module: inception3a.branch1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.44596
Module: inception3a.branch1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.14946
Module: inception3a.branch1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.11876
Module: inception3a.branch1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.06162
Module: inception3a.branch1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.68978
Module: inception3a.branch1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.6688
Module: inception3a.branch1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.61338
Module: inception3a.branch1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.5421
Module: inception3a.branch1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.3667
Module: inception3a.branch1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.27108
Module: inception3a.branch1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.09656
Module: inception3a.branch1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.02646
Module: inception3a.branch1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.0131
Module: inception3a.branch2.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68132
Module: inception3a.branch2.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.6697
Module: inception3a.branch2.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.62762
Module: inception3a.branch2.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.61034
Module: inception3a.branch2.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.3543
Module: inception3a.branch2.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.29116
Module: inception3a.branch2.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.24284
Module: inception3a.branch2.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.04512
Module: inception3a.branch2.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.0231
Module: inception3a.branch2.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6952
Module: inception3a.branch2.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.6671
Module: inception3a.branch2.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.64556
Module: inception3a.branch2.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.61814
Module: inception3a.branch2.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.4827
Module: inception3a.branch2.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.42462
Module: inception3a.branch2.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.18602
Module: inception3a.branch2.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.08404
Module: inception3a.branch2.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.02488
Module: inception3a.branch2.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69588
Module: inception3a.branch2.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68014
Module: inception3a.branch2.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.65384
Module: inception3a.branch2.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.61996
Module: inception3a.branch2.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.5773
Module: inception3a.branch2.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.50196
Module: inception3a.branch2.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.44118
Module: inception3a.branch2.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.30338
Module: inception3a.branch2.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.09374
Module: inception3a.branch2.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69192
Module: inception3a.branch2.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68598
Module: inception3a.branch2.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.67672
Module: inception3a.branch2.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.65928
Module: inception3a.branch2.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.63932
Module: inception3a.branch2.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.58164
Module: inception3a.branch2.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.47066
Module: inception3a.branch2.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.33744
Module: inception3a.branch2.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.15504
Module: inception3a.branch3.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.58224
Module: inception3a.branch3.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.504
Module: inception3a.branch3.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.4986
Module: inception3a.branch3.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.49516
Module: inception3a.branch3.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.4816
Module: inception3a.branch3.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.47466
Module: inception3a.branch3.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.4705
Module: inception3a.branch3.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.42346
Module: inception3a.branch3.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.45232
Module: inception3a.branch3.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.698
Module: inception3a.branch3.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69722
Module: inception3a.branch3.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6968
Module: inception3a.branch3.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69534
Module: inception3a.branch3.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69482
Module: inception3a.branch3.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.69486
Module: inception3a.branch3.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.69156
Module: inception3a.branch3.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.65916
Module: inception3a.branch3.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.62014
Module: inception3a.branch3.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69802
Module: inception3a.branch3.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69762
Module: inception3a.branch3.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69602
Module: inception3a.branch3.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.69682
Module: inception3a.branch3.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.61492
Module: inception3a.branch3.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.61002
Module: inception3a.branch3.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.6098
Module: inception3a.branch3.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.59244
Module: inception3a.branch3.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.58462
Module: inception3a.branch3.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6979
Module: inception3a.branch3.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69704
Module: inception3a.branch3.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6974
Module: inception3a.branch3.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69704
Module: inception3a.branch3.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69658
Module: inception3a.branch3.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.69392
Module: inception3a.branch3.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.69066
Module: inception3a.branch3.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.65324
Module: inception3a.branch3.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.65216
Module: inception3a.branch4.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.67146
Module: inception3a.branch4.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.49732
Module: inception3a.branch4.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.4891
Module: inception3a.branch4.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.43552
Module: inception3a.branch4.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.40648
Module: inception3a.branch4.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.3687
Module: inception3a.branch4.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.3129
Module: inception3a.branch4.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.01284
Module: inception3a.branch4.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.0094
Module: inception3a.branch4.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69586
Module: inception3a.branch4.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69538
Module: inception3a.branch4.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68744
Module: inception3a.branch4.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68438
Module: inception3a.branch4.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.67536
Module: inception3a.branch4.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.67138
Module: inception3a.branch4.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.64774
Module: inception3a.branch4.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.63696
Module: inception3a.branch4.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.58172
Module: inception3b.branch1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69084
Module: inception3b.branch1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.67892
Module: inception3b.branch1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.6596
Module: inception3b.branch1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.64948
Module: inception3b.branch1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.62596
Module: inception3b.branch1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.57084
Module: inception3b.branch1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.48286
Module: inception3b.branch1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.43296
Module: inception3b.branch1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.3134
Module: inception3b.branch1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69536
Module: inception3b.branch1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69338
Module: inception3b.branch1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68932
Module: inception3b.branch1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68382
Module: inception3b.branch1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.67378
Module: inception3b.branch1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.65958
Module: inception3b.branch1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.63438
Module: inception3b.branch1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.57824
Module: inception3b.branch1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.46772
Module: inception3b.branch2.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68452
Module: inception3b.branch2.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.65954
Module: inception3b.branch2.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.59722
Module: inception3b.branch2.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.52396
Module: inception3b.branch2.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.38384
Module: inception3b.branch2.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.26054
Module: inception3b.branch2.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.18654
Module: inception3b.branch2.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.1502
Module: inception3b.branch2.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.1081
Module: inception3b.branch2.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69674
Module: inception3b.branch2.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69598
Module: inception3b.branch2.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68974
Module: inception3b.branch2.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.67912
Module: inception3b.branch2.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.66506
Module: inception3b.branch2.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.57898
Module: inception3b.branch2.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.47796
Module: inception3b.branch2.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.39154
Module: inception3b.branch2.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.19718
Module: inception3b.branch2.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68438
Module: inception3b.branch2.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.67204
Module: inception3b.branch2.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.64804
Module: inception3b.branch2.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.61494
Module: inception3b.branch2.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.55138
Module: inception3b.branch2.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.47936
Module: inception3b.branch2.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.4255
Module: inception3b.branch2.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.35578
Module: inception3b.branch2.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.29556
Module: inception3b.branch2.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6953
Module: inception3b.branch2.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68996
Module: inception3b.branch2.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68056
Module: inception3b.branch2.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.66954
Module: inception3b.branch2.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.65586
Module: inception3b.branch2.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.63752
Module: inception3b.branch2.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.60202
Module: inception3b.branch2.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.5296
Module: inception3b.branch2.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.43
Module: inception3b.branch3.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.6975
Module: inception3b.branch3.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69682
Module: inception3b.branch3.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.66058
Module: inception3b.branch3.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.65936
Module: inception3b.branch3.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.65406
Module: inception3b.branch3.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.65
Module: inception3b.branch3.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.63656
Module: inception3b.branch3.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.63938
Module: inception3b.branch3.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.63212
Module: inception3b.branch3.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69786
Module: inception3b.branch3.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69708
Module: inception3b.branch3.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69588
Module: inception3b.branch3.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69394
Module: inception3b.branch3.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69018
Module: inception3b.branch3.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.6873
Module: inception3b.branch3.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.68388
Module: inception3b.branch3.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.66948
Module: inception3b.branch3.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.64394
Module: inception3b.branch3.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.6957
Module: inception3b.branch3.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69404
Module: inception3b.branch3.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69232
Module: inception3b.branch3.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.69016
Module: inception3b.branch3.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.6865
Module: inception3b.branch3.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.68204
Module: inception3b.branch3.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.6755
Module: inception3b.branch3.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.6658
Module: inception3b.branch3.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.65832
Module: inception3b.branch3.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69764
Module: inception3b.branch3.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69674
Module: inception3b.branch3.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69416
Module: inception3b.branch3.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69196
Module: inception3b.branch3.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.6874
Module: inception3b.branch3.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.68332
Module: inception3b.branch3.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.67856
Module: inception3b.branch3.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.6644
Module: inception3b.branch3.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.66526
Module: inception3b.branch4.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69558
Module: inception3b.branch4.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68578
Module: inception3b.branch4.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.67132
Module: inception3b.branch4.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.65104
Module: inception3b.branch4.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.64554
Module: inception3b.branch4.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.63088
Module: inception3b.branch4.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.60228
Module: inception3b.branch4.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.52422
Module: inception3b.branch4.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.14544
Module: inception3b.branch4.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69486
Module: inception3b.branch4.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.67792
Module: inception3b.branch4.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.58656
Module: inception3b.branch4.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.52064
Module: inception3b.branch4.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.37688
Module: inception3b.branch4.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.24976
Module: inception3b.branch4.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.12232
Module: inception3b.branch4.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.07278
Module: inception3b.branch4.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.05584
Module: inception4a.branch1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69024
Module: inception4a.branch1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68206
Module: inception4a.branch1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.66522
Module: inception4a.branch1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.6496
Module: inception4a.branch1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.62708
Module: inception4a.branch1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.60282
Module: inception4a.branch1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.52802
Module: inception4a.branch1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.41508
Module: inception4a.branch1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.27844
Module: inception4a.branch1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69098
Module: inception4a.branch1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.66518
Module: inception4a.branch1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6187
Module: inception4a.branch1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.5332
Module: inception4a.branch1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.43722
Module: inception4a.branch1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.37914
Module: inception4a.branch1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.3377
Module: inception4a.branch1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.27136
Module: inception4a.branch1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.20912
Module: inception4a.branch2.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69364
Module: inception4a.branch2.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68646
Module: inception4a.branch2.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.6825
Module: inception4a.branch2.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.67492
Module: inception4a.branch2.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.66266
Module: inception4a.branch2.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.64748
Module: inception4a.branch2.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.63496
Module: inception4a.branch2.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.61604
Module: inception4a.branch2.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.57194
Module: inception4a.branch2.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69658
Module: inception4a.branch2.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69462
Module: inception4a.branch2.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69318
Module: inception4a.branch2.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69102
Module: inception4a.branch2.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.68734
Module: inception4a.branch2.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.67968
Module: inception4a.branch2.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.66966
Module: inception4a.branch2.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.65796
Module: inception4a.branch2.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.6301
Module: inception4a.branch2.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69112
Module: inception4a.branch2.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68692
Module: inception4a.branch2.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.68008
Module: inception4a.branch2.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.67438
Module: inception4a.branch2.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.65912
Module: inception4a.branch2.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.63992
Module: inception4a.branch2.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.61126
Module: inception4a.branch2.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.52782
Module: inception4a.branch2.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.44364
Module: inception4a.branch2.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69456
Module: inception4a.branch2.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69092
Module: inception4a.branch2.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68404
Module: inception4a.branch2.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.67614
Module: inception4a.branch2.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.6684
Module: inception4a.branch2.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.65968
Module: inception4a.branch2.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.63894
Module: inception4a.branch2.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.59594
Module: inception4a.branch2.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.52952
Module: inception4a.branch3.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69624
Module: inception4a.branch3.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.6955
Module: inception4a.branch3.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69458
Module: inception4a.branch3.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.69502
Module: inception4a.branch3.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.6887
Module: inception4a.branch3.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.67608
Module: inception4a.branch3.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.66914
Module: inception4a.branch3.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.66758
Module: inception4a.branch3.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.6636
Module: inception4a.branch3.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69796
Module: inception4a.branch3.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69766
Module: inception4a.branch3.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69702
Module: inception4a.branch3.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.6966
Module: inception4a.branch3.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69594
Module: inception4a.branch3.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.69518
Module: inception4a.branch3.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.69404
Module: inception4a.branch3.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.69304
Module: inception4a.branch3.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.6893
Module: inception4a.branch3.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.635
Module: inception4a.branch3.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.63566
Module: inception4a.branch3.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.63328
Module: inception4a.branch3.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.63306
Module: inception4a.branch3.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.6307
Module: inception4a.branch3.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.62756
Module: inception4a.branch3.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.62504
Module: inception4a.branch3.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.62288
Module: inception4a.branch3.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.6165
Module: inception4a.branch3.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69756
Module: inception4a.branch3.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.6976
Module: inception4a.branch3.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69622
Module: inception4a.branch3.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69614
Module: inception4a.branch3.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69288
Module: inception4a.branch3.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.6862
Module: inception4a.branch3.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.67398
Module: inception4a.branch3.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.6547
Module: inception4a.branch3.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.6393
Module: inception4a.branch4.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68096
Module: inception4a.branch4.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.63948
Module: inception4a.branch4.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.62604
Module: inception4a.branch4.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.56506
Module: inception4a.branch4.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.43806
Module: inception4a.branch4.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.3916
Module: inception4a.branch4.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.31428
Module: inception4a.branch4.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.20098
Module: inception4a.branch4.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.03422
Module: inception4a.branch4.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69248
Module: inception4a.branch4.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68106
Module: inception4a.branch4.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.65932
Module: inception4a.branch4.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.61862
Module: inception4a.branch4.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.5753
Module: inception4a.branch4.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.46204
Module: inception4a.branch4.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.31692
Module: inception4a.branch4.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.23782
Module: inception4a.branch4.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.18752
Module: inception4b.branch1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69018
Module: inception4b.branch1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68408
Module: inception4b.branch1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.67202
Module: inception4b.branch1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.6516
Module: inception4b.branch1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.62332
Module: inception4b.branch1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.60076
Module: inception4b.branch1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.5613
Module: inception4b.branch1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.5284
Module: inception4b.branch1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.40756
Module: inception4b.branch1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69456
Module: inception4b.branch1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.6897
Module: inception4b.branch1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68168
Module: inception4b.branch1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.6761
Module: inception4b.branch1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.66772
Module: inception4b.branch1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.6552
Module: inception4b.branch1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.63384
Module: inception4b.branch1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.59378
Module: inception4b.branch1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.4936
Module: inception4b.branch2.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69006
Module: inception4b.branch2.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68328
Module: inception4b.branch2.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.66858
Module: inception4b.branch2.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.6614
Module: inception4b.branch2.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.65072
Module: inception4b.branch2.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.63802
Module: inception4b.branch2.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.61176
Module: inception4b.branch2.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.57264
Module: inception4b.branch2.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.54534
Module: inception4b.branch2.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69632
Module: inception4b.branch2.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69254
Module: inception4b.branch2.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68646
Module: inception4b.branch2.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.67954
Module: inception4b.branch2.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.67782
Module: inception4b.branch2.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.67226
Module: inception4b.branch2.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.66132
Module: inception4b.branch2.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.63714
Module: inception4b.branch2.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.5977
Module: inception4b.branch2.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69374
Module: inception4b.branch2.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68778
Module: inception4b.branch2.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.68052
Module: inception4b.branch2.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.66846
Module: inception4b.branch2.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.65636
Module: inception4b.branch2.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.63662
Module: inception4b.branch2.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.60276
Module: inception4b.branch2.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.55874
Module: inception4b.branch2.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.47748
Module: inception4b.branch2.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6964
Module: inception4b.branch2.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68964
Module: inception4b.branch2.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.67798
Module: inception4b.branch2.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.6726
Module: inception4b.branch2.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.66804
Module: inception4b.branch2.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.6527
Module: inception4b.branch2.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.64368
Module: inception4b.branch2.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.62056
Module: inception4b.branch2.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.57044
Module: inception4b.branch3.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69768
Module: inception4b.branch3.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69634
Module: inception4b.branch3.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69526
Module: inception4b.branch3.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.69236
Module: inception4b.branch3.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.68972
Module: inception4b.branch3.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.6886
Module: inception4b.branch3.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.68592
Module: inception4b.branch3.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.68422
Module: inception4b.branch3.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.67634
Module: inception4b.branch3.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69796
Module: inception4b.branch3.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69824
Module: inception4b.branch3.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69792
Module: inception4b.branch3.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69732
Module: inception4b.branch3.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69686
Module: inception4b.branch3.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.69606
Module: inception4b.branch3.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.6943
Module: inception4b.branch3.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.69214
Module: inception4b.branch3.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.68954
Module: inception4b.branch3.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69784
Module: inception4b.branch3.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68688
Module: inception4b.branch3.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.68574
Module: inception4b.branch3.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.67396
Module: inception4b.branch3.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.67408
Module: inception4b.branch3.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.67142
Module: inception4b.branch3.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.66452
Module: inception4b.branch3.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.5912
Module: inception4b.branch3.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.58752
Module: inception4b.branch3.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6984
Module: inception4b.branch3.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.6969
Module: inception4b.branch3.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6958
Module: inception4b.branch3.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69238
Module: inception4b.branch3.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.6913
Module: inception4b.branch3.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.69054
Module: inception4b.branch3.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.68846
Module: inception4b.branch3.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.68784
Module: inception4b.branch3.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.6796
Module: inception4b.branch4.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69538
Module: inception4b.branch4.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69386
Module: inception4b.branch4.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69298
Module: inception4b.branch4.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.67698
Module: inception4b.branch4.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.67016
Module: inception4b.branch4.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.6612
Module: inception4b.branch4.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.65194
Module: inception4b.branch4.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.568
Module: inception4b.branch4.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.10502
Module: inception4b.branch4.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69394
Module: inception4b.branch4.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68678
Module: inception4b.branch4.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.67894
Module: inception4b.branch4.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.61258
Module: inception4b.branch4.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.48034
Module: inception4b.branch4.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.27774
Module: inception4b.branch4.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.0544
Module: inception4b.branch4.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.01696
Module: inception4b.branch4.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.00878
Module: inception4c.branch1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.6911
Module: inception4c.branch1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.67826
Module: inception4c.branch1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.64308
Module: inception4c.branch1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.62
Module: inception4c.branch1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.5577
Module: inception4c.branch1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.50106
Module: inception4c.branch1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.44246
Module: inception4c.branch1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.35074
Module: inception4c.branch1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.27288
Module: inception4c.branch1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6976
Module: inception4c.branch1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69488
Module: inception4c.branch1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69278
Module: inception4c.branch1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69054
Module: inception4c.branch1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.68562
Module: inception4c.branch1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.68084
Module: inception4c.branch1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.67132
Module: inception4c.branch1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.64114
Module: inception4c.branch1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.5721
Module: inception4c.branch2.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.6888
Module: inception4c.branch2.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.66826
Module: inception4c.branch2.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.64474
Module: inception4c.branch2.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.61396
Module: inception4c.branch2.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.57352
Module: inception4c.branch2.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.54424
Module: inception4c.branch2.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.4961
Module: inception4c.branch2.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.46164
Module: inception4c.branch2.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.4309
Module: inception4c.branch2.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6941
Module: inception4c.branch2.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68884
Module: inception4c.branch2.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6745
Module: inception4c.branch2.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.67066
Module: inception4c.branch2.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.66352
Module: inception4c.branch2.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.65982
Module: inception4c.branch2.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.64446
Module: inception4c.branch2.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.61066
Module: inception4c.branch2.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.5637
Module: inception4c.branch2.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68988
Module: inception4c.branch2.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.6796
Module: inception4c.branch2.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.66286
Module: inception4c.branch2.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.64548
Module: inception4c.branch2.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.5961
Module: inception4c.branch2.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.54052
Module: inception4c.branch2.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.43994
Module: inception4c.branch2.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.36888
Module: inception4c.branch2.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.30222
Module: inception4c.branch2.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69428
Module: inception4c.branch2.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69016
Module: inception4c.branch2.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68632
Module: inception4c.branch2.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.67606
Module: inception4c.branch2.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.6653
Module: inception4c.branch2.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.64536
Module: inception4c.branch2.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.60516
Module: inception4c.branch2.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.53616
Module: inception4c.branch2.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.40522
Module: inception4c.branch3.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69746
Module: inception4c.branch3.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69726
Module: inception4c.branch3.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69722
Module: inception4c.branch3.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.69644
Module: inception4c.branch3.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.69544
Module: inception4c.branch3.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.6935
Module: inception4c.branch3.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.69296
Module: inception4c.branch3.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.68676
Module: inception4c.branch3.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.67626
Module: inception4c.branch3.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69774
Module: inception4c.branch3.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69838
Module: inception4c.branch3.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69862
Module: inception4c.branch3.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69864
Module: inception4c.branch3.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69822
Module: inception4c.branch3.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.69706
Module: inception4c.branch3.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.69616
Module: inception4c.branch3.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.6958
Module: inception4c.branch3.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.69428
Module: inception4c.branch3.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69598
Module: inception4c.branch3.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69534
Module: inception4c.branch3.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69276
Module: inception4c.branch3.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.68948
Module: inception4c.branch3.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.68868
Module: inception4c.branch3.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.68786
Module: inception4c.branch3.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.68676
Module: inception4c.branch3.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.68338
Module: inception4c.branch3.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.67274
Module: inception4c.branch3.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69784
Module: inception4c.branch3.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69682
Module: inception4c.branch3.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6953
Module: inception4c.branch3.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69438
Module: inception4c.branch3.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69448
Module: inception4c.branch3.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.6918
Module: inception4c.branch3.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.68836
Module: inception4c.branch3.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.67828
Module: inception4c.branch3.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.63378
Module: inception4c.branch4.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68888
Module: inception4c.branch4.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.67906
Module: inception4c.branch4.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.67546
Module: inception4c.branch4.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.66786
Module: inception4c.branch4.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.66234
Module: inception4c.branch4.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.6294
Module: inception4c.branch4.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.61532
Module: inception4c.branch4.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.6104
Module: inception4c.branch4.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.57796
Module: inception4c.branch4.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6969
Module: inception4c.branch4.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69358
Module: inception4c.branch4.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6893
Module: inception4c.branch4.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68526
Module: inception4c.branch4.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.68192
Module: inception4c.branch4.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.66528
Module: inception4c.branch4.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.65378
Module: inception4c.branch4.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.6257
Module: inception4c.branch4.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.58878
Module: inception4d.branch1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68928
Module: inception4d.branch1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.67176
Module: inception4d.branch1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.65414
Module: inception4d.branch1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.6299
Module: inception4d.branch1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.60204
Module: inception4d.branch1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.5604
Module: inception4d.branch1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.5081
Module: inception4d.branch1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.44082
Module: inception4d.branch1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.36716
Module: inception4d.branch1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69788
Module: inception4d.branch1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69526
Module: inception4d.branch1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69434
Module: inception4d.branch1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69214
Module: inception4d.branch1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.68852
Module: inception4d.branch1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.68772
Module: inception4d.branch1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.66574
Module: inception4d.branch1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.64416
Module: inception4d.branch1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.59432
Module: inception4d.branch2.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68378
Module: inception4d.branch2.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.65822
Module: inception4d.branch2.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.62946
Module: inception4d.branch2.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.59678
Module: inception4d.branch2.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.5534
Module: inception4d.branch2.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.50778
Module: inception4d.branch2.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.4723
Module: inception4d.branch2.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.42682
Module: inception4d.branch2.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.41842
Module: inception4d.branch2.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69748
Module: inception4d.branch2.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69356
Module: inception4d.branch2.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69146
Module: inception4d.branch2.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68056
Module: inception4d.branch2.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.66864
Module: inception4d.branch2.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.65744
Module: inception4d.branch2.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.63032
Module: inception4d.branch2.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.5982
Module: inception4d.branch2.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.51014
Module: inception4d.branch2.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69208
Module: inception4d.branch2.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68358
Module: inception4d.branch2.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.67342
Module: inception4d.branch2.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.66106
Module: inception4d.branch2.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.63704
Module: inception4d.branch2.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.60376
Module: inception4d.branch2.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.56196
Module: inception4d.branch2.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.51498
Module: inception4d.branch2.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.44988
Module: inception4d.branch2.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6914
Module: inception4d.branch2.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68348
Module: inception4d.branch2.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.67572
Module: inception4d.branch2.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.65924
Module: inception4d.branch2.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.65078
Module: inception4d.branch2.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.63134
Module: inception4d.branch2.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.60482
Module: inception4d.branch2.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.56782
Module: inception4d.branch2.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.4978
Module: inception4d.branch3.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.6967
Module: inception4d.branch3.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69456
Module: inception4d.branch3.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69352
Module: inception4d.branch3.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.69006
Module: inception4d.branch3.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.6844
Module: inception4d.branch3.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.68198
Module: inception4d.branch3.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.67772
Module: inception4d.branch3.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.67142
Module: inception4d.branch3.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.66042
Module: inception4d.branch3.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69746
Module: inception4d.branch3.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69542
Module: inception4d.branch3.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69556
Module: inception4d.branch3.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69592
Module: inception4d.branch3.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69464
Module: inception4d.branch3.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.69182
Module: inception4d.branch3.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.6886
Module: inception4d.branch3.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.68444
Module: inception4d.branch3.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.67958
Module: inception4d.branch3.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69562
Module: inception4d.branch3.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69434
Module: inception4d.branch3.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.6921
Module: inception4d.branch3.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.68408
Module: inception4d.branch3.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.6821
Module: inception4d.branch3.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.68044
Module: inception4d.branch3.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.67908
Module: inception4d.branch3.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.66898
Module: inception4d.branch3.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.64426
Module: inception4d.branch3.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69604
Module: inception4d.branch3.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69496
Module: inception4d.branch3.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69352
Module: inception4d.branch3.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69258
Module: inception4d.branch3.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69024
Module: inception4d.branch3.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.6867
Module: inception4d.branch3.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.684
Module: inception4d.branch3.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.66256
Module: inception4d.branch3.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.643
Module: inception4d.branch4.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69106
Module: inception4d.branch4.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.66424
Module: inception4d.branch4.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.65246
Module: inception4d.branch4.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.62744
Module: inception4d.branch4.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.61286
Module: inception4d.branch4.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.5903
Module: inception4d.branch4.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.57842
Module: inception4d.branch4.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.48794
Module: inception4d.branch4.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.39074
Module: inception4d.branch4.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6975
Module: inception4d.branch4.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69476
Module: inception4d.branch4.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68906
Module: inception4d.branch4.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68058
Module: inception4d.branch4.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.66372
Module: inception4d.branch4.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.63378
Module: inception4d.branch4.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.58142
Module: inception4d.branch4.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.5207
Module: inception4d.branch4.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.45578
Module: inception4e.branch1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69086
Module: inception4e.branch1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68442
Module: inception4e.branch1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.67664
Module: inception4e.branch1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.66676
Module: inception4e.branch1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.6587
Module: inception4e.branch1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.64356
Module: inception4e.branch1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.62392
Module: inception4e.branch1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.60418
Module: inception4e.branch1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.57768
Module: inception4e.branch1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69656
Module: inception4e.branch1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69482
Module: inception4e.branch1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69148
Module: inception4e.branch1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68974
Module: inception4e.branch1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.6851
Module: inception4e.branch1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.67786
Module: inception4e.branch1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.66656
Module: inception4e.branch1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.6479
Module: inception4e.branch1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.63426
Module: inception4e.branch2.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69232
Module: inception4e.branch2.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.6828
Module: inception4e.branch2.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.67294
Module: inception4e.branch2.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.65284
Module: inception4e.branch2.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.6341
Module: inception4e.branch2.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.60426
Module: inception4e.branch2.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.57732
Module: inception4e.branch2.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.5504
Module: inception4e.branch2.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.53322
Module: inception4e.branch2.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69558
Module: inception4e.branch2.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69408
Module: inception4e.branch2.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68722
Module: inception4e.branch2.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.67398
Module: inception4e.branch2.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.65514
Module: inception4e.branch2.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.63448
Module: inception4e.branch2.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.60038
Module: inception4e.branch2.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.52606
Module: inception4e.branch2.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.45594
Module: inception4e.branch2.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69268
Module: inception4e.branch2.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68388
Module: inception4e.branch2.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.67864
Module: inception4e.branch2.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.66748
Module: inception4e.branch2.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.65758
Module: inception4e.branch2.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.63856
Module: inception4e.branch2.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.62132
Module: inception4e.branch2.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.59734
Module: inception4e.branch2.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.557
Module: inception4e.branch2.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69306
Module: inception4e.branch2.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68538
Module: inception4e.branch2.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6724
Module: inception4e.branch2.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.66032
Module: inception4e.branch2.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.64894
Module: inception4e.branch2.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.62374
Module: inception4e.branch2.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.60068
Module: inception4e.branch2.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.58046
Module: inception4e.branch2.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.55034
Module: inception4e.branch3.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68876
Module: inception4e.branch3.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68774
Module: inception4e.branch3.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.68234
Module: inception4e.branch3.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.68026
Module: inception4e.branch3.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.67492
Module: inception4e.branch3.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.6628
Module: inception4e.branch3.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.65
Module: inception4e.branch3.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.64144
Module: inception4e.branch3.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.63722
Module: inception4e.branch3.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69796
Module: inception4e.branch3.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.697
Module: inception4e.branch3.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69654
Module: inception4e.branch3.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69584
Module: inception4e.branch3.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69342
Module: inception4e.branch3.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.6914
Module: inception4e.branch3.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.68802
Module: inception4e.branch3.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.68566
Module: inception4e.branch3.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.67518
Module: inception4e.branch3.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69666
Module: inception4e.branch3.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.6951
Module: inception4e.branch3.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69152
Module: inception4e.branch3.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.68712
Module: inception4e.branch3.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.685
Module: inception4e.branch3.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.68122
Module: inception4e.branch3.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.67244
Module: inception4e.branch3.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.64156
Module: inception4e.branch3.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.63308
Module: inception4e.branch3.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69484
Module: inception4e.branch3.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69394
Module: inception4e.branch3.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69048
Module: inception4e.branch3.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68824
Module: inception4e.branch3.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.68374
Module: inception4e.branch3.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.6815
Module: inception4e.branch3.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.67372
Module: inception4e.branch3.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.66076
Module: inception4e.branch3.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.62908
Module: inception4e.branch4.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.6937
Module: inception4e.branch4.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.6902
Module: inception4e.branch4.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.6848
Module: inception4e.branch4.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.68004
Module: inception4e.branch4.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.6683
Module: inception4e.branch4.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.65924
Module: inception4e.branch4.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.64904
Module: inception4e.branch4.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.62992
Module: inception4e.branch4.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.6203
Module: inception4e.branch4.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69586
Module: inception4e.branch4.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69408
Module: inception4e.branch4.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68818
Module: inception4e.branch4.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68322
Module: inception4e.branch4.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.67478
Module: inception4e.branch4.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.65342
Module: inception4e.branch4.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.63392
Module: inception4e.branch4.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.61358
Module: inception4e.branch4.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.5977
Module: inception5a.branch1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69236
Module: inception5a.branch1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68342
Module: inception5a.branch1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.67306
Module: inception5a.branch1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.66618
Module: inception5a.branch1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.65752
Module: inception5a.branch1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.64532
Module: inception5a.branch1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.63734
Module: inception5a.branch1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.6265
Module: inception5a.branch1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.61356
Module: inception5a.branch1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69406
Module: inception5a.branch1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69244
Module: inception5a.branch1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68872
Module: inception5a.branch1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68326
Module: inception5a.branch1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.67766
Module: inception5a.branch1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.67114
Module: inception5a.branch1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.66086
Module: inception5a.branch1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.64718
Module: inception5a.branch1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.62816
Module: inception5a.branch2.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.64238
Module: inception5a.branch2.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.63028
Module: inception5a.branch2.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.61038
Module: inception5a.branch2.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.588
Module: inception5a.branch2.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.56806
Module: inception5a.branch2.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.54814
Module: inception5a.branch2.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.52536
Module: inception5a.branch2.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.50636
Module: inception5a.branch2.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.47288
Module: inception5a.branch2.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69634
Module: inception5a.branch2.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69378
Module: inception5a.branch2.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68972
Module: inception5a.branch2.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68208
Module: inception5a.branch2.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.6736
Module: inception5a.branch2.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.66276
Module: inception5a.branch2.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.64702
Module: inception5a.branch2.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.62666
Module: inception5a.branch2.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.58056
Module: inception5a.branch2.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.68316
Module: inception5a.branch2.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.66802
Module: inception5a.branch2.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.654
Module: inception5a.branch2.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.62112
Module: inception5a.branch2.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.59852
Module: inception5a.branch2.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.57232
Module: inception5a.branch2.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.54218
Module: inception5a.branch2.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.47988
Module: inception5a.branch2.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.4021
Module: inception5a.branch2.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.68904
Module: inception5a.branch2.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.6798
Module: inception5a.branch2.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.66548
Module: inception5a.branch2.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.6478
Module: inception5a.branch2.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.62808
Module: inception5a.branch2.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.59958
Module: inception5a.branch2.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.56406
Module: inception5a.branch2.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.48184
Module: inception5a.branch2.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.4554
Module: inception5a.branch3.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69506
Module: inception5a.branch3.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69082
Module: inception5a.branch3.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.6844
Module: inception5a.branch3.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.68242
Module: inception5a.branch3.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.68136
Module: inception5a.branch3.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.67966
Module: inception5a.branch3.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.67932
Module: inception5a.branch3.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.67558
Module: inception5a.branch3.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.67594
Module: inception5a.branch3.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69734
Module: inception5a.branch3.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69678
Module: inception5a.branch3.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69564
Module: inception5a.branch3.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69478
Module: inception5a.branch3.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69272
Module: inception5a.branch3.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.69054
Module: inception5a.branch3.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.68978
Module: inception5a.branch3.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.68656
Module: inception5a.branch3.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.68152
Module: inception5a.branch3.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69168
Module: inception5a.branch3.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.6901
Module: inception5a.branch3.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.68796
Module: inception5a.branch3.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.68368
Module: inception5a.branch3.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.68106
Module: inception5a.branch3.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.6764
Module: inception5a.branch3.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.6728
Module: inception5a.branch3.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.66906
Module: inception5a.branch3.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.66242
Module: inception5a.branch3.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69504
Module: inception5a.branch3.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68786
Module: inception5a.branch3.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.67678
Module: inception5a.branch3.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.66324
Module: inception5a.branch3.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.6583
Module: inception5a.branch3.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.65716
Module: inception5a.branch3.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.6697
Module: inception5a.branch3.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.6705
Module: inception5a.branch3.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.66794
Module: inception5a.branch4.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69256
Module: inception5a.branch4.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.68666
Module: inception5a.branch4.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.68166
Module: inception5a.branch4.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.67288
Module: inception5a.branch4.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.66584
Module: inception5a.branch4.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.6603
Module: inception5a.branch4.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.65248
Module: inception5a.branch4.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.64386
Module: inception5a.branch4.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.63408
Module: inception5a.branch4.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6957
Module: inception5a.branch4.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69152
Module: inception5a.branch4.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.68632
Module: inception5a.branch4.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68144
Module: inception5a.branch4.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.67704
Module: inception5a.branch4.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.66638
Module: inception5a.branch4.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.65442
Module: inception5a.branch4.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.6449
Module: inception5a.branch4.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.62724
Module: inception5b.branch1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69718
Module: inception5b.branch1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.6948
Module: inception5b.branch1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.6926
Module: inception5b.branch1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.68958
Module: inception5b.branch1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.68668
Module: inception5b.branch1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.68232
Module: inception5b.branch1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.67578
Module: inception5b.branch1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.669
Module: inception5b.branch1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.66046
Module: inception5b.branch1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69662
Module: inception5b.branch1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69622
Module: inception5b.branch1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69396
Module: inception5b.branch1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69308
Module: inception5b.branch1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69218
Module: inception5b.branch1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.69032
Module: inception5b.branch1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.68654
Module: inception5b.branch1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.6788
Module: inception5b.branch1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.66336
Module: inception5b.branch2.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.67192
Module: inception5b.branch2.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.64318
Module: inception5b.branch2.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.60856
Module: inception5b.branch2.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.59432
Module: inception5b.branch2.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.53764
Module: inception5b.branch2.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.5566
Module: inception5b.branch2.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.51428
Module: inception5b.branch2.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.48136
Module: inception5b.branch2.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.50674
Module: inception5b.branch2.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6948
Module: inception5b.branch2.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.68912
Module: inception5b.branch2.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6794
Module: inception5b.branch2.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.67392
Module: inception5b.branch2.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.65638
Module: inception5b.branch2.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.63422
Module: inception5b.branch2.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.6144
Module: inception5b.branch2.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.58738
Module: inception5b.branch2.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.5316
Module: inception5b.branch2.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69644
Module: inception5b.branch2.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69384
Module: inception5b.branch2.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69072
Module: inception5b.branch2.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.68604
Module: inception5b.branch2.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.68018
Module: inception5b.branch2.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.67078
Module: inception5b.branch2.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.65834
Module: inception5b.branch2.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.64008
Module: inception5b.branch2.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.61228
Module: inception5b.branch2.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.6865
Module: inception5b.branch2.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.67108
Module: inception5b.branch2.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.65472
Module: inception5b.branch2.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.63586
Module: inception5b.branch2.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.6169
Module: inception5b.branch2.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.59534
Module: inception5b.branch2.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.5714
Module: inception5b.branch2.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.55298
Module: inception5b.branch2.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.5485
Module: inception5b.branch3.0.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69362
Module: inception5b.branch3.0.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.67514
Module: inception5b.branch3.0.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.66766
Module: inception5b.branch3.0.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.64922
Module: inception5b.branch3.0.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.65796
Module: inception5b.branch3.0.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.6455
Module: inception5b.branch3.0.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.64908
Module: inception5b.branch3.0.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.64434
Module: inception5b.branch3.0.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.63744
Module: inception5b.branch3.0.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69774
Module: inception5b.branch3.0.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69668
Module: inception5b.branch3.0.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.69406
Module: inception5b.branch3.0.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69244
Module: inception5b.branch3.0.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.69012
Module: inception5b.branch3.0.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.6841
Module: inception5b.branch3.0.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.67768
Module: inception5b.branch3.0.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.67286
Module: inception5b.branch3.0.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.65976
Module: inception5b.branch3.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69756
Module: inception5b.branch3.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.69656
Module: inception5b.branch3.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69466
Module: inception5b.branch3.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.69328
Module: inception5b.branch3.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.69076
Module: inception5b.branch3.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.68814
Module: inception5b.branch3.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.68454
Module: inception5b.branch3.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.68106
Module: inception5b.branch3.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.67696
Module: inception5b.branch3.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69566
Module: inception5b.branch3.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69194
Module: inception5b.branch3.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6903
Module: inception5b.branch3.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.68704
Module: inception5b.branch3.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.68548
Module: inception5b.branch3.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.68186
Module: inception5b.branch3.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.67902
Module: inception5b.branch3.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.67736
Module: inception5b.branch3.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.67372
Module: inception5b.branch4.1.conv, Pruning Rate: 0.1, Dim: 0, Accuracy: 0.69656
Module: inception5b.branch4.1.conv, Pruning Rate: 0.2, Dim: 0, Accuracy: 0.6956
Module: inception5b.branch4.1.conv, Pruning Rate: 0.3, Dim: 0, Accuracy: 0.69442
Module: inception5b.branch4.1.conv, Pruning Rate: 0.4, Dim: 0, Accuracy: 0.69216
Module: inception5b.branch4.1.conv, Pruning Rate: 0.5, Dim: 0, Accuracy: 0.6906
Module: inception5b.branch4.1.conv, Pruning Rate: 0.6, Dim: 0, Accuracy: 0.68858
Module: inception5b.branch4.1.conv, Pruning Rate: 0.7, Dim: 0, Accuracy: 0.68732
Module: inception5b.branch4.1.conv, Pruning Rate: 0.8, Dim: 0, Accuracy: 0.68516
Module: inception5b.branch4.1.conv, Pruning Rate: 0.9, Dim: 0, Accuracy: 0.68272
Module: inception5b.branch4.1.conv, Pruning Rate: 0.1, Dim: 1, Accuracy: 0.69746
Module: inception5b.branch4.1.conv, Pruning Rate: 0.2, Dim: 1, Accuracy: 0.69584
Module: inception5b.branch4.1.conv, Pruning Rate: 0.3, Dim: 1, Accuracy: 0.6945
Module: inception5b.branch4.1.conv, Pruning Rate: 0.4, Dim: 1, Accuracy: 0.69354
Module: inception5b.branch4.1.conv, Pruning Rate: 0.5, Dim: 1, Accuracy: 0.6923
Module: inception5b.branch4.1.conv, Pruning Rate: 0.6, Dim: 1, Accuracy: 0.69116
Module: inception5b.branch4.1.conv, Pruning Rate: 0.7, Dim: 1, Accuracy: 0.69112
Module: inception5b.branch4.1.conv, Pruning Rate: 0.8, Dim: 1, Accuracy: 0.68886
Module: inception5b.branch4.1.conv, Pruning Rate: 0.9, Dim: 1, Accuracy: 0.68686
"""


def plotLocal(data):
    # process the data
    records = [line.split(", ") for line in data.strip().split("\n")]

    columns = ["Module", "Pruning Rate", "Accuracy"]
    df = pd.DataFrame(records, columns=columns)

    # convert data types
    df["Pruning Rate"] = df["Pruning Rate"].str.split(
        ": ").str[1].astype(float)
    df["Accuracy"] = df["Accuracy"].str.split(": ").str[1].astype(float)

    # attribute indices to modules
    df['Module'] = df['Module'].astype('category').cat.codes

    # create a grid for the surface area
    pruning_rate_grid, module_grid = np.meshgrid(
        np.linspace(df['Pruning Rate'].min(), df['Pruning Rate'].max(), 50),
        np.linspace(df['Module'].min(), df['Module'].max(), 50)
    )

    # interpolate the accuracies
    accuracy_grid = griddata(
        (df['Pruning Rate'], df['Module']),
        df['Accuracy'],
        (pruning_rate_grid, module_grid),
        method='linear'
    )

    # create the 3d plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(pruning_rate_grid, module_grid,
                           accuracy_grid, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Pruning Rate')
    ax.set_ylabel('Module Index')
    ax.set_zlabel('Accuracy')

    # legend
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Accuracy')

    plt.show()


def plotGlobal(data):
    # process the data
    records = [line.split(", ") for line in data.strip().split("\n")]

    columns = ["Pruning Rate", "Accuracy"]
    df = pd.DataFrame(records, columns=columns)

    # convert data types
    df["Pruning Rate"] = df["Pruning Rate"].str.split(
        ": ").str[1].astype(float)
    df["Accuracy"] = df["Accuracy"].str.split(": ").str[1].astype(float)

    # plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df["Pruning Rate"], df["Accuracy"],
             marker='o', linestyle='-', color='b')
    plt.xlabel('Pruning Rate')
    plt.ylabel('Accuracy')
    plt.title('Pruning Rate vs Accuracy')
    plt.grid(True)
    plt.show()


def plotLocalWorkAround(data):
    # process the data
    records = [line.split(", ") for line in data.strip().split("\n")]

    # add the dim dimension
    columns = ["Module", "Pruning Rate", "Dim", "Accuracy"]
    df = pd.DataFrame(records, columns=columns)

    # convert data types
    df["Pruning Rate"] = df["Pruning Rate"].str.split(
        ": ").str[1].astype(float)
    df["Dim"] = df["Dim"].str.split(": ").str[1].astype(int)
    df["Accuracy"] = df["Accuracy"].str.split(": ").str[1].astype(float)

    # to only select convolutional layers
    df = df[df["Dim"] == 0]

    # attribute indices to modules
    df['Module'] = df['Module'].astype('category').cat.codes

    # grid for the surface area
    pruning_rate_grid, module_grid = np.meshgrid(
        np.linspace(df['Pruning Rate'].min(), df['Pruning Rate'].max(), 50),
        np.linspace(df['Module'].min(), df['Module'].max(), 50)
    )

    # interpolate the accuracies
    accuracy_grid = griddata(
        (df['Pruning Rate'], df['Module']),
        df['Accuracy'],
        (pruning_rate_grid, module_grid),
        method='linear'
    )

    # create a 3d plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(pruning_rate_grid, module_grid,
                           accuracy_grid, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Pruning Rate')
    ax.set_ylabel('Module Index')
    ax.set_zlabel('Accuracy')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Accuracy')

    plt.show()

# examines which pruning rate makes the accuracy drop below a certain threshold 
def find_pruning_rate_below_threshold(data, threshold):

    # Parsing data into a DataFrame
    records = [line.split(", ") for line in data.split("\n") if line.strip()]
    df = pd.DataFrame(records, columns=["Module", "Pruning Rate", "Accuracy"])
    
    # Clean up data
    df["Module"] = df["Module"].str.split(": ").str[1]
    df["Pruning Rate"] = df["Pruning Rate"].str.split(": ").str[1].astype(float)
    df["Accuracy"] = df["Accuracy"].str.split(": ").str[1].astype(float)
    result = {}
    modules = df["Module"].unique()

    for module in modules:
        module_df = df[df["Module"] == module]
        below_threshold = module_df[module_df["Accuracy"] < threshold]

        if not below_threshold.empty:
            pruning_rate = below_threshold.iloc[0]["Pruning Rate"]
            result[module] = pruning_rate
        else:
            result[module] = 0.9

    return result

# plots which modules have a significant impact on the accuracy if pruned
def plot_threshold_rates(result, threshold):

    modules = list(result.keys())
    pruning_rates = list(result.values())

    module_indices = list(range(1, len(modules) + 1))

    plt.figure(figsize=(10, 6))
    plt.bar(module_indices, pruning_rates)

    plt.xlabel('Module Index')
    plt.ylabel('Pruning Rate')
    plt.title(
        f'Pruning Rates for Modules below Accuracy Threshold {threshold}')

    # smaller font size to prevent overlapping
    plt.xticks(module_indices, module_indices, fontsize=6)

    plt.show()

    # map the module indices onto the module names
    index_module_map = {index: module for index,
                        module in enumerate(modules, start=1)}
    for index, module in index_module_map.items():
        print(f'Index: {index}, Module: {module}')

    return index_module_map


def filter_modules_below_pruning_threshold(index_module_map, result, pruning_threshold):
    # filter out modules below the pruning threshold
    filtered_modules = {index: module for index, module in index_module_map.items(
    ) if result[module] < pruning_threshold}

    return filtered_modules


# define a threshold 
threshold = 0.68
pruning_threshold = 0.8

# find pruning rates
#result = find_pruning_rate_below_threshold(
#    data_local_unstructured_l1, threshold)

# plot and show threshold rates
#index_module_map = plot_threshold_rates(result, threshold)

# filter out modules
#filtered_modules = filter_modules_below_pruning_threshold(
#    index_module_map, result, pruning_threshold)
#print(filtered_modules)


# Output the result
# for module, rate in result.items():
#     print(f"Module: {module}, Pruning Rate below threshold {threshold}: {rate}")

def number_weights():
    # load the GoogleNet
    googlenet = models.googlenet(pretrained=True)

    num_weights = []

    # traverse through each layer and append number of output channels
    for name, layer in googlenet.named_modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            number_weights = layer.weight.numel()
            num_weights.append({"Layer Name": name, "Number of Weights": number_weights})

def number_channels():
    # load the GoogleNet
    googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)

    num_channel = [] 

    # traverse through each layer and append number of output channels
    for name, layer in googlenet.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            num_channel.append({"Layer Name": name, "Out Channels": layer.out_channels})

    df = pd.DataFrame(num_channel)

    return df

def correlation_channel_accuracy_pruning(data):
    # process the data
    records = [line.split(", ") for line in data.strip().split("\n")]

    # add the dim dimension
    columns = ["Module", "Pruning Rate", "Dim", "Accuracy"]
    df = pd.DataFrame(records, columns=columns)

    # convert data types
    df["Pruning Rate"] = df["Pruning Rate"].str.split(": ").str[1].astype(float)
    df["Dim"] = df["Dim"].str.split(": ").str[1].astype(int)
    df["Accuracy"] = df["Accuracy"].str.split(": ").str[1].astype(float)

    # to only select convolutional layers
    df = df[df["Dim"] == 1]

    # extract the module names from the "Module" column
    df['Module'] = df['Module'].str.split(": ").str[1]

    num_channels = number_channels()

    # Merge the number of channels with the provided data
    df = df.merge(num_channels, left_on='Module', right_on='Layer Name', how='left')

    # Drop rows where Out Channels is NaN (not a convolutional layer)
    df.dropna(subset=['Out Channels'], inplace=True)

    # Calculate correlation
    correlation_matrix = df[['Out Channels', 'Pruning Rate', 'Accuracy']].corr()

    return correlation_matrix

def visualize_correlation_matrix(correlation_matrix):
    # Plot heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

def overall_correlation_score(correlation_matrix):
    # Calculate the average absolute correlation
    avg_abs_corr = correlation_matrix.abs().mean().mean()
    return avg_abs_corr

correlation_matrix = correlation_channel_accuracy_pruning(data_local_structured_l1)
print(correlation_matrix)

# Visualize the correlation matrix
visualize_correlation_matrix(correlation_matrix)

# Calculate overall correlation score
overall_score = overall_correlation_score(correlation_matrix)
# we obtain a correlation score of 0.43, which is a moderate correlation
print(f'Overall Correlation Score: {overall_score}')