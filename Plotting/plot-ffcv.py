import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Daten für jede Pruningsrate
accuracy_0_442 = [
    0.66, 0.52908, 0.56306, 0.5645, 0.5879, 0.58322, 0.59812, 0.60502, 0.62498,
    0.62386, 0.63788, 0.64072, 0.64416, 0.65274, 0.65744, 0.65692, 0.66568,
    0.67186, 0.67296, 0.68094, 0.68278, 0.68568, 0.68824, 0.69198, 0.69626,
    0.69782, 0.69868, 0.70168, 0.70296, 0.7064, 0.70632, 0.70808, 0.70958,
    0.71062, 0.71224, 0.71306, 0.7148, 0.71428, 0.71526, 0.71592, 0.71664,
    0.7165, 0.71642, 0.7195, 0.71912, 0.71884, 0.71598, 0.71836, 0.718,
    0.71936, 0.71938
]

accuracy_0_602 = [
    0.69024, 0.48614, 0.5134, 0.55564, 0.55286, 0.57556, 0.59708, 0.60324, 0.61904,
    0.62008, 0.62396, 0.63218, 0.64178, 0.65082, 0.65548, 0.66302, 0.66766,
    0.67572, 0.67706, 0.67728, 0.68436, 0.6848, 0.68946, 0.68584, 0.69288,
    0.69558, 0.69648, 0.6997, 0.70102, 0.70524, 0.70594, 0.7066, 0.71016,
    0.7109, 0.7111, 0.7124, 0.71248, 0.71202, 0.71432, 0.71444, 0.71488,
    0.7158, 0.71726, 0.71708, 0.71648, 0.716, 0.71578, 0.71622, 0.71794,
    0.71668, 0.718
]

accuracy_0_748 = [
    0.53044, 0.45616, 0.51008, 0.54384, 0.54538, 0.5773, 0.5856, 0.60034, 0.5917,
    0.61446, 0.61896, 0.61754, 0.6355, 0.63602, 0.64686, 0.65034, 0.65844,
    0.6613, 0.66798, 0.6714, 0.6672, 0.68178, 0.68206, 0.68302, 0.68788,
    0.68654, 0.69202, 0.69284, 0.69258, 0.69354, 0.69584, 0.70098, 0.7004,
    0.70166, 0.70434, 0.70548, 0.70408, 0.70524, 0.70452, 0.70464, 0.70544,
    0.70556, 0.70648, 0.70718, 0.70768, 0.7054, 0.7072, 0.70786, 0.7079,
    0.70654, 0.70828
]

accuracy_0_838 = [
    0.18154, 0.4571, 0.4975, 0.50824, 0.52788, 0.53846, 0.56206, 0.55906, 0.57982,
    0.58106, 0.59602, 0.60846, 0.61196, 0.6136, 0.62432, 0.62658, 0.63482,
    0.64078, 0.64188, 0.64552, 0.65122, 0.65702, 0.65788, 0.6598, 0.6612,
    0.66586, 0.66834, 0.66748, 0.67172, 0.67058, 0.67274, 0.67644, 0.67846,
    0.67816, 0.68002, 0.67796, 0.68094, 0.6816, 0.68346, 0.6833, 0.68302,
    0.68296, 0.6843, 0.68696, 0.68484, 0.68492, 0.68626, 0.68726, 0.68736,
    0.68652, 0.6859
]

# Daten erstellen
epochs = list(range(1, 52))

# Initialer Accuracy-Wert vor dem Pruning
initial_accuracy = 0.698

# Pruning Raten und dazugehörige Accuracy-Werte
actual_pruning_rate = [
    0.44231335547447703,
    0.6024873112779936,
    0.7475423470358473,
    0.838124713209701
]

accuracy_after_pruning = [
    0.66,  # nach Actual Pruning Rate: 0.44231335547447703
    0.69024,  # nach Actual Pruning Rate: 0.6024873112779936
    0.53044,  # nach Actual Pruning Rate: 0.7475423470358473
    0.18154   # nach Actual Pruning Rate: 0.838124713209701
]

accuracy_after_retraining = [
    0.71938,
    0.718,
    0.70828,
    0.6859
]

# Plot erstellen
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Startpunkt mit initialer Accuracy
ax.plot([0, 0], [0.0, actual_pruning_rate[0]], [
        initial_accuracy, accuracy_after_pruning[0]], marker='o')

# Plot für jede Pruningsrate und Verbinden der letzten Punkte
ax.plot(epochs, [actual_pruning_rate[0]]*51, accuracy_0_442,
        label='Pruning Rate: 0.4423', marker='o')
ax.plot([51, 0], [actual_pruning_rate[0], actual_pruning_rate[1]],
        [accuracy_0_442[-1], accuracy_after_pruning[1]], marker='o')

ax.plot(epochs, [actual_pruning_rate[1]]*51, accuracy_0_602,
        label='Pruning Rate: 0.6025', marker='^')
ax.plot([51, 0], [actual_pruning_rate[1], actual_pruning_rate[2]],
        [accuracy_0_602[-1], accuracy_after_pruning[2]], marker='^')

ax.plot(epochs, [actual_pruning_rate[2]]*51, accuracy_0_748,
        label='Pruning Rate: 0.7475', marker='s')
ax.plot([51, 0], [actual_pruning_rate[2], actual_pruning_rate[3]],
        [accuracy_0_748[-1], accuracy_after_pruning[3]], marker='s')

ax.plot(epochs, [actual_pruning_rate[3]]*51, accuracy_0_838,
        label='Pruning Rate: 0.8381', marker='d')

# Letzter Punkt
ax.plot([51, 52], [actual_pruning_rate[3], actual_pruning_rate[3]], [
        accuracy_0_838[-1], accuracy_after_retraining[3]], marker='d')

ax.set_xlabel('Epochs')
ax.set_ylabel('Actual Pruning Rate')
ax.set_zlabel('Accuracy')
ax.set_title('Accuracy vs. Actual Pruning Rate and Epochs')
ax.legend()

plt.show()
