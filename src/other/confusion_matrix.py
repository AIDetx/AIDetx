from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Data 1 Human
# Total samples: 16223
# Samples classified as human: 16219, 99.98%
# Samples classified as AI: 4, 0.02%

# Data 1 AI
# Total samples: 17963
# Samples classified as human: 121, 0.67%
# Samples classified as AI: 17842, 99.33%

# Data 2 Human
# Total samples: 3593
# Samples classified as human: 3462, 96.35%
# Samples classified as AI: 131, 3.65%

# Data 2 AI
# Total samples: 2619
# Samples classified as human: 20, 0.76%
# Samples classified as AI: 2599, 99.24%

data1_human_correct = 16219
data1_human_incorrect = 4
data1_ai_correct = 17842
data1_ai_incorrect = 121

data2_human_correct = 3462
data2_human_incorrect = 131
data2_ai_correct = 2599
data2_ai_incorrect = 20

data1_y_pred = np.array([0 for _ in range(data1_human_correct)] + [1 for _ in range(data1_ai_correct)] + [0 for _ in range(data1_ai_incorrect)] + [1 for _ in range(data1_human_incorrect)])
data1_y_test = np.array([0 for _ in range(data1_human_correct)] + [1 for _ in range(data1_ai_correct)] + [1 for _ in range(data1_ai_incorrect)] + [0 for _ in range(data1_human_incorrect)])

data2_y_pred = np.array([0 for _ in range(data2_human_correct)] + [1 for _ in range(data2_ai_correct)] + [0 for _ in range(data2_ai_incorrect)] + [1 for _ in range(data2_human_incorrect)])
data2_y_test = np.array([0 for _ in range(data2_human_correct)] + [1 for _ in range(data2_ai_correct)] + [1 for _ in range(data2_ai_incorrect)] + [0 for _ in range(data2_human_incorrect)])

labels = ["Human", "AI"]

cm = confusion_matrix(data1_y_test, data1_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.savefig("src/other/data1_results_matrix.png")

cm2 = confusion_matrix(data2_y_test, data2_y_pred)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=labels)
disp2.plot(cmap=plt.cm.Blues)
plt.savefig("src/other/data2_results_matrix.png")

