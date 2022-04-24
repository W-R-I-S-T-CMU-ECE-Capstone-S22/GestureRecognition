import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cf_matrix = np.array([[50, 2, 0, 6, 8, 0, 0],
[0, 44, 3, 1, 0, 1, 0],
[0, 4, 45, 0, 2, 0, 1],
[0, 0, 2, 41, 1, 0, 0],
[0, 0, 0, 2, 39, 0, 0],
[0, 0, 0, 0, 0, 39, 12],
[0, 0, 0, 0, 0, 10, 37]])

ax = sns.heatmap(cf_matrix.T/50, annot=True, fmt='.0%', cmap='Blues')

ax.set_title('WRIST Gesture Classification')
ax.set_ylabel('\nPerformed Gesture')
ax.set_xlabel('3D model action')

ax.yaxis.set_ticklabels(["No gesture", "Swipe left", "Swipe right", "Swipe up",
                         "Swipe down", "Two finger swipe left",
                         "Two finger swipe right"])
ax.xaxis.set_ticklabels(["Nothing", "Rotate CCW along y", "Rotate CW along y",
                         "Rotate up along x",
                         "Rotate down along x", "Zoom out",
                         "Zoom in"])

plt.xticks(rotation=90)
plt.yticks(rotation=0)

# plt.show()
plt.tight_layout()
plt.savefig("confusion_matrix.png")
