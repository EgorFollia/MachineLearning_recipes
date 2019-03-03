# Is it a greyhound or a labrador
import numpy as np
import matplotlib.pyplot as plt

# population
greyhounds = 500
labradors = 500

# average height
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labradors)

# height histograms
fig, axs = plt.subplots(1, 3, sharey = True, tight_layout = True)
axs[0].hist(grey_height, stacked = True, color = 'grey', label = 'Greyhound')
axs[1].hist(lab_height, stacked = True, color = 'brown', label = 'Labrador')

axs[0].set_xlabel('Height')
axs[0].set_ylabel('# of dogs')

axs[1].set_xlabel('Height')
axs[1].set_ylabel('# of dogs')

axs[2].hist([grey_height, lab_height], width = 1.8, stacked = True, color = ['grey', 'brown'])
axs[2].set_xlabel('Correlation')

fig.tight_layout()
fig.legend()
plt.show()

# need more features
# - Informative, independent, simple
