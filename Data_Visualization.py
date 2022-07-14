#########################################
# Data Visualization : Matplotlib & Seaborn
#########################################

#####################################
# MATPLOTLIB
#####################################

# Kategorik Değişken: sütun grafik. countplot bar
# Sayısal Değişken: hist, boxplot

######################################
# Kategorik Değişken Görselleştirme
######################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind='bar')
plt.show()

#######################################
# Sayısal Değişken Görselleştirme
#######################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

#######################################
# Matplotlib'in Özellikleri
#######################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#####################
# plot
#####################

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

#######################
# Marker
#######################
y = np.array([13, 28, 11, 100])

plt.plot(y, marker='o')
plt.show()

######################
# line
######################

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dotted", color="r")
plt.show()

######################
# Multiple Lines
######################

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])

plt.plot(x)
plt.plot(y)
plt.show()

########################
# Labels
########################

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)
# Başlık
plt.title("Bu ana başlık")

# X eksenine isimlendirme
plt.xlabel("X ekseni isimlendirme")

plt.ylabel("Y ekseni isimlendirme")

plt.grid()
plt.show()

#########################
# Subplots
#########################

# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)
plt.show()

#############################
# Seaborn
#############################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

# Sayısal değişkenleri görselleştirme

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()
