import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('accelerometer_gyro_mobile_phone_dataset.csv')
# Удалил поле с временным отпечатком, так как он тут не несет никакой смысловой нагрузки
df = df.drop('timestamp', axis=1)

print(df.describe())

print("Названия столбцов:", df.columns)
print("Названия строк:", df.index)


# Визулизация каждого признака

# аккселерометр
sns.histplot(df['accX'])
plt.title('Распределение значений accX')
plt.show()

sns.histplot(x='Activity', y='accX', data=df)
plt.title('accX для активных и неактивных пользователей')
plt.show()

sns.histplot(df['accY'])
plt.title('Распределение значений accY')
plt.show()

sns.histplot(x='Activity', y='accY', data=df)
plt.title('accY для активных и неактивных пользователей')
plt.show()

sns.histplot(df['accZ'])
plt.title('Распределение значений accZ')
plt.show()

sns.histplot(x='Activity', y='accZ', data=df)
plt.title('accZ для активных и неактивных пользователей')
plt.show()

# гироскоп
sns.histplot(df['gyroX'])
plt.title('Распределение значений gyroX')
plt.show()

sns.histplot(x='Activity', y='gyroX', data=df)
plt.title('gyroX для активных и неактивных пользователей')
plt.show()

sns.histplot(df['gyroY'])
plt.title('Распределение значений gyroY')
plt.show()

sns.histplot(x='Activity', y='gyroY', data=df)
plt.title('gyroY для активных и неактивных пользователей')
plt.show()


sns.histplot(df['gyroZ'])
plt.title('Распределение значений gyroZ')
plt.show()

sns.histplot(x='Activity', y='gyroZ', data=df)
plt.title('gyroZ для активных и неактивных пользователей')
plt.show()



# Групповые визуализации

sns.pairplot(df[['accX', 'accY', 'accZ', 'Activity']], hue='Activity', kind='scatter')
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()

sns.pairplot(df[['gyroX', 'gyroY', 'gyroZ', 'Activity']], hue='Activity', kind='scatter')
plt.show()