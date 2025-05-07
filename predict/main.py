import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from clover2.clover.locart import LocartSplit
from clover2.clover.scores import RegressionScore
from mapie.regression import MapieRegressor
from sklearn.decomposition import FastICA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.ensemble import GradientBoostingRegressor


# from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA, KernelPCA

def calculate_coverage(y_true, y_pred_lower, y_pred_upper):
    in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
    coverage = np.mean(in_interval.astype(float))
    return coverage

def calculate_interval_width(y_pred_lower, y_pred_upper):
    """
    计算区间宽度

    Args:
        y_pred_lower (torch.Tensor): 预测区间的下界
        y_pred_upper (torch.Tensor): 预测区间的上界

    Returns:
        float: 平均区间宽度
    """
    # 计算区间宽度
    interval_width = np.mean(y_pred_upper - y_pred_lower).item()
    return interval_width


# lr001 sheet2 .xlsx是展示结果的数据来源
# lr001 sheet5
path = "/Users/ranli/PycharmProjects/PythonProject/lr001.xlsx"
df = pd.read_excel(path, sheet_name="Sheet1", skiprows=1)

# path = r"E:\算法学习\算法练习\combine_try/cleaned_data.xlsx"
# df = pd.read_excel(path, sheet_name="Sheet1", skiprows=1)
array = df.values

data_train, data_temp = train_test_split(array, test_size=0.2, shuffle=False)
data_validation, data_test = train_test_split(data_temp, test_size=0.6, shuffle=False) # 0.6
x0 = data_train[1:, 20:]
xcal0 = data_validation[1:, 20:]
x_t0 = data_test[455:, 20:]  # 455
# 出口厚度
# x0 = data_train[1:, :-20]
# y0 = data_train[1:, -2]
# xcal0 = data_validation[1:, :-20]
# ycal0 = data_validation[1:, -2]
# x_t0 = data_test[205:, :-20]
# y_t0 = data_test[205:, -2]

y0 = data_train[1:, -1]
ycal0 = data_validation[1:, -1]
y_t0 = data_test[455:, -1]

scaler = StandardScaler()
x = scaler.fit_transform(x0)
xcal = scaler.transform(xcal0)
x_t = scaler.transform(x_t0)
scaler2 = StandardScaler()
y = scaler2.fit_transform(y0.reshape(-1, 1))
ycal = scaler2.transform(ycal0.reshape(-1, 1))
y_t = scaler2.transform(y_t0.reshape(-1, 1))

# Rdi = PCA(n_components=20)   #Test_MAE: 0.178 能稍微提升点效果
# x = Rdi.fit_transform(x)
# xcal = Rdi.transform(xcal)
# x_t = Rdi.transform(x_t)



# svr = svm.NuSVR(C=5, gamma=0.5, kernel='rbf')

# svr = svm.SVR(C=0.5, gamma=0.05, kernel='rbf')  # Test_MAE: 0.82825
# svr = svm.SVR(C=0.9, gamma=0.05, kernel='rbf')  # Test_MAE: 0.790
# svr = svm.SVR(C=1, gamma=0.05, kernel='rbf')  # Test_MAE: 0.7898930260926385

svr = svm.SVR(C=2, gamma=0.05, kernel='rbf')  # Test_MAE: 0.7738
# svr = svm.SVR(kernel='linear', C=1.0)


# svr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
svr.fit(x, y.ravel());



# Set miscalibration level
alpha = 0.05 # 95%
# alpha = 0.10  # 90%
# alpha = 0.20  # 80%

# Defining the class with conformity score, base model, miscalibration level and is_fitted paramter (optional)
locart = LocartSplit(RegressionScore, svr, alpha=alpha, is_fitted=True)
# fitting the base model to the training set
locart.fit(x, y.ravel())

# computing local cutoffs by fitting regression tree to our calibration set
locart.calib(xcal, ycal.ravel());

# changing cart_type to "RF" to fit loforest
loforest = LocartSplit(RegressionScore, svr, alpha=alpha, cart_type="RF", is_fitted = True)

# fitting base model
loforest.fit(x, y.ravel())

# computing local cutoffs by fitting random forest to our calibration set
loforest.calib(xcal, ycal.ravel());



# fitting mapie
mapie = MapieRegressor(svr, method='base', cv='prefit')
mapie.fit(xcal, ycal.ravel())

# values for prediction intervals
# x_values = np.linspace(val_df['x'].min(), val_df['x'].max(), 500).reshape(-1, 1)
y_values = svr.predict(x_t)

mae = mean_absolute_error(y_t, y_values)

print("Test_MAE:", mae)

y_pred, mapie_values = mapie.predict(x_t, alpha=alpha)

locart_values = locart.predict(x_t)
loforest_values = loforest.predict(x_t)

# # 计算覆盖率
# coverage1 = calculate_coverage(y_t0, mapie_values[:, 1], mapie_values[:, 0]);
# print(f"Coverage1: {coverage1}")
# coverage2 = calculate_coverage(y_t0, locart_values[:, 0], locart_values[:, 1]);
# print(f"Coverage2: {coverage2}")
# coverage3 = calculate_coverage(y_t0, loforest_values[:, 0], loforest_values[:, 1]);
# print(f"Coverage3: {coverage3}")
#
# # 计算区间宽度
# interval_width1 = calculate_interval_width(mapie_values[:, 0], mapie_values[:, 1])
# print(f"Interval Width1: {interval_width1}")
# interval_width2 = calculate_interval_width(locart_values[:, 0], locart_values[:, 1])
# print(f"Interval Width2: {interval_width2}")
# interval_width3 = calculate_interval_width(loforest_values[:, 0], loforest_values[:, 1])
# print(f"Interval Width3: {interval_width3}")

y_values = scaler2.inverse_transform(y_values.reshape(-1, 1))
# # 如果 y_values 是一维数组，将其 reshape 为二维数组

# df = pd.DataFrame(y_values, columns=['Y_Values'])
# df.to_excel('y_values.xlsx', index=False)

y_pis1 = scaler2.inverse_transform(mapie_values[:, 0].reshape(-1, 1))
y_pis2 = scaler2.inverse_transform(mapie_values[:, 1].reshape(-1, 1))
y_pis3 = scaler2.inverse_transform(locart_values[:, 0].reshape(-1, 1))
y_pis4 = scaler2.inverse_transform(locart_values[:, 1].reshape(-1, 1))
y_pis5 = scaler2.inverse_transform(loforest_values[:, 0].reshape(-1, 1))
y_pis6 = scaler2.inverse_transform(loforest_values[:, 1].reshape(-1, 1))


# 计算覆盖率
coverage1 = calculate_coverage(y_t0, y_pis1, y_pis2);
print(f"Coverage1: {coverage1}")
coverage2 = calculate_coverage(y_t0, y_pis3, y_pis4);
print(f"Coverage2: {coverage2}")
coverage3 = calculate_coverage(y_t0, y_pis5, y_pis6);
print(f"Coverage3: {coverage3}")

# 计算区间宽度
interval_width1 = calculate_interval_width(y_pis1, y_pis2)
print(f"Interval Width1: {interval_width1}")
interval_width2 = calculate_interval_width(y_pis3, y_pis4)
print(f"Interval Width2: {interval_width2}")
interval_width3 = calculate_interval_width(y_pis5, y_pis6)
print(f"Interval Width3: {interval_width3}")

# yy = np.ravel(y_pis1)
# yy2 = np.ravel(y_pis2)

plt.figure(1)
# plot regression split prediction intervals
# plt.plot(range(len(y_t0)), y_t0, color='black', label='True Values')

plt.plot(range(len(y_t0)), y_t0, color='black', label='True Values')
plt.plot(range(len(y_t0)), y_values, color='tab:blue', label='Predict Values')
plt.fill_between(range(len(y_t0)), np.ravel(y_pis1), np.ravel(y_pis2), color='#E6E6FA', label='PI')
# plt.fill_between(range(len(y_t0)), np.ravel(y_pis1), np.ravel(y_pis2), color='#E6E6FA', label='PI')
# plt.scatter(range(len(y_t0)), y_t0, color='red', label='True Values', marker='o',s=5)
plt.title('GCM-CP')
# plt.title('LSSVM-CP')
plt.legend()

plt.figure(2)
# plot locart prediction intervals
# plt.plot(range(len(y_t0)), y_t0, color='black', label='True Values')
plt.plot(range(len(y_t0)), y_t0, color='black', label='True Values')
plt.plot(range(len(y_t0)), y_values, color='tab:blue', label='Predict Values')

plt.fill_between(range(len(y_t0)), np.ravel(y_pis3), np.ravel(y_pis4), color='#E6E6FA', label='PI')
# plt.scatter(range(len(y_t0)), y_t0, color='red', label='True Values', marker='o',s=5)

plt.title('GCM-FCP')
# plt.title('LSSVM-FCP')
plt.legend()
# plt.ylim(bottom=-50, top=250)  # 例如，设置 y 轴范围从 0 到 100


plt.figure(3)
# plot loforest prediction intervals


plt.plot(range(len(y_t0)), y_t0, color='black', label='True Values')
plt.plot(range(len(y_t0)), y_values, color='tab:blue', label='Predict Values')
plt.fill_between(range(len(y_t0)), np.ravel(y_pis5), np.ravel(y_pis6), color='#E6E6FA', label='PI')
# plt.scatter(range(len(y_t0)), y_t0, color='red', label='True Values', marker='o',s=5)
# plt.scatter(range(len(y_t0)), np.ravel(y_pis5), color='blue', label='SW', marker='x',s=5)
# plt.scatter(range(len(y_t0)), np.ravel(y_pis6), color='blue', label='LW', marker='x',s=5)

plt.title('GCMAC2')
plt.legend()
plt.tight_layout()
plt.show()

# lr001 sheet1 Test_MAE: 0.6281798276667301  sheet2  0.44131046597513296
