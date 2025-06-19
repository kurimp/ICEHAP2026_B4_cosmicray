import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import glob #ファイル探索
from natsort import natsorted #ソート
import pandas as pd
from scipy.optimize import curve_fit

#カレントディレクトリを本ファイルが存在するフォルダに変更。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#実験結果をまとめた最新ファイルを参照し、データフレーム化。
result_filename_list = natsorted(glob.glob('./pocket_counter_output/results/*.csv'))
result_filename = result_filename_list[-1]
df_res_base = pd.read_csv(result_filename, sep='\t', header = 0)

#検出器のefficiencyデータを参照し、データフレーム化。
eff_filename = './performance/results/performance.csv'
df_eff = pd.read_csv(eff_filename, sep='\t')
eff_dict = df_eff.set_index('COM')['value'].to_dict()

# 1. 重み定義DataFrameから、COMの種類と重み係数のマップ（辞書）を作成
# set_indexでCOM_typeをインデックスにし、weight_factor列を選択してto_dict()で辞書化
com_weights_map = df_eff.set_index('COM')['value'].to_dict()

print("\n--- 作成されたCOMタイプと重みのマップ (辞書) ---")
print(com_weights_map)
print("-" * 30)

# 2. 'COM' 列の値をキーとして、対応する重み係数を取得する新しいSeriesを作成
# map() メソッドは、Seriesの値を使って、辞書のキーに対応する値をルックアップします。
# マップに存在しないCOMタイプ (例: 'COM4', 'COM9') の場合は NaN になります。
mapped_weights = df_res_base['COM'].map(com_weights_map)

print("\n--- df_data['COM']にマップされた重み ---")
print(mapped_weights)
print("-" * 30)

# 3. totalcount 列に重みを適用
# NaNでない場合にのみ掛け算を適用するための処理
# オプション: 元の値を保持しつつ、重みがない場合はNaNを結果にしたい場合
# df_data['weighted_totalcount'] = df_data['totalcount'] * mapped_weights

# 推奨: 重みが設定されていないCOMタイプの場合は、totalcountをそのまま残す（または別のデフォルト値を設定）
# mapped_weights の NaN を 1.0 (重みなし) に置き換える
final_weights = mapped_weights.fillna(1.0) # 重みがない場合は1倍（元の値のまま）

# totalcount 列を更新 (または新しい列を作成)
df_res_base['totalcount_weighted'] = df_res_base['totalcount'] * final_weights

print("\n--- 重みづけ後のDataFrame (df_data) ---")
print(df_res_base)

# df_for_list を使わず、df_res_base から直接集計する
df_res = df_res_base.groupby(['x(cm)', 'h(cm)']).agg(
    duration=('duration', 'sum'),
    totalcount=('totalcount_weighted', 'sum')
).reset_index() # groupbyのキーを列に戻す

print(df_res)

df_res['mean'] = df_res['totalcount'] / df_res['duration']
df_res['std'] = np.sqrt(df_res['totalcount']) / df_res['duration']

print(df_res)

#シミュレーション結果をまとめた最新ファイルを参照し、データフレーム化。
sim_filename_list = natsorted(glob.glob('./CosmicRaySimulation/CosmicRaySimulation_flux.csv'))
sim_filename = sim_filename_list[-1]
df_sim = pd.read_csv(sim_filename, sep='\t', header = 0)

#BackGroundを参照し、変数化。
BG_filename = os.path.join(".", "pocket_counter_output", "BG.csv")
with open(BG_filename) as BG:
  reader = csv.reader(BG, delimiter='\t')
  l = [row for row in reader]
  Mean_BG = float(l[1][2])
  Std_BG = float(l[1][3])

#実験結果のフレームについて、xとzを参照し、cosθの平均と標準偏差、立体角の平均と標準偏差を追記。
df_res = pd.merge(df_res, df_sim[['plate1_move_x(cm)', 'plate1_move_z(cm)', 'Mean_CosTheta_CosSquaredTheta', 'Std_CosTheta_CosSquaredTheta', 'Mean_SolidAngle_CosSquaredTheta(sr)', 'Std_SolidAngle_CosSquaredTheta(sr)']], left_on=['x(cm)', 'h(cm)'], right_on = ['plate1_move_x(cm)', 'plate1_move_z(cm)'], how='left')

#(M1±ε1)/(M2±ε2)の計算において統計誤差を算出する関数
def stat_err_01(M1, e1, M2, e2):
  Nres = M1/M2
  eres = np.sqrt(np.power((1/M2*e1), 2)+np.power((M1/(np.power(M2, 2))*e2),2))
  print(Nres, eres)
  return Nres, eres

#(M1±ε1)-(M2±ε2)の計算において統計誤差を算出する関数
def stat_err_02(M1, e1, M2, e2):
  Nres = M1-M2
  eres = np.sqrt(np.power(e1, 2)+np.power(e2, 2))
  return Nres, eres

#((M1±ε1)-(M2±ε2))/(M3±ε3)の計算において統計誤差を算出する関数
def stat_err_03(M1, e1, M2, e2, M3, e3):
  Nres02, eres02 = stat_err_02(M1, e1, M2, e2)
  Nres, eres = stat_err_01(Nres02, eres02, M3, e3)
  return Nres, eres

#実験結果のフレームについて、FluxのMeanおよび統計誤差を考慮したStdを追記。
calc_list = stat_err_03(df_res["mean"], df_res["std"], Mean_BG, Std_BG, df_res["Mean_SolidAngle_CosSquaredTheta(sr)"], df_res["Std_SolidAngle_CosSquaredTheta(sr)"])
df_res["Mean_Flux"] = calc_list[0]/100
df_res["Std_Flux"] = calc_list[1]/100

df_res.to_csv("Result.csv")

#実験結果のフレームについて、Fluxが0以上の数なっていない行を削除。
df_res = df_res[(df_res["Mean_Flux"]>=0)]

df_res_h10 = df_res[(df_res["h(cm)"]==10)]
df_res_h325 = df_res[(df_res["h(cm)"]==32.5)]

print(df_res)

#フィット
def model01(x, a, b, n):
  return a + b * np.power(x, n)
def model02(x, a, b):
  return a + b * np.power(x, 2)
def model03(x, b):
  return b * np.power(x, 2)

popt01, pcov01 = curve_fit(model01, df_res['Mean_CosTheta_CosSquaredTheta'], df_res["Mean_Flux"], sigma = df_res["Std_Flux"], p0=[0, 0.005, 2], maxfev=50000,absolute_sigma=True)
popt02, pcov02 = curve_fit(model02, df_res['Mean_CosTheta_CosSquaredTheta'], df_res["Mean_Flux"], sigma = df_res["Std_Flux"], p0=[0.001, 0.005], maxfev=50000,absolute_sigma=True)
popt03, pcov03 = curve_fit(model03, df_res['Mean_CosTheta_CosSquaredTheta'], df_res["Mean_Flux"], sigma = df_res["Std_Flux"], p0=0.005, maxfev=50000,absolute_sigma=True)

a01 = float(f"{popt01[0]:.4f}")
b01 = float(f"{popt01[1]:.4f}")
n01 = float(f"{popt01[2]:.4f}")

a02 = float(f"{popt02[0]:.4f}")
b02 = float(f"{popt02[1]:.4f}")

b03 = float(f"{popt03[0]:.4f}")

err01 = np.sqrt(np.diag(pcov01))
err02 = np.sqrt(np.diag(pcov02))
err03 = np.sqrt(np.diag(pcov03))

# データ点の数 N
N = len(df_res['Mean_CosTheta_CosSquaredTheta'])
# フィットパラメータの数 P
P01 = 3
P02 = 2
P03 = 1

# フィットされたモデルからの予測値を計算
y_predicted_01 = model01(df_res['Mean_CosTheta_CosSquaredTheta'], *popt01)
y_predicted_02 = model02(df_res['Mean_CosTheta_CosSquaredTheta'], *popt02)
y_predicted_03 = model03(df_res['Mean_CosTheta_CosSquaredTheta'], *popt03)

# chi^2 の計算
chi_squared_01 = np.sum(((df_res["Mean_Flux"] - y_predicted_01) / df_res["Std_Flux"])**2)
chi_squared_02 = np.sum(((df_res["Mean_Flux"] - y_predicted_02) / df_res["Std_Flux"])**2)
chi_squared_03 = np.sum(((df_res["Mean_Flux"] - y_predicted_03) / df_res["Std_Flux"])**2)

# 自由度あたりの chi^2 (reduced chi-squared) の計算
degrees_of_freedom_01 = N - P01
degrees_of_freedom_02 = N - P02
degrees_of_freedom_03 = N - P03
reduced_chi_squared_01 = chi_squared_01 / degrees_of_freedom_01
reduced_chi_squared_02 = chi_squared_02 / degrees_of_freedom_02
reduced_chi_squared_03 = chi_squared_03 / degrees_of_freedom_03

#グラフの生成
plt.figure(figsize=(12, 6)) # グラフのサイズを調整
plt.style.use("ggplot")

plt.errorbar(df_res['Mean_CosTheta_CosSquaredTheta'], df_res["Mean_Flux"], xerr=df_res['Std_CosTheta_CosSquaredTheta'], yerr=df_res["Std_Flux"], fmt='none', ecolor='black', elinewidth=0.5, capsize=1, label='std', alpha=0.8, zorder=1)

plt.scatter(df_res_h10['Mean_CosTheta_CosSquaredTheta'], df_res_h10["Mean_Flux"], marker = "x", label="Mean(h=10cm)")
plt.scatter(df_res_h325['Mean_CosTheta_CosSquaredTheta'], df_res_h325["Mean_Flux"], marker = "x", label="Mean(h=32.5cm)")

plt.plot(np.linspace(0, 1, 10000), model01(np.linspace(0, 1, 10000), a01, b01, n01), linewidth=0.5,label=f'Fitted Curve: Flux=({a01:.4f}±{err01[0]:.4f}) + ({b01:.4f}±{err01[1]:.4f})(cosθ)^({n01:.4f}±{err01[2]:.4f}) (reduced χ^2:{reduced_chi_squared_01:.1f})')
plt.plot(np.linspace(0, 1, 10000), model02(np.linspace(0, 1, 10000), a02, b02), linewidth=0.5, label=f'Fitted Curve: Flux=({a02:.4f}±{err02[0]:.4f}) + ({b02:.4f}±{err02[1]:.4f})(cosθ)^2 (reduced χ^2:{reduced_chi_squared_02:.1f})')
plt.plot(np.linspace(0, 1, 10000), model03(np.linspace(0, 1, 10000), b03), linewidth=0.5, label=f'Fitted Curve: Flux=({b03:.4f}±{err03[0]:.4f})(cosθ)^2 (reduced χ^2:{reduced_chi_squared_03:.1f})')

plt.title("Flux vs. Average Cosine of Zenith Angle")
plt.xlabel(r"$\cos\theta$")
plt.ylabel(r"Flux (/${cm}^2$ /$s$ /$sr$)")
plt.legend(ncol=1)
plt.tight_layout()

plt.xlim(0, 1)
plt.ylim(-0.001, 0.006)

plt.savefig("./analyzation_result.png", dpi = 600)
plt.show()