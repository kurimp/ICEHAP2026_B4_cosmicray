import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import glob #ファイル探索
from natsort import natsorted #ソート
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2

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

# 効率の辞書の作成。
com_weights_map = df_eff.set_index('COM')['value'].to_dict()

print("\n--- 作成されたCOMと効率の辞書 ---")
print(com_weights_map)
print("-" * 30)

# 各行に対応する効率の値を羅列。
mapped_weights = df_res_base['COM'].map(com_weights_map)
final_weights = mapped_weights.fillna(1.0)

# 効率を適用した方のdataframeの作成。
df_res_base_weighted = df_res_base.copy()
df_res_base_weighted['totalcount'] = df_res_base_weighted['totalcount'] / final_weights

print("\n--- 重みづけ前後のDataFrame (df_res_base, df_res_base_weighted) ---")
print(df_res_base)
print(df_res_base_weighted)

# df_for_list を使わず、df_res_base から直接集計する
df_res = df_res_base.groupby(['x(cm)', 'h(cm)']).agg(
    duration=('duration', 'sum'),
    totalcount=('totalcount', 'sum')
).reset_index() # groupbyのキーを列に戻す
df_res_weighted = df_res_base_weighted.groupby(['x(cm)', 'h(cm)']).agg(
    duration=('duration', 'sum'),
    totalcount=('totalcount', 'sum')
).reset_index() # groupbyのキーを列に戻す

def add_mean_std(df):
  df['mean'] = df['totalcount'] / df['duration']
  df['std'] = np.sqrt(df['totalcount']) / df['duration']
  return df

df_res_base = add_mean_std(df_res_base)
df_res_base_weighted = add_mean_std(df_res_base_weighted)
df_res = add_mean_std(df_res)
df_res_weighted = add_mean_std(df_res_weighted)

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
def merging(df):
  df = pd.merge(df, df_sim[['plate1_move_x(cm)', 'plate1_move_z(cm)', 'Mean_CosTheta_CosSquaredTheta', 'Std_CosTheta_CosSquaredTheta', 'Mean_SolidAngle_CosSquaredTheta(sr)', 'Std_SolidAngle_CosSquaredTheta(sr)']], left_on=['x(cm)', 'h(cm)'], right_on = ['plate1_move_x(cm)', 'plate1_move_z(cm)'], how='left')
  return df

df_res_base = merging(df_res_base)
df_res_base_weighted = merging(df_res_base_weighted)
df_res = merging(df_res)
df_res_weighted = merging(df_res_weighted)

#(M1±ε1)/(M2±ε2)の計算において統計誤差を算出する関数
def stat_err_01(M1, e1, M2, e2):
  Nres = M1/M2
  eres = np.sqrt(np.power((1/M2*e1), 2)+np.power((M1/(np.power(M2, 2))*e2),2))
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
def add_err(df):
  calc_list = stat_err_03(df["mean"], df["std"], Mean_BG, Std_BG, df["Mean_SolidAngle_CosSquaredTheta(sr)"], df["Std_SolidAngle_CosSquaredTheta(sr)"])
  df["Mean_Flux"] = calc_list[0]/100
  df["Std_Flux"] = calc_list[1]/100
  return df

df_res_base = add_err(df_res_base)
df_res_base_weighted = add_err(df_res_base_weighted)
df_res = add_err(df_res)
df_res_weighted = add_err(df_res_weighted)

df_res_base['Mean_Flux_weighted'] = df_res_base_weighted['Mean_Flux']
df_res_base['Std_Flux_weighted'] = df_res_base_weighted['Std_Flux']
df_res['Mean_Flux_weighted'] = df_res_weighted['Mean_Flux']
df_res['Std_Flux_weighted'] = df_res_weighted['Std_Flux']

df_res_base.to_csv("Result_individual.csv")
df_res.to_csv("Result_integrated.csv")

#実験結果のフレームについて、Fluxがnaになっている行を削除。
def deleting(df):
  df = df[(df["Mean_Flux"].notna())]
  return df

df_res_base = deleting(df_res_base)
df_res_base_weighted = deleting(df_res_base_weighted)
df_res = deleting(df_res)
df_res_weighted = deleting(df_res_weighted)

#フィット
def model01(x, a, b, n):
  return a + b * np.power(x, n)
def model02(x, a, b):
  return a + b * np.power(x, 2)
def model03(x, b):
  return b * np.power(x, 2)

def fitting(df):
  popt01, pcov01 = curve_fit(model01, df['Mean_CosTheta_CosSquaredTheta'], df["Mean_Flux"], sigma = df["Std_Flux"], p0=[0, 0.005, 2], maxfev=50000, absolute_sigma=True)
  popt02, pcov02 = curve_fit(model02, df['Mean_CosTheta_CosSquaredTheta'], df["Mean_Flux"], sigma = df["Std_Flux"], p0=[0.001, 0.005], maxfev=50000, absolute_sigma=True)
  popt03, pcov03 = curve_fit(model03, df['Mean_CosTheta_CosSquaredTheta'], df["Mean_Flux"], sigma = df["Std_Flux"], p0=0.005, maxfev=50000, absolute_sigma=True)
  return [popt01, popt02, popt03], pcov01, pcov02, pcov03

popt, pcov01, pcov02, pcov03 = fitting(df_res)
popt_weighted, pcov01_weighted, pcov02_weighted, pcov03_weighted = fitting(df_res_weighted)

a01 = float(f"{popt[0][0]}")
b01 = float(f"{popt[0][1]}")
n01 = float(f"{popt[0][2]:}")

a02 = float(f"{popt[1][0]:}")
b02 = float(f"{popt[1][1]:}")

b03 = float(f"{popt[2][0]:}")

err01 = np.sqrt(np.diag(pcov01))
err02 = np.sqrt(np.diag(pcov02))
err03 = np.sqrt(np.diag(pcov03))

a01_weighted = float(f"{popt_weighted[0][0]:}")
b01_weighted = float(f"{popt_weighted[0][1]:}")
n01_weighted = float(f"{popt_weighted[0][2]:}")

a02_weighted = float(f"{popt_weighted[1][0]:}")
b02_weighted = float(f"{popt_weighted[1][1]:}")

b03_weighted = float(f"{popt_weighted[2][0]:}")

err01_weighted = np.sqrt(np.diag(pcov01_weighted))
err02_weighted = np.sqrt(np.diag(pcov02_weighted))
err03_weighted = np.sqrt(np.diag(pcov03_weighted))

# データ点の数 N
N = len(df_res['Mean_CosTheta_CosSquaredTheta'])
# フィットパラメータの数 P
P = [3, 2, 1]
P01 = 3
P02 = 2
P03 = 1

def test(df, popt):
  # フィットされたモデルからの予測値を計算
  y_predicted = []
  y_predicted.append(model01(df['Mean_CosTheta_CosSquaredTheta'], *popt[0]))
  y_predicted.append(model02(df['Mean_CosTheta_CosSquaredTheta'], *popt[1]))
  y_predicted.append(model03(df['Mean_CosTheta_CosSquaredTheta'], *popt[2]))

  # chi^2 の計算
  chi_squared = []
  chi_squared.append(np.sum(((df["Mean_Flux"] - y_predicted[0]) / df["Std_Flux"])**2))
  chi_squared.append(np.sum(((df["Mean_Flux"] - y_predicted[1]) / df["Std_Flux"])**2))
  chi_squared.append(np.sum(((df["Mean_Flux"] - y_predicted[2]) / df["Std_Flux"])**2))

  # 自由度あたりの chi^2 (reduced chi-squared) の計算
  degrees_of_freedom = [N - P[0], N - P[1], N - P[2]]
  reduced_chi_squared = [chi_squared[i] / degrees_of_freedom[i] for i in range(len(chi_squared))]

  # P値の計算
  p_value = 1-chi2.cdf(chi_squared, degrees_of_freedom)
  
  print()
  return y_predicted, chi_squared, degrees_of_freedom, reduced_chi_squared, p_value

y_predicted, chi_squared, degrees_of_freedom, reduced_chi_squared, p_value = test(df_res, popt)
y_predicted_weighted, chi_squared_weighted, degrees_of_freedom_weighted, reduced_chi_squared_weighted, p_value_weighted = test(df_res_weighted, popt_weighted)

print(f"\n--- Chi-squared Test Results ---")
print(f"Model 01 (a + b*x^n):")
print(f"  Chi-squared: {chi_squared[0]:.2f}")
print(f"  Degrees of Freedom: {degrees_of_freedom[0]}")
print(f"  Reduced Chi-squared: {reduced_chi_squared[0]:.2f}")
print(f"  P-value: {p_value[0]:.4f}")

print(f"\nModel 02 (a + b*x^2):")
print(f"  Chi-squared: {chi_squared[1]:.2f}")
print(f"  Degrees of Freedom: {degrees_of_freedom[1]}")
print(f"  Reduced Chi-squared: {reduced_chi_squared[1]:.2f}")
print(f"  P-value: {p_value[1]:.4f}")

print(f"\nModel 03 (b*x^2):")
print(f"  Chi-squared: {chi_squared[2]:.2f}")
print(f"  Degrees of Freedom: {degrees_of_freedom[2]}")
print(f"  Reduced Chi-squared: {reduced_chi_squared[2]:.2f}")
print(f"  P-value: {p_value[2]:.4f}")
print("-" * 30)

print(f"Model 01 (a + b*x^n):")
print(f"  Chi-squared: {chi_squared_weighted[0]:.2f}")
print(f"  Degrees of Freedom: {degrees_of_freedom_weighted[0]}")
print(f"  Reduced Chi-squared: {reduced_chi_squared_weighted[0]:.2f}")
print(f"  P-value: {p_value_weighted[0]:.4f}")

print(f"\nModel 02 (a + b*x^2):")
print(f"  Chi-squared: {chi_squared_weighted[1]:.2f}")
print(f"  Degrees of Freedom: {degrees_of_freedom_weighted[1]}")
print(f"  Reduced Chi-squared: {reduced_chi_squared_weighted[1]:.2f}")
print(f"  P-value: {p_value_weighted[1]:.4f}")

print(f"\nModel 03 (b*x^2):")
print(f"  Chi-squared: {chi_squared_weighted[2]:.2f}")
print(f"  Degrees of Freedom: {degrees_of_freedom_weighted[2]}")
print(f"  Reduced Chi-squared: {reduced_chi_squared_weighted[2]:.2f}")
print(f"  P-value: {p_value_weighted[2]:.4f}")
print("-" * 30)

df_list = [df_res_base, df_res_base_weighted, df_res, df_res_weighted]
title_list = ["individual(non-weighted)", "individual(weighted)", "integrated(non-weighted)", "integrated(weighted)"]

#グラフの生成
fig, axes = plt.subplots(4, 1, figsize=(12, 24), squeeze=False) # squeeze=Falseで常に2D配列を返す

fig.suptitle('Flux vs. Average Cosine of Zenith Angle', fontsize=16, y=1)
plt.style.use("ggplot")

i=0
for df_res in df_list:
  df_res_h10 = df_res[(df_res["h(cm)"]==10)]
  df_res_h325 = df_res[(df_res["h(cm)"]==32.5)]
  df_res_h45 = df_res[(df_res["h(cm)"]==45)]
  
  ax = axes[i, 0]
  ax.errorbar(df_res['Mean_CosTheta_CosSquaredTheta'], df_res["Mean_Flux"], xerr=df_res['Std_CosTheta_CosSquaredTheta'], yerr=df_res["Std_Flux"], fmt='none', ecolor='black', elinewidth=0.5, capsize=1, label='std', alpha=0.8, zorder=1)
  
  ax.scatter(df_res_h10['Mean_CosTheta_CosSquaredTheta'], df_res_h10["Mean_Flux"], marker = "x", label="Mean(h=10cm)")
  ax.scatter(df_res_h325['Mean_CosTheta_CosSquaredTheta'], df_res_h325["Mean_Flux"], marker = "x", label="Mean(h=32.5cm)")
  ax.scatter(df_res_h45['Mean_CosTheta_CosSquaredTheta'], df_res_h45["Mean_Flux"], marker = "x", label="Mean(mayoko)")
  
  if i % 2 == 0:
    v_a01 = a01
    v_b01 = b01
    v_n01 = n01
    v_a02 = a02
    v_b02 = b02
    v_b03 = b03
    v_err01 = np.sqrt(np.diag(pcov01))
    v_err02 = np.sqrt(np.diag(pcov02))
    v_err03 = np.sqrt(np.diag(pcov03))
    v_reduced_chi_squared01 = reduced_chi_squared[0]
    v_reduced_chi_squared02 = reduced_chi_squared[1]
    v_reduced_chi_squared03 = reduced_chi_squared[2]
  elif i % 2 == 1:
    v_a01 = a01_weighted
    v_b01 = b01_weighted
    v_n01 = n01_weighted
    v_a02 = a02_weighted
    v_b02 = b02_weighted
    v_b03 = b03_weighted
    v_err01 = np.sqrt(np.diag(pcov01_weighted))
    v_err02 = np.sqrt(np.diag(pcov02_weighted))
    v_err03 = np.sqrt(np.diag(pcov03_weighted))
    v_reduced_chi_squared01 = reduced_chi_squared_weighted[0]
    v_reduced_chi_squared02 = reduced_chi_squared_weighted[1]
    v_reduced_chi_squared03 = reduced_chi_squared_weighted[2]
  
  
  ax.plot(np.linspace(0, 1, 10000), model01(np.linspace(0, 1, 10000), v_a01, v_b01, v_n01), linewidth=0.5,label=f'Fitted Curve: Flux=({v_a01:.4f}±{v_err01[0]:.4f}) + ({v_b01:.4f}±{v_err01[1]:.4f})(cosθ)^({v_n01:.4f}±{v_err01[2]:.4f}) (reduced χ^2:{v_reduced_chi_squared01:.2f})')
  ax.plot(np.linspace(0, 1, 10000), model02(np.linspace(0, 1, 10000), v_a02, v_b02), linewidth=0.5, label=f'Fitted Curve: Flux=({v_a02:.4f}±{v_err02[0]:.4f}) + ({v_b02:.4f}±{v_err02[1]:.4f})(cosθ)^2 (reduced χ^2:{v_reduced_chi_squared02:.2f})')
  ax.plot(np.linspace(0, 1, 10000), model03(np.linspace(0, 1, 10000), v_b03), linewidth=0.5, label=f'Fitted Curve: Flux=({v_b03:.4f}±{v_err03[0]:.4f})(cosθ)^2 (reduced χ^2:{v_reduced_chi_squared03:.2f})')

  ax.set_title(title_list[i])
  ax.set_xlabel(r"$\cos\theta$")
  ax.set_ylabel(r"Flux (/${cm}^2$ /$s$ /$sr$)")
  ax.legend(ncol=1)
  
  ax.set_xlim(0, 1)
  ax.set_ylim(-0.001, 0.006)
  i += 1

plt.tight_layout()
plt.savefig("./analyzation_result.png", dpi = 300)
plt.show()