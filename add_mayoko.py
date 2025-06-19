import numpy as np
from tqdm import tqdm
import os
import pandas as pd

# カレントディレクトリを本ファイルが存在するフォルダに変更。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sim_file_path = './CosmicRaySimulation/CosmicRaySimulation_flux.csv'

df_res = pd.read_csv(sim_file_path, sep='\t', header = 0)

x = 200
z = 45

df_res = df_res[(df_res['plate1_move_x(cm)']==0)&(df_res['plate1_move_z(cm)']==45)]

Coincidence_CosSquaredTheta = df_res['Coincidence_CosSquaredTheta(/s)'].iloc[0]
mean_cos_theta_through_values_squared = 0
std_cos_theta_through_values_squared = df_res['Std_CosTheta_CosSquaredTheta'].iloc[0]
mean_solid_angles_squared = df_res['Mean_SolidAngle_CosSquaredTheta(sr)'].iloc[0]
std_solid_angles_squared = df_res['Std_SolidAngle_CosSquaredTheta(sr)'].iloc[0]
mean_cos_theta_through_values_squared_np = df_res['Mean_CosSquaredTheta_CosSquaredTheta'].iloc[0]

with open(sim_file_path, "a") as out:
  # mayoko情報を追記。
  out.write(f"\n{x}\t"
            f"{z}\t"
            f"{Coincidence_CosSquaredTheta:.4e}\t"
            f"{mean_cos_theta_through_values_squared:.4e}\t"
            f"{std_cos_theta_through_values_squared:.4e}\t"
            f"{mean_solid_angles_squared:.4e}\t"
            f"{std_solid_angles_squared:.4e}\t"
            f"{mean_cos_theta_through_values_squared_np:.4e}")