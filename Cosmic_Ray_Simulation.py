import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad
from tqdm import tqdm
from scipy.optimize import curve_fit
import os
import configparser # 設定ファイルを扱うためのモジュール

# カレントディレクトリを本ファイルが存在するフォルダに変更。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- シミュレーション設定 (プログラムに直接記述) ---
# これらの値がシミュレーションで使用され、
# そのまま simulation_config.cfg ファイルに出力されます。

# 板の情報
plate_thickness = 1.0   # cm (このコードでは未使用)
plate_side = 10.0       # cm
plate2_z = 0.0          # cm

# plate1 の x,y 方向の移動量及びzの値を定義
plate1_move_x_values_start = 0.0
plate1_move_x_values_end = 50.0
plate1_move_x_values_num = 51
plate1_move_y = 0.0     # cm
plate1_move_z_values = [10, 32.5, 45] # cm - リスト

# 実験のパラメータ
collection_time = 60 * 60 * 1  # 計測時間(s)
cosmicray_rate = 0.7 # 板の間隔0 cmでのCoincidence/100cm^2/sr/s

# 出力ディレクトリ
output_dir = 'CosmicRaySimulation'

# プログラム内の設定値を simulation_config.cfg に出力
config = configparser.ConfigParser()
config_file_path = 'simulation_config.cfg'

config['SIMULATION_PARAMS'] = {
    'plate_thickness': str(plate_thickness),
    'plate_side': str(plate_side),
    'plate2_z': str(plate2_z),
    'plate1_move_x_values_start': str(plate1_move_x_values_start),
    'plate1_move_x_values_end': str(plate1_move_x_values_end),
    'plate1_move_x_values_num': str(plate1_move_x_values_num),
    'plate1_move_y': str(plate1_move_y),
    'plate1_move_z_values': ','.join(map(str, plate1_move_z_values)), # リストをカンマ区切り文字列に変換
    'collection_time': str(collection_time),
    'cosmicray_rate': str(cosmicray_rate),
    'output_dir': output_dir
}

with open(config_file_path, 'w') as configfile:
    config.write(configfile)
print(f"現在のシミュレーション設定が '{config_file_path}' に出力されました。")

plate1_move_x_values = np.linspace(
    plate1_move_x_values_start,
    plate1_move_x_values_end,
    int(plate1_move_x_values_num)
)

# 宇宙線の発生数を設定 
detector_area = plate_side * plate_side  # cm^2
num_cosmic_rays =  int(cosmicray_rate * collection_time)

# 宇宙線の発生源となる板1上の点をランダムに生成する関数
def generate_origin_point():
    x = np.random.uniform(-plate_side / 2 , plate_side / 2 )
    y = np.random.uniform(-plate_side / 2 , plate_side / 2 )
    return np.array([x, y, 0.0])


# cosθについてランダムな方向の単位ベクトルを生成する関数
def generate_random_direction():
    cos_theta = np.random.uniform(0, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.uniform(0, 2 * np.pi)
    dx = sin_theta * np.cos(phi)
    dy = sin_theta * np.sin(phi)
    dz = -cos_theta  # z軸負の方向へ飛ぶ宇宙線を考える
    return np.array([dx, dy, dz]), cos_theta # 方向ベクトルとcos_thetaを別々に返す


# cosθの二乗に比例するランダムな方向ベクトルを生成する関数
def generate_cos_theta_random_direction():
    # cosθの二乗に比例する確率密度関数に従うようにcosθを生成する
    cos_theta = (np.random.uniform(0, 1))**(1/3)  # 0から1までの一様分布の平方根
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.uniform(0, 2 * np.pi)
    dx = sin_theta * np.cos(phi)
    dy = sin_theta * np.sin(phi)
    dz = -cos_theta  # z軸負の方向へ飛ぶ宇宙線を考える
    return np.array([dx, dy, dz]), cos_theta # 方向ベクトルとcos_thetaを返す


# 宇宙線が板2を通過するかどうかを判定し、通過する場合はその通過点を計算する関数
def calculate_intersection(origin, direction):
    # z方向の成分が0の場合は通過しない
    if direction[2] >= 0:
        return None

    # 板2 (z = 0) との交点のz座標は0
    t = (plate2_z - origin[2]) / direction[2]
    intersection_x = origin[0] + t * direction[0]
    intersection_y = origin[1] + t * direction[1]
    intersection_z = plate2_z

    # 交点が板2の範囲内にあるか判定
    if -plate_side / 2 <= intersection_x <= plate_side / 2 and \
      -plate_side / 2 <= intersection_y <= plate_side / 2:
      return np.array([intersection_x, intersection_y, intersection_z])
    else:
        return None


# 立体角を計算する関数
def calculate_exact_solid_angle(origin, intersection, plate1_move, plate1_z):

    origin = np.array(origin)
    x1, y1, _ = origin

    def integrand(x2, y2):
        r_squared = (x1 - x2)**2 + (y1 - y2)**2 + (plate1_z - plate2_z)**2
        return (plate1_z - plate2_z) / (r_squared**1.5)

    # 板2の領域で二重積分
    solid_angle, _ = dblquad(integrand, -plate_side / 2, plate_side / 2,
                            lambda x: -plate_side / 2, lambda x: plate_side / 2)
    return solid_angle


def Simulation(plate1_move_x, plate1_move_y, plate1_z):
    origins = []
    intersections = []
    exact_solid_angles = []
    cos_theta_through_values = []

    plate1_move = np.array([plate1_move_x, plate1_move_y, plate1_z])

    # シミュレーションの実行とデータの収集
    for _ in range(num_cosmic_rays):
        origin = generate_origin_point() + plate1_move
        direction, cos_theta = generate_random_direction() # 戻り値をunpack
        intersection = calculate_intersection(origin, direction)

        origins.append(origin)
        intersections.append(intersection)
        exact_solid_angle = calculate_exact_solid_angle(origin, intersection, plate1_move, plate1_z)
        exact_solid_angles.append(exact_solid_angle)

        if intersection is not None:
            cos_theta_through_values.append(cos_theta) # 板2を通過した場合のみcosθを記録

    # NumPy配列に変換 (None要素を除去)
    origins_np = np.array(origins)
    intersections_valid = np.array([i for i in intersections if i is not None])
    exact_solid_angles_valid = np.array([sa for sa in exact_solid_angles if sa is not None])
    cos_theta_through_values_np = np.array(cos_theta_through_values)

    return origins_np, intersections_valid, exact_solid_angles_valid, cos_theta_through_values_np


def Simulation_squared(plate1_move_x, plate1_move_y, plate1_z):
    origins = []
    intersections = []
    exact_solid_angles = []
    cos_theta_through_values = []
    cos_theta_through_values_squared = []

    plate1_move = np.array([plate1_move_x, plate1_move_y, plate1_z])

    # シミュレーションの実行とデータの収集
    for _ in range(num_cosmic_rays):
        origin = generate_origin_point() + plate1_move
        direction, cos_theta = generate_cos_theta_random_direction() # 戻り値をunpack
        intersection = calculate_intersection(origin, direction)

        origins.append(origin)
        intersections.append(intersection)
        exact_solid_angle = calculate_exact_solid_angle(origin, intersection, plate1_move, plate1_z)
        exact_solid_angles.append(exact_solid_angle)

        if intersection is not None:
            cos_theta_through_values.append(cos_theta) # 板2を通過した場合のみcosθを記録
            cos_theta_through_values_squared.append(cos_theta**2)

    # NumPy配列に変換 (None要素を除去)
    origins_np = np.array(origins)
    intersections_valid = np.array([i for i in intersections if i is not None])
    exact_solid_angles_valid = np.array([sa for sa in exact_solid_angles if sa is not None])
    cos_theta_through_values_np = np.array(cos_theta_through_values)
    cos_theta_through_values_squared_np = np.array(cos_theta_through_values_squared)

    return origins_np, intersections_valid, exact_solid_angles_valid, cos_theta_through_values_np, cos_theta_through_values_squared_np

#位置情報
x_list = []
z_list = []

#cosθについてランダム
number_of_Coincidence = []
mean_solid_angles = [] # 立体角の平均を格納するリスト
std_solid_angles = []  # 立体角の標準偏差を格納するリスト
mean_cos_theta_through_values = [] # cosθの平均を格納するリスト
std_cos_theta_through_values = [] # cosθの標準偏差を格納するリスト

#cosθ_squaredについてランダム
number_of_Coincidence_squared = []
mean_solid_angles_squared = [] # 立体角の平均を格納するリスト
std_solid_angles_squared = []  # 立体角の標準偏差を格納するリスト
mean_cos_theta_through_values_squared = [] # cosθの平均を格納するリスト
std_cos_theta_through_values_squared = [] # cosθの標準偏差を格納するリスト
mean_cos_theta_through_values_squared_np = [] # cosθの二乗の平均を格納するリスト


# シミュレーションを実行し、同時計数と立体角の平均、標準偏差を計算
for z_move in tqdm(plate1_move_z_values, desc="z:"):
  for x_move in tqdm(plate1_move_x_values, desc="x:"):
    x_list.append(x_move)
    z_list.append(z_move)
    #cosθについてランダム
    _, intersections_valid, exact_solid_angles_valid, cos_theta_through_values_np = Simulation(x_move, plate1_move_y, z_move)
    number_of_Coincidence.append(len(intersections_valid))
    mean_solid_angles.append(np.mean(exact_solid_angles_valid)) # 立体角の平均を計算してリストに追加
    std_solid_angles.append(np.std(exact_solid_angles_valid))  # 立体角の標準偏差を計算してリストに追加
    mean_cos_theta_through_values.append(np.mean(cos_theta_through_values_np)) # cosθの平均を計算してリストに追加
    std_cos_theta_through_values.append(np.std(cos_theta_through_values_np)) # cosθの標準偏差を計算してリストに追加

    #cosθ_squaredについてランダム
    _, intersections_valid_squared, exact_solid_angles_valid_squared, cos_theta_through_values_np_squared, cos_theta_through_values_squared_np = Simulation_squared(x_move, plate1_move_y, z_move)
    number_of_Coincidence_squared.append(len(intersections_valid_squared))
    mean_solid_angles_squared.append(np.mean(exact_solid_angles_valid_squared)) # 立体角の平均を計算してリストに追加
    std_solid_angles_squared.append(np.std(exact_solid_angles_valid_squared))  # 立体角の標準偏差を計算してリストに追加
    mean_cos_theta_through_values_squared.append(np.mean(cos_theta_through_values_np_squared)) # cosθの平均を計算してリストに追加
    std_cos_theta_through_values_squared.append(np.std(cos_theta_through_values_np_squared)) # cosθの標準偏差を計算してリストに追加
    mean_cos_theta_through_values_squared_np.append(np.mean(cos_theta_through_values_squared_np)) # cosθの平均を計算してリストに追加

print(f"データ収集時間: {collection_time / (60 * 60):.2f} 時間")
print(f"検出器面積: {detector_area} cm^2")

# CosmicRaySimulation/CosmicRaySimulation_flux.txt ファイルを作成
make_dir_path = './CosmicRaySimulation'
#作成しようとしているディレクトリが存在するかどうかを判定する
if os.path.isdir(make_dir_path):
    #既にディレクトリが存在する場合は何もしない
    pass
else:
    #ディレクトリが存在しない場合のみ作成する
    os.makedirs(make_dir_path)
with open("CosmicRaySimulation/CosmicRaySimulation_flux.csv", "w") as out:
    # ヘッダー行を書き込む
    out.write("plate1_move_x(cm)\t"
              "plate1_move_z(cm)\t"
              "Coincidence_CosSquaredTheta(/s)\t"
              "Mean_CosTheta_CosSquaredTheta\t"
              "Std_CosTheta_CosSquaredTheta\t"
              "Mean_SolidAngle_CosSquaredTheta(sr)\t"
              "Std_SolidAngle_CosSquaredTheta(sr)\t"
              "Mean_CosSquaredTheta_CosSquaredTheta") # 新しい列の追加
    for i in range(len(x_list)):
      out.write(f"\n{x_list[i]:.2f}\t"
                f"{z_list[i]:.2f}\t"
                f"{np.array(number_of_Coincidence_squared)[i] / collection_time:.4e}\t"
                f"{mean_cos_theta_through_values_squared[i]:.4e}\t"
                f"{std_cos_theta_through_values_squared[i]:.4e}\t"
                f"{mean_solid_angles_squared[i]:.4e}\t"
                f"{std_solid_angles_squared[i]:.4e}\t"
                f"{mean_cos_theta_through_values_squared_np[i]:.4e}") # 新しい列の追加