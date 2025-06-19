import glob
from natsort import natsorted
import os
import numpy as np

#カレントディレクトリを本ファイルが存在するフォルダに変更。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def treatment(input_filename):
  data = np.loadtxt(input_filename, delimiter="\t", skiprows=1)
  
  filename = os.path.splitext(os.path.basename(path))[0]
  info = filename.split("_")
  
  t = data[:, 0]  # 時間データ (Unixタイムスタンプ)
  coin01 = data[:, 1]  # COM3のコインシデンスカウント
  coin02 = data[:, 4]  # もうひとつのCOMのコインシデンスカウント
  
  # 各タイムスタンプにおけるコインシデンスの"差分"
  dcoin01_events = np.diff(coin01)
  dcoin02_events = np.diff(coin02)
  # dcoin_eventsに対応する時間点
  t_events = 0.5 * (t[1:] + t[:-1])
  
  dcoin01_events_count = 0
  dcoin02_events_count = 0
  for i in range(1, len(dcoin01_events)-1):
    if dcoin01_events[i] == 1:
      dcoin01_events_count += 1
      if (dcoin02_events[i] == 1) or (dcoin02_events[i-1] == 1) or (dcoin02_events[i+1] == 1):
        dcoin02_events_count += 1
  
  print(dcoin01_events_count)
  print(dcoin02_events_count)
  
  efficiency = dcoin02_events_count / dcoin01_events_count
  
  print(efficiency)
  
  print(dcoin01_events)
  print(dcoin02_events)
  
  return efficiency, info[4]

pathlist = glob.glob('./performance/datas/*.txt')

pathlist = natsorted(pathlist)

make_dir_path = './performance/results'

if os.path.isdir(make_dir_path):
  #既にディレクトリが存在する場合は何もしない
  pass
else:
  #ディレクトリが存在しない場合のみ作成する
  os.makedirs(make_dir_path)

with open(make_dir_path + "/performance.csv", "w") as out:
  # ヘッダー行を書き込む
  out.write("COM\t"
            "value")
  out.write(f"\nCOM3\t"
            f"1.0000")
  for path in pathlist:
    eff, COM = treatment(path)
    out.write(f"\n{COM}\t"
              f"{eff:.4f}")