import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # 日時フォーマット
from datetime import datetime # タイムスタンプ変換
import os # ファイルパス操作のために追加
import re #数字の抽出
import glob #ファイル探索
from natsort import natsorted #ソート

#カレントディレクトリを本ファイルが存在するフォルダに変更。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def treatment(input_filename, tf):
  # データの読み込み
  data = np.loadtxt(input_filename, delimiter="\t")
  
  t = data[:, 0]  # 時間データ (Unixタイムスタンプ)
  coin = data[:, 1]  # コインシデンスカウント
  
  bin_width_sec = 30*60 #30分を秒に変換
  
  # 時間データの最小値と最大値を取得
  t_min = t.min()
  t_max = t.max()
  
  # ビンの端点を計算
  bins = np.arange(t_min, t_max + bin_width_sec, bin_width_sec)
  
  dcoin_events = np.diff(coin)    # 各タイムスタンプにおけるコインシデンスの"差分"
  t_events = 0.5 * (t[1:] + t[:-1]) # dcoin_eventsに対応する時間点
  
  counts, bin_edges = np.histogram(t_events, bins=bins, weights=dcoin_events)
  
  bin_centers_unix = 0.5 * (bin_edges[:-1] + bin_edges[1:])    # 各ビンの中心点を計算 (Unixタイムスタンプ)
  
  # --- ここから修正 ---
  # 各ビンの実際の時間長を計算
  # 最後のビン以外はbin_width_sec、最後のビンはt_max - bin_edges[-2]
  bin_durations = np.full(len(bin_edges) - 1, bin_width_sec, dtype=float)
  if len(bin_edges) > 1: # 少なくとも2つのビンエッジがあることを確認
    # 最後のビンのdurationをt_maxから最後のビンの開始点までとする
    # ただし、最後のビンが完全に空の場合（つまりt_eventsがbins範囲外の場合）、t_max - bin_edges[-2]が適切でないことがあるため、
    # countsが0の場合はdurationをbin_width_secのままにするか、0にするかを検討する必要がある。
    # ここでは、データが存在する限り有効なdurationを計算する方針
    last_bin_start = bin_edges[-2]
    if t_max > last_bin_start:
      bin_durations[-1] = t_max - last_bin_start
    else:
      # t_maxが最後のビンの開始点以下の場合（例えば、データが少ない場合）は、durationを適切に設定
      # ここでは、データがないビンは計算に含めないという前提で、bin_width_secとする
      # あるいは、そのビンのcountsが0ならdurationも0とすることも考えられる
      pass
      
  # 実際の時間長でレートを計算
  rates = np.zeros_like(counts, dtype=float)
  # bin_durationsが0でないビンに対してのみ計算
  valid_indices = bin_durations > 0
  rates[valid_indices] = counts[valid_indices] / bin_durations[valid_indices]
  # --- ここまで修正 ---
  
  # X軸の日時表示のための変更
  # Unixタイムスタンプをdatetimeオブジェクトに変換
  bin_centers_datetime = [datetime.fromtimestamp(ts) for ts in bin_centers_unix]
  
  # === ここから追加/変更 ===
  # データ全体における総コインシデンス数と総測定時間
  total_coin_count = coin[-1] - coin[0]
  total_duration_sec = t[-1] - t[0]
  
  # データ全体における1秒あたりの平均レート
  mean = total_coin_count / total_duration_sec
  
  # データ全体における平均レートの標準偏差（誤差）
  # ポアソン統計を仮定
  stds = np.sqrt(total_coin_count) / total_duration_sec
  # === ここまで追加/変更 ===
  
  if tf == "true":
    print(input_filename)
    print(f"Mean: {mean:.4f}, Std: {stds:.4f}")
    plt.figure(figsize=(12, 6)) # グラフのサイズを調整
    plt.title(input_filename)
    plt.bar(bin_centers_datetime, rates, width=bin_width_sec/(24*60*60) * 0.8, color="black", alpha=0.5, label=f"Coin Rate ({bin_width_sec/60:.0f}-min avg)") # widthも日付単位に合わせる
    plt.errorbar(bin_centers_datetime, rates, yerr=np.sqrt(counts)/bin_width_sec, fmt='none', ecolor='black', capsize=0, label='Error Bars')
    plt.xlabel("Time") # X軸ラベルを "Time" に変更
    plt.ylabel("Coin Rate (counts/sec)")
    ax = plt.gca() # X軸の目盛りを日時形式に設定
    
    #locator = mdates.HourLocator(interval=6) # interval = 何時間ごとかを指定
    #ax.xaxis.set_major_locator(locator)
    
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10) # この値を調整
    ax.xaxis.set_major_locator(locator)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')) # 例: 2025-05-23 15:00
    plt.xticks(rotation=45, ha='right') # X軸のラベルが重ならないように回転
    
    plt.legend()
    plt.tight_layout()
    
    # 入力ファイル名からPNGファイル名を生成
    make_dir_path = './pocket_counter_output/images/'
    #作成しようとしているディレクトリが存在するかどうかを判定する
    if os.path.isdir(make_dir_path):
      #既にディレクトリが存在する場合は何もしない
      pass
    else:
      #ディレクトリが存在しない場合のみ作成する
      os.makedirs(make_dir_path)
    # os.path.splitext() でファイル名と拡張子を分割
    filename = input_filename.split("/")[len(input_filename.split("/"))-1]
    base_name = make_dir_path + os.path.splitext(filename)[0]
    output_png_filename = f"{base_name}_ana{bin_width_sec/60:.0f}-min_date.png"

    # グラフをPNGファイルとして保存
    plt.savefig(output_png_filename)
  return mean, stds, total_coin_count, total_duration_sec

#数字を抽出する関数
def numb(text):
  match = re.search(r'\d+(\.\d+)?', text)
  if match:
    return float(match.group(0)) # 抽出した文字列をfloat型に変換して返す
  else:
    return None # マッチしない場合はNoneを返す（必要に応じて0などのデフォルト値でも可）

def main():
  # ワイルドカードで条件を満たすパスの文字列を指定
  pathlist = glob.glob('./pocket_counter_output/datas/*.txt')
  
  pathlist = natsorted(pathlist)
  
  now = datetime.now()
  now_str = now.strftime('%y%m%d_%H%M%S')
  
  make_dir_path = './pocket_counter_output/results'
  #作成しようとしているディレクトリが存在するかどうかを判定する
  if os.path.isdir(make_dir_path):
    #既にディレクトリが存在する場合は何もしない
    pass
  else:
    #ディレクトリが存在しない場合のみ作成する
    os.makedirs(make_dir_path)
  
  with open(make_dir_path + "/pocket_counter_output_" + now_str + ".csv", "w") as out:
    # ヘッダー行を書き込む
    out.write("Filename\t"
              "x(cm)\t"
              "h(cm)\t"
              "date\t"
              "time\t"
              "duration\t"
              "COM\t"
              "totalcount\t"
              "mean\t"
              "std")
    pathlist_BG = []
    pathlist_main = []
    for path in pathlist:
      filename = os.path.splitext(path.split("/")[len(path.split("/"))-1])[0]
      info = filename.split("_")
      if int(numb(info[1]))==100:
        pathlist_BG.append(path)
      else:
        pathlist_main.append(path)
    totalcount_BG = 0
    totalduration_BG = 0
    for path in pathlist_BG:
      i, j, c, t = treatment(path, "false")
      totalcount_BG = totalcount_BG + c
      totalduration_BG = totalduration_BG + t
    print(totalcount_BG, totalduration_BG)
    totalmean_BG = totalcount_BG / totalduration_BG
    totalstd_BG = np.sqrt(totalcount_BG) / totalduration_BG
    print(f"Background Mean:{totalmean_BG:.4f}")
    print(f"Background Std:{totalstd_BG:.4f}")
    with open(os.path.join(".", "pocket_counter_output", "BG.csv"), "w") as out_BG:
      out_BG.write("totalcount_BG\t"
              "totalduration_BG\t"
              "totalmean_BG\t"
              "totalstd_BG")
      out_BG.write(f"\n{totalcount_BG}\t"
                f"{totalduration_BG:.4f}\t"
                f"{totalmean_BG:.4f}\t"
                f"{totalstd_BG:.4f}")
    for path in pathlist:
      i, j, c, t= treatment(path, "true")
      filename = os.path.splitext(path.split("/")[len(path.split("/"))-1])[0]
      info = filename.split("_")
      out.write(f"\n{filename}.txt\t"
                f"{numb(info[1])}\t"
                f"{numb(info[2])}\t"
                f"{info[3]}\t"
                f"{info[4]}\t"
                f"{t:.4f}\t"
                f"{info[5]}\t"
                f"{c:.4f}\t"
                f"{i:.4f}\t"
                f"{j:.4f}")


#このファイルが直接実行されたときにmain関数を呼び出す。
if __name__ == '__main__':
  main()