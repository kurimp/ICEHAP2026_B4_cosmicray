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

def treatment(input_filename):
  # データの読み込み
  data = np.loadtxt(input_filename, delimiter="\t")
  
  print(input_filename)
  
  t = data[:, 0]  # 時間データ (Unixタイムスタンプ)
  coin = data[:, 1]  # コインシデンスカウント
  
  bin_width_sec = 30*60 #30分を秒に変換 (コメントを修正: 元々は60分と記載されていましたが、コードは30分でした)
  
  # 時間データの最小値と最大値を取得
  t_min = t.min()
  t_max = t.max()
  
  # ビンの端点を計算
  bins = np.arange(t_min, t_max + bin_width_sec, bin_width_sec)
  
  dcoin_events = np.diff(coin)    # 各タイムスタンプにおけるコインシデンスの"差分"
  t_events = 0.5 * (t[1:] + t[:-1]) # dcoin_eventsに対応する時間点
  
  counts, bin_edges = np.histogram(t_events, bins=bins, weights=dcoin_events)
  
  bin_centers_unix = 0.5 * (bin_edges[:-1] + bin_edges[1:])    # 各ビンの中心点を計算 (Unixタイムスタンプ)
  
  # rates = counts / bin_width_sec  # この行は今回は直接使わないためコメントアウトまたは削除可能
  
  # X軸の日時表示のための変更
  # Unixタイムスタンプをdatetimeオブジェクトに変換
  bin_centers_datetime = [datetime.fromtimestamp(ts) for ts in bin_centers_unix]
  
  # --- ここから修正 ---
  # 各ビンの実際の時間長を計算
  bin_durations = np.diff(bin_edges) # 各ビンの実際の時間長 (秒)
  
  # ただし、最後のビンはt_maxで終了するため、そのdurationを調整
  if len(bin_durations) > 0:
    # 最後のビンがt_maxまでであることを考慮
    # numpyのhistogramはbin_edgesの最後の値がt_maxを超える場合でもそのビンを生成するため、
    # 最後のビンのdurationはt_maxからそのビンの開始点までとするのが適切です。
    # しかし、t_maxは実際のデータ範囲の最大値なので、最後のビンの終了点はt_maxとなります。
    # そのため、bin_edgesの最後にt_maxを加えたbin_edgesをnp.histogramに渡すのではなく、
    # np.histogramが生成したbin_edgesの差分をそのまま使うのが最もシンプルで一般的です。
    # bin_edgesの最後の要素はbinsで指定した最大値 (t_max + bin_width_sec) になるため、
    # diff(bin_edges)で計算される最後のビンのdurationはbin_width_secとなります。
    # 実際には、t_maxまでのデータしかないので、最後のビンのdurationはt_maxから最後から2番目のbin_edgeまでとなるべきです。
    
    # 最後のビンのdurationをt_maxからそのビンの開始点 (bin_edges[-2]) までとする
    # t_maxが最後のビンの開始点より大きい場合のみ調整
    if t_max > bin_edges[-2]:
        bin_durations[-1] = t_max - bin_edges[-2]
    elif len(bin_durations) > 0: # t_maxが最後のビンの開始点と一致する場合など、durationが0になるケースの対応
        # データが非常に短い場合などにbin_durations[-1]が負になるのを防ぐ
        bin_durations[-1] = max(0, t_max - bin_edges[-2])
  
  count_accumulation = 0
  time_accumulation_sec = 0.0 # 累積時間 (秒)
  Mean_count_accumulation = []
  Std_count_accumulation = []
  
  for i in range(len(counts)):
    count_accumulation += counts[i]
    time_accumulation_sec += bin_durations[i] # 実際のビンの時間長を累積
    
    if time_accumulation_sec > 0: # ゼロ除算を避ける
      Mean_count_accumulation.append(count_accumulation / time_accumulation_sec)
      Std_count_accumulation.append(np.sqrt(count_accumulation) / time_accumulation_sec)
    else:
      Mean_count_accumulation.append(0.0) # 時間が0の場合はレートも0
      Std_count_accumulation.append(0.0)
  # --- ここまで修正 ---
  
  plt.figure(figsize=(12, 6)) # グラフのサイズを調整
  
  # バーの幅は引き続きbin_width_secを基準にしても問題ないですが、
  # より正確を期すなら、bin_durationsの平均値などを利用しても良いでしょう。
  # 今回は視覚的な連続性を保つため、元のbin_width_secを使用します。
  plt.bar(bin_centers_datetime, Mean_count_accumulation, width=bin_width_sec/(24*60*60) * 0.8, color="black", alpha=0.5, label=f"Coin Rate (Cumulative Average Rate)")
  plt.errorbar(bin_centers_datetime, Mean_count_accumulation, yerr=Std_count_accumulation, fmt='none', ecolor='black', capsize=0, label='Std of Cumulative Average Rate')
  
  plt.xlabel("Time") # X軸ラベルを "Time" に変更
  plt.ylabel("Coin Rate (counts/sec)")
  
  ax = plt.gca() # X軸の目盛りを日時形式に設定

  locator = mdates.AutoDateLocator(minticks=5, maxticks=10) # この値を調整
  ax.xaxis.set_major_locator(locator)

  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')) # 例: 2025-05-23 15:00
  plt.xticks(rotation=45, ha='right') # X軸のラベルが重ならないように回転
  plt.title(input_filename)
  plt.legend()
  plt.tight_layout()

  # 入力ファイル名からPNGファイル名を生成
  make_dir_path = './pocket_counter_output/changes/'
  #作成しようとしているディレクトリが存在するかどうかを判定する
  if os.path.isdir(make_dir_path):
    #既にディレクトリが存在する場合は何もしない
    pass
  else:
    os.makedirs(make_dir_path)
  filename = input_filename.split("/")[len(input_filename.split("/"))-1]
  base_name = make_dir_path + os.path.splitext(filename)[0]
  output_png_filename = f"{base_name}_ana{bin_width_sec/60:.0f}-min_date.png"

  # グラフをPNGファイルとして保存
  plt.savefig(output_png_filename)

#数字を抽出する関数
def numb(text):
    return re.findall(r'\d+', text)[0]

def main():
  # ワイルドカードで条件を満たすパスの文字列を指定
  pathlist = glob.glob('./pocket_counter_output/datas/*.txt')
  
  pathlist = natsorted(pathlist)
  
  for path in pathlist:
    treatment(path)


#このファイルが直接実行されたときにmain関数を呼び出す。
if __name__ == '__main__':
  main()