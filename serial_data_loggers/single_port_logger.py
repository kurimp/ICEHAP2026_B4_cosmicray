import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
ser = serial.Serial('COM3', 115200, timeout=3)
#測定前に記録
x_in = 12 #cm
h_in = 10 #cm
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp_YMD = datetime.datetime.now().strftime("%Y%m%d")
filename = f"output_x{x_in}cm_h{h_in}cm_{timestamp}_COM3.txt"
out = open(filename, "w")
t0 = time.time()
times = []
ch0s = []
ch1s = []
coins = []
c = 0
try:
    while(True):
        try:
            ser.write("getcounter\n".encode())
            line = ser.readline().decode().replace("\n", "")
            vals = line.split(",")
            coin = int(vals[0].replace("COUNT:", "").split(" (")[0])
            ch0  = int(vals[1].replace("ch0:", ""))
            ch1  = int(vals[2].replace("ch1:", ""))
            t = time.time()
            txt = "{0}\t{1}\t{2}\t{3}\n".format(t, coin, ch0, ch1)
            out.write(txt)
            print(txt, end="")
            times.append(t-t0)
            ch0s.append(ch0)
            ch1s.append(ch1)
            coins.append(coin)
            #if(c<100 or (c%100==0)):
                #plt.plot(times, coins, "o", color="black")
                #plt.plot(times, ch0s, "o", color="red")
                #plt.plot(times, ch1s, "o", color="blue")
                #plt.pause(1)
            time.sleep(5)
            #c += 1
        except serial.SerialException as e:
            print(f"シリアル通信エラーが発生しました: {e}")
            time.sleep(5)
            continue
        except ValueError as e:
            print(f"データ変換エラーが発生しました: {line} - {e}")
            # 不正なデータを受信した場合の処理
            continue
        except Exception as e: #KeyboardInterrupt以外の予期せぬエラー
            print(f"ループ内で予期せぬエラーが発生しました: {e}")
            break #ループを抜ける
except KeyboardInterrupt:
    print("\nCtrl+C が押されました。ループを終了し、後処理を実行します。")
except Exception as e:
    print(f"メイン処理で予期せぬエラーが発生しました: {e}")
finally: #例外発生の有無に関わらず実行
    print("\nクリーンアップ処理を実行します。")
    if 'out' in locals() and not out.closed:
        out.close()
    if 'ser' in locals() and ser.is_open:
        ser.close()
# ----- ここからSlackへのアップロード処理 -----
# 設定値
SLACK_BOT_TOKEN = #APIキーを入力。
CHANNEL_ID = "C08TW4BKEFM"
# サブタスクのリスト（ファイルパスと親メッセージのタイムスタンプ）
SUBTASKS = [
    {"file_path": filename, "thread_ts": "1748414219.526629", "initial_comment": f"{timestamp_YMD}のx={x_in} cm, h={h_in} cmのCOM3測定データです。"},
]
if not os.path.exists(filename):
    print(f"エラー: アップロードするファイルが見つかりません: {filename}")
else:
    # Slackクライアントの初期化
    client = WebClient(token=SLACK_BOT_TOKEN)
    # 各サブタスクを処理
    for subtask in SUBTASKS:
        file_path = subtask.get("file_path")
        thread_ts = subtask.get("thread_ts")
        initial_comment = subtask.get("initial_comment", "サブタスクの関連ファイルです。")
        filename_on_slack = subtask.get("filename_on_slack", os.path.basename(file_path) if file_path else "uploaded_file")
        if not file_path:
            print("エラー: サブタスクのファイルパスが指定されていません。")
            continue
        if not os.path.exists(file_path): # 再度チェック
            print(f"エラー: 指定されたファイルが見つかりません（アップロード直前）: {file_path}")
            continue
        try:
            upload_params = {
                "channel": CHANNEL_ID,
                "file": file_path,
                "initial_comment": initial_comment, # initial_comment を有効化
                "filename": filename_on_slack,
            }
            if thread_ts:
                upload_params["thread_ts"] = thread_ts
            response = client.files_upload_v2(**upload_params)
            if response.get("ok"):
                if thread_ts:
                    print(f"ファイル「{response['file']['name']}」がスレッド (親TS: {thread_ts}) へのアップロードに成功しました。")
                else:
                    print(f"ファイル「{response['file']['name']}」がチャンネルへのアップロードに成功しました。")
            else:
                print(f"ファイルのアップロードに失敗しました。エラー: {response.get('error', 'Unknown error')}")
        except SlackApiError as e:
            print(f"Slack APIエラーが発生しました: {e.response['error']} (ファイル: {file_path})")
        except FileNotFoundError:
            print(f"エラー: 指定されたファイルが見つかりません（アップロード処理中）: {file_path}")
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e} (ファイル: {file_path})")
        print("-" * 30)
print("スクリプトの処理が完了しました。")