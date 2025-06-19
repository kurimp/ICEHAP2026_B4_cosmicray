import serial
import time
import datetime
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
# --- 設定項目 ---
# 測定前に記録
COM_PORTS = ['COM3', 'COM4']
BAUD_RATE = 115200
TIMEOUT_SEC = 3
LOOP_INTERVAL_SEC = 0.5
# Slackへのアップロード設定
SLACK_BOT_TOKEN = #APIキーを入力。
CHANNEL_ID = "C08TW4BKEFM"
THREAD_TS = "1748414219.526629"  # 親メッセージのタイムスタンプ
# --- ファイル名の生成 ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp_YMD = datetime.datetime.now().strftime("%Y%m%d")
filename = f"output_{COM_PORTS[0]}_{COM_PORTS[1]}.txt"
# --- シリアルポートからデータを取得する関数 ---
def get_data_from_port(ser):
    """
    指定されたシリアルポートからカウンターデータを取得し、パースして返す。
    エラーが発生した場合は (None, None, None) を返す。
    """
    if not ser or not ser.is_open:
        return None, None, None
    try:
        ser.write("getcounter\n".encode())
        line = ser.readline().decode().strip()
        if not line:  # タイムアウトした場合
            print(f"Warning: No data from {ser.port} (timeout).")
            return None, None, None
        vals = line.split(",")
        coin = int(vals[0].replace("COUNT:", "").split(" (")[0])
        ch0 = int(vals[1].replace("ch0:", ""))
        ch1 = int(vals[2].replace("ch1:", ""))
        return coin, ch0, ch1
    except (serial.SerialException, ValueError, IndexError) as e:
        print(f"Error reading or parsing data from {ser.port}: {e}")
        return None, None, None
# --- メイン処理 ---
serial_connections = {}
for port in COM_PORTS:
    try:
        serial_connections[port] = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT_SEC)
        print(f"Successfully opened {port}.")
    except serial.SerialException as e:
        print(f"Error: Could not open {port}. {e}")
        serial_connections[port] = None
if all(ser is None for ser in serial_connections.values()):
    print("Error: No COM ports could be opened. Exiting.")
    exit()
output_file = None
try:
    output_file = open(filename, "w")
    # ヘッダーを書き込む
    header = "Timestamp\tCOM3_Coin\tCOM3_CH0\tCOM3_CH1\tCOM4_Coin\tCOM4_CH0\tCOM4_CH1\n"
    output_file.write(header)
    print(header.strip())
    while True:
        current_time = time.time()
        # 各ポートからデータを取得
        data_com3 = get_data_from_port(serial_connections.get('COM3'))
        data_com4 = get_data_from_port(serial_connections.get('COM4'))
        # データを文字列に変換（データがない場合は 'N/A'）
        coin3, ch0_3, ch1_3 = (str(d) if d is not None else "N/A" for d in data_com3)
        coin4, ch0_4, ch1_4 = (str(d) if d is not None else "N/A" for d in data_com4)
        # 出力するテキストを生成
        # コンソールには主要なコインシデンスのみ表示
        console_txt = f"{current_time:.2f}\tCOM3_Coin: {coin3}\tCOM4_Coin: {coin4}"
        # ファイルには全データを記録
        file_txt = f"{current_time}\t{coin3}\t{ch0_3}\t{ch1_3}\t{coin4}\t{ch0_4}\t{ch1_4}\n"
        output_file.write(file_txt)
        print(console_txt)
        time.sleep(LOOP_INTERVAL_SEC)
except KeyboardInterrupt:
    print("\nCtrl+C detected. Exiting loop and cleaning up.")
except Exception as e:
    print(f"\nAn unexpected error occurred in the main loop: {e}")
finally:
    print("\nCleaning up resources...")
    if output_file and not output_file.closed:
        output_file.close()
        print("Output file closed.")
    for port, ser in serial_connections.items():
        if ser and ser.is_open:
            ser.close()
            print(f"{port} closed.")
# --- Slackへのアップロード処理 ---
if not os.path.exists(filename) or os.path.getsize(filename) <= 0:
    print(f"Error: File '{filename}' not found or is empty. Skipping upload.")
else:
    print(f"\nAttempting to upload '{filename}' to Slack...")
    client = WebClient(token=SLACK_BOT_TOKEN)
    initial_comment_text = f"{timestamp_YMD}の測定データです（{COM_PORTS[0]} & {COM_PORTS[1]}）。"
    try:
        response = client.files_upload_v2(
            channel=CHANNEL_ID,
            file=filename,
            initial_comment=initial_comment_text,
            thread_ts=THREAD_TS,
            filename=os.path.basename(filename)
        )
        if response.get("ok"):
            print(f"Successfully uploaded file '{response['file']['name']}' to the thread.")
        else:
            print(f"Failed to upload file. Error: {response.get('error', 'Unknown error')}")
    except SlackApiError as e:
        print(f"Slack API Error: {e.response['error']} (File: {filename})")
    except FileNotFoundError:
        print(f"Error: File not found during upload process: {filename}")
    except Exception as e:
        print(f"An unexpected error occurred during upload: {e}")
print("\nScript finished.")