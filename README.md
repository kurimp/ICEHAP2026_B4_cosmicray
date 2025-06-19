実行順序
Cosmic_Ray_Simulation.py

目的: 宇宙線が検出器を通過するシミュレーションを実行し、さまざまな検出器配置での同時計数率、立体角、および平均コサイン角度を計算します。結果は CosmicRaySimulation_flux.csv ファイルに保存されます。
出力: CosmicRaySimulation/CosmicRaySimulation_flux.csv と simulation_config.cfg。
注意事項: このスクリプトはシミュレーションの基盤となるデータを作成するため、必ず最初に実行する必要があります。シミュレーションパラメータはスクリプト内で直接設定されており、simulation_config.cfg にも出力されます。
add_mayoko.py

目的: 特定のシミュレーション結果（x=200, z=45）を CosmicRaySimulation_flux.csv に追加します。このスクリプトは既存のシミュレーション結果ファイルに依存しています。
入力: CosmicRaySimulation/CosmicRaySimulation_flux.csv
出力: 更新された CosmicRaySimulation/CosmicRaySimulation_flux.csv
注意事項: このスクリプトは Cosmic_Ray_Simulation.py が実行され、CosmicRaySimulation_flux.csv が存在した後に実行する必要があります。
performance.py

目的: 検出器の性能（効率）を評価します。これは、異なるCOMポートからのデータを比較して、各検出器の相対的な効率を算出します。結果は performance.csv に保存され、後の分析で重み付けとして使用されます。
入力: ./performance/datas/*.txt (実験データ)
出力: ./performance/results/performance.csv
注意事項: このスクリプトは、実際の検出器からのデータファイル (./performance/datas/ に格納されている.txtファイル) を必要とします。
ana_hist_date.py

目的: 生の実験データ（コインシデンスカウント）を読み込み、時間経過に伴うコインシデンスレートの平均と標準偏差を計算し、グラフとして出力します。また、バックグラウンド測定の結果を別途ファイルに保存します。
入力: ./pocket_counter_output/datas/*.txt (実験データ)
出力:
./pocket_counter_output/images/ にレートの時系列グラフ (PNG)
./pocket_counter_output/results/pocket_counter_output_*.csv に各測定の集計結果
./pocket_counter_output/BG.csv にバックグラウンド測定の集計結果
注意事項: このスクリプトは、検出器の実験データが ./pocket_counter_output/datas/ ディレクトリに配置されていることを前提としています。バックグラウンド測定のデータもこのディレクトリに含まれている必要があります。
analyzation.py

目的: これまでのステップで生成されたシミュレーション結果、検出器効率、および実験データの集計結果を統合し、最終的な宇宙線フラックスの計算とコサイン角度に対するフィットを行います。結果はグラフとして出力され、Result.csv に保存されます。
入力:
./pocket_counter_output/results/*.csv (複数の pocket_counter_output_*.csv のうち最新のもの)
./performance/results/performance.csv
./CosmicRaySimulation/CosmicRaySimulation_flux.csv
./pocket_counter_output/BG.csv
出力:
Result.csv
analyzation_result.png (フィット結果のグラフ)
注意事項: このスクリプトは、上記のすべての前提となるファイルが生成されていることを確認してから実行してください。
coin_change.py

目的: 各測定の累積平均コインシデンスレートとその標準偏差を時間経過でプロットします。これは、個々の測定の安定性や傾向を視覚的に確認するために使用できます。
入力: ./pocket_counter_output/datas/*.txt (実験データ)
出力: ./pocket_counter_output/changes/*.png
注意事項: このスクリプトは、分析のメインフローとは独立して実行できますが、ana_hist_date.py と同じ実験データファイルを使用します。分析の補助的な可視化として利用できます。
全体的な注意事項
ディレクトリ構造: スクリプトは特定のディレクトリ構造を期待しています。実行前に以下のディレクトリが存在し、関連ファイルが適切に配置されていることを確認してください。
./CosmicRaySimulation/
./performance/datas/
./performance/results/
./pocket_counter_output/datas/
./pocket_counter_output/images/
./pocket_counter_output/results/
./pocket_counter_output/changes/
ファイル名: スクリプトは glob と natsort を使用して最新のファイルを自動的に選択しますが、ファイル名の命名規則（特に日付やCOMポート情報を含むもの）がスクリプトの期待するものと一致していることを確認してください。
データ形式: すべての入力データファイルはタブ区切り (\t) で、特定のヘッダーや列の順序に従っている必要があります。
数値精度: スクリプト内で :.4e や :.4f のように浮動小数点数の出力精度が指定されている箇所があります。必要に応じて調整してください。
