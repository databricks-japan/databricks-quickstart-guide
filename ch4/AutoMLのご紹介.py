# Databricks notebook source
# MAGIC %md # Databricks AutoMLのご紹介
# MAGIC 
# MAGIC Databricksのサンプルデータから与信データを使用し、貸し倒れ予測のモデルをAutoMLで構築します。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/07/10</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricks AutoMLのご紹介 : 機械学習開発の自動化に対するガラスボックスアプローチ \- Qiita](https://qiita.com/taka_yayoi/items/3bf4f4e89990b299c6c9)
# MAGIC - [Databricks AutoML \- Qiita](https://qiita.com/taka_yayoi/items/0ed860131a02aeade14f)
# MAGIC - [Databricks AutoML \| Databricks on AWS](https://docs.databricks.com/applications/machine-learning/automl.html)

# COMMAND ----------

# MAGIC %md ## データ準備
# MAGIC サンプルデータとして、アメリカのLendingClubという会社が公開している与信データを用います。

# COMMAND ----------

import re
from pyspark.sql.types import * 

# ログインIDからUsernameを取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# データベース名
db_name = f"automl_{username}"

# Hiveメタストアのデータベースの準備:データベースの作成
spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
# Hiveメタストアのデータベースの選択
####spark.sql(f"USE {db_name}")

print("database name: " + db_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### データの読み込み

# COMMAND ----------

# パス指定 Sparkではdbfs:を使用してアクセス
source_path = 'dbfs:/databricks-datasets/samples/lending_club/parquet/'

# ソースディレクトリにあるParquetファイルをデータフレームに読み込む
data = spark.read.parquet(source_path)

# 読み込まれたデータを参照
display(data)

# レコード件数確認
print("レコード件数:" , data.count())

# COMMAND ----------

# 一次ビューとして登録
data.createOrReplaceTempView("bronze_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### データクレンジング(不要なカラム/レコードの削除)

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- クレンジングを実施します
# MAGIC DROP TABLE IF EXISTS loan_stats_work;
# MAGIC CREATE TABLE loan_stats_work AS
# MAGIC SELECT
# MAGIC   annual_inc,
# MAGIC   addr_state,
# MAGIC   chargeoff_within_12_mths,
# MAGIC   delinq_2yrs,
# MAGIC   delinq_amnt,
# MAGIC   dti,
# MAGIC   --emp_length,
# MAGIC   emp_title,
# MAGIC   grade,
# MAGIC   home_ownership,
# MAGIC   cast(replace(int_rate, '%', '') as float) as int_rate,
# MAGIC   installment,
# MAGIC   loan_amnt,
# MAGIC   open_acc,
# MAGIC   pub_rec,
# MAGIC   purpose,
# MAGIC   pub_rec_bankruptcies,
# MAGIC   revol_bal,
# MAGIC   cast(replace(revol_util, '%', '') as float) as revol_util,
# MAGIC   sub_grade,
# MAGIC   total_acc,
# MAGIC   verification_status,
# MAGIC   zip_code,
# MAGIC   case
# MAGIC     when loan_status = 'Fully Paid' then 0
# MAGIC     else 1
# MAGIC   end as bad_loan
# MAGIC FROM
# MAGIC   bronze_data
# MAGIC WHERE
# MAGIC   loan_status in (
# MAGIC     'Fully Paid',
# MAGIC     'Default',
# MAGIC     'Charged Off',
# MAGIC     'Late (31-120 days)',
# MAGIC     'Late (16-30 days)'
# MAGIC   )
# MAGIC   AND addr_state is not null;

# COMMAND ----------

# MAGIC %md
# MAGIC ### クレンジングデータの確認

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from loan_stats_work;

# COMMAND ----------

# MAGIC %sql
# MAGIC --特定の特徴量が予測対象にどの程度影響しているかをBoxチャートで確認
# MAGIC select case when bad_loan = 1 then '貸し倒れ' else '完済' end as bad_loan , annual_inc as `年収` from loan_stats_work

# COMMAND ----------

# MAGIC %md 
# MAGIC ### SQLによるデータ加工

# COMMAND ----------

# MAGIC %sql
# MAGIC --(1) emp_titleの種類が多いので職業についている場合は"EMPLOYED"を代入
# MAGIC update loan_stats_work set emp_title = 'EMPLOYED' where emp_title is not null;

# COMMAND ----------

# MAGIC %sql
# MAGIC --(2) emp_titleの欠損値には"UNEMPLOYED"を代入
# MAGIC update loan_stats_work set emp_title = 'UNEMPLOYED' where emp_title is null;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from loan_stats_work;

# COMMAND ----------

# テーブルのデータをデータフレームに読み込み
loan_stats_df = spark.sql("select * from loan_stats_work")

# randomSplit()を使って、5%のサンプルを使用
(train_df, test_df) = loan_stats_df.randomSplit([0.05, 0.95], seed=123)

# COMMAND ----------

print("トレーニングデータ件数:", train_df.count())

# COMMAND ----------

# AutoML UIから参照できるようにデータフレームをテーブルに保存します
train_df.write.saveAsTable("train_df")

# COMMAND ----------

# MAGIC %md これでモデルを作成する準備ができました。

# COMMAND ----------

# MAGIC %md 
# MAGIC ## AutoMLによるハイパーパラメーターチューニング＋分散学習の実行
# MAGIC 
# MAGIC 上のトレーニングデータを用いてGUIからAutoMLを実行します。

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks AutoMLでは、GUIでの操作に加え、以下のようにAPI経由での実行も可能です。
# MAGIC 
# MAGIC 次のコードでは、AutoMLを使用してモデルの分散トレーニングを実行し、同時にハイパーパラメータのチューニングを実行します。また、MLflowにより学習に使用されたハイパーパラメータ設定とそれぞれの精度をトラッキングし、モデルアーティファクトとともに保存します。

# COMMAND ----------

from databricks import automl
summary = automl.classify(train_df, # 特徴量と目的変数を含むトレーニング用データ、これがさらにトレーニングデータ、テストデータ、検証用データに分割される
                          target_col="bad_loan", # 目的変数のカラム名
                          primary_metric="roc_auc", # モデルを評価するためのメトリクス
                          timeout_minutes=5, # タイムアウト
                          max_trials=2) # トライアル数の上限

# COMMAND ----------

# MAGIC %md
# MAGIC MLflowが持つオートロギング機能によって、以下の項目がモデルアーチファクトとして自動で保存されます。
# MAGIC 
# MAGIC 1. ハイパーパラメータ
# MAGIC 1. 作成されたモデル
# MAGIC 1. 特徴量の重要度

# COMMAND ----------

# MAGIC %md  
# MAGIC #### MLflowのUIで結果を参照してみる
# MAGIC 
# MAGIC MLflow の実行結果を見るには**Experiment**サイドバーを開きます。下向き矢印の隣の日付(Date)をクリックしてメニューを表示し`auc`を選択して、`auc`メトリックでソートされたMLflowランを表示します。
# MAGIC 
# MAGIC MLflowは、各ランのパラメータとパフォーマンス指標を追跡します。**Experiment**サイドバーの上部にある外部リンクアイコン <img src="https://docs.microsoft.com/azure/databricks/_static/images/external-link.png"/>をクリックすると、MLflowのランテーブルに移動します。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
