# Databricks notebook source
# MAGIC %md # ワイン品質データを利用した機械学習モデル学習、モデルレジストリ、推論までのエンドトゥエンドの例
# MAGIC 
# MAGIC 本チュートリアルでは下記を実施します:
# MAGIC - ローカルのCSVデータを Databricks ファイルシステム (DBFS) にインポート
# MAGIC - Seabornとmatplotlibを使ってデータを可視化
# MAGIC - 並列ハイパーパラメータスイープを実行して、機械モデル学習を実行
# MAGIC - MLflowでハイパーパラメータスイープの結果を調べる
# MAGIC - MLflowに最も性能の良いモデルを登録する
# MAGIC - 登録したモデルをSpark UDFを使って別のデータセットに適用する
# MAGIC - 低レイテンシーのリクエストに対応したモデルサービスの設定
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2020/11/19</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>7.4ML</td></tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md ## データのインポート
# MAGIC   
# MAGIC  このセクションでは、ウェブからデータセットをダウンロードし、Databricks File System (DBFS)にアップロードします。
# MAGIC 
# MAGIC 1. ダウンロードサイトである https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ に移動して、 `winequality-red.csv` 及び `winequality-white.csv` をローカルマシーンにダウンロード。
# MAGIC 
# MAGIC 1. このDatabricksノートブックから *File* > *Upload Data*　を選択して、これらのファイルをドラッグアンドドロップターゲットにドラッグして、Databricksファイルシステム(DBFS)にアップロード。
# MAGIC 
# MAGIC 1. ボタン *Next* をクリックして、データをロードするためのいくつかの自動生成されたコードが表示されますので、pandas*を選択し、サンプルコードをコピー。
# MAGIC 
# MAGIC 1. 新しいセルを作成し、サンプルコードを貼り付け。以下のように変更。
# MAGIC   - pd.read_csv` に `sep=';'` を渡します。
# MAGIC   - 以下のセルのように、変数名を `df1`, `df2` から `white_wine`, `red_wine` に変更します。 

# COMMAND ----------

import pandas as pd
white_wine = pd.read_csv("https://psajpstorage.blob.core.windows.net/commonfiles/winequality-white.csv", sep=";")
red_wine = pd.read_csv("https://psajpstorage.blob.core.windows.net/commonfiles/winequality-red.csv", sep=";")

# COMMAND ----------

#import pandas as pd
# <username>を自分用に変更してください
#white_wine = pd.read_csv("/dbfs/FileStore/shared_uploads/<username>/winequality_white.csv", sep=';')
#red_wine = pd.read_csv("/dbfs/FileStore/shared_uploads/<username>/winequality_red.csv", sep=';')
#white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
#red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

# MAGIC %md 2つのDataFramesを1つのデータセットに統合し、ワインが赤か白かを示す新しいバイナリフィーチャ "is_red "を追加します。

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# カラム名から空白スペースを削除
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md ##データの可視化
# MAGIC 
# MAGIC モデルを学習する前に、SeabornとMatplotlibを使ってデータセットを探索します。

# COMMAND ----------

# MAGIC %md 従属変数である品質(quality)のヒストグラムをプロットします。

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md 品質のスコアは3と9の間に正規分布しているように見えます。品質≧7であれば高品質のワインと定義します。

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md ボックスプロットは、特徴量とバイナリラベルの間の相関関係に気づくのに便利です。

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md 上のボックスプロットでは、品質の優れた一変量予測因子としていくつかの変数が際立っています。
# MAGIC 
# MAGIC - アルコール度数のボックス・プロットでは、高品質のワインのアルコール度数の中央値は、低品質のワインの75分の1よりも大きい。高いアルコール度数は品質と相関している。
# MAGIC - 密度のボックスプロットでは、低品質のワインは高品質のワインよりも密度が高い。密度は品質と逆相関している。

# COMMAND ----------

# MAGIC %md ## データプロセシング
# MAGIC モデルを訓練する前に、欠落値をチェックし、データを訓練セットと検証セットに分割します。

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md 欠損値は無いようです。

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["quality"], axis=1)
X_test = test.drop(["quality"], axis=1)
y_train = train.quality
y_test = test.quality

# COMMAND ----------

# MAGIC %md ## ベースとなるモデルを作成
# MAGIC 
# MAGIC このタスクはランダムフォレスト分類器に適していると思われます。
# MAGIC 
# MAGIC 以下のコードは、scikit-learnを用いて単純な分類器を構築しています。モデルの精度を追跡し、後で使用するためにモデルを保存するためにMLflowを使用しています。

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature

# sklearnのRandomForestClassifierのpredictメソッドは，バイナリ分類（0または1）を返します．
# 次のコードにてラッパー関数SklearnModelWrapperを作成します．
# 観測値が各クラスに属する確率を返すための predict_proba メソッドを利用します．

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# このモデルのパフォーマンスを追跡するために、 mlflow.start_run にて MLflow の実行します。
# コンテキスト内では、使用されたパラメータを追跡するために mlflow.log_param を呼び出します。
# 精度などのメトリクスを記録するための mlflow.log_metric を呼び出します。

with mlflow.start_run(run_name='untuned_random_forest'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_proba は [prob_negative, prob_positive]　を返します。
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # Area under the ROC curve をメトリックスとして利用します。
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
  # モデルインプットとアウトプットを定義する signature を利用してロギングします。 
  # モデルがデプロイされる際には、この signature によりモデルインプットの検証が可能です。
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, signature=signature)

# COMMAND ----------

# MAGIC %md 特徴量のImportanceをチェックします。

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md 前述のボックスプロットで示されているように、アルコール(alcohol)と密度(density)の両方が品質(quality)を予測する上で重要です。

# COMMAND ----------

# MAGIC %md 右の **Experiment** タブを参照すると、AUCの値を確認可能です。AUCの値は０．８９となります。

# COMMAND ----------

# MAGIC %md #### このモデルを MLflow モデルレジストリに登録します。
# MAGIC 
# MAGIC このモデルをModel Registryに登録することで、Databricks内のどこからでも簡単にモデルを参照することができます。
# MAGIC 
# MAGIC 次項ではプログラムで行う方法を紹介しますが、[モデルレジストリにモデルを登録](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-registry)の手順を踏んで、GUIベースでモデルを登録することもできます。

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

# COMMAND ----------

# "PERMISSION_DENIED: User does not have any permission level assigned to registered model "というエラーが表示された場合. 
# 原因は, "wine_quality "という名前のモデルが既に存在している可能性があります. 別の名前を使ってみてください。
model_name = "wine_quality_202202"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# COMMAND ----------

# MAGIC %md これで、モデルページにワイン品質のモデルが表示されるはずです。［モデル］ページを表示するには、左サイドバーの［モデル］アイコンをクリックします。
# MAGIC 
# MAGIC 次に、このモデルを"Production"に移行し、モデルレジストリからこのノートブックにロードします。

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md モデル・ページには、ステージ "Production" のモデル・バージョンが表示されます。
# MAGIC "models:/-win-quality/production" というパスを使ってモデルを参照できるようになります。

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ##実験で新しいモデルを作成してみます。
# MAGIC 
# MAGIC 前述のランダムフォレストモデルは、ハイパーパラメータをチューニングしなくても良好な結果が得られました。
# MAGIC 
# MAGIC 以下のコードは、より正確なモデルを学習するためにxgboostライブラリを使用しています。これは，並列ハイパーパラメータ・スイープを実行して，
# MAGIC 複数のモデルをHyperoptとSparkTrialsを使って並列に作成しています。これまでと同様に、コードはMLflowを用いて各パラメータ設定の性能をトラッキングします。

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Set a seed for deterministic training
}

def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)
    # Pass in the test set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
    # is no longer improving.
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(test, "test")], early_stopping_rounds=50)
    predictions_test = booster.predict(test)
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_metric('auc', auc_score)

    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=10)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(run_name='xgboost_models'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=96,
    trials=spark_trials, 
    rstate=np.random.RandomState(123)
  )

# COMMAND ----------

# MAGIC %md  #### MLflow にて結果を参照してみる
# MAGIC 
# MAGIC MLflow の実行結果を見るには、Experiment Runs サイドバーを開きます。下向き矢印の隣の日付(Date)をクリックしてメニューを表示し、「auc」を選択して、auc メトリックでソートされた実行を表示します。最高の auc 値は 0.91 です。ベースラインを超えました。
# MAGIC 
# MAGIC MLflow は、各ランのパラメータとパフォーマンス指標を追跡します。Experiment Runs サイドバーの上部にある外部リンクアイコン <img src="https://docs.microsoft.com/azure/databricks/_static/images/external-link.png"/> をクリックすると、MLflow のランテーブルに移動します。

# COMMAND ----------

# MAGIC %md 次に、ハイパーパラメータの選択がAUCとどのように相関するかを調べます。"+"アイコンをクリックして親ランを展開し、Parent Run を除くすべてのRunを選択して、"Compare "をクリックします。平行座標プロット(Parallel Coordinates Plot)を選択します。
# MAGIC 
# MAGIC 平行座標プロットは、メトリックに対するパラメータの影響を理解するのに便利です。プロットの右上隅にあるピンクのスライダバーをドラッグして、AUC値のサブセットと対応するパラメータ値を強調表示することができます。以下のプロットでは、最も高いAUC値を強調表示しています。
# MAGIC 
# MAGIC <img src="https://docs.microsoft.com/azure/databricks/_static/images/mlflow/end-to-end-example/parallel-coordinates-plot.png"/>
# MAGIC 
# MAGIC 上位の実行はすべて、reg_lambdaとlearning_rateの値が低いことに注目してください。
# MAGIC 
# MAGIC 別のハイパーパラメータスイープを実行して、これらのパラメータの値をさらに低くすることができます。ここではこのステップは省略します。

# COMMAND ----------

# MAGIC %md 
# MAGIC MLflow を使用して、各ハイパーパラメータ設定によって生成されたモデルをログに記録しました。以下のコードは、最も性能の良いRunを見つけ、モデルをモデルレジストリに保存します。

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

# MAGIC %md #### MLflow モデル・レジストリでのProductionｍにあるワイン品質モデルの更新
# MAGIC 先ほど、ベースラインモデルをモデルレジストリに「wine_quality」の下に保存しました。より正確なモデルを作成したので、wine_qualityを更新します。

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)

# COMMAND ----------

# MAGIC %md 左サイドバーの **Models** をクリックすると、wine_qualityモデルに2つのバージョンがあることがわかります。
# MAGIC 
# MAGIC 以下のコードは、新しいバージョンをProductionへプロモーションします。

# COMMAND ----------

# 古いバージョンをアーカイブ
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Archived",
)

# 新しいモデルをProductionへ
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md load_model を呼び出すと、クライアントは新しいモデルを受け取ることができます。

# COMMAND ----------

# このコードは、前述の　"ベースラインモデル"と同様です。変更不要です。
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ##バッチ推論
# MAGIC 
# MAGIC 新しいデータを利用してモデルを評価するシナリオは複数あります。例えば、新しいバッチデータに適用したり、2つのモデルの性能を比較したりする場合などです。
# MAGIC 以下のコードは、Deltaテーブルに格納されたデータに対してモデルを評価し、Sparkを使って並列計算を実行しています。

# COMMAND ----------

# 新しいデータでシミュレーションするには, 既存のX_trainデータをデルタテーブルに保存します. 
# 現実の世界では, これは新しいデータのバッチになります.
spark_df = spark.createDataFrame(X_train)
# <username> を自身の名前に変更
table_path = "dbfs:/<username>/delta/wine_data"
# このパスのデータを事前にクリーンアップ
dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)

# COMMAND ----------

# MAGIC %md モデルを Spark UDF に読み込みDeltaテーブルに適用可能とします。

# COMMAND ----------

import mlflow.pyfunc

apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# COMMAND ----------

# Deltaから "new data" を読み込み
new_data = spark.read.format("delta").load(table_path)

# COMMAND ----------

display(new_data)

# COMMAND ----------

from pyspark.sql.functions import struct

# モデルを new data　に適用
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

# これで、各行は関連する予測値を持つようになりました。xgboost関数はデフォルトで確率を出力しないので、予測値は[0, 1]の範囲に制限されないことに注意してください。
display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのサービング
# MAGIC 
# MAGIC 低レイテンシー予測のためのモデルを本番適用させるには、MLflow [model serving](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-serving) を使用して、モデルをエンドポイントにデプロイします。
# MAGIC 
# MAGIC 次のコードは、展開したモデルから予測値を取得するために、REST API を使用してリクエストを発行する方法を示しています。

# COMMAND ----------

# MAGIC %md
# MAGIC モデルのエンドポイントにリクエストを発行するには、Databricksトークンが必要です。トークンは、ユーザー設定ページ（右上のプロファイルアイコンの下）から生成できます。トークンを次のセルにコピーします。

# COMMAND ----------

import os
os.environ["DATABRICKS_TOKEN"] = "dapi17939b87ebd7c2c55b7b59172d9e2a95"

# COMMAND ----------

# MAGIC %md
# MAGIC 左サイドバーの**モデル**をクリックし、登録されているワインモデルに移動します。サービングタブをクリックし、**Enable Serving**をクリックします。
# MAGIC 
# MAGIC 次に、**Call The Model**の下の**Python**ボタンをクリックして、リクエストを発行するためのPythonコードスニペットを表示します。コードをこのノートブックにコピーします。次のセルのコードと似ているはずです。
# MAGIC 
# MAGIC トークンを使用して、Databricksの外部ノートブックからもリクエストを発行することができます。

# COMMAND ----------

import os
import requests
import pandas as pd

def score_model(dataset: pd.DataFrame):
#  url = 'https://DATABRICKS_URL/model/wine_quality_20201127/Production/invocations'
  url = 'https://westus.azuredatabricks.net/model/wine_quality_20201127/2/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split')
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC エンドポイントからのモデル予測は、モデルをローカルで評価した結果と一致している必要があります。

# COMMAND ----------

# モデル・サービングは、より小さなデータ・バッチでの低レイテンシー予測のために設計されています。
num_predictions = 5
served_predictions = score_model(X_test[:num_predictions])
model_evaluations = model.predict(X_test[:num_predictions])
# デプロイしたモデルと学習したモデルの結果を比較します。
pd.DataFrame({
  "Model Prediction": model_evaluations,
  "Served Model Prediction": served_predictions,
})

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h1>END</h1>  
