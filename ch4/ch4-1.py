# Databricks notebook source
dbutils.fs.ls("dbfs:/databricks-datasets/")

# COMMAND ----------

dbutils.fs.ls("dbfs:/databricks-datasets/online_retail/data-001")

# COMMAND ----------

df = spark.read.csv("dbfs:/databricks-datasets/online_retail/data-001/data.csv")

# COMMAND ----------

display(df)

# COMMAND ----------

df = spark.read.option("header", True).csv("dbfs:/databricks-datasets/online_retail/data-001/data.csv")

# COMMAND ----------

display(df)

# COMMAND ----------

df2 = spark.read.parquet("dbfs:/databricks-datasets/amazon/test4K/")

# COMMAND ----------

display(df2)

# COMMAND ----------

from pyspark.sql.functions import col

df = df.withColumn("total", col("Quantity") * col("UnitPrice")) 

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.filter(df.CustomerID.isNull()))

# COMMAND ----------

df_null = df.na.fill({'CustomerID' : '123456'})
display(df_null.filter(df_null.CustomerID.isNull()))

# COMMAND ----------

display(df_null.filter(df_null.CustomerID=='123456'))

# COMMAND ----------

display(df.filter(df.Quantity <'0'))

# COMMAND ----------

df.select('Quantity').where(df.Quantity <'0').count()

# COMMAND ----------

display(df.filter(df.Quantity >'0'))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col


spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

dfWithDate = df.withColumn("date", F.to_date(F.to_timestamp(col("InvoiceDate"), "M/d/y H:mm")))


# COMMAND ----------

display(dfWithDate)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col


df_trans = df.filter(df.Quantity >'0') \
             .withColumn("total", col("Quantity") * col("UnitPrice")) \
             .withColumn("Date", F.to_date(F.to_timestamp(col("InvoiceDate"), "M/d/y H:mm"))) \
             .na.fill({'CustomerID' : '123456'})

# COMMAND ----------

display(df_trans)

# COMMAND ----------

save_path = '/tmp/delta/online_retail'

df_trans.write \
  .format('delta') \
  .save(save_path)

# COMMAND ----------

table_name = 'default.online_retail'

spark.sql("CREATE TABLE " + table_name + " USING DELTA LOCATION '" + save_path + "'")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DESCRIBE EXTENDED online_retail;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select count(*) from online_retail;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select sum(total) as total_sum, CustomerId from online_retail group by CustomerId order by total_sum desc limit 10; 

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select sum(total) as total_sum, Country from online_retail group by Country order by Country;

# COMMAND ----------

aws_bucket_name = "Enter Bucker Name"
mount_name = "Enter Mount Name"
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % mount_name)
display(dbutils.fs.ls("/mnt/%s" % mount_name))

# COMMAND ----------

display(dbutils.fs.ls("/mnt/%s/fashion" % mount_name))

# COMMAND ----------

df = spark.read.option("header", True).csv("/mnt/%s/fashion/extend/fashion-dataset/fashion-dataset/images.csv" % mount_name)

# COMMAND ----------

display(df)
