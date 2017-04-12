import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans

// 利用 StructType 定义字段格式，与数据集中各个字段一一映射。
// StructField 中的的三个参数分别为字段名称、字段数据类型和是否不允许为空
val fieldSchema = StructType(Array(
StructField("TID", StringType, true),
StructField("Lat", DoubleType, true),
StructField("Lon", DoubleType, true),
StructField("Time", StringType, true)
))

// deprecated from spark 2.0
// val sqlContext = new SQLContext(sc) //sc: SparkContext
// val taxiDF = sqlContext.builder.read.format("com.databricks.spark.csv").option("header", "false").schema(fieldSchema).load("/home/henry/data/spark-taxi/taxi.csv") 

// new from spark 2.0
val taxiDF = sparkSession.read.schema(fieldSchema).option("header", "false").csv("/home/henry/data/spark-taxi/taxi.csv")


val columns = Array("Lat", "Lon")
// 设置参数
val va = new VectorAssembler().setInputCols(columns).setOutputCol("features")
// 将数据集按照指定的特征向量进行转化
val taxiDF2 = va.transform(taxiDF)

// 设置训练集与测试集的比例
val trainTestRatio = Array(0.7, 0.3)
// 对数据集进行随机划分，randomSplit 的第二个参数为随机数的种子
val Array(trainingData, testData) = taxiDF2.randomSplit(trainTestRatio, 2333)

// 设置模型的参数
val km = new KMeans().setK(10).setFeaturesCol("features").setPredictionCol("prediction")
// 训练 KMeans 模型，此步骤比较耗时
val kmModel = km.fit(taxiDF2)

val kmResult = kmModel.clusterCenters

// 先将结果转化为 RDD 以便于保存
val kmRDD1 = sc.parallelize(kmResult)
// 保存前将经纬度进行位置上的交换
val kmRDD2 = kmRDD1.map(x => (x(1), x(0)))
// 调用 saveAsTextFile 方法保存到文件中，
kmRDD2.saveAsTextFile("/home/henry/data/spark-taxi/kmResult")

val predictions = kmModel.transform(testData)
predictions.show()

predictions.registerTempTable("predictions")

/* 使用 select 方法选取字段，
* substring 用于提取时间的前 2 位作为小时，
* alias 方法是为选取的字段命名一个别名，
* 选择字段时用符号 $ ，
* groupBy 方法对结果进行分组。
*/
val tmpQuery = predictions.select(substring($"Time",0,2).alias("hour"), $"prediction").groupBy("hour", "prediction")

/* agg 是聚集函数，count 为其中的一种实现，
* 用于统计某个字段的数量。
* 最后的结果按照预测命中数来降序排列（Desc）。
*/
val predictCount = tmpQuery.agg(count("prediction").alias("count")).orderBy(desc("count"))
predictCount.show()
//save as cvs
predictCount.write.format("com.databricks.spark.csv").save("/home/henry/data/spark-taxi/predictCount") //path check

val busyZones = predictions.groupBy("prediction").count()
busyZones.show()
//save as cvs
busyZones.write.format("com.databricks.spark.csv").save("/home/henry/data/spark-taxi/busyZones") //path check

// allPredictions = sqlContext.sql("SELECT * FROM predictions")


