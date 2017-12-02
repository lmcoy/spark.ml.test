package de.lmcoy.sparkmltest

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml._
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.types._

object Main {

  def getPath(filename: String) = {
    Option(getClass.getResource("/" + filename))
      .getOrElse(throw new java.io.FileNotFoundException(filename))
      .getPath
  }

  def main(args: Array[String]): Unit = {
    // create spark session
    val sparkConf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("testML")
      .set("Spark.logging", "false")

    val sparkSession =
      SparkSession.builder().config(sparkConf).getOrCreate()

    sparkSession.sparkContext.setLogLevel("ERROR")

    // load data
    val df = sparkSession.read
      .format("csv")
      .schema(schema)
      .load(getPath("flights.csv"))

    df.show()

    // convert strings CarrierCode, OriginAirportCode, and DestinationAirportCode into IDs
    val carrierIndexer = new StringIndexer()
      .setInputCol("CarrierCode")
      .setOutputCol("CarrierID")
      .fit(df)
    val originAirportCodeIndexer = new StringIndexer()
      .setInputCol("OriginAirportCode")
      .setOutputCol("OAirportID")
      .fit(df)
    val destinationAirportCodeIndexer = new StringIndexer()
      .setInputCol("DestinationAirportCode")
      .setOutputCol("DAirportID")
      .fit(df)
    // set a flight to delayed if it is more than 40 minutes delayed
    val delayed = new Binarizer()
      .setInputCol("DepartureDelayInMinutes")
      .setOutputCol("Delayed")
      .setThreshold(40)

    // convert IDs and days to categories
    val carrierCategorizer = new OneHotEncoder()
      .setInputCol("CarrierID")
      .setOutputCol("CarrierCategory")
    val dayOfMonthCategorizer = new OneHotEncoder()
      .setInputCol("DayOfMonth")
      .setOutputCol("DayOfMonthCategory")
    val dayOfWeekCategorizer = new OneHotEncoder()
      .setInputCol("DayOfWeek")
      .setOutputCol("DayOfWeekCategory")
    val originAirportCategorizer = new OneHotEncoder()
      .setInputCol("OAirportID")
      .setOutputCol("OAirportCategory")
    val destinationAirportCategorizer = new OneHotEncoder()
      .setInputCol("DAirportID")
      .setOutputCol("DAirportCategory")

    // put all features in one column with a features vector
    val featureArray = Array(
      "DayOfMonthCategory",
      "DayOfWeekCategory",
      "ScheduledDepartureTime",
      "ScheduledArrivalTime",
      "CarrierCategory",
      "ElapsedTime",
      "OAirportCategory",
      "DAirportCategory"
    )
    val features = new VectorAssembler()
      .setInputCols(featureArray)
      .setOutputCol("Features")

    // use a decision tree to predict if a flight is delayed
    val decisionTree = new DecisionTreeClassifier()
    decisionTree.setFeaturesCol("Features")
    decisionTree.setMaxDepth(9)
    decisionTree.setImpurity("gini")
    decisionTree.setMaxBins(7000)
    decisionTree.setLabelCol("Delayed")

    // create the machine learning pipeline
    val pipeline = new Pipeline().setStages(
      Array(
        carrierIndexer,
        originAirportCodeIndexer,
        destinationAirportCodeIndexer,
        delayed,
        carrierCategorizer,
        dayOfMonthCategorizer,
        dayOfWeekCategorizer,
        originAirportCategorizer,
        destinationAirportCategorizer,
        features,
        decisionTree
      ))

    // training of the model
    val Array(trainingDF, validationDF) = df.randomSplit(Array(0.8, 0.2), 1234)
    val model = pipeline.fit(trainingDF)

    model.save("Model")

    val loadedModel = PipelineModel.load("Model")

    val vali = loadedModel.transform(validationDF).drop(featureArray: _*)

    val evaluator =
      new MulticlassClassificationEvaluator()
        .setLabelCol("Delayed")
        .setPredictionCol("prediction")

    val wrongPredictions = vali.filter(vali("prediction") =!= vali("Delayed"))

    vali.show()

    println(
      "Percentage of wrongly predicted delays: " + wrongPredictions
        .count()
        .toDouble / vali.count().toDouble * 100.0)
    println(evaluator.evaluate(vali))

  }

  // schema of CSV file
  val schema = StructType(
    Seq(
      StructField("DayOfMonth", IntegerType),
      StructField("DayOfWeek", IntegerType),
      StructField("CarrierCode", StringType),
      StructField("AirPlane", StringType),
      StructField("FlightNumber", IntegerType),
      StructField("OriginAirportID", StringType),
      StructField("OriginAirportCode", StringType),
      StructField("DestinationAirportID", StringType),
      StructField("DestinationAirportCode", StringType),
      StructField("ScheduledDepartureTime", DoubleType),
      StructField("ActualDepartureTime", DoubleType),
      StructField("DepartureDelayInMinutes", DoubleType),
      StructField("ScheduledArrivalTime", DoubleType),
      StructField("ActualArrivalTime", DoubleType),
      StructField("ArrivalDelayInMinutes", DoubleType),
      StructField("ElapsedTime", DoubleType),
      StructField("Distance", IntegerType)
    ))

}
