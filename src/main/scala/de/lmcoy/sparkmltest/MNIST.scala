package de.lmcoy.sparkmltest

import java.io.{BufferedInputStream, DataInputStream, File, FileInputStream}

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vectors, Vector => VectorML}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object MNIST {
  case class Vec(ID: Int, rows: Int, cols: Int, data: VectorML)
  case class Label(ID: Int, label: Int)

  def withFileInput[T](filename: String)(f: (DataInputStream) => T): T = {
    val fileInputStream: FileInputStream = new FileInputStream(
      new File(filename))
    val bufferedStream = new BufferedInputStream(fileInputStream)
    val dataInputStream = new DataInputStream(bufferedStream)
    val result = f(dataInputStream)

    dataInputStream.close()
    bufferedStream.close()
    fileInputStream.close()

    result
  }

  def getPath(filename: String) = {
    Option(getClass.getResource("/" + filename))
      .getOrElse(throw new java.io.FileNotFoundException(filename))
      .getPath
  }

  def loadImageData(filename: String) = withFileInput(filename) {
    (in: DataInputStream) =>
      {
        val magicNumber = in.readInt()
        if (magicNumber != 0x00000803)
          throw new Exception("wrong file format " + magicNumber)
        val nb = in.readInt()

        val rows = in.readInt()
        val cols = in.readInt()

        for (i <- 0 until nb) yield {
          val n = rows * cols

          val data = for (_ <- 0 until n)
            yield in.readUnsignedByte().toDouble / 256.0
          Vec(i, rows, cols, Vectors.dense(data.toArray))
        }
      }
  }

  def loadLabelData(filename: String) = withFileInput(filename) {
    (in: DataInputStream) =>
      {
        val magicNumber = in.readInt()
        if (magicNumber != 0x00000801)
          throw new Exception("wrong file format " + magicNumber)
        val nb = in.readInt()

        val data = Array.fill[Byte](nb)(0)
        in.read(data)

        data.zipWithIndex.map(b => Label(b._2, b._1.toInt))
      }
  }

  def main(args: Array[String]): Unit = {
    // load data: http://yann.lecun.com/exdb/mnist/
    val filenameImageData = getPath("train-images-idx3-ubyte")
    val imageData = loadImageData(filenameImageData)

    val filenameLabelData = getPath("train-labels-idx1-ubyte")
    val labelData = loadLabelData(filenameLabelData)

    // create spark session
    val sparkConf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("testML")

    val sparkSession =
      SparkSession.builder().config(sparkConf).getOrCreate()

    sparkSession.sparkContext.setLogLevel("ERROR")

    val labels = sparkSession.createDataFrame(labelData.take(10000))
    val images = sparkSession.createDataFrame(imageData.take(10000))

    val Array(data, vali) = images
      .join(labels, Seq("ID"), "left")
      .randomSplit(Array(0.8, 0.2), 12345L)
    data.cache()
    vali.cache()
    data.show()
    data.printSchema()

    // transformers
    val labelCategory =
      new OneHotEncoder().setInputCol("label").setOutputCol("labelCategory")
    val classifier = new MultilayerPerceptronClassifier()
      .setLabelCol("label")
      .setFeaturesCol("data")
      .setMaxIter(100)
      .setTol(1E-4)
      .setLayers(Array(784, 80, 10))

    // define the pipeline
    val pipeline = new Pipeline()
    pipeline.setStages(Array(labelCategory, classifier))

    // train the model
    val model = pipeline.fit(data)

    val df = model.transform(vali)

    val evaluator1 = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setLabelCol("label")
    val evaluator2 = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")
      .setLabelCol("label")
    val evaluator3 = new MulticlassClassificationEvaluator()
      .setMetricName("weightedRecall")
      .setLabelCol("label")
    val evaluator4 = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("label")

    println("accuracy: " + evaluator1.evaluate(df))
    println("weightedPrecision: " + evaluator2.evaluate(df))
    println("weightedRecall: " + evaluator3.evaluate(df))
    println("f1: " + evaluator4.evaluate(df))

  }

}
