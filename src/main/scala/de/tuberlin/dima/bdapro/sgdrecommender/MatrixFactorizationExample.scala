package de.tuberlin.dima.bdapro.sgdrecommender

import java.io.{BufferedWriter, File, FileWriter}

import de.tuberlin.dima.bdapro.sgdrecommender.DSGD.SGDforMatrixFactorization
import de.tuberlin.dima.bdapro.sgdrecommender.FlinkGradientDescent.{GenericLossFunction, RecommenderPrediction}
import org.apache.flink.api.java.aggregation.Aggregations
import org.apache.flink.api.java.aggregation.SumAggregationFunction.SumAggregationFunctionFactory
import org.apache.flink.api.scala._
import org.apache.flink.core.fs.FileSystem.WriteMode
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.VectorBuilder
import org.apache.flink.ml.optimization.SquaredLoss
import org.apache.flink.streaming.api.functions.aggregation.AggregationFunction.AggregationType

/**
  * Created by duy on 12/7/16.
  */
object MatrixFactorizationExample extends App{

  val HOME_PDUY_LOCAL = "/home/duy/TUB/"
  val HOME_PDUY_CLUSTER = "/home/hadoop/output/"
  val HOME_ROD = "/home/rparra/"

  val toLabeledVector = { (t: (Double, Double, Double, Double)) =>
    val features = t match {
      case (user, item, rank, _)
      => VectorBuilder.vectorBuilder.build(user :: item :: rank :: Nil )
    }
    new LabeledVector(t._3, features)
  }

  val env  = ExecutionEnvironment.getExecutionEnvironment


  /* params: <training file> <test file> <delimiter> <number of blocks> <number of iterations>
   * example: /home/pduy/yahoo-artist-train.txt /home/pduy/yahoo-artist-test.txt "\t" 200 10
   * example: /home/pduy/r1.train /home/pduy/r1.test "::" 100
   */
  val train_file = args(0)
  val test_file = args(1)
  val delimiter = args(2)
  val nBlocks = if (args.length > 3) args(3).toInt else 50
  val iterations = if (args.length > 4) args(4).toInt else 1

  //  val train = env.readCsvFile[(Int, Int, Double, Double)](HOME_PDUY_LOCAL + "1_semester/BDAPRO/ml-100k/u1.base", fieldDelimiter = "\t")
  //  val test = env.readCsvFile[(Int, Int, Double, Double)](HOME_PDUY_LOCAL + "1_semester/BDAPRO/ml-100k/u1.test", fieldDelimiter = "\t")
  //    val train = env.readCsvFile[(Int, Int, Double, Double)](HOME_PDUY_LOCAL + "1_semester/BDAPRO/ml-10m/r1.train", fieldDelimiter = "::")
  //    val test = env.readCsvFile[(Int, Int, Double, Double)](HOME_PDUY_LOCAL + "1_semester/BDAPRO/ml-10m/r1.test", fieldDelimiter = "::")
  //  val train = env.readCsvFile[(Int, Int, Double, Double)](HOME_PDUY_LOCAL + "1_semester/BDAPRO/ml-1m/ml-train.csv", fieldDelimiter = "::")
  //  val test = env.readCsvFile[(Int, Int, Double, Double)](HOME_PDUY_LOCAL + "1_semester/BDAPRO/ml-1m/ml-test.csv", fieldDelimiter = "::")
  //  val train = env.readCsvFile[(Int, Int, Double)](HOME_PDUY_LOCAL + "1_semester/BDAPRO/ml-latest/ml-train.csv", fieldDelimiter = ",")
  //  val test = env.readCsvFile[(Int, Int, Double)](HOME_PDUY_LOCAL + "1_semester/BDAPRO/ml-latest/ml-test.csv", fieldDelimiter = ",")
  //  val train = env.readCsvFile[(Int, Int, Double )](HOME_PDUY_LOCAL + "1_semester/BDAPRO/yahoo-artist-train.txt", fieldDelimiter = "\t")
  //  val test = env.readCsvFile[(Int, Int, Double )](HOME_PDUY_LOCAL + "1_semester/BDAPRO/yahoo-artist-test.txt", fieldDelimiter = "\t")

  val train = env.readCsvFile[(Int, Int, Double)](args(0), fieldDelimiter = delimiter)
  val test = env.readCsvFile[(Int, Int, Double)](args(1), fieldDelimiter = delimiter)

  val trainData = train.map(x => (x._1, x._2, x._3))
  val trainLabel = train.map(x => (x._1 + "-" + x._2, x._3))
  val testData = test.map(x => (x._1, x._2))
  val testLabel = test.map(x => (x._1 + "-" + x._2, x._3))

  val trainSize = train.map((_, 1))
    .sum(1)
    .map(_._2)
  val testSize = test.map((_, 1))
    .sum(1)
    .map(_._2)

  //  val lossFunction = GenericLossFunction(SquaredLoss, RecommenderPrediction)

  //  val begin = System.nanoTime()

  //  val resultDS = (1 to 200).map({
  //    iteration => {
  val sgd = SGDforMatrixFactorization()
    .setIterations(iterations)
    .setLambda(0.01)
    .setBlocks(nBlocks)
    .setNumFactors(40)
    .setLearningRate(0.001)
    .setSeed(43L)

  sgd.fit(trainData)
  val trainPredictions = sgd.predict(trainData.map(x => (x._1, x._2)))
  val testPredictions = sgd.predict(testData)

  val trainMSE = trainPredictions.map(x => (x._1 + "-" + x._2, x._3))
    .join(trainLabel)
    .where(0)
    .equalTo(0)
    .cross(trainSize)
    .map(x => (2 * SquaredLoss.loss(x._1._1._2, x._1._2._2) / x._2, 1))
    .sum(0)
    .map(x => ("Train MSE", x._1))
    .writeAsCsv(HOME_PDUY_CLUSTER + "pduy_output_train", "\n", "\t", WriteMode.OVERWRITE)

  val testMSE = testPredictions.map(x => (x._1 + "-" + x._2, x._3))
    .join(testLabel)
    .where(0)
    .equalTo(0)
    .cross(testSize)
    .map(x => (2 * SquaredLoss.loss(x._1._1._2, x._1._2._2) / x._2, 1))
    .sum(0)
    .map(x => ("Test MSE", x._1))
    //        .print
    .writeAsCsv(HOME_PDUY_CLUSTER + "pduy_output_test", "\n", "\t", WriteMode.OVERWRITE)

  env.execute("Writing")

  //      trainMSE.union(testMSE).print
  //    }
  //  })

  //  val finalResult = resultDS.foldLeft(List[(String, Double)]())((left, right) => left ++ right.collect())
  //
  //  val file = new File(HOME_PDUY_CLUSTER + "output.txt")
  //  val bw = new BufferedWriter(new FileWriter(file))
  //  finalResult.foreach(result => bw.write(result._1 + "\t" + result._2 + "\n"))
  //  bw.close()

  //finalResult.writeAsCsv(HOME_PDUY + "output_train.txt", "\n", "\t", WriteMode.OVERWRITE)

  //  finalResult._1.writeAsCsv(HOME_PDUY + "output_train.txt", "\n", "\t", WriteMode.OVERWRITE)
  //  finalResult._2.writeAsCsv(HOME_PDUY + "output_test.txt", "\n", "\t", WriteMode.OVERWRITE)

  //    .print()
  //  val end = System.nanoTime()
  //
  //  println("n epochs = " + iterations)
  //  println("Elapsed Time = " + (end - begin)/1000000000 + "seconds")
}
