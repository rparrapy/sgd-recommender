package de.tuberlin.dima.bdapro.sgdrecommender

import de.tuberlin.dima.bdapro.sgdrecommender.DSGD.SGDforMatrixFactorization
import de.tuberlin.dima.bdapro.sgdrecommender.FlinkGradientDescent.{GenericLossFunction, RecommenderPrediction}
import org.apache.flink.api.scala._
import org.apache.flink.core.fs.FileSystem.WriteMode
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.VectorBuilder
import org.apache.flink.ml.optimization.SquaredLoss

/**
  * Created by duy on 12/7/16.
  */
object MatrixFactorizationExample extends App{

  val HOME_PDUY = "/home/duy/TUB/"
  val HOME_ROD = "/home/rparra/"
  val YAHOO_ARTIST_TRAIN= "yahoo-artist-train.txt"
  val YAHOO_ARTIST_TEST = "yahoo-artist-test.txt"
  val YAHOO_SONG_TRAIN = "yahoo-song-train.txt"
  val YAHOO_SONG_TEST = "yahoo-song-test.txt"

  val toLabeledVector = { (t: (Double, Double, Double, Double)) =>
    val features = t match {
      case (user, item, rank, _)
      => VectorBuilder.vectorBuilder.build(user :: item :: rank :: Nil )
    }
    new LabeledVector(t._3, features)
  }

  val env  = ExecutionEnvironment.getExecutionEnvironment

//  val train = env.readCsvFile[(Int, Int, Double, Double)](HOME_PDUY + YAHOO_ARTIST_NAME, fieldDelimiter = "\t")
  val train = env.readCsvFile[(Int, Int, Double, Double)]("/Users/rparra/Workspace/tub/bdapro/ml-100k/u1.base", fieldDelimiter = "\t")
  val test = env.readCsvFile[(Int, Int, Double, Double)]("/Users/rparra/Workspace/tub/bdapro/ml-100k/u1.test", fieldDelimiter = "\t")


  val trainData = train.map(x => (x._1, x._2, x._3))
  val trainLabel = train.map(x => (x._1 + "-" + x._2, x._3))
  val testData = test.map(x => (x._1, x._2))
  val testLabel = test.map(x => (x._1 + "-" + x._2, x._3))

  val trainSize = trainData.collect.size
  val testSize = testData.collect.size

//  val lossFunction = GenericLossFunction(SquaredLoss, RecommenderPrediction)

//  val begin = System.nanoTime()

  val resultDS = (1 to 5).map({
    iteration => {
      val sgd = SGDforMatrixFactorization()
        .setIterations(iteration)
        .setLambda(0.01)
        .setBlocks(16)
        .setNumFactors(40)
        .setLearningRate(0.001).setSeed(43L)

      sgd.fit(trainData)
      val trainPredictions = sgd.predict(trainData.map(x => (x._1, x._2)))
      val testPredictions = sgd.predict(testData)

      val trainMSE = trainPredictions.map(x => (x._1 + "-" + x._2, x._3))
        .join(trainLabel)
        .where(0)
        .equalTo(0)
        .map(x => (2 * SquaredLoss.loss(x._1._2, x._2._2) / trainSize, 1))
        .sum(0)
        .map(x => ("Train MSE", x._1))

      val testMSE = testPredictions.map(x => (x._1 + "-" + x._2, x._3))
        .join(testLabel)
        .where(0)
        .equalTo(0)
        .map(x => (2 * SquaredLoss.loss(x._1._2, x._2._2) / testSize, 1))
        .sum(0)
        .map(x => ("Test MSE", x._1))

      trainMSE.union(testMSE)
    }
  })

  val finalResult = resultDS.foldLeft(List[(String, Double)]())((left, right) => left ++ right.collect())
  println(finalResult)
  //finalResult.writeAsCsv(HOME_PDUY + "output_train.txt", "\n", "\t", WriteMode.OVERWRITE)

  //  finalResult._1.writeAsCsv(HOME_PDUY + "output_train.txt", "\n", "\t", WriteMode.OVERWRITE)
//  finalResult._2.writeAsCsv(HOME_PDUY + "output_test.txt", "\n", "\t", WriteMode.OVERWRITE)

//    .print()
//  val end = System.nanoTime()
//
//  println("n epochs = " + iteration)
//  println("Elapsed Time = " + (end - begin)/1000000000 + "seconds")
}
