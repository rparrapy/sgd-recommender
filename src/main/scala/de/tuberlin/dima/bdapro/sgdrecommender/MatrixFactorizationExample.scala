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

  val HOME_PDUY = "/home/duy/"
  val HOME_ROD = "/home/rparra/"

  val toLabeledVector = { (t: (Double, Double, Double, Double)) =>
    val features = t match {
      case (user, item, rank, _)
      => VectorBuilder.vectorBuilder.build(user :: item :: rank :: Nil )
    }
    new LabeledVector(t._3, features)
  }

  val env  = ExecutionEnvironment.getExecutionEnvironment

  val train = env.readCsvFile[(Int, Int, Double, Double)]("/home/duy/TUB/1_semester/BDAPRO/ml-100k/u1.base", fieldDelimiter = "\t")
  val test = env.readCsvFile[(Int, Int, Double, Double)]("/home/duy/TUB/1_semester/BDAPRO/ml-100k/u1.test", fieldDelimiter = "\t")


  val trainData = train.map(x => (x._1, x._2, x._3))
  val testData = test.map(x => (x._1, x._2))
  val testLabel= test.map(x => (x._1 + "-" + x._2, x._3))

  val lossFunction = GenericLossFunction(SquaredLoss, RecommenderPrediction)

  val iteration = 200
  val begin = System.nanoTime()
  val sgd = SGDforMatrixFactorization()
    .setIterations(iteration)
    .setLambda(0.01)
    .setBlocks(16)
    .setNumFactors(40)
    .setLearningRate(0.001) .setSeed(43L)

  sgd.fit(trainData)

  val testPredictions = sgd.predict(testData)

  testPredictions.map(x => (x._1 + "-" + x._2, x._3))
    .join(testLabel)
    .where(0)
    .equalTo(0)
    .map(x => (2 * SquaredLoss.loss(x._1._2, x._2._2) / 20000, 1))
    .sum(0)
    .map(x => ("Test MSE", x._1))
    .writeAsCsv(HOME_PDUY + "output.txt", "\n", "\t", WriteMode.OVERWRITE)
//    .print()

  val end = System.nanoTime()

  println("n epochs = " + iteration)
  println("Elapsed Time = " + (end - begin)/1000000000 + "seconds")
}
