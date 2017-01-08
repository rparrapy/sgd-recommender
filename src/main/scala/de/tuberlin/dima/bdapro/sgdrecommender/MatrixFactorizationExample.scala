package de.tuberlin.dima.bdapro.sgdrecommender

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.common.io.FileOutputFormat
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.VectorBuilder
import org.apache.flink.ml.optimization.SquaredLoss

/**
  * Created by duy on 12/7/16.
  */
object MatrixFactorizationExample extends App{
  val toLabeledVector = { (t: (Double, Double, Double, Double)) =>
    val features = t match {
      case (user, item, rank, _)
      => VectorBuilder.vectorBuilder.build(user :: item :: rank :: Nil )
    }
    new LabeledVector(t._3, features)
  }

  val env  = ExecutionEnvironment.getExecutionEnvironment
  val dimension = 40
  val dimensionDS = env.fromCollection[Int](List(dimension))

  val train = env.readCsvFile[(Int, Int, Double, Double)](args(0), fieldDelimiter = "\t")
  val test = env.readCsvFile[(Int, Int, Double, Double)](args(1), fieldDelimiter = "\t")

  //Count number of users and items
  val nItems = 1682
  val nUsers = 943
//  train.max(0).map(_._1).print()

//  val labeledTraining = train.map(toLabeledVector)
//  val labeledTest = test.map(toLabeledVector)
  val trainData = train.map(x => (x._1, x._2, x._3))
  val testData = test.map(x => (x._1, x._2))
  val testLabel= test.map(x => (x._1 + "-" + x._2, x._3))

  val lossFunction = GenericLossFunction(SquaredLoss, RecommenderPrediction)

//  val sgd = GradientDescentL2()
//    .setLossFunction(lossFunction)
//    .setRegularizationConstant(0.01)
//    .setIterations(200)
//    .setStepsize(0.001)
//    .setLearningRateMethod(LearningRateMethod.Constant)
//    .setConvergenceThreshold(0.0)


  for (iteration <- Array(25, 50, 75, 100, 125, 150, 175, 200)) {
    val begin = System.nanoTime()
    val sgd = SGDforMatrixFactorization()
      .setIterations(iteration)
      .setLambda(0.01)
      .setBlocks(16)
      .setNumFactors(40)
      .setLearningRate(0.001)
      .setSeed(43L)

    sgd.fit(trainData)

//    val trainPredictions = sgd.predict(trainData.map(x => (x._1, x._2)))
    val testPredictions = sgd.predict(testData)

//    trainPredictions.map(x => (x._1 + "-" + x._2, x._3))
//      .join(trainData.map(x => (x._1 + "-" + x._2, x._3)))
//      .where(0)
//      .equalTo(0)
//      .map(x => (2 * SquaredLoss.loss(x._1._2, x._2._2) / 80000, 1))
//      .sum(0)
//      .map("Training MSE:" + _._1)
//      .print()
//      .writeAsText("/home/duy/TUB/1_semester/BDAPRO/train-test-score/training-predictions" + iteration)
//    env.execute("Training")

    testPredictions.map(x => (x._1 + "-" + x._2, x._3))
      .join(testLabel)
      .where(0)
      .equalTo(0)
      .map(x => (2 * SquaredLoss.loss(x._1._2, x._2._2) / 20000, 1))
      .sum(0)
      .map("Test MSE:" + _._1)
      .print()
    val end = System.nanoTime()

    println("n epochs = " + iteration)
    println("Elapsed Time = " + (end - begin)/1000000000 + "seconds")
//      .writeAsText("/home/duy/TUB/1_semester/BDAPRO/train-test-score/test-predictions" + iteration)
//    env.execute("Testing")
  }

//  val weights = sgd.optimize(labeledTraining, Some(labeledTest), None, dimensionDS, nItems.toInt, nUsers.toInt)

//  val itemWeightDS = weights.flatMap(weight => weight.itemWeights).print()
//  val userWeightDS = weights.flatMap(weight => weight.userWeights).print()

//  labeledTest.cross(weights)
//    .map(x => Predictor.predict(x._1, x._2))
//    .map(x =>(SquaredLoss.loss(x._1, x._2) / 20000, 1))
//    .sum(0)
//    .print()
}
