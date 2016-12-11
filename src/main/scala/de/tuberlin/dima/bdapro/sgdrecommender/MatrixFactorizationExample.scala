package de.tuberlin.dima.bdapro.sgdrecommender

import org.apache.flink.api.common.functions.RichMapFunction
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

  val train = env.readCsvFile[(Double, Double, Double, Double)](args(0), fieldDelimiter = "\t")
  val test = env.readCsvFile[(Double, Double, Double, Double)](args(1), fieldDelimiter = "\t")

  //Count number of users and items
  val nItems = 1682
  val nUsers = 943
  train.max(0).map(_._1).print()

  val labeledTraining = train.map(toLabeledVector)
  val labeledTest = test.map(toLabeledVector)

  val lossFunction = GenericLossFunction(SquaredLoss, RecommenderPrediction)

  val sgd = GradientDescentL2()
    .setLossFunction(lossFunction)
    .setRegularizationConstant(0.01)
    .setIterations(50)
    .setStepsize(0.001)
    .setLearningRateMethod(LearningRateMethod.Constant)
    .setConvergenceThreshold(0.0)

  val weights = sgd.optimize(labeledTraining, None, dimensionDS, nItems.toInt, nUsers.toInt)

//  val itemWeightDS = weights.flatMap(weight => weight.itemWeights).print()
//  val userWeightDS = weights.flatMap(weight => weight.userWeights).print()

  labeledTest.cross(weights)
    .map(x => Predictor.predict(x._1, x._2))
    .map(x =>(SquaredLoss.loss(x._1, x._2) / 20000, 1))
    .sum(0)
    .print()
}
