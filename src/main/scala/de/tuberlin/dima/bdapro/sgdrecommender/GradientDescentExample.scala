package de.tuberlin.dima.bdapro.sgdrecommender

import org.apache.flink.api.scala._
import org.apache.flink.ml.common.{LabeledVector, WeightVector}
import org.apache.flink.ml.math.{DenseVector, VectorBuilder}
import org.apache.flink.ml.optimization.{GenericLossFunction, LearningRateMethod, LinearPrediction, SquaredLoss}
/**
  * Created by rparra on 26/11/16.
  * Test Flink's SGD for a simple linear regression use case using the advertisement dataset from ISL
  * @see https://raw.githubusercontent.com/nguyen-toan/ISLR/master/dataset/Advertising.csv
  *
  * args(0) should be the local path of the dataset.
  */
object GradientDescentExample extends App{
  val env  = ExecutionEnvironment.getExecutionEnvironment

  val data = env.readCsvFile[(String, Double, Double, Double, Double)](args(0), ignoreFirstLine = true)

  val toLabeledVector = { (t: (String, Double, Double, Double, Double)) =>
    val features = t match {
      case (_, tv, radio, newspaper, _)
        => VectorBuilder.vectorBuilder.build(tv :: radio :: newspaper :: Nil )
    }
    new LabeledVector(t._5, features)
  }

  val training = data.filter(_._1.replace("\"", "").toInt <= 150).map(toLabeledVector)
  val test = data.filter(_._1.replace("\"", "").toInt > 150).map(toLabeledVector)
//  val lossFunction = GenericLossFunction(SquaredLoss, LinearPrediction)

//  //training.print()
//  val sgd = SimpleGradientDescent()
//    .setLossFunction(lossFunction)
//    //.setRegularizationConstant(0.2)
//    .setIterations(1000)
//    .setStepsize(0.0001)
//    .setConvergenceThreshold(0.001)
//    //.setLearningRateMethod(LearningRateMethod.Xu(-0.75))
//
//  val initialWeights = Some(env.fromCollection(Some(new WeightVector(DenseVector.zeros(3), 0.0))))
//  val weights = sgd.optimize(training, initialWeights)
//
//  test.cross(weights)
//    .map(x => (x._1.vector.dot(x._2.weights), x._1.label, x._2))
//    .map(x =>(((SquaredLoss.loss(x._1, x._2)) / 50), x._3))
//    .sum(0)
//    .print()
}
