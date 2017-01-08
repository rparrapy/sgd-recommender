package de.tuberlin.dima.bdapro.sgdrecommender

import org.apache.flink.api.scala._
import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml.common.{LabeledVector, WeightVector}
import org.apache.flink.ml.math.{DenseVector, VectorBuilder}
import org.apache.flink.ml.optimization.{GenericLossFunction, LearningRateMethod, SquaredLoss}

import scala.util.Random
/**
  * Created by rparra on 10/12/16.
  */
object SGDRecommenderExample extends App{

//  val customConfiguration = new Configuration()
//  customConfiguration.setInteger("parallelism", 1)
//  customConfiguration.setInteger("jobmanager.heap.mb",2560)
//  customConfiguration.setInteger("taskmanager.heap.mb",2560)
//  customConfiguration.setString("akka.ask.timeout", "10000 s")
  // val env = ExecutionEnvironment.createLocalEnvironment(customConfiguration)
  val t0 = System.nanoTime()
  val env  = ExecutionEnvironment.getExecutionEnvironment


  val toLabeledVector = { (t: (Double, Double, Double, Double)) =>
    val features = t match {
      case (user, item, rating, timestamp)
      => VectorBuilder.vectorBuilder.build( user :: item :: timestamp :: Nil )
    }
    new LabeledVector(t._3, features)
  }

  val training = env.readCsvFile[(Double, Double, Double, Double)](args(0), fieldDelimiter = "\t").map(toLabeledVector)
  val test = env.readCsvFile[(Double, Double, Double, Double)](args(1), fieldDelimiter = "\t").map(toLabeledVector)

  val numberOfUsers = 943
  val numberOfItems = 1682
  val numberOfFactors = 40
  val meanRating = 3.526463

  val predictor = new RedisRecommenderPrediction(numberOfFactors, meanRating)
  val lossFunction = GenericLossFunction(SquaredLoss, predictor)

  val sgd = RedisGradientDescentL2()
    .setLossFunction(lossFunction)
    .setRegularizationConstant(0.001)
    .setIterations(125)
    .setStepsize(0.001)
    .setLearningRateMethod(LearningRateMethod.Constant)
  //.setConvergenceThreshold(0.0)
  //.setLearningRateMethod(LearningRateMethod.Xu(-0.75))

//  val r = new Random(1000L)
//  //val w0 = (0 to ((numberOfUsers + numberOfItems) * numberOfFactors)).map(_ => r.nextGaussian() * 1 / numberOfFactors).toArray
//  val w0 = (0 to ((numberOfUsers + numberOfItems) * (numberOfFactors + 1))).map(x => {
//    if (x % (numberOfFactors + 1) == 0) 0.0
//    else r.nextGaussian() * 1 / numberOfFactors
//  }).toArray
//  val initialWeights = Some(env.fromCollection(Some(new WeightVector(DenseVector(w0), meanRating))))


  RedisModelManager.setup(numberOfUsers, numberOfItems, numberOfFactors)
  val weights = sgd.optimize(training, None, numberOfFactors)
  weights.collect()


  //  test.cross(weights)
//    .map(x => (predictor.predict(x._1.vector, x._2), x._1.label, x._2))
//    .map(x =>(((SquaredLoss.loss(x._1, x._2)) / 20000), 1))
//    .sum(0)
//    .print()

    test
      .map(x => {
        val weights = RedisModelManager.getWeights(x.vector, numberOfFactors)
        (x, weights)
      })
      .map(x => (predictor.predict(x._1.vector, x._2), x._1.label, x._2))
      .map(x =>(((SquaredLoss.loss(x._1, x._2)) / 20000), 1))
      .sum(0)
      .print()

  RedisModelManager.tearDown
  val t1 = System.nanoTime()
  println("Elapsed time: " + (t1 - t0) + "ns")


}
