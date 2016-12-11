package de.tuberlin.dima.bdapro.sgdrecommender

import org.apache.flink.ml.common.WeightVector
import org.apache.flink.ml.math.{DenseVector, Vector}
import org.apache.flink.ml.optimization.PredictionFunction
import org.apache.flink.ml.math.BLAS


/**
  * Created by rparra on 10/12/16.
  */
case class RecommenderPrediction(numberOfUsers: Int, numberOfItems: Int, numberOfFactors: Int) extends PredictionFunction{
  val itemsMatrixOffset = numberOfUsers * numberOfFactors

  override def predict(features: Vector, weights: WeightVector): Double = {
    val (userWeights, itemWeights) = getWeightVectors(features, weights)
    BLAS.dot(userWeights, itemWeights)
  }

  /**
    * Return the gradients for a particular user-item pair.
    * First {numberOfFactor} elements correspond to the gradient to update the user weights.
    * Posterior {numberOfFactor} elements correspond to the gradient to update the user weights.
    *
    * @param features
    * @param weights
    * @return a vector of weights of size 2 * numberOfFactors
    */
  override def gradient(features: Vector, weights: WeightVector): WeightVector = {
    val (userWeights, itemWeights) = getWeightVectors(features, weights)
    val (userOffset, itemOffset) = getOffsets(features, weights)
    val gradient = DenseVector.zeros(weights.weights.size)
    for (i <- 0 to numberOfFactors - 1) {
      gradient(userOffset + i) = itemWeights(i)
      gradient(itemsMatrixOffset + itemOffset + i) = userWeights(i)
    }

    WeightVector(gradient, 0)
  }

  private def getWeightVectors(features: Vector, weights: WeightVector) = {
    val (userOffset, itemOffset) = getOffsets(features, weights)
    val userWeights = weights.weights.slice(userOffset, userOffset + numberOfFactors).map(_._2).toArray
    val itemWeights = weights.weights.slice(itemsMatrixOffset + itemOffset, itemsMatrixOffset + itemOffset + numberOfFactors)
      .map(_._2).toArray
    (DenseVector(userWeights), DenseVector(itemWeights))
  }

  private def getOffsets(features: Vector, weights: WeightVector) = {
    val user = features(0).toInt
    val item = features(1).toInt
    val userOffset = (user - 1) * numberOfFactors
    val itemOffset = (item - 1) * numberOfFactors
    (userOffset, itemOffset)
  }
}
