package de.tuberlin.dima.bdapro.sgdrecommender

import org.apache.flink.ml.common.WeightVector
import org.apache.flink.ml.math.{BLAS, DenseVector, Vector}
import org.apache.flink.ml.optimization.PredictionFunction

/**
  * Created by rparra on 11/12/16.
  */
case class RecommenderPredictionWithBias(numberOfUsers: Int, numberOfItems: Int, numberOfFactors: Int) extends PredictionFunction{
  val itemsMatrixOffset = numberOfUsers * numberOfFactors

  override def predict(features: Vector, weights: WeightVector): Double = {
    val (userWeights, itemWeights) = getWeightVectors(features, weights)
    val (userBias, itemBias) = getBiases(features, weights)
    BLAS.dot(userWeights, itemWeights) + userBias + itemBias + weights.intercept
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
    val (userOffset, itemOffset) = getOffsets(features)
    val gradient = DenseVector.zeros(weights.weights.size)
    for (i <- 0 to numberOfFactors - 1) {
      gradient(userOffset + i) = itemWeights(i)
      gradient(itemsMatrixOffset + itemOffset + i) = userWeights(i)
    }
    gradient(userOffset + numberOfFactors) = 1.0
    gradient(itemsMatrixOffset + itemOffset + numberOfFactors) = 1.0

    WeightVector(gradient, 0)
  }

  private def getWeightVectors(features: Vector, weights: WeightVector) = {
    val (userOffset, itemOffset) = getOffsets(features)
    val userWeights = weights.weights.slice(userOffset, userOffset + numberOfFactors).map(_._2).toArray
    val itemWeights = weights.weights.slice(itemsMatrixOffset + itemOffset, itemsMatrixOffset + itemOffset + numberOfFactors)
      .map(_._2).toArray
    (DenseVector(userWeights), DenseVector(itemWeights))
  }

  private def getBiases(features: Vector, weights: WeightVector) = {
    val (userOffset, itemOffset) = getOffsets(features)
    val userBias = weights.weights(userOffset + numberOfFactors)
    val itemBias =  weights.weights(itemsMatrixOffset + itemOffset + numberOfFactors)
    (userBias, itemBias)
  }

  private def getOffsets(features: Vector) = {
    val user = features(0).toInt
    val item = features(1).toInt
    val userOffset = (user - 1) * (numberOfFactors + 1)
    val itemOffset = (item - 1) * (numberOfFactors + 1)
    (userOffset, itemOffset)
  }
}

