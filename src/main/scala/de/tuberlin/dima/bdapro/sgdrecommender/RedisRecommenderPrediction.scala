package de.tuberlin.dima.bdapro.sgdrecommender

import org.apache.flink.ml.common.WeightVector
import org.apache.flink.ml.math.{BLAS, DenseVector, Vector}
import org.apache.flink.ml.optimization.PredictionFunction

/**
  * Created by rparra on 5/1/17.
  *
  * We stick to the PredictionFunction interface for backwards compatibility reasons.
  * The weights we receive are a concatenation of user weights and bias and item weights and bias
  *
  */
class RedisRecommenderPrediction (numberOfFactors: Int, globalBias: Double) extends PredictionFunction{
  /**
    * Predict a user/item rating
    *
    * @param features vector [userid, itemid,...]
    * @param weights (user weights, user bias, item weights, item bias)
    * @return predicted rating
    */
  override def predict(features: Vector, weights: WeightVector): Double = {
    val (userWeights, itemWeights) = getWeightVectors(weights)
    val (userBias, itemBias) = getBiases(weights)
    BLAS.dot(userWeights, itemWeights) + userBias + itemBias + globalBias

  }

  /**
    * ATTENTION!!!!
    *
    * This method does not return ONLY the gradient.
    *
    * Sadly the interface enforces a single WeightVector as the return value, because we need to return two things:
    * - user gradient
    * - item gradient
    *
    *
    * Luckily, both elements are Vectors of equal size (#factors + 1 elements), hence we concatenate them and return
    * that as a result.
    *
    *
    * @param features vector [userid, itemid,...]
    * @param weights (user weights, user bias, item weights, item bias)
    * @return
    */
  override def gradient(features: Vector, weights: WeightVector): WeightVector = {
    val (userWeights, itemWeights) = getWeightVectors(weights)
    val gradient = DenseVector.zeros((numberOfFactors + 1) * 2)
    for (i <- 0 to numberOfFactors - 1) {
      gradient(i) = itemWeights(i)
      gradient(numberOfFactors + 1 + i) = userWeights(i)
    }

    gradient(numberOfFactors) = 1.0
    gradient(2 * numberOfFactors + 1) = 1.0

    WeightVector(gradient, 0)

  }

  private def getWeightVectors(weights: WeightVector) = {
    val userWeights = weights.weights.slice(0, numberOfFactors).map(_._2).toArray
    val itemWeights = weights.weights.slice(numberOfFactors + 1, 2 * numberOfFactors + 1)
      .map(_._2).toArray
    (DenseVector(userWeights), DenseVector(itemWeights))
  }

  private def getBiases(weights: WeightVector) = {
    val userBias = weights.weights(numberOfFactors)
    val itemBias =  weights.weights(2 * numberOfFactors + 1)
    (userBias, itemBias)
  }


}
