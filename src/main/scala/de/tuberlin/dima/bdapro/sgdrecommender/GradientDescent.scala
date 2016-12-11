/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package de.tuberlin.dima.bdapro.sgdrecommender


import de.tuberlin.dima.bdapro.sgdrecommender.IterativeSolver._
import de.tuberlin.dima.bdapro.sgdrecommender.LearningRateMethod.LearningRateMethodTrait
import de.tuberlin.dima.bdapro.sgdrecommender.Solver.{LossFunction, RegularizationConstant}
import org.apache.flink.api.scala._
import org.apache.flink.ml._
import org.apache.flink.ml.common._
import org.apache.flink.ml.math._

/** Base class which performs Stochastic Gradient Descent optimization using mini batches.
  *
  * For each labeled vector in a mini batch the gradient is computed and added to a partial
  * gradient. The partial gradients are then summed and divided by the size of the batches. The
  * average gradient is then used to updated the weight values, including regularization.
  *
  * At the moment, the whole partition is used for SGD, making it effectively a batch gradient
  * descent. Once a sampling operator has been introduced, the algorithm can be optimized
  *
  * The parameters to tune the algorithm are:
  * [[Solver.LossFunction]] for the loss function to be used,
  * [[Solver.RegularizationConstant]] for the regularization parameter,
  * [[IterativeSolver.Iterations]] for the maximum number of iteration,
  * [[IterativeSolver.LearningRate]] for the learning rate used.
  * [[IterativeSolver.ConvergenceThreshold]] when provided the algorithm will
  * stop the iterations if the relative change in the value of the objective
  * function between successive iterations is is smaller than this value.
  * [[IterativeSolver.LearningRateMethodValue]] determines functional form of
  * effective learning rate.
  */
abstract class GradientDescent extends IterativeSolver {

  /** Provides a solution for the given optimization problem
    *
    * @param data           A Dataset of LabeledVector (label, features) pairs
    * @param initialWeights The initial weights that will be optimized
    * @return The weights, optimized for the provided data.
    */

  override def optimize(
                         data: DataSet[LabeledVector],
                         initialWeights: Option[DataSet[RecommenderWeights]],
                         f: DataSet[Int],
                         nItems: Int,
                         nUsers: Int): DataSet[RecommenderWeights] = {

    val numberOfIterations: Int = parameters(Iterations)
    val convergenceThresholdOption: Option[Double] = parameters.get(ConvergenceThreshold)
    val lossFunction = parameters(LossFunction)
    val learningRate = parameters(LearningRate)
    val regularizationConstant = parameters(RegularizationConstant)
    val learningRateMethod = parameters(LearningRateMethodValue)
    // Initialize weights
    val initialWeightsDS: DataSet[RecommenderWeights] = createInitialWeightsDS(initialWeights, f, nItems, nUsers)

    //     Perform the iterations
    convergenceThresholdOption match {
      //     No convergence criterion
      case None =>
        optimizeWithoutConvergenceCriterion(
          data,
          initialWeightsDS,
          numberOfIterations,
          regularizationConstant,
          learningRate,
          lossFunction,
          learningRateMethod)
      case Some(convergence) =>
        optimizeWithConvergenceCriterion(
          data,
          initialWeightsDS,
          numberOfIterations,
          regularizationConstant,
          learningRate,
          convergence,
          lossFunction,
          learningRateMethod)
    }
  }

  def optimizeWithConvergenceCriterion(
                                        dataPoints: DataSet[LabeledVector],
                                        initialWeightsDS: DataSet[RecommenderWeights],
                                        numberOfIterations: Int,
                                        regularizationConstant: Double,
                                        learningRate: Double,
                                        convergenceThreshold: Double,
                                        lossFunction: LossFunction,
                                        learningRateMethod: LearningRateMethodTrait)
  : DataSet[RecommenderWeights] = {
    // We have to calculate for each weight vector the sum of squared residuals,
    // and then sum them and apply regularization
    val initialLossSumDS = calculateLoss(dataPoints, initialWeightsDS, lossFunction)

    // Combine weight vector with the current loss
    val initialWeightsWithLossSum = initialWeightsDS.mapWithBcVariable(initialLossSumDS) {
      (weights, loss) => (weights, loss)
    }

    val resultWithLoss = initialWeightsWithLossSum.iterateWithTermination(numberOfIterations) {
      weightsWithPreviousLossSum =>

        // Extract weight vector and loss
        val previousWeightsDS = weightsWithPreviousLossSum.map {
          _._1
        }
        val previousLossSumDS = weightsWithPreviousLossSum.map {
          _._2
        }

        val currentWeightsDS = SGDStep(
          dataPoints,
          previousWeightsDS,
          lossFunction,
          regularizationConstant,
          learningRate,
          learningRateMethod)

        val currentLossSumDS = calculateLoss(dataPoints, currentWeightsDS, lossFunction)

        // Check if the relative change in the loss is smaller than the
        // convergence threshold. If yes, then terminate i.e. return empty termination data set
        val termination = previousLossSumDS.filterWithBcVariable(currentLossSumDS) {
          (previousLoss, currentLoss) => {
            if (previousLoss <= 0) {
              false
            } else {
              scala.math.abs((previousLoss - currentLoss) / previousLoss) >= convergenceThreshold
            }
          }
        }

        // Result for new iteration
        (currentWeightsDS.mapWithBcVariable(currentLossSumDS)((w, l) => (w, l)), termination)
    }
    // Return just the weights
    resultWithLoss.map {
      _._1
    }
  }

  def optimizeWithoutConvergenceCriterion(
                                           data: DataSet[LabeledVector],
                                           initialWeightsDS: DataSet[RecommenderWeights],
                                           numberOfIterations: Int,
                                           regularizationConstant: Double,
                                           learningRate: Double,
                                           lossFunction: LossFunction,
                                           optimizationMethod: LearningRateMethodTrait)
  : DataSet[RecommenderWeights] = {
    initialWeightsDS.iterate(numberOfIterations) {
      weightVectorDS => {
        SGDStep(data,
          weightVectorDS,
          lossFunction,
          regularizationConstant,
          learningRate,
          optimizationMethod)
      }
    }
  }

  /** Performs one iteration of Stochastic Gradient Descent using mini batches
    *
    * @param data           A Dataset of LabeledVector (label, features) pairs
    * @param currentWeights A Dataset with the current weights to be optimized as its only element
    * @return A Dataset containing the weights after one stochastic gradient descent step
    */
  def SGDStep(
               data: DataSet[(LabeledVector)],
               currentWeights: DataSet[RecommenderWeights],
               lossFunction: LossFunction,
               regularizationConstant: Double,
               learningRate: Double,
               learningRateMethod: LearningRateMethodTrait)
  : DataSet[RecommenderWeights] = {

    data.mapWithBcVariable(currentWeights) {
      (data, weightVector) => {
        val itemIndex = data.vector.apply(1).toInt
        val userIndex = data.vector.apply(0).toInt
        val gradientVectors = lossFunction.gradient(data, weightVector)
        (itemIndex, userIndex, weightVector, gradientVectors, false)
      }
    }.reduce {
      (left, right) =>
        val (leftItemIndex, leftUserIndex, leftWeightVector, leftGradientVectors, leftProcessed) = left
        val (rightItemIndex, rightUserIndex, _ , rightGradientVectors, _) = right

        if (!leftProcessed) {
          val currentItemWeight = leftWeightVector.itemWeights.apply(leftItemIndex - 1)
          val newItemWeightVector = takeStep(currentItemWeight.weights,
            leftGradientVectors._1.weights,
            regularizationConstant,
            learningRate)

          val currentUserWeight = leftWeightVector.userWeights.apply(leftUserIndex - 1)
          val newUserWeightVector = takeStep(currentUserWeight.weights,
            leftGradientVectors._2.weights,
            regularizationConstant,
            learningRate)

          leftWeightVector.itemWeights.update(leftItemIndex - 1,
            WeightVector(newItemWeightVector,
              takeStepIntercept(currentItemWeight.intercept, leftGradientVectors._1.intercept, regularizationConstant, learningRate)))
          leftWeightVector.userWeights.update(leftUserIndex - 1,
            WeightVector(newUserWeightVector,
              takeStepIntercept(currentUserWeight.intercept, leftGradientVectors._2.intercept, regularizationConstant, learningRate)))
        }

        val currentItemWeight = leftWeightVector.itemWeights.apply(rightItemIndex - 1)
        val newItemWeightVector = takeStep(currentItemWeight.weights,
          rightGradientVectors._1.weights,
          regularizationConstant,
          learningRate)

        val currentUserWeight = leftWeightVector.userWeights.apply(rightUserIndex - 1)
        val newUserWeightVector = takeStep(currentUserWeight.weights,
          rightGradientVectors._2.weights,
          regularizationConstant,
          learningRate)

        leftWeightVector.itemWeights.update(rightItemIndex - 1,
          WeightVector(newItemWeightVector,
            takeStepIntercept(currentItemWeight.intercept, rightGradientVectors._1.intercept, regularizationConstant, learningRate)))
        leftWeightVector.userWeights.update(rightUserIndex - 1,
          WeightVector(newUserWeightVector,
            takeStepIntercept(currentUserWeight.intercept, rightGradientVectors._2.intercept, regularizationConstant, learningRate)))

        (rightItemIndex, rightUserIndex, leftWeightVector, rightGradientVectors, true)
    }.map(item => item._3)
  }

  /** Calculates the new weights based on the gradient
    *
    * @param weightVector
    * @param gradient
    * @param regularizationConstant
    * @param learningRate
    * @return
    */
  def takeStep(weightVector: Vector,
               gradient: Vector,
               regularizationConstant: Double,
               learningRate: Double): Vector

  def takeStepIntercept(intercept: Double,
                                 gradient: Double,
                                 regularizationConstant: Double,
                                 learningRate: Double)
  : Double = {
    val newGradient = gradient + regularizationConstant*intercept
    intercept - learningRate*newGradient
  }

  /** Calculates the regularized loss, from the data and given weights.
    *
    * @param data
    * @param weightDS
    * @param lossFunction
    * @return
    */
  private def calculateLoss(
                             data: DataSet[LabeledVector],
                             weightDS: DataSet[RecommenderWeights],
                             lossFunction: LossFunction)
  : DataSet[Double] = {
    data.mapWithBcVariable(weightDS) {
      (data, weightVector) => (lossFunction.loss(data, weightVector), 1)
    }.reduce {
      (left, right) => (left._1 + right._1, left._2 + right._2)
    }.map {
      lossCount => {
        val loss = lossCount._1 / lossCount._2
        println("TRAINING LOSS + " + loss)
        loss
      }
    }
  }
}

/** Implementation of a SGD solver with L2 regularization.
  *
  * The regularization function is `1/2 ||w||_2^2` with `w` being the weight vector.
  */
class GradientDescentL2 extends GradientDescent {

  /** Calculates the new weights based on the gradient
    *
    * @param weightVector
    * @param gradient
    * @param regularizationConstant
    * @param learningRate
    * @return
    */
  override def takeStep(
                         weightVector: Vector,
                         gradient: Vector,
                         regularizationConstant: Double,
                         learningRate: Double)
  : Vector = {
    // add the gradient of the L2 regularization
    BLAS.axpy(regularizationConstant, weightVector, gradient)

    // update the weights according to the learning rate
    BLAS.axpy(-learningRate, gradient, weightVector)

    weightVector
  }

}

object GradientDescentL2 {
  def apply() = new GradientDescentL2
}

/** Implementation of a SGD solver with L1 regularization.
  *
  * The regularization function is `||w||_1` with `w` being the weight vector.
  */
class GradientDescentL1 extends GradientDescent {

  /** Calculates the new weights based on the gradient.
    *
    * @param weightVector
    * @param gradient
    * @param regularizationConstant
    * @param learningRate
    * @return
    */
  override def takeStep(
                         weightVector: Vector,
                         gradient: Vector,
                         regularizationConstant: Double,
                         learningRate: Double)
  : Vector = {
    // Update weight vector with gradient. L1 regularization has no gradient, the proximal operator
    // does the job.
    BLAS.axpy(-learningRate, gradient, weightVector)

    // Apply proximal operator (soft thresholding)
    val shrinkageVal = regularizationConstant * learningRate
    var i = 0
    while (i < weightVector.size) {
      val wi = weightVector(i)
      weightVector(i) = scala.math.signum(wi) *
        scala.math.max(0.0, scala.math.abs(wi) - shrinkageVal)
      i += 1
    }

    weightVector
  }
}

object GradientDescentL1 {
  def apply() = new GradientDescentL1
}

/** Implementation of a SGD solver without regularization.
  *
  * No regularization is applied.
  */
class SimpleGradientDescent extends GradientDescent {

  /** Calculates the new weights based on the gradient.
    *
    * @param weightVector
    * @param gradient
    * @param regularizationConstant
    * @param learningRate
    * @return
    */
  override def takeStep(
                         weightVector: Vector,
                         gradient: Vector,
                         regularizationConstant: Double,
                         learningRate: Double)
  : Vector = {
    // Update the weight vector
    BLAS.axpy(-learningRate, gradient, weightVector)
    weightVector
  }
}

object SimpleGradientDescent {
  def apply() = new SimpleGradientDescent
}

object Predictor {
  def predict(x: LabeledVector, w: RecommenderWeights): (Double, Double) = {
    val itemIndex = x.vector.apply(1).toInt
    val userIndex = x.vector.apply(0).toInt
    val label = x.vector.apply(2)

    (w.itemWeights.apply(itemIndex).weights.dot(w.userWeights.apply(userIndex).weights), label)
  }
}
