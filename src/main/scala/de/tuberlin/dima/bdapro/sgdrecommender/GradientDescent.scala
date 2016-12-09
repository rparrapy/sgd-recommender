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
import de.tuberlin.dima.bdapro.sgdrecommender.Solver.{LossFunction, RegularizationConstant}
import org.apache.flink.api.scala._
import org.apache.flink.ml._
import org.apache.flink.ml.common._
import org.apache.flink.ml.math._
import org.apache.flink.ml.optimization.LearningRateMethod.LearningRateMethodTrait
import de.tuberlin.dima.bdapro.sgdrecommender.IterativeSolver

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
                         f: DataSet[Int]): DataSet[RecommenderWeights] = {

    val numberOfIterations: Int = parameters(Iterations)
    val convergenceThresholdOption: Option[Double] = parameters.get(ConvergenceThreshold)
    val lossFunction = parameters(LossFunction)
    val learningRate = parameters(LearningRate)
    val regularizationConstant = parameters(RegularizationConstant)
    val learningRateMethod = parameters(LearningRateMethodValue)
    // Initialize weights
    val initialWeightsDS: DataSet[RecommenderWeights] = createInitialWeightsDS(initialWeights, f)

    optimizeWithoutConvergenceCriterion(
      data,
      initialWeightsDS,
      numberOfIterations,
      regularizationConstant,
      learningRate,
      lossFunction,
      learningRateMethod)

    // Perform the iterations
    //    convergenceThresholdOption match {
    // No convergence criterion
    //      case None =>
    //        optimizeWithoutConvergenceCriterion(
    //          data,
    //          initialWeightsDS,
    //          numberOfIterations,
    //          regularizationConstant,
    //          learningRate,
    //          lossFunction,
    //          learningRateMethod)
    //      case Some(convergence) =>
    //        optimizeWithConvergenceCriterion(
    //          data,
    //          initialWeightsDS,
    //          numberOfIterations,
    //          regularizationConstant,
    //          learningRate,
    //          convergence,
    //          lossFunction,
    //          learningRateMethod)
    //    }
    //  }

    //  def optimizeWithConvergenceCriterion(
    //                                        dataPoints: DataSet[LabeledVector],
    //                                        initialWeightsDS: DataSet[WeightVector],
    //                                        numberOfIterations: Int,
    //                                        regularizationConstant: Double,
    //                                        learningRate: Double,
    //                                        convergenceThreshold: Double,
    //                                        lossFunction: LossFunction,
    //                                        learningRateMethod: LearningRateMethodTrait)
    //  : DataSet[WeightVector] = {
    //    // We have to calculate for each weight vector the sum of squared residuals,
    //    // and then sum them and apply regularization
    //    val initialLossSumDS = calculateLoss(dataPoints, initialWeightsDS, lossFunction)
    //
    //    // Combine weight vector with the current loss
    //    val initialWeightsWithLossSum = initialWeightsDS.mapWithBcVariable(initialLossSumDS) {
    //      (weights, loss) => (weights, loss)
    //    }
    //
    //    val resultWithLoss = initialWeightsWithLossSum.iterateWithTermination(numberOfIterations) {
    //      weightsWithPreviousLossSum =>
    //
    //        // Extract weight vector and loss
    //        val previousWeightsDS = weightsWithPreviousLossSum.map {
    //          _._1
    //        }
    //        val previousLossSumDS = weightsWithPreviousLossSum.map {
    //          _._2
    //        }
    //
    //        val currentWeightsDS = SGDStep(
    //          dataPoints,
    //          previousWeightsDS,
    //          lossFunction,
    //          regularizationConstant,
    //          learningRate,
    //          learningRateMethod)
    //
    //        val currentLossSumDS = calculateLoss(dataPoints, currentWeightsDS, lossFunction)
    //
    //        // Check if the relative change in the loss is smaller than the
    //        // convergence threshold. If yes, then terminate i.e. return empty termination data set
    //        val termination = previousLossSumDS.filterWithBcVariable(currentLossSumDS) {
    //          (previousLoss, currentLoss) => {
    //            if (previousLoss <= 0) {
    //              false
    //            } else {
    //              scala.math.abs((previousLoss - currentLoss) / previousLoss) >= convergenceThreshold
    //            }
    //          }
    //        }
    //
    //        // Result for new iteration
    //        (currentWeightsDS.mapWithBcVariable(currentLossSumDS)((w, l) => (w, l)), termination)
    //    }
    //    // Return just the weights
    //    resultWithLoss.map {
    //      _._1
    //    }
    //  }
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
        (data, weightVector) => (lossFunction.gradient(data, weightVector), 1)
      }.reduce {
        (left, right) =>
          val (leftGradVector, leftCount) = left
          val (rightGradVector, rightCount) = right

          // make the left gradient dense so that the following reduce operations (left fold) reuse
          // it. This strongly depends on the underlying implementation of the ReduceDriver which
          // always passes the new input element as the second parameter
          val itemResult = leftGradVector.itemWeights.weights match {
            case d: DenseVector => d
            case s: SparseVector => s.toDenseVector
          }
          val userResult = leftGradVector.userWeights.weights match {
            case d: DenseVector => d
            case s: SparseVector => s.toDenseVector
          }

          // Add the right gradient to the result
          BLAS.axpy(1.0, rightGradVector.itemWeights.weights, itemResult)
          BLAS.axpy(1.0, rightGradVector.userWeights.weights, userResult)

          val itemGradients = WeightVector(
            itemResult, leftGradVector.itemWeights.intercept + rightGradVector.itemWeights.intercept)
          val userGradients = WeightVector(
            userResult, leftGradVector.userWeights.intercept + rightGradVector.userWeights.intercept)
          val gradients = RecommenderWeights(itemGradients, userGradients)

          (gradients, leftCount + rightCount)
      }.mapWithBcVariableIteration(currentWeights) {
        (gradientCount, weightVector, iteration) => {
          val (RecommenderWeights(WeightVector(itemWeights, itemIntercept),
          WeightVector(userWeights, userIntercept)), count) = gradientCount

          BLAS.scal(1.0 / count, itemWeights)
          BLAS.scal(1.0 / count, userWeights)

          val itemGradient = WeightVector(itemWeights, itemIntercept / count)
          val userGradient = WeightVector(userWeights, userIntercept / count)

          val effectiveLearningRate = learningRateMethod.calculateLearningRate(
            learningRate,
            iteration,
            regularizationConstant)

          val tempNewItemWeights = takeStep(
            weightVector.itemWeights.weights,
            itemGradient.weights,
            regularizationConstant,
            effectiveLearningRate)
          val tempNewUserWeights = takeStep(
            weightVector.userWeights.weights,
            userGradient.weights,
            regularizationConstant,
            effectiveLearningRate)

          val newItemWeights = WeightVector(
            tempNewItemWeights,
            weightVector.itemWeights.intercept - effectiveLearningRate * itemGradient.intercept)
          val newUserWeights = WeightVector(
            tempNewUserWeights,
            weightVector.userWeights.intercept - effectiveLearningRate * userGradient.intercept)

          RecommenderWeights(newItemWeights, newUserWeights)
        }
      }
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
               ngradient: Vector,
               regularizationConstant: Double,
               learningRate: Double): Vector

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
      lossCount => lossCount._1 / lossCount._2
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