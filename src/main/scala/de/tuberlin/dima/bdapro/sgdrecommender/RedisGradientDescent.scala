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


import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.scala.{DataSet, _}
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml._
import org.apache.flink.ml.common._
import org.apache.flink.ml.math._
import org.apache.flink.ml.optimization.IterativeSolver._
import org.apache.flink.ml.optimization.LearningRateMethod.LearningRateMethodTrait
import org.apache.flink.ml.optimization.Solver._
import org.apache.flink.ml.optimization.{IterativeSolver, LossFunction}

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
abstract class RedisGradientDescent extends IterativeSolver {

  //This does not work anymore, as it does not have a Redis manager
  override def optimize(
                         data: DataSet[LabeledVector],
                         initialWeights: Option[DataSet[WeightVector]]): DataSet[WeightVector] = {
    //optimize(data, None, initialWeights, 0)
    null
  }


  def optimize(
                data: DataSet[LabeledVector],
                numberOfFactors: Int): Unit = {
    optimize(data, None, numberOfFactors)
  }

  /** Provides a solution for the given optimization problem
    *
    * @param data A Dataset of LabeledVector (label, features) pairs
    * @return The weights, optimized for the provided data.
    */
  def optimize(
                data: DataSet[LabeledVector],
                test: Option[DataSet[LabeledVector]],
                numberOfFactors: Int): DataSet[LabeledVector] = {

    val numberOfIterations: Int = parameters(Iterations)
    val convergenceThresholdOption: Option[Double] = parameters.get(ConvergenceThreshold)
    val lossFunction = parameters(LossFunction)
    val learningRate = parameters(LearningRate)
    val regularizationConstant = parameters(RegularizationConstant)
    val learningRateMethod = parameters(LearningRateMethodValue)
    // Initialize weights
    //val initialWeightsDS: DataSet[WeightVector] = createInitialWeightsDS(initialWeights, data)


    // Perform the iterations
    convergenceThresholdOption match {
      // No convergence criterion
      case None =>
        optimizeWithoutConvergenceCriterion(
          data,
          numberOfIterations,
          regularizationConstant,
          learningRate,
          lossFunction,
          learningRateMethod,
          numberOfFactors)
      case Some(convergence) =>
        optimizeWithConvergenceCriterion(
          data,
          test,
          numberOfIterations,
          regularizationConstant,
          learningRate,
          convergence,
          lossFunction,
          learningRateMethod,
          numberOfFactors)
    }
  }

  def optimizeWithConvergenceCriterion(
                                        dataPoints: DataSet[LabeledVector],
                                        testDataPoints: Option[DataSet[LabeledVector]],
                                        numberOfIterations: Int,
                                        regularizationConstant: Double,
                                        learningRate: Double,
                                        convergenceThreshold: Double,
                                        lossFunction: LossFunction,
                                        learningRateMethod: LearningRateMethodTrait,
                                        numberOfFactors: Int)
  : DataSet[LabeledVector] = {
    // We have to calculate for each weight vector the sum of squared residuals,
    // and then sum them and apply regularization
    val initialLossSumDS = calculateLoss(dataPoints, lossFunction, numberOfFactors)
    val initialTestLossDS = testDataPoints.map(calculateLoss(_, lossFunction, numberOfFactors, "Test MSE: "))

    // Combine weight vector with the current loss
    val initialWeightsWithLossSum = initialTestLossDS.map { testDS => {
      val withTestLoss = dataPoints.mapWithBcVariable(testDS)((w, l) => (w, l))
      withTestLoss.mapWithBcVariable(initialLossSumDS) {
        (wl, l) => (wl._1, l, wl._2)
      }
    }
    }.getOrElse {
      dataPoints.mapWithBcVariable(initialLossSumDS) {
        (weights, loss) => (weights, loss, -1.0)
      }
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
          previousWeightsDS,
          lossFunction,
          regularizationConstant,
          learningRate,
          learningRateMethod,
          numberOfFactors)

        val currentLossSumDS = calculateLoss(previousWeightsDS, lossFunction, numberOfFactors)
        val currentTestLossDS = testDataPoints.map(calculateLoss(_, lossFunction, numberOfFactors, "Test MSE: ", Some(previousWeightsDS)))

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
        currentTestLossDS.map { testDS => {
          val withTestLoss = currentWeightsDS.mapWithBcVariable(testDS)((w, l) => (w, l))
          (withTestLoss.mapWithBcVariable(currentLossSumDS)((wl, l) => (wl._1, l, wl._2)), termination)
        }
        }.getOrElse {
          (currentWeightsDS.mapWithBcVariable(currentLossSumDS)((w, l) => (w, l, -1.0)), termination)
        }
    }
    // Return just the weights
    resultWithLoss map (_._1)
  }

  def optimizeWithoutConvergenceCriterion(
                                           data: DataSet[LabeledVector],
                                           numberOfIterations: Int,
                                           regularizationConstant: Double,
                                           learningRate: Double,
                                           lossFunction: LossFunction,
                                           optimizationMethod: LearningRateMethodTrait,
                                           numberOfFactors: Int)
  : DataSet[LabeledVector] = {

    data.iterate(numberOfIterations) {
      data => {
        SGDStep(data,
          lossFunction,
          regularizationConstant,
          learningRate,
          optimizationMethod,
          numberOfFactors)
      }
    }
  }

  /** Performs one iteration of Stochastic Gradient Descent using mini batches
    *
    * @param data A Dataset of LabeledVector (label, features) pairs
    * @return A Dataset containing the weights after one stochastic gradient descent step
    */
  private def SGDStep(
                       data: DataSet[(LabeledVector)],
                       lossFunction: LossFunction,
                       regularizationConstant: Double,
                       learningRate: Double,
                       learningRateMethod: LearningRateMethodTrait,
                       numberOfFactors: Int)
  : DataSet[LabeledVector] = {
    data.map {
      new RichMapFunction[LabeledVector, LabeledVector] {

        override def close(): Unit = {
          super.close()
          //println("Iteration finished wooohooo")
        }

        override def map(data: LabeledVector): LabeledVector = {
          val weightVector = RedisModelManager.getWeights(data.vector, numberOfFactors)
          val gradient = lossFunction.gradient(data, weightVector)

          val newWeights = takeStep(
            weightVector.weights,
            gradient.weights,
            regularizationConstant,
            learningRate)

          val newWeightsVector = WeightVector(
            newWeights,
            weightVector.intercept - learningRate * gradient.intercept)

          RedisModelManager.updateParameters(data.vector, newWeightsVector, numberOfFactors)
          data
        }
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
  def takeStep(
                weightVector: Vector,
                gradient: Vector,
                regularizationConstant: Double,
                learningRate: Double
              ): Vector

  /** Calculates the regularized loss, from the data and given weights.
    *
    * @param data
    * @param weightDS
    * @param lossFunction
    * @return
    */
  private def calculateLoss(
                             data: DataSet[LabeledVector],
                             lossFunction: LossFunction,
                             numberOfFactors: Int,
                             prefix: String = "Training MSE: ",
                             dummy: Option[DataSet[LabeledVector]] = None)
  : DataSet[Double] = {
    val d = dummy match {
      case None => data
      case Some(dd) => dd.map((_, 1)).sum(1).cross(data).map(_._2)
    }

    d.map(x => {
      val weights = RedisModelManager.getWeights(x.vector, numberOfFactors)
      (lossFunction.loss(x, weights), 1)
    }).reduce {
      (left, right) => (left._1 + right._1, left._2 + right._2)
    }.map {
      lossCount => {
        val loss = lossCount._1 / lossCount._2
        println(prefix + s"$loss")
        loss
      }
    }
  }
}

/** Implementation of a SGD solver with L2 regularization.
  *
  * The regularization function is `1/2 ||w||_2^2` with `w` being the weight vector.
  */
class RedisGradientDescentL2 extends RedisGradientDescent {

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

object RedisGradientDescentL2 {
  def apply() = new RedisGradientDescentL2
}

/** Implementation of a SGD solver with L1 regularization.
  *
  * The regularization function is `||w||_1` with `w` being the weight vector.
  */
class RedisGradientDescentL1 extends RedisGradientDescent {

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

object RedisGradientDescentL1 {
  def apply() = new RedisGradientDescentL1
}

/** Implementation of a SGD solver without regularization.
  *
  * No regularization is applied.
  */
class RedisSimpleGradientDescent extends RedisGradientDescent {

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

object RedisSimpleGradientDescent {
  def apply() = new RedisSimpleGradientDescent
}