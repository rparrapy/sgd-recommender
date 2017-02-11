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

package de.tuberlin.dima.bdapro.sgdrecommender.FlinkGradientDescent

import de.tuberlin.dima.bdapro.sgdrecommender.FlinkGradientDescent.IterativeSolver.{ConvergenceThreshold, Iterations, LearningRate, LearningRateMethodValue}
import de.tuberlin.dima.bdapro.sgdrecommender.FlinkGradientDescent.LearningRateMethod.LearningRateMethodTrait
import org.apache.flink.api.scala.{DataSet, _}
import org.apache.flink.ml.common._
import org.apache.flink.ml.math.{DenseVector, SparseVector}

import scala.util.Random

/** Base class for optimization algorithms
 *
 */
abstract class Solver extends Serializable with WithParameters {
  import Solver._

  /** Provides a solution for the given optimization problem
    *
    * @param data A Dataset of LabeledVector (input, output) pairs
    * @param initialWeights The initial weight that will be optimized
    * @return A Vector of weights optimized to the given problem
    */
  def optimize(
                data: DataSet[LabeledVector],
                initialWeights: Option[DataSet[RecommenderWeights]],
                f: DataSet[Int],
                nItems: Int,
                nUsers: Int)
    : DataSet[RecommenderWeights]


  def createInitialWeightsDS(initialWeights: Option[DataSet[RecommenderWeights]],
                             f: DataSet[Int],
                             nItems: Int,
                             nUsers: Int): DataSet[RecommenderWeights] = {
    def iterativeInit(wv: Array[WeightVector]): Array[WeightVector] = {
      if (wv.nonEmpty) {
        val denseItemWeights = wv.head.weights match {
          case dv: DenseVector => dv
          case sv: SparseVector => sv.toDenseVector
        }
        val itemWeightVector = WeightVector(denseItemWeights, wv.head.intercept)

        itemWeightVector +: iterativeInit(wv.tail)
      }

      Array.empty[WeightVector]
    }

    // TODO: Faster way to do this?
    initialWeights match {
      // Ensure provided weight vector is a DenseVector
      case Some(wvDS) =>
        wvDS.map {
          wv => {

            val denseItemWeights = iterativeInit(wv.itemWeights)
            val denseUserWeights = iterativeInit(wv.userWeights)

            RecommenderWeights(denseItemWeights, denseUserWeights, wv.intercept)
          }
        }
      case None => createInitialWeightVector(f, nItems, nUsers)
    }
  }

  /** Creates a DataSet with one zero vector. The zero vector has dimension d, which is given
    * by the dimensionDS.
    *
    * @param dimensionDS DataSet with one element d, denoting the dimension of the returned zero
    *                    vector
    * @return DataSet of a zero vector of dimension d
    */
  def createInitialWeightVector(dimensionDS: DataSet[Int], nItems: Int, nUsers: Int): DataSet[RecommenderWeights] = {
    val rand = new Random(1000L)
    dimensionDS.map {
      dimension =>
        val itemWeightVector = Array.fill(nItems)(WeightVector(DenseVector(Array.fill(dimension)(rand.nextGaussian / dimension)), 0))
        val userWeightVector = Array.fill(nUsers)(WeightVector(DenseVector(Array.fill(dimension)(rand.nextGaussian / dimension)), 0))

        RecommenderWeights(itemWeightVector, userWeightVector, 3.52986)
    }
  }

  //Setters for parameters
  // TODO(tvas): Provide an option to fit an intercept or not
  def setLossFunction(lossFunction: LossFunction): this.type = {
    parameters.add(LossFunction, lossFunction)
    this
  }

  def setRegularizationConstant(regularizationConstant: Double): this.type = {
    parameters.add(RegularizationConstant, regularizationConstant)
    this
  }
}

object Solver {
  // Define parameters for Solver
  case object LossFunction extends Parameter[LossFunction] {
    // TODO(tvas): Should depend on problem, here is where differentiating between classification
    // and regression could become useful
    val defaultValue = None
  }

  case object RegularizationConstant extends Parameter[Double] {
    val defaultValue = Some(0.0001) // TODO(tvas): Properly initialize this, ensure Parameter > 0!
  }
}

/** An abstract class for iterative optimization algorithms
  *
  * See [[https://en.wikipedia.org/wiki/Iterative_method Iterative Methods on Wikipedia]] for more
  * info
  */
abstract class IterativeSolver() extends Solver {

  //Setters for parameters
  def setIterations(iterations: Int): this.type = {
    parameters.add(Iterations, iterations)
    this
  }

  def setStepsize(stepsize: Double): this.type = {
    parameters.add(LearningRate, stepsize)
    this
  }

  def setConvergenceThreshold(convergenceThreshold: Double): this.type = {
    parameters.add(ConvergenceThreshold, convergenceThreshold)
    this
  }

  def setLearningRateMethod(learningRateMethod: LearningRateMethodTrait): this.type = {
    parameters.add(LearningRateMethodValue, learningRateMethod)
    this
  }
}

object IterativeSolver {

  val MAX_DLOSS: Double = 1e12

  // Define parameters for IterativeSolver
  case object LearningRate extends Parameter[Double] {
    val defaultValue = Some(0.1)
  }

  case object Iterations extends Parameter[Int] {
    val defaultValue = Some(10)
  }

  case object ConvergenceThreshold extends Parameter[Double] {
    val defaultValue = None
  }

  case object LearningRateMethodValue extends Parameter[LearningRateMethodTrait] {
    val defaultValue = Some(LearningRateMethod.Default)
  }
}

object LearningRateMethod {

  sealed trait LearningRateMethodTrait extends Serializable {
    def calculateLearningRate(
      initialLearningRate: Double,
      iteration: Int,
      regularizationConstant: Double)
    : Double
  }

  object Default extends LearningRateMethodTrait {
    override def calculateLearningRate(
      initialLearningRate: Double,
      iteration: Int,
      regularizationConstant: Double)
    : Double = {
      initialLearningRate / Math.sqrt(iteration)
    }
  }

  object Constant extends LearningRateMethodTrait {
    override def calculateLearningRate(
      initialLearningRate: Double,
      iteration: Int,
      regularizationConstant: Double)
    : Double = {
      initialLearningRate
    }
  }

  case class Bottou(optimalInit: Double) extends LearningRateMethodTrait {
    override def calculateLearningRate(
      initialLearningRate: Double,
      iteration: Int,
      regularizationConstant: Double)
    : Double = {
      1 / (regularizationConstant * (optimalInit + iteration - 1))
    }
  }

  case class InvScaling(decay: Double) extends LearningRateMethodTrait {
    override def calculateLearningRate(
      initialLearningRate: Double,
      iteration: Int,
      regularizationConstant: Double)
    : Double = {
      initialLearningRate / Math.pow(iteration, decay)
    }
  }

  case class Xu(decay: Double) extends LearningRateMethodTrait {
    override def calculateLearningRate(
      initialLearningRate: Double,
      iteration: Int,
      regularizationConstant: Double)
    : Double = {
      initialLearningRate *
        Math.pow(1 + regularizationConstant * initialLearningRate * iteration, -decay)
    }
  }
}
