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

import org.apache.flink.ml.common.WeightVector
import org.apache.flink.ml.math.{BLAS, VectorBuilder, Vector => FlinkVector}

/** An abstract class for prediction functions to be used in optimization **/
abstract class PredictionFunction extends Serializable {
  def predict(weights: (WeightVector, WeightVector, Double)): Double

  def gradient(weights: (WeightVector, WeightVector)): (WeightVector, WeightVector)
}

/** A linear prediction function **/
object RecommenderPrediction extends PredictionFunction {
  override def predict(weight: (WeightVector, WeightVector, Double)): Double = {
    BLAS.dot(weight._1.weights, weight._2.weights) + weight._1.intercept + weight._2.intercept + weight._3
  }

  override def gradient(weight: (WeightVector, WeightVector)): (WeightVector, WeightVector) = {
    val gradientItem = WeightVector(weight._2.weights.copy, 1)
    val gradientUser = WeightVector(weight._1.weights.copy, 1)

    (gradientItem, gradientUser)
  }
}
