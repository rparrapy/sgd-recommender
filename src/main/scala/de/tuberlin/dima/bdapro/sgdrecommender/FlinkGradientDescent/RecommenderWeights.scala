package de.tuberlin.dima.bdapro.sgdrecommender.FlinkGradientDescent

/**
  * Created by duy on 12/9/16.
  */
import org.apache.flink.ml.common.WeightVector

case class RecommenderWeights (itemWeights: Array[WeightVector], userWeights: Array[WeightVector], intercept: Double) extends Serializable
