package de.tuberlin.dima.bdapro.sgdrecommender

/**
  * Created by duy on 12/9/16.
  */
import org.apache.flink.ml.common.WeightVector

case class RecommenderWeights (itemWeights: WeightVector, userWeights:WeightVector) extends Serializable
