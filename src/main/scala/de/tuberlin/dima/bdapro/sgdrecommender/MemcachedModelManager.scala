package de.tuberlin.dima.bdapro.sgdrecommender

import de.tuberlin.dima.bdapro.sgdrecommender.util.de.tuberlin.dima.bdapro.sgdrecommender.util.PooledMemcachedClient
import org.apache.flink.ml.common.WeightVector
import org.apache.flink.ml.math.{DenseVector, Vector}
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.util.Random

/**
  * Implements the logic needed to update model parameters in the Redis server
  *
  * Created by rparra on 6/1/17.
  */
object MemcachedModelManager extends Serializable{
  lazy val pool = new PooledMemcachedClient(8, "localhost:11211")
  val r = new Random(1000L)
  implicit val formats = DefaultFormats
  val timeout = 3600
  case class Weights(weights: List[Double])

  /**
    * Updates model parameters in the Redis server.
    *
    * Note: it is useful to remember that the user gradient == current item weights
    * and viceversa.
    *
    * @param features
    * @return
    */
  def updateParameters(features: Vector, weightVector: WeightVector, numberOfFactors: Int) = {

    val user = features(0).toInt
    val item = features(1).toInt

    val userWeightsBias = weightVector.weights.slice(0, numberOfFactors + 1)
    val itemWeightsBias = weightVector.weights.slice(numberOfFactors + 1, 2 * (numberOfFactors + 1))

    val client = pool.getCache
    client.set("user-" + user, timeout, compact(render(userWeightsBias.toList.map(_._2))))
    client.set("item-" + item, timeout, compact(render(itemWeightsBias.toList.map(_._2))))
  }

  /**
    * Fetches the weight vectors from a (user, item) pair, concatenates them and returns a WeightVector
    *
    * @param features
    * @return
    */
  def getWeights(features: Vector, numberOfFactors: Int): WeightVector = {
    val user = features(0).toInt
    val item = features(1).toInt

    val client = pool.getCache
    val userSerialized = client.get("user-" + user).toString
    val userWeightsBias = parse(userSerialized).extract[List[Double]].toArray
    val itemSerialized = client.get("item-" + user).toString
    val itemWeightsBias = parse(itemSerialized).extract[List[Double]].toArray
    val weights = DenseVector.zeros((numberOfFactors + 1) * 2)
    for (i <- 0 to numberOfFactors) {
      weights(i) = userWeightsBias(i)
      weights(numberOfFactors + 1 + i) = itemWeightsBias(i)
    }
    WeightVector(weights, 0)
  }

  def setup(numberOfUsers: Int, numberOfItems: Int, numberOfFactors: Int) = {

    val client = pool.getCache
    for (uid <- 1 to numberOfUsers) {
      client.set("user-" + uid, timeout, compact(render(getInitialWeights(numberOfFactors))))
    }

    for (iid <- 1 to numberOfItems) {
      client.set("item-" + iid, timeout, compact(render(getInitialWeights(numberOfFactors))))
    }
  }

  def tearDown = pool.teardown

  private def getInitialWeights(numberOfFactors: Int): List[Double] = {
    (0 to numberOfFactors).map(x => {
      if (x % numberOfFactors == 0) 0.0
      else r.nextGaussian() * 1 / numberOfFactors
    }).toList
  }
}
