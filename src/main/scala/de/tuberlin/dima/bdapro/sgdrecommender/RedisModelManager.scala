package de.tuberlin.dima.bdapro.sgdrecommender

import com.redis.{RedisClient, RedisClientPool}
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
object RedisModelManager extends Serializable{
  lazy val clients = new RedisClientPool("localhost", 6379)
  val r = new Random(1000L)
  implicit val formats = DefaultFormats
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

    clients.withClient(rc => {
      rc.set("user-" + user, compact(render(userWeightsBias.toList.map(_._2))))
      rc.set("item-" + item, compact(render(itemWeightsBias.toList.map(_._2))))
    })
  }

  /**
    * Fetches the weight vectors from a (user, item) pair, concatenates them and returns a WeightVector
    * @param features
    * @return
    */
  def getWeights(features: Vector, numberOfFactors: Int): WeightVector = {
    val user = features(0).toInt
    val item = features(1).toInt

    clients.withClient(rc => {
      val userSerialized = rc.get("user-" + user).get
      val userWeightsBias = parse(userSerialized).extract[List[Double]].toArray
      val itemSerialized = rc.get("item-" + user).get
      val itemWeightsBias = parse(itemSerialized).extract[List[Double]].toArray
      val weights = DenseVector.zeros((numberOfFactors + 1) * 2)
      for (i <- 0 to numberOfFactors) {
        weights(i) = userWeightsBias(i)
        weights(numberOfFactors + 1 + i) = itemWeightsBias(i)
      }
      WeightVector(weights, 0)
    })
  }

  def setup(numberOfUsers: Int, numberOfItems: Int, numberOfFactors: Int) = {

    clients.withClient(rc => {
      for (uid <- 1 to numberOfUsers) {
        rc.set("user-" + uid, compact(render(getInitialWeights(numberOfFactors))))
      }

      for (iid <- 1 to numberOfItems) {
        rc.set("item-" + iid, compact(render(getInitialWeights(numberOfFactors))))
      }
    })
  }

  def tearDown = clients.close

  private def getInitialWeights(numberOfFactors: Int): List[Double] = {
    (0 to numberOfFactors).map(x => {
      if (x % numberOfFactors == 0) 0.0
      else r.nextGaussian() * 1 / numberOfFactors
    }).toList
  }
}
