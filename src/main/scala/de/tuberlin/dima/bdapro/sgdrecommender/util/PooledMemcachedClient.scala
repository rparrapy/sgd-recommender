package de.tuberlin.dima.bdapro.sgdrecommender.util
package de.tuberlin.dima.bdapro.sgdrecommender.util

import net.spy.memcached.AddrUtil
import net.spy.memcached.BinaryConnectionFactory
import net.spy.memcached.MemcachedClient

/**
  * Scala version of
  * https://github.com/mcascallares/pooled-memcached/blob/master/src/main/java/org/matias/memcached/PooledMemcachedClient.java
  * @param connectionPoolSize
  * @param serverAddresses
  */
class PooledMemcachedClient(val connectionPoolSize: Int, val serverAddresses: String) extends Serializable{

  private var connectionPool: Array[MemcachedClient] = null

  try {
    connectionPool = new Array[MemcachedClient](connectionPoolSize)
    var i: Int = 0
    while (i < connectionPoolSize) {
      {
        connectionPool(i) = new MemcachedClient(new BinaryConnectionFactory, AddrUtil.getAddresses(serverAddresses))
      }
      ({
        i += 1; i - 1
      })
    }
  }
  catch {
    case e: Exception => {
      throw new RuntimeException(e)
    }
  }

  def getCache: MemcachedClient = {
    val idx = (Math.random * connectionPoolSize).toInt
    //println("Returning client #: " + idx)
    return connectionPool(idx)
  }

  def teardown: Unit = {
    connectionPool.foreach(c => c.shutdown())
  }
}
