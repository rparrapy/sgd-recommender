import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.flink.ml.common.WeightVector
import org.apache.flink.ml.math.{DenseVector, VectorBuilder}
/**
  * Created by duy on 12/7/16.
  */
object MatrixFactorizationExample extends App{
  override def main(args: _root_.scala.Array[_root_.scala.Predef.String]): Unit = {
    val env  = ExecutionEnvironment.getExecutionEnvironment
    val dimension = 10

    val data = env.readCsvFile[(Int, Int, Int, Int)](args(0), ignoreFirstLine = true)
    val q = Some(env.fromCollection(Some(new WeightVector(DenseVector.zeros(dimension), 0.0))))
    val p = Some(env.fromCollection(Some(new WeightVector(DenseVector.zeros(dimension), 0.0))))
  }

}
