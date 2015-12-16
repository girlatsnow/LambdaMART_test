package org.apache.spark.mllib.util

object TreeUtils {
  def getPartitionOffsets(upper: Int, numPartitions: Int): (Array[Int], Array[Int]) = {
    val kpp = {
      val npp = upper / numPartitions
      if (npp * numPartitions == upper) npp else npp + 1
    }
    val startPP = Array.tabulate(numPartitions)(_ * kpp)
    val lcLensPP = Array.tabulate(numPartitions)(pi =>
      if (pi < numPartitions - 1) kpp else upper - startPP(pi)
    )
    (startPP, lcLensPP)
  }
}
