package org.apache.spark.mllib.util

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkConf
import org.apache.spark.deploy.SparkHadoopUtil


object TreeUtils {
  def getFileSystem(conf: SparkConf, path: Path): FileSystem = {
    val hadoopConf = SparkHadoopUtil.get.newConfiguration(conf)
    if (sys.env.contains("HADOOP_CONF_DIR") || sys.env.contains("YARN_CONF_DIR")) {
      val hdfsConfPath = if (sys.env.get("HADOOP_CONF_DIR").isDefined) {
        sys.env.get("HADOOP_CONF_DIR").get + "/core-site.xml"
      } else {
        sys.env.get("YARN_CONF_DIR").get + "/core-site.xml"
      }
      hadoopConf.addResource(new Path(hdfsConfPath))
    }
    path.getFileSystem(hadoopConf)
  }

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
