package org.apache.spark.mllib.util

import java.io.{File, PrintWriter, FileOutputStream}


object treeAggregatorFormat{
  def appendTreeAggregator(filePath: String,
    index: Int,
    evalNodes: Array[Int],
    evalWeights: Array[Double] = null,
    bias: Double = 0.0,
    Type: String = "Linear"): Unit = {
    val pw = new PrintWriter(new FileOutputStream(new File(filePath), true))

    pw.append(s"[Evaluator:$index]").write("\r\n")
    pw.append(s"EvaluatorType=Aggregator").write("\r\n")

    val numNodes = evalNodes.length
    val defaultWeight = 1.0
    if (evalNodes == null) {
      throw new IllegalArgumentException("there is no evaluators to be aggregated")
    } else {
      pw.append(s"NumNodes=$numNodes").write("\r\n")
      pw.append(s"Nodes=").write("")
      for (eval <- evalNodes) {
        pw.append(s"E:$eval").write("\t")
      }
      pw.write("\r\n")
    }

    var weights = new Array[Double](numNodes)
    if (evalWeights == null) {
      for (i <- 0 until numNodes) {
        weights(i) = defaultWeight
      }
    } else {
      weights = evalWeights
    }

    pw.append(s"Weights=").write("")
    for (weight <- weights) {
      pw.append(s"$weight").write("\t")
    }

    pw.write("\r\n")

    pw.append(s"Bias=$bias").write("\r\n")
    pw.append(s"Type=$Type").write("\r\n")

    pw.close()
  }
}
