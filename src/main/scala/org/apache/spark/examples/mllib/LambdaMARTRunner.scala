package org.apache.spark.examples.mllib

import scopt.OptionParser
import scala.language.reflectiveCalls
import scala.util.Random

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.LambdaMART
import org.apache.spark.mllib.tree.configuration._
import org.apache.spark.util.Utils
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.model.SplitInfo


object LambdaMARTRunner {

  case class Params(
      trainingData: String = null,
      testData: String = null,
      targetScores: String = null,
      initScores: String = null,
      queryBoundy: String = null,
      algo: String = "Regression",
      maxDepth: Int = 8,
      numIterations: Int = 10) extends AbstractParams[Params]

  def main(args: Array[String]) {
  
    val defaultParams = Params()

    val parser = new OptionParser[Params]("LambdaMART") {
      head("LambdaMART: an implementation of LambdaMART for FastRank.")        
      opt[String]("trainingData")
        .text("trainingData path")
        .required()
        .action((x, c) => c.copy(trainingData = x))        
      opt[String]("testData")
        .text("testData path")
        .required()
        .action((x, c) => c.copy(testData = x))        
      opt[String]("targetScores")
        .text("targetScores path to training dataset")
        .required()
        .action((x, c) => c.copy(targetScores = x))
      opt[String]("initScores")
        .text(s"initScores path to training dataset. If not given, initScores will be {0 ...}.")
        .action((x, c) => c.copy(initScores = x))  
      opt[String]("queryBoundy")
        .text("queryBoundy path")
        .required()
        .action((x, c) => c.copy(queryBoundy = x))       
      opt[String]("algo")
        .text(s"algorithm (${Algo.values.mkString(",")}), default: ${defaultParams.algo}")
        .action((x, c) => c.copy(algo = x))
      opt[Int]("maxDepth")
        .text(s"max depth of the tree, default: ${defaultParams.maxDepth}")
        .action((x, c) => c.copy(maxDepth = x))
      opt[Int]("numIterations")
        .text(s"number of iterations of boosting," + s" default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      checkConfig { params =>
        if (params.maxDepth > 30) {
          failure(s"maxDepth ${params.maxDepth} value incorrect; should be less than or equals to 30.")
        } else {
          success
        }
      }
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {

    val conf = new SparkConf().setAppName(s"LambdaMARTRunner with $params")
    val sc = new SparkContext(conf)

    println(s"LambdaMARTRunner with parameters:\n$params")
        
    val trainingDataRaw = loadTrainingData(sc, params.trainingData, sc.defaultMinPartitions)
    //val numfeatures = trainingDataRaw.count()
    val samplePercent = 1.0/3.0
    //val numfeaturesAfterSampling = samplePercent * numfeatures 
    val trainingData = trainingDataRaw.sample(false, samplePercent, Random.nextLong())
    println(trainingDataRaw.count(), trainingData.count())
    // val testData = MLUtils.loadLibSVMFile(sc, params.testData).cache()
    val (lengthOfTargetScores, targetScores) = loadScores(sc, params.targetScores)
    targetScores.map(_.toShort)
    val (lengthOfInitScores, initScores) = if (params.initScores == null) {
      (lengthOfTargetScores, Array.fill[Double](lengthOfTargetScores)(0))
    }
    else {
      loadScores(sc, params.initScores)
    }
    val queryBoundy = loadQueryBoundy(sc, params.queryBoundy)

    require(lengthOfTargetScores == lengthOfInitScores,
      s"lengthOfTargetScores: $lengthOfTargetScores != lengthOfInitScores: $lengthOfTargetScores")
    

    val boostingStrategy = BoostingStrategy.defaultParams(params.algo)
    boostingStrategy.treeStrategy.maxDepth = params.maxDepth
    boostingStrategy.numIterations = params.numIterations
    
    var targetScores_sh = targetScores.map(_.toShort)
    if (params.algo == "Regression") {
      val startTime = System.nanoTime()
      val model = LambdaMART.train(trainingData, targetScores_sh, initScores, queryBoundy, boostingStrategy)
      val elapsedTime = (System.nanoTime() - startTime) / 1e9
      println(s"Training time: $elapsedTime seconds")
      if (model.totalNumNodes < 30) {
        println(model.toDebugString) // Print full model.
      } else {
        println(model) // Print model summary.
      }
      // val testMSE = meanSquaredError(model, testData)
      // println(s"Test mean squared error = $testMSE")
    }

    sc.stop()
  } 
  
  def loadTrainingData(
      sc: SparkContext,
      path: String,
      minPartitions: Int): RDD[(Int, Array[Byte], Array[SplitInfo])] = {
    sc.textFile(path, minPartitions).map { line =>
      val parts = line.split('#')
      val num = parts(0).toInt
      val features = parts(1).split(',').map(_.toByte)
      val splits = if (parts.length > 2)
          parts(2).split(',').map(_.toDouble)
          .map(threshold => new SplitInfo(num, threshold))
        else
          (0 to (if (features.max > 0) features.max - 1 else 0) toArray).map(_.toDouble)
          .map(threshold => new SplitInfo(num, threshold))

      (num, features, splits)
    }
  }
  
  def loadScores(
      sc: SparkContext,
      path: String): (Int, Array[Double]) = {
    val scores = sc.textFile(path).first().split(',').map(_.toDouble)
    (scores.length, scores)
  }
  
  def loadQueryBoundy(
    sc: SparkContext,
    path: String): Array[Int] = {
    val boundy = sc.textFile(path).first().split(',').map(_.toInt)
    boundy
  }

  /***
  def meanSquaredError(
      model: { def predict(features: Vector): Double },
      data: RDD[LabeledPoint]): Double = {
    data.map { y =>
      val err = model.predict(y.features) - y.label
      err * err
    }.mean()
  }***/

}