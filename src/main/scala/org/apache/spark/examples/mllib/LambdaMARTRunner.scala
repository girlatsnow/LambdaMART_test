package org.apache.spark.examples.mllib

import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.tree.LambdaMART
import org.apache.spark.mllib.tree.configuration._
import org.apache.spark.mllib.tree.model.SplitInfo
import org.apache.spark.mllib.util.TreeUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import scopt.OptionParser

import scala.language.reflectiveCalls


object LambdaMARTRunner {
  case class Params(trainingData: String = null,
    testData: String = null,
    queryBoundy: String = null,
    modelOutput: String = null,
    labelScores: String = null,
    initScores: String = null,
    algo: String = "Regression",
    maxDepth: Int = 8,
    numLeaves: Int = 0,
    numIterations: Int = 10,
    maxSplits: Int = 128) extends AbstractParams[Params]

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
      opt[String]("queryBoundy")
        .text("queryBoundy path")
        .required()
        .action((x, c) => c.copy(queryBoundy = x))
      opt[String]("modelOutput")
        .text("modelOutput path")
        .required()
        .action((x, c) => c.copy(modelOutput = x))
      opt[String]("labelScores")
        .text("labelScores path to training dataset")
        .required()
        .action((x, c) => c.copy(labelScores = x))
      opt[String]("initScores")
        .text(s"initScores path to training dataset. If not given, initScores will be {0 ...}.")
        .action((x, c) => c.copy(initScores = x))
      opt[String]("algo")
        .text(s"algorithm (${Algo.values.mkString(",")}), default: ${defaultParams.algo}")
        .action((x, c) => c.copy(algo = x))
      opt[Int]("maxDepth")
        .text(s"max depth of the tree, default: ${defaultParams.maxDepth}")
        .action((x, c) => c.copy(maxDepth = x))
      opt[Int]("numLeaves")
        .text(s"num of leaves per tree, default: ${defaultParams.numLeaves}. Take precedence over --maxDepth.")
        .action((x, c) => c.copy(numLeaves = x))
      opt[Int]("numIterations")
        .text(s"number of iterations of boosting," + s" default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("maxSplits")
        .text(s"max Nodes to be split simultaneously, default: ${defaultParams.maxSplits}")
        .action((x, c) => c.copy(maxSplits = x))
      checkConfig(params =>
        if (params.maxDepth > 30) {
          failure(s"maxDepth ${params.maxDepth} value incorrect; should be less than or equals to 30.")
        } else if (params.maxSplits > 128 || params.maxSplits <= 0) {
          failure(s"maxSplits ${params.maxSplits} value incorrect; should be between 1 and 128.")
        } else {
          success
        }
      )
    }

    parser.parse(args, defaultParams).map(run).getOrElse(sys.exit(1))
  }

  def run(params: Params) {
    println(s"LambdaMARTRunner with parameters:\n$params")
    val conf = new SparkConf().setAppName(s"LambdaMARTRunner with $params")
    val sc = new SparkContext(conf)
    try {
      val labelScores = loadlabelScores(sc, params.labelScores)
      val numSamples = labelScores.length
      val initScores = if (params.initScores == null) {
        new Array[Double](numSamples)
      } else {
        val loaded = loadInitScores(sc, params.initScores)
        require(loaded.length == numSamples, s"lengthOfInitScores: ${loaded.length} != numSamples: $numSamples")
        loaded
      }

      val queryBoundy = loadQueryBoundy(sc, params.queryBoundy)
      require(queryBoundy.last == numSamples, s"QueryBoundy file does not match with data!")

      val trainingData = loadTrainingData(sc, params.trainingData, sc.defaultParallelism)
      val numFeats = trainingData.count().toInt

      val trainingData_T = genTransposedData(trainingData, numFeats, numSamples)

      //val samplePercent = 1.0/3.0
      //val numfeaturesAfterSampling = samplePercent * numfeatures
      //val trainingData = trainingDataRaw.sample(false, samplePercent, Random.nextLong())
      //println(trainingDataRaw.count(), trainingData.count())
      // val testData = MLUtils.loadLibSVMFile(sc, params.testData).cache()

      val boostingStrategy = BoostingStrategy.defaultParams(params.algo)
      boostingStrategy.treeStrategy.maxDepth = params.maxDepth
      boostingStrategy.numIterations = params.numIterations

      if (params.algo == "Regression") {
        val startTime = System.nanoTime()
        val model = LambdaMART.train(trainingData, trainingData_T, labelScores, initScores, queryBoundy,
          boostingStrategy, params.numLeaves, params.maxSplits)
        val elapsedTime = (System.nanoTime() - startTime) / 1e9
        println(s"Training time: $elapsedTime seconds")

        val outPath = new Path(params.modelOutput)
        val fs = TreeUtils.getFileSystem(trainingData.context.getConf, outPath)
        fs.copyFromLocalFile(true, true, new Path("treeEnsemble.ini"), outPath)

        if (model.totalNumNodes < 30) {
          println(model.toDebugString) // Print full model.
        } else {
          println(model) // Print model summary.
        }
        // val testMSE = meanSquaredError(model, testData)
        // println(s"Test mean squared error = $testMSE")
      }
    } finally {
      sc.stop()
    }
  }

  def loadTrainingData(sc: SparkContext, path: String, minPartitions: Int)
  : RDD[(Int, Array[Short], Array[SplitInfo])] = {
    sc.textFile(path, minPartitions).map { line =>
      val parts = line.split("#")
      val feat = parts(0).toInt
      val samples = parts(1).split(',').map(_.toShort)
      val splits = if (parts.length > 2) {
        parts(2).split(',').map(threshold => new SplitInfo(feat, threshold.toDouble))
      } else {
        val maxFeat = samples.max
        Array.tabulate(maxFeat)(threshold => new SplitInfo(feat, threshold))
      }
      (feat, samples, splits)
    }.persist(StorageLevel.MEMORY_AND_DISK).setName("trainingData")
  }

  def loadlabelScores(sc: SparkContext, path: String): Array[Short] = {
    sc.textFile(path).first().split(',').map(_.toShort)
  }

  def loadInitScores(sc: SparkContext, path: String): Array[Double] = {
    sc.textFile(path).first().split(',').map(_.toDouble)
  }

  def loadQueryBoundy(sc: SparkContext, path: String): Array[Int] = {
    sc.textFile(path).first().split(',').map(_.toInt)
  }

  def loadThresholdMap(sc: SparkContext, path: String, numFeats: Int): Array[Array[Double]] = {
    val thresholdMapTuples = sc.textFile(path).map { line =>
      val fields = line.split("#", 2)
      (fields(0).toInt, fields(1).split(',').map(_.toDouble))
    }.collect()
    val numFeatsTM = thresholdMapTuples.length
    assert(numFeats == numFeatsTM, s"ThresholdMap file contains $numFeatsTM features that != $numFeats")
    val thresholdMap = new Array[Array[Double]](numFeats)
    thresholdMapTuples.foreach { case (fi, thresholds) =>
      thresholdMap(fi) = thresholds
    }
    thresholdMap
  }

  def genTransposedData(trainingData: RDD[(Int, Array[Short], Array[SplitInfo])],
    numFeats: Int,
    numSamples: Int): RDD[(Int, Array[Array[Short]])] = {
    println("generating transposed data...")
    // validate that the original data is ordered
    val denseAsc = trainingData.mapPartitions { iter =>
      var prev = iter.next()._1
      val remaining = iter.dropWhile { case (fi, _, _) =>
        val goodNext = fi - prev == 1
        prev = fi
        goodNext
      }
      Iterator.single(!remaining.hasNext)
    }.reduce(_ && _)
    assert(denseAsc, "the original data must be ordered.")

    val numPartitions = trainingData.partitions.length
    val (siMinPP, lcNumSamplesPP) = TreeUtils.getPartitionOffsets(numSamples, numPartitions)
    val trainingData_T = trainingData.mapPartitions { iter =>
      val (metaIter, dataIter) = iter.duplicate
      val fiMin = metaIter.next()._1
      val lcNumFeats = metaIter.length + 1
      val blocksPP = Array.tabulate(numPartitions)(pi => Array.ofDim[Short](lcNumFeats, lcNumSamplesPP(pi)))
      dataIter.foreach { case (fi, samples, _) =>
        val lfi = fi - fiMin
        var pi = 0
        while (pi < numPartitions) {
          Array.copy(samples, siMinPP(pi), blocksPP(pi)(lfi), 0, lcNumSamplesPP(pi))
          pi += 1
        }
      }
      Range(0, numPartitions).iterator.map(pi => (pi, (fiMin, blocksPP(pi))))
    }.partitionBy(new HashPartitioner(numPartitions)).mapPartitionsWithIndex((pid, iter) => {
      val siMin = siMinPP(pid)
      val sampleSlice = new Array[Array[Short]](numFeats)
      iter.foreach { case (_, (fiMin, blocks)) =>
        var lfi = 0
        while (lfi < blocks.length) {
          sampleSlice(lfi + fiMin) = blocks(lfi)
          lfi += 1
        }
      }
      Iterator.single((siMin, sampleSlice))
    }, preservesPartitioning=true)
    trainingData_T.persist(StorageLevel.MEMORY_AND_DISK).setName("trainingData_T").count()
    trainingData_T
  }

  /***
    * def meanSquaredError(
    * model: { def predict(features: Vector): Double },
    * data: RDD[LabeledPoint]): Double = {
    * data.map { y =>
    * val err = model.predict(y.features) - y.label
    * err * err
    * }.mean()
    * }***/
}
