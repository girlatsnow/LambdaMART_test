package org.apache.spark.examples.mllib

import java.io.IOException
import scala.language.reflectiveCalls

import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.mllib.tree.LambdaMART
import org.apache.spark.mllib.tree.configuration._
import org.apache.spark.mllib.tree.model.SplitInfo
import org.apache.spark.mllib.util.TreeUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import scopt.OptionParser


object LambdaMARTRunner {

  case class Params(
      trainingData: String = null,
      trainingData_T: String = null,
      testData: String = null,
      labelScores: String = null,
      initScores: String = null,
      queryBoundy: String = null,
      algo: String = "Regression",
      maxDepth: Int = 8,
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
      opt[String]("transposedData")
        .text("transposed trainingData path")
        .required()
        .action((x, c) => c.copy(trainingData_T = x))
      opt[String]("testData")
        .text("testData path")
        .required()
        .action((x, c) => c.copy(testData = x))        
      opt[String]("labelScores")
        .text("labelScores path to training dataset")
        .required()
        .action((x, c) => c.copy(labelScores = x))
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

    val trainingData = loadTrainingData(sc, params.trainingData, sc.defaultMinPartitions)
    val numFeats = trainingData.count().toInt

    val trainingData_T = loadOrGenTransposedData(trainingData, params.trainingData_T, numFeats, numSamples)

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
        boostingStrategy, params.maxSplits)
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
    sc.textFile(path, minPartitions).map(line => {
      val parts = line.split("#", 2)
      val num = parts(0).toInt
      val features = parts(1).split(',').map(_.toByte)
      val splits = if (parts.length > 2) {
        parts(2).split(',').map(threshold => new SplitInfo(num, threshold.toDouble))
      } else {
        val max = math.max(0, features.max)
        Array.tabulate(max)(threshold => new SplitInfo(num, threshold.toDouble))
      }
      (num, features, splits)
    }).persist(StorageLevel.MEMORY_AND_DISK).setName("trainingData")
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

  def loadOrGenTransposedData(
      trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
      path: String,
      numFeats: Int,
      numSamples: Int): RDD[(Int, Array[Array[Byte]])] = {
    val sc = trainingData.context
    val fs = FileSystem.get(sc.hadoopConfiguration)
    if (!fs.exists(new Path(path))) {
      println("transposed data does not exist, generating...")
      // validate that the original data is ordered
      val denseAsc = trainingData.mapPartitions(iter => {
        var prev = iter.next()._1
        val remaining = iter.dropWhile(Function.tupled((fi, _, _) => {
          val goodNext = fi - prev == 1
          prev = fi
          goodNext
        }))
        Iterator.single(!remaining.hasNext)
      }).reduce(_ && _)
      assert(denseAsc, "the original data must be ordered.")

      val numPartitions = trainingData.partitions.length
      val (siMinPP, lcNumSamplesPP) = TreeUtils.getPartitionOffsets(numSamples, numPartitions)
      val trainingData_T = trainingData.mapPartitions(iter => {
        val (metaIter, dataIter) = iter.duplicate
        val fiMin = metaIter.next()._1
        val lcNumFeats = metaIter.length + 1
        val blocksPP = Array.tabulate(numPartitions)(pi => Array.ofDim[Byte](lcNumFeats, lcNumSamplesPP(pi)))
        dataIter.foreach(Function.tupled((fi, samples, _) => {
          val lfi = fi - fiMin
          var pi = 0
          while (pi < numPartitions) {
            Array.copy(samples, siMinPP(pi), blocksPP(pi)(lfi), 0, lcNumSamplesPP(pi))
            pi += 1
          }
        }))
        Range(0, numPartitions).iterator.map(pi => (pi, (fiMin, blocksPP(pi))))
      }).partitionBy(new HashPartitioner(numPartitions)).mapPartitionsWithIndex((pid, iter) => {
        val siMin = siMinPP(pid)
        val sampleSlice = new Array[Array[Byte]](numFeats)
        iter.foreach { case (_, (fiMin, blocks)) =>
          var lfi = 0
          while (lfi < blocks.length) {
            sampleSlice(lfi + fiMin) = blocks(lfi)
            lfi += 1
          }
        }
        Iterator.single((siMin, sampleSlice))
      }, preservesPartitioning=true)
      trainingData_T.persist(StorageLevel.MEMORY_AND_DISK).setName("trainingData_T")

      val savDir = path + ".sav"
      val savPath = new Path(savDir)
      val cpmgPath = new Path(path + ".cpmg")
      fs.delete(savPath, true)
      trainingData_T.mapPartitions(iter => {
        val sb = new StringBuilder
        val (siMin, sampleSlice) = iter.next()
        val lcNumSamples = sampleSlice(0).length
        Range(0, lcNumSamples).iterator.map(lsi => {
          sb.clear()
          val si = lsi + siMin
          sb ++= s"$si#${sampleSlice(0)(lsi)}"
          var fi = 1
          while (fi < numFeats) {
            sb ++= s",${sampleSlice(fi)(lsi)}"
            fi += 1
          }
          sb.mkString
        })
      }).saveAsTextFile(savDir)
      fs.delete(cpmgPath, true)
      var suc = FileUtil.copyMerge(fs, savPath, fs, cpmgPath, true, sc.hadoopConfiguration, null)
      if (suc) {
        suc = fs.rename(cpmgPath, new Path(path))
      }
      fs.delete(cpmgPath, true)
      if (!suc) {
        throw new IOException("Save error!")
      }
      println("transposed data generated and saved.")

      trainingData_T
    } else {
      val trainingData_T = sc.textFile(path).mapPartitions(iter => {
        val (metaIter, dataIter) = iter.duplicate
        val siMin = metaIter.next().split("#", 2)(0).toInt
        val lcNumSamples = metaIter.length + 1
        val sampleSlice = Array.ofDim[Byte](numFeats, lcNumSamples)
        dataIter.foreach(line => {
          val parts = line.split("#", 2)
          val si = parts(0).toInt
          val lsi = si - siMin
          val feats = parts(1).split(',').map(_.toByte)
          assert(feats.length == numFeats)
          var fi = 0
          while (fi < numFeats) {
            sampleSlice(fi)(lsi) = feats(fi)
            fi += 1
          }
        })
        Iterator.single((siMin, sampleSlice))
      }, preservesPartitioning=true)
      trainingData_T.persist(StorageLevel.MEMORY_AND_DISK).setName("trainingData_T").count()
      trainingData_T
    }
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
