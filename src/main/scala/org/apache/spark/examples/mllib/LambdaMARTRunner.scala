package org.apache.spark.examples.mllib

import breeze.linalg.SparseVector
import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.tree.configuration._
import org.apache.spark.mllib.tree.model.SplitInfo
import org.apache.spark.mllib.tree.model.ensemblemodels.GradientBoostedDecisionTreesModel
import org.apache.spark.mllib.tree.{DerivativeCalculator, LambdaMART}
import org.apache.spark.mllib.util.{TreeUtils, treeAggregatorFormat}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import scopt.OptionParser

import scala.language.reflectiveCalls
import scala.util.Random


object LambdaMARTRunner {
  case class Params(trainingData: String = null,
                    testData: String = null,
                    queryBoundy: String = null,
                    testQueryBound: String = null,
                    outputTreeEnsemble: String = null,
                    labelScores: String = null,
                    testLabel: String = null,
                    initScores: String = null,
                    initialTreeEnsemble: String = null,
                    featureIniFile: String = null,
                    gainTableStr: String = null,
                    algo: String = "Regression",
                    learningStrategy: String = "sgd",
                    maxDepth: Int = 8,
                    numLeaves: Int = 0,
                    numIterations: Int = 10,
                    maxSplits: Int = 128,
                    learningRate: Double = 1.0,
                    minInstancesPerNode: Int = 2000,
                    testRate: Int = 0,
                    sampleFeaturePercent: Double = 1.0,
                    sampleQueryPercent: Double = 1.0) extends AbstractParams[Params]

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
      opt[String]("outputTreeEnsemble")
        .text("outputTreeEnsemble path")
        .required()
        .action((x, c) => c.copy(outputTreeEnsemble = x))
      opt[String]("labelScores")
        .text("labelScores path to training dataset")
        .required()
        .action((x, c) => c.copy(labelScores = x))
      opt[String]("initScores")
        .text(s"initScores path to training dataset. If not given, initScores will be {0 ...}.")
        .action((x, c) => c.copy(initScores = x))
	  opt[String]("initialTreeEnsemble")
        .text(s"path to initialTreeEnsemble")
        .action((x, c) => c.copy(initialTreeEnsemble = x))
      opt[String]("featureIniFile")
        .text(s"path to featureIniFile")
        .action((x, c) => c.copy(featureIniFile = x))
      opt[String]("gainTableStr")
        .text(s"gainTableStr parameters")
        .action((x, c) => c.copy(gainTableStr = x))
      opt[String]("algo")
        .text(s"algorithm (${Algo.values.mkString(",")}), default: ${defaultParams.algo}")
        .action((x, c) => c.copy(algo = x))
      opt[String]("learningStrategy")
        .text(s"algorithm to adjust learning rate, default: ${defaultParams.learningStrategy}")
        .action((x, c) => c.copy(learningStrategy = x))
      opt[Int]("maxDepth")
        .text(s"max depth of the tree, default: ${defaultParams.maxDepth}")
        .action((x, c) => c.copy(maxDepth = x))
      opt[Int]("numLeaves")
        .text(s"num of leaves per tree, default: ${defaultParams.numLeaves}. Take precedence over --maxDepth.")
        .action((x, c) => c.copy(numLeaves = x))
      opt[Int]("numIterations")
        .text(s"number of iterations of boosting," + s" default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("minInstancesPerNode")
        .text(s"the minimum number of documents allowed in a leaf of the tree, default: ${defaultParams.minInstancesPerNode}")
        .action((x, c) => c.copy(minInstancesPerNode = x))
      opt[Int]("maxSplits")
        .text(s"max Nodes to be split simultaneously, default: ${defaultParams.maxSplits}")
        .action((x, c) => c.copy(maxSplits = x))
      opt[Double]("learningRate")
        .text(s"learning rate of the score update, default: ${defaultParams.learningRate}")
        .action((x, c) => c.copy(learningRate = x))
      opt[String]("testQueryBound")
        .text("test queryBoundy path")
        .action((x, c) => c.copy(testQueryBound = x))
      opt[String]("testLabel")
        .text("test labelScores path to training dataset")
        .action((x, c) => c.copy(testLabel = x))
      opt[Int]("testRate")
        .text(s"frequecy of test NDCG, default: ${defaultParams.testRate}")
        .action((x, c) => c.copy(testRate = x))
      opt[Double]("sampleFeaturePercent")
        .text(s"percentage of sampling feature, default: ${defaultParams.sampleFeaturePercent}")
        .action((x, c) => c.copy( sampleFeaturePercent= x))
      opt[Double]("sampleQueryPercent")
        .text(s"percentage of sampling queries, default: ${defaultParams.sampleQueryPercent}")
        .action((x, c) => c.copy( sampleQueryPercent= x))

      checkConfig(params =>
        if (params.maxDepth > 30) {
          failure(s"maxDepth ${params.maxDepth} value incorrect; should be less than or equals to 30.")
        } else if (params.maxSplits > 128 || params.maxSplits <= 0) {
          failure(s"maxSplits ${params.maxSplits} value incorrect; should be between 1 and 128.")
        } else if(params.testRate!=0 && params.testQueryBound==null) {
          failure(s"testQueryBound missing")
        }else if(params.testRate!=0 && params.testLabel==null) {
          failure(s"testLabel missing")
        }else if(params.sampleFeaturePercent>1.0 || params.sampleFeaturePercent<=0.0){
          failure(s"sampleFeaturePercent ${params.sampleFeaturePercent} invalid")
        }else if(params.sampleQueryPercent>1.0 || params.sampleQueryPercent<=0.0){
          failure(s"sampleQueryPercent ${params.sampleQueryPercent} invalid")
        }
        else {
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
      var labelScores = loadlabelScores(sc, params.labelScores)
      val numSamples = labelScores.length
      println(s"numSamples: $numSamples")

      var initScores = if (params.initScores == null) {
        new Array[Double](numSamples)
      } else {
        val loaded = loadInitScores(sc, params.initScores)
        require(loaded.length == numSamples, s"lengthOfInitScores: ${loaded.length} != numSamples: $numSamples")
        loaded
      }
      var queryBoundy = loadQueryBoundy(sc, params.queryBoundy)
      require(queryBoundy.last == numSamples, s"QueryBoundy ${queryBoundy.last} does not match with data $numSamples !")
      val numQuery = queryBoundy.length-1
      println(s"num of data query: $numQuery")

      val numSampleQuery = if(params.sampleQueryPercent<1) (numQuery * params.sampleQueryPercent).toInt else numQuery
      println(s"num of sampling query: $numSampleQuery")
      val sampleQueryId = if(params.sampleQueryPercent<1) {
        (new Random(Random.nextInt)).shuffle((0 until queryBoundy.length - 1).toList).take(numSampleQuery).toArray
      }else null//query index for training

      if(params.sampleQueryPercent<1) { // sampling
        labelScores = getSampleLabels(sampleQueryId, queryBoundy, labelScores)
        println(s"num of sampling labels: ${labelScores.length}")
        initScores = getSampleInitScores(sampleQueryId, queryBoundy, initScores, labelScores.length)
        require(labelScores.length==initScores.length, s"num of labels ${labelScores.length} does not match with initScores ${initScores.length}!")
      }
      val trainingData = loadTrainingData(sc, params.trainingData, sc.defaultMinPartitions, params.sampleFeaturePercent, sampleQueryId, queryBoundy, labelScores.length)

      if(params.sampleQueryPercent<1) {
        queryBoundy = getSampleQueryBound(sampleQueryId, queryBoundy)
        require(queryBoundy.last == labelScores.length, s"QueryBoundy ${queryBoundy.last} does not match with data ${labelScores.length} !")
      }
      val numFeats = trainingData.count().toInt
      println(s"numFeats: $numFeats")
      val trainingData_T = genTransposedData(trainingData, numFeats, labelScores.length)
      val gainTable = params.gainTableStr.split(':').map(_.toDouble)

      val boostingStrategy = BoostingStrategy.defaultParams(params.algo)
      boostingStrategy.treeStrategy.setMaxDepth(params.maxDepth)
      boostingStrategy.setNumIterations(params.numIterations)
      boostingStrategy.setLearningRate(params.learningRate)
      boostingStrategy.treeStrategy.setMinInstancesPerNode(params.minInstancesPerNode)


      if (params.algo == "Regression") {
        val startTime = System.nanoTime()
        val model = LambdaMART.train(trainingData, trainingData_T, labelScores, initScores, queryBoundy, gainTable,
          boostingStrategy, params.numLeaves, params.maxSplits, params.initialTreeEnsemble, params.learningStrategy)
        val elapsedTime = (System.nanoTime() - startTime) / 1e9
        println(s"Training time: $elapsedTime seconds")

        // test
        if(params.testRate!=0) {
          val testNDCG = testModel(sc, model, params, gainTable)
          for(i<- 0 until testNDCG.length){
            val it = i*params.testRate
            println(s"testNDCG error $it: "+ testNDCG(i))
          }
        }

        if(params.featureIniFile!=null) {
          val featureIniPath = new Path(params.featureIniFile)
          val featurefs = TreeUtils.getFileSystem(trainingData.context.getConf, featureIniPath)
          featurefs.copyToLocalFile(false, featureIniPath, new Path("treeEnsemble.ini"))
        }

        for(i <- 0 until model.trees.length){
          val evaluator = model.trees(i)
          evaluator.sequence("treeEnsemble.ini", evaluator, i + 1)
        }
        println(s"save succeed")
        val totalEvaluators = model.trees.length
        val evalNodes = Array.tabulate[Int](totalEvaluators)(_ + 1)
        treeAggregatorFormat.appendTreeAggregator(params.initialTreeEnsemble, "treeEnsemble.ini", totalEvaluators + 1, evalNodes)

        val outPath = new Path(params.outputTreeEnsemble)
        val fs = TreeUtils.getFileSystem(trainingData.context.getConf, outPath)
        fs.copyFromLocalFile(false, true, new Path("treeEnsemble.ini"), outPath)

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

  def loadTrainingData(sc: SparkContext, path: String, minPartitions: Int,
                       sampleFeatPercentage: Double = 1.0, sampleQueryId: Array[Int], QueryBound: Array[Int], numSampling: Int)
  : RDD[(Int, SparseVector[Short], Array[SplitInfo])] = {
    val data = sc.textFile(path, minPartitions).map { line =>
      val parts = line.split("#")
      val feat = parts(0).toInt
      val samples = parts(1).split(',').map(_.toShort)  // input samples
      var is = 0
      // sampling data
      val samplingData = if(sampleQueryId==null) samples else{
        val sd = new Array[Short](numSampling)
        var it = 0
        var icur = 0
        while(it<sampleQueryId.length){
          val query = sampleQueryId(it)
          for(is <- QueryBound(query) until QueryBound(query+1)){
            sd(icur)=samples(is)
            icur+=1
          }
          it+=1
        }
        sd
      }
      // Sparse data
      is = 0
      var nnz = 0
      while (is < samplingData.length) {
        if (samplingData(is) != 0) {
          nnz += 1
        }
        is += 1
      }
      val idx = new Array[Int](nnz)
      val vas = new Array[Short](nnz)
      is = 0
      nnz = 0
      while (is < samplingData.length) {
        if (samplingData(is) != 0) {
          idx(nnz) = is
          vas(nnz) = samplingData(is)
          nnz += 1
        }
        is += 1
      }
      val sparseSamples = new SparseVector[Short](idx, vas, nnz, is)

      val splits = if (parts.length > 2) {
        parts(2).split(',').map(threshold => new SplitInfo(feat, threshold.toDouble))
      } else {
        val maxFeat = samples.max + 1
        Array.tabulate(maxFeat)(threshold => new SplitInfo(feat, threshold))
      }
      (feat, sparseSamples, splits)
    }

    val trainingData = if(sampleFeatPercentage<1) data.sample(false, sampleFeatPercentage).zipWithIndex().map{
      case((fid,samples, splits),id) => (id.toInt, samples, splits)
    } else data

    trainingData.persist(StorageLevel.MEMORY_AND_DISK).setName("trainingData").count()
    trainingData
  }
  def loadTestData(sc: SparkContext, path: String): RDD[Vector] = {
    sc.textFile(path).map{line => Vectors.dense(line.split(',').map(_.toDouble))
    }
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

  def genTransposedData(trainingData: RDD[(Int, SparseVector[Short], Array[SplitInfo])],
    numFeats: Int,
    numSamples: Int): RDD[(Int, Array[Array[Short]])] = {
    println("generating transposed data...")
    // validate that the original data is ordered
    val denseAsc = trainingData.mapPartitions { iter =>
      var res = Iterator.single(true)
      if(iter.hasNext) {
        var prev = iter.next()._1
        val remaining = iter.dropWhile { case (fi, _, _) =>
          val goodNext = fi - prev == 1
          prev = fi
          goodNext
        }
        res = Iterator.single(!remaining.hasNext)
      }
      res
    }.reduce(_ && _)
    assert(denseAsc, "the original data must be ordered.")
    println("pass data check in transposing")

    val numPartitions = trainingData.partitions.length
    val (siMinPP, lcNumSamplesPP) = TreeUtils.getPartitionOffsets(numSamples, numPartitions)
    val trainingData_T = trainingData.mapPartitions { iter =>
      val (metaIter, dataIter) = iter.duplicate
      val fiMin = metaIter.next()._1
      val lcNumFeats = metaIter.length + 1
      val blocksPP = Array.tabulate(numPartitions)(pi => Array.ofDim[Short](lcNumFeats, lcNumSamplesPP(pi)))
      dataIter.foreach { case (fi, sparseSamples, _) =>
        val samples = sparseSamples.toArray
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
  def getSampleLabels(testQueryId: Array[Int], QueryBound: Array[Int], labels: Array[Short]): Array[Short]={
    println("parse test labels")
    val testLabels = new Array[Short](labels.length)
    var it = 0
    var icur = 0
    while(it<testQueryId.length){
      val query = testQueryId(it)
      for(is <- QueryBound(query) until QueryBound(query+1)){
        testLabels(icur)=labels(is)
        icur+=1
      }
      it+=1
    }
    testLabels.dropRight(labels.length - icur)
  }
  def getSampleInitScores(trainingQueryId: Array[Int], QueryBound: Array[Int], scores: Array[Double], len:Int): Array[Double]={
    println("parse init scores")
    val trainingScores = new Array[Double](len)
    var it = 0
    var icur = 0
    while(it<trainingQueryId.length){
      val query = trainingQueryId(it)
      for(is <- QueryBound(query) until QueryBound(query+1)){
        trainingScores(icur)=scores(is)
        icur+=1
      }
      it+=1
    }
    trainingScores
  }
  def getSampleQueryBound(QueryId: Array[Int], queryBoundy: Array[Int]):Array[Int]={
    println("get query bound")
    val res = new Array[Int](QueryId.length+1)
    res(0) = 0
    var qid = 0
    while(qid<QueryId.length){
      res(qid+1)=res(qid)+queryBoundy(QueryId(qid)+1)-queryBoundy(QueryId(qid))
      qid+=1
    }
    res
  }

  def testModel(sc: SparkContext, model: GradientBoostedDecisionTreesModel, params: Params, gainTable:Array[Double]):Array[Double]={
    val testData = loadTestData(sc, params.testData).cache().setName("TestData")
    val numTest = testData.count()
    println(s"numTest: $numTest")
    val testLabels = loadlabelScores(sc, params.testLabel)
    println(s"numTestLabels: ${testLabels.length}")
    require(testLabels.length == numTest, s"lengthOfLabels: ${testLabels.length} != numTestSamples: $numTest")
    val testQueryBound = loadQueryBoundy(sc, params.testQueryBound)
    require(testQueryBound.last == numTest, s"TestQueryBoundy ${testQueryBound.last} does not match with test data $numTest!")

    val rate = params.testRate
    val predictions = testData.map { features =>
      val scores = model.trees.map(_.predict(features))
      for (it <- 1 until model.trees.length) {
        scores(it) += scores(it - 1)
      }

      scores.zipWithIndex.collect {
        case (score, it) if it % rate == 0 => score
      }
    }.collect().transpose

    val dc = new DerivativeCalculator
    dc.init(testLabels, gainTable,  testQueryBound)
    val numQueries = testQueryBound.length - 1
    val (qiMinPP, lcNumQueriesPP) = TreeUtils.getPartitionOffsets(numQueries, sc.defaultParallelism)
    val pdcRDD = sc.parallelize(qiMinPP.zip(lcNumQueriesPP)).cache().setName("testPDCCtrl")

    val dcBc = sc.broadcast(dc)
    predictions.map{scores =>
      val currentScoresBc = sc.broadcast(scores)
      val sumErrors = pdcRDD.mapPartitions { iter =>
        val dc = dcBc.value
        val currentScores = currentScoresBc.value
        iter.map { case (qiMin, lcNumQueries) =>
          dc.getPartErrors(currentScores, qiMin, qiMin + lcNumQueries)
        }
      }.sum()
      currentScoresBc.destroy(blocking=false)
      sumErrors / numQueries
    }

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
