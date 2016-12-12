package org.apache.spark.examples.mllib

import org.apache.spark.mllib.dataSet.dataSetLoader
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.tree.DerivativeCalculator
import org.apache.spark.mllib.tree.config.Algo
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.tree.model.ensemblemodels.GradientBoostedDecisionTreesModel
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import scopt.OptionParser

/**
  * Created by v-cuili on 10/17/2016.
  */
object Predictor {
  class Params(var trainingData: String = null,
               var queryBoundy: String = null,
               var label: String = null,
               var initScores: String = null,
               var testData: String = null,
               var testQueryBound: String = null,
               var testLabel: String = null,
               var validationData: String = null,
               var queryBoundyValidate: String = null,
               var initScoreValidate: String = null,
               var labelValidate: String = null,
               var featureNoToFriendlyName: String = null,
               var outputTreeEnsemble: String = null,
               var expandTreeEnsemble: Boolean = false,
               var featureIniFile: String = null,
               var gainTableStr: String = null,
               var algo: String = "LambdaMart",
               var learningStrategy: String = "sgd",
               var maxDepth: Array[Int] = null,
               var numLeaves: Int = 0,
               var numPruningLeaves: Array[Int] = null,
               var numIterations: Array[Int] = null,
               var maxSplits: Int = 128,
               var learningRate: Array[Double] = null,
               var minInstancesPerNode: Array[Int] = null,
               var testSpan: Int = 0,
               var sampleFeaturePercent: Double = 1.0,
               var sampleQueryPercent: Double = 1.0,
               var sampleDocPercent: Double = 1.0,
               var numPartitions: Int = 160,
               var ffraction: Double = 1.0,
               var sfraction: Double = 1.0,
               var secondaryMS: Double = 0.0,
               var secondaryLE: Boolean = false,
               var sigma: Double = 1.0,
               var distanceWeight2: Boolean = false,
               var baselineAlpha: Array[Double] = null,
               var baselineAlphaFilename: String = null,
               var entropyCoefft: Double = 0.0,
               var featureFirstUsePenalty: Double = 0.0,
               var featureReusePenalty: Double = 0.0,
               var outputNdcgFilename: String = null,
               var active_lambda_learningStrategy: Boolean = false,
               var rho_lambda: Double = 0.5,
               var active_leaves_value_learningStrategy: Boolean = false,
               var rho_leave: Double = 0.5,
               var GainNormalization: Boolean = false,
               var feature2NameFile: String = null,
               var validationSpan: Int = 10,
               var useEarlystop: Boolean = true,
               var secondGainsFileName: String = null,
               var secondaryInverseMaxDcgFileName: String = null,
               var secondGains: Array[Double] = null,
               var secondaryInverseMaxDcg: Array[Double] = null,
               var discountsFilename: String = null,
               var discounts: Array[Double] = null,
               var sampleWeightsFilename: String = null,
               var sampleWeights: Array[Double] = null,
               var baselineDcgsFilename: String = null,
               var baselineDcgs: Array[Double] = null) extends java.io.Serializable {

    override def toString: String = {
      val propertiesStr =
        s"testData = $testData\ntestQueryBound = $testQueryBound\ntestLabel = $testLabel\ntestSpan = $testSpan\n" +
        s"sampleFeaturePercent = $sampleFeaturePercent\nsampleQueryPercent = $sampleQueryPercent\nsampleDocPercent = $sampleDocPercent\n" +
        s"outputTreeEnsemble = $outputTreeEnsemble\nfeatureNoToFriendlyName = $featureNoToFriendlyName\nvalidationData = $validationData\n" +
        s"queryBoundyValidate = $queryBoundyValidate\ninitScoreValidate = $initScoreValidate\nlabelValidate = $labelValidate\n" +
        s"expandTreeEnsemble = $expandTreeEnsemble\nfeatureIniFile = $featureIniFile\ngainTableStr = $gainTableStr\n" +
        s"algo = $algo\nmaxDepth = ${maxDepth.mkString(":")}\nnumLeaves = $numLeaves\nnumPruningLeaves = ${numPruningLeaves.mkString(":")}\nnumIterations = ${numIterations.mkString(":")}\nmaxSplits = $maxSplits\n" +
        s"learningRate = ${learningRate.mkString(":")}\nminInstancesPerNode = ${minInstancesPerNode.mkString(":")}\nffraction = $ffraction\nsfraction = $sfraction\n"


      propertiesStr
    }
  }


  def main(args: Array[String]): Unit = {
    val defaultParams = new Params()

    val parser = new OptionParser[Unit]("LambdaMART") {
      head("LambdaMART: an implementation of LambdaMART for FastRank.")

      opt[String]("trainingData") required() foreach { x =>
        defaultParams.trainingData = x
      } text ("trainingData path")
      opt[String]("queryBoundy") optional() foreach { x =>
        defaultParams.queryBoundy = x
      } text ("queryBoundy path")
      opt[String]("label") required() foreach { x =>
        defaultParams.label = x
      } text ("label path to training dataset")
      opt[String]("initScores") optional() foreach { x =>
        defaultParams.initScores = x
      } text (s"initScores path to training dataset. If not given, initScores will be {0 ...}.")

      opt[String]("testData") optional() foreach { x =>
        defaultParams.testData = x
      } text ("testData path")
      opt[String]("testQueryBound") optional() foreach { x =>
        defaultParams.testQueryBound = x
      } text ("test queryBoundy path")
      opt[String]("testLabel") optional() foreach { x =>
        defaultParams.testLabel = x
      } text ("label path to test dataset")

      opt[String]("vd") optional() foreach { x =>
        defaultParams.validationData = x
      } text ("validationData path")
      opt[String]("qbv") optional() foreach { x =>
        defaultParams.queryBoundyValidate = x
      } text ("path to queryBoundy for validation data")
      opt[String]("lv") optional() foreach { x =>
        defaultParams.labelValidate = x
      } text ("path to label for validation data")
      opt[String]("isv") optional() foreach { x =>
        defaultParams.initScoreValidate = x
      } text (s"path to initScore for validation data. If not given, initScores will be {0 ...}.")

      opt[String]("outputTreeEnsemble") required() foreach { x =>
        defaultParams.outputTreeEnsemble = x
      } text ("outputTreeEnsemble path")
      opt[String]("ftfn") optional() foreach { x =>
        defaultParams.featureNoToFriendlyName = x
      } text ("path to featureNoToFriendlyName")
      opt[Boolean]("expandTreeEnsemble") optional() foreach { x =>
        defaultParams.expandTreeEnsemble = x
      } text (s"expandTreeEnsemble")
      opt[String]("featureIniFile") optional() foreach { x =>
        defaultParams.featureIniFile = x
      } text (s"path to featureIniFile")
      opt[String]("gainTableStr") required() foreach { x =>
        defaultParams.gainTableStr = x
      } text (s"gainTableStr parameters")
      opt[String]("algo") optional() foreach { x =>
        defaultParams.algo = x
      } text (s"algorithm (${Algo.values.mkString(",")}), default: ${defaultParams.algo}")
      opt[String]("maxDepth") optional() foreach { x =>
        defaultParams.maxDepth = x.split(":").map(_.toInt)
      } text (s"max depth of the tree, default: ${defaultParams.maxDepth}")

      opt[Int]("numLeaves") optional() foreach { x =>
        defaultParams.numLeaves = x
      } text (s"num of leaves per tree, default: ${defaultParams.numLeaves}. Take precedence over --maxDepth.")
      opt[String]("numPruningLeaves") optional() foreach { x =>
        defaultParams.numPruningLeaves = x.split(":").map(_.toInt)
      } text (s"num of leaves per tree after pruning, default: ${defaultParams.numPruningLeaves}.")
      opt[String]("numIterations") optional() foreach { x =>
        defaultParams.numIterations = x.split(":").map(_.toInt)
      } text (s"number of iterations of boosting," + s" default: ${defaultParams.numIterations}")
      opt[String]("minInstancesPerNode") optional() foreach { x =>
        defaultParams.minInstancesPerNode = x.split(":").map(_.toInt)
      } text (s"the minimum number of documents allowed in a leaf of the tree, default: ${defaultParams.minInstancesPerNode}")
      opt[Int]("maxSplits") optional() foreach { x =>
        defaultParams.maxSplits = x
      } text (s"max Nodes to be split simultaneously, default: ${defaultParams.maxSplits}") validate { x =>
        if (x > 0 && x <= 512) success else failure("value <maxSplits> incorrect; should be between 1 and 512.")
      }
      opt[String]("learningRate") optional() foreach { x =>
        defaultParams.learningRate = x.split(":").map(_.toDouble)
      } text (s"learning rate of the score update, default: ${defaultParams.learningRate}")
      opt[Int]("testSpan") optional() foreach { x =>
        defaultParams.testSpan = x
      } text (s"test span")
      opt[Int]("numPartitions") optional() foreach { x =>
        defaultParams.numPartitions = x
      } text (s"number of partitions, default: ${defaultParams.numPartitions}")
      opt[Double]("sampleFeaturePercent") optional() foreach { x =>
        defaultParams.sampleFeaturePercent = x
      } text (s"global feature percentage used for training")
      opt[Double]("sampleQueryPercent") optional() foreach { x =>
        defaultParams.sampleQueryPercent = x
      } text (s"global query percentage used for training")
      opt[Double]("sampleDocPercent") optional() foreach { x =>
        defaultParams.sampleDocPercent = x
      } text (s"global doc percentage used for classification")
      opt[Double]("ffraction") optional() foreach { x =>
        defaultParams.ffraction = x
      } text (s"feature percentage used for training for each tree")
      opt[Double]("sfraction") optional() foreach { x =>
        defaultParams.sfraction = x
      } text (s"sample percentage used for training for each tree")
      opt[Double]("secondaryMS") optional() foreach { x =>
        defaultParams.secondaryMS = x
      } text (s"secondaryMetricShare")
      opt[Boolean]("secondaryLE") optional() foreach { x =>
        defaultParams.secondaryLE = x
      } text (s"secondaryIsoLabelExclusive")
      opt[Double]("sigma") optional() foreach { x =>
        defaultParams.sigma = x
      } text (s"parameter for init sigmoid table")
      opt[Boolean]("dw") optional() foreach { x =>
        defaultParams.distanceWeight2 = x
      } text (s"Distance weight 2 adjustment to cost")
      opt[String]("bafn") optional() foreach { x =>
        defaultParams.baselineAlphaFilename = x
      } text (s"Baseline alpha for tradeoffs of risk (0 is normal training)")
      opt[Double]("entropyCoefft") optional() foreach { x =>
        defaultParams.entropyCoefft = x
      } text (s"The entropy (regularization) coefficient between 0 and 1")
      opt[Double]("ffup") optional() foreach { x =>
        defaultParams.featureFirstUsePenalty = x
      } text (s"The feature first use penalty coefficient")
      opt[Double]("frup") optional() foreach { x =>
        defaultParams.featureReusePenalty = x
      } text (s"The feature re-use penalty (regularization) coefficient")
      opt[String]("learningStrategy") optional() foreach { x =>
        defaultParams.learningStrategy = x
      } text (s"learningStrategy for adaptive gradient descent")
      opt[String]("oNDCG") optional() foreach { x =>
        defaultParams.outputNdcgFilename = x
      } text (s"save ndcg of training phase in this file")
      opt[Boolean]("allr") optional() foreach { x =>
        defaultParams.active_lambda_learningStrategy = x
      } text (s"active lambda learning strategy or not")
      opt[Double]("rhol") optional() foreach { x =>
        defaultParams.rho_lambda = x
      } text (s"rho lambda value")
      opt[Boolean]("alvst") optional() foreach { x =>
        defaultParams.active_leaves_value_learningStrategy = x
      } text (s"active leave value learning strategy or not")
      opt[Double]("rholv") optional() foreach { x =>
        defaultParams.rho_leave = x
      } text (s"rho parameter for leave value learning strategy")
      opt[Boolean]("gn") optional() foreach { x =>
        defaultParams.GainNormalization = x
      } text (s"normalize the gian value in the comment or not")
      opt[String]("f2nf") optional() foreach { x =>
        defaultParams.feature2NameFile = x
      } text (s"path to feature to name map file")
      opt[Int]("vs") optional() foreach { x =>
        defaultParams.validationSpan = x
      } text (s"validation span")
      opt[Boolean]("es") optional() foreach { x =>
        defaultParams.useEarlystop = x
      } text (s"apply early stop or not")
      opt[String]("sgfn") optional() foreach { x =>
        defaultParams.secondGainsFileName = x
      }
      opt[String]("simdfn") optional() foreach { x =>
        defaultParams.secondaryInverseMaxDcgFileName = x
      }
      opt[String]("swfn") optional() foreach { x =>
        defaultParams.sampleWeightsFilename = x
      }
      opt[String]("bdfn") optional() foreach { x =>
        defaultParams.baselineDcgsFilename = x
      }
      opt[String]("dfn") optional() foreach { x =>
        defaultParams.discountsFilename = x
      }
    }
    parser.parse(args)
    run(defaultParams)
  }

  def run(params: Params){
    val conf = new SparkConf().setAppName("GradientBoostedTreesRegressionExample")
    val sc = new SparkContext(conf)

    val model = GradientBoostedTreesModel.load(sc, params.outputTreeEnsemble)

    val gainTable = params.gainTableStr.split(':').map(_.toDouble)

    if (params.algo == "LambdaMart" && params.testSpan != 0) {
      val testNDCG = testModel(sc, model, params, gainTable)
      println(s"testNDCG error 0 = " + testNDCG(0))
      for (i <- 1 until testNDCG.length) {
        val it = i * params.testSpan
        println(s"testNDCG error $it = " + testNDCG(i))
      }
    }
  }

  def loadTestData(sc: SparkContext, path: String): RDD[Vector] = {
    sc.textFile(path).map { line =>
      if (line.contains("#"))
        Vectors.dense(line.split("#")(1).split(",").map(_.toDouble))
      else
        Vectors.dense(line.split(",").map(_.toDouble))
    }
  }

  def testModel(sc: SparkContext, model: GradientBoostedTreesModel, params: Params, gainTable: Array[Double]): Array[Double] = {
    val testData = loadTestData(sc, params.testData).cache().setName("TestData")
    println(s"numTestFeature: ${testData.first().toArray.length}")
    val numTest = testData.count()
    println(s"numTest: $numTest")
    val testLabels = dataSetLoader.loadlabelScores(sc, params.testLabel)

    println(s"numTestLabels: ${testLabels.length}")
    require(testLabels.length == numTest, s"lengthOfLabels: ${testLabels.length} != numTestSamples: $numTest")
    val testQueryBound = dataSetLoader.loadQueryBoundy(sc, params.testQueryBound)
    require(testQueryBound.last == numTest, s"TestQueryBoundy ${testQueryBound.last} does not match with test data $numTest!")

    val rate = params.testSpan
    val predictions = testData.map { features =>
      val scores = model.trees.map(_.predict(features))
      for (it <- 1 until model.trees.length) {
        scores(it) += scores(it - 1)
      }

      scores.zipWithIndex.collect {
        case (score, it) if it == 0 || (it + 1) % rate == 0 => score
      }
    }
    val predictionsByIter = predictions.zipWithIndex.flatMap {
      case (row, rowIndex) => row.zipWithIndex.map {
        case (number, columnIndex) => columnIndex ->(rowIndex, number)
      }
    }.groupByKey.sortByKey().values
      .map {
        indexedRow => indexedRow.toArray.sortBy(_._1).map(_._2)
      }

    val learningRates = params.learningRate
    val distanceWeight2 = params.distanceWeight2
    val baselineAlpha = params.baselineAlpha
    val secondMs = params.secondaryMS
    val secondLe = params.secondaryLE
    val secondGains = params.secondGains
    val secondaryInverseMacDcg = params.secondaryInverseMaxDcg
    val discounts = params.discounts
    val baselineDcgs = params.baselineDcgs
    val dc = new DerivativeCalculator
    dc.init(testLabels, gainTable, testQueryBound,
      learningRates(0), distanceWeight2, baselineAlpha,
      secondMs, secondLe, secondGains, secondaryInverseMacDcg, discounts, baselineDcgs)
    val numQueries = testQueryBound.length - 1
    val dcBc = sc.broadcast(dc)

    predictionsByIter.map { scores =>
      val dc = dcBc.value
      dc.getPartErrors(scores, 0, numQueries) / numQueries
    }.collect()

  }


}
