package org.apache.spark.mllib.tree

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impl.TimeTracker
import org.apache.spark.mllib.tree.impurity.Variance
//import org.apache.spark.mllib.tree.model.{DecisionTreeModel, GradientBoostedTreesModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.tree.model.SplitInfo
import org.apache.spark.mllib.tree.model.GetDerivatives
import org.apache.spark.mllib.tree.model.opdtmodel.OptimizedDecisionTreeModel
import org.apache.spark.mllib.tree.model.ensemblemodels.GradientBoostedDecisionTreesModel

@Experimental
class LambdaMART(private val boostingStrategy: BoostingStrategy)
  extends Serializable with Logging {
  
  def run(
      trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
      labelScores: Array[Short],
      initScores: Array[Double],
      queryBoundy: Array[Int]): GradientBoostedDecisionTreesModel = {
    val algo = boostingStrategy.treeStrategy.algo
    algo match {
      case Regression =>
        LambdaMART.boost(trainingData, labelScores, initScores, queryBoundy, boostingStrategy)
      case _ =>
        throw new IllegalArgumentException(s"$algo is not supported by the implementation of LambdaMART.")
    }
  }
}

object LambdaMART extends Logging {

  def train(
      trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
      labelScores: Array[Short],
      initScores: Array[Double],
      queryBoundy: Array[Int],
      boostingStrategy: BoostingStrategy): GradientBoostedDecisionTreesModel = {
    new LambdaMART(boostingStrategy).run(trainingData, labelScores, initScores, queryBoundy)
  }
  
  private def boost(
      trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
      labelScores: Array[Short],
      initScores: Array[Double],
      queryBoundy: Array[Int],
      boostingStrategy: BoostingStrategy): GradientBoostedDecisionTreesModel = {
    val timer = new TimeTracker()
    timer.start("total")
    timer.start("init")

    boostingStrategy.assertValid()

    // Initialize gradient boosting parameters
    val numIterations = boostingStrategy.numIterations
    val baseLearners = new Array[OptimizedDecisionTreeModel](numIterations)
    val baseLearnerWeights = new Array[Double](numIterations)
    val loss = boostingStrategy.loss
    val learningRate = boostingStrategy.learningRate
    
    // Prepare strategy for individual trees, which use regression with variance impurity.
    val treeStrategy = boostingStrategy.treeStrategy.copy
    val validationTol = boostingStrategy.validationTol
    treeStrategy.algo = Regression
    treeStrategy.impurity = Variance
    treeStrategy.assertValid()

    // Cache trainingData
    val persistedInput = if (trainingData.getStorageLevel == StorageLevel.NONE) {
      trainingData.persist(StorageLevel.MEMORY_AND_DISK)
      true
    } else {
      false
    }

    timer.stop("init")
    // calculate discount

    val maxDocument = labelScores.length
    var aDiscount = Array.tabulate(maxDocument) { index =>
        1.0/scala.math.log(1.0 + index.toDouble + 1.0)
    }

    var aSecondaryGains = new Array[Double](labelScores.length)
    var aGainLabels = labelScores.map(_.toDouble)

    var aLabels = Array.tabulate(labelScores.length) { index =>
        (scala.math.log(labelScores(index))/scala.math.log(2)).toShort
    }
    
    var sigmoidTable = GetDerivatives.FillSigmoidTable()
    
    var lambdas = new Array[Double](labelScores.length)
    var weights = new Array[Double](labelScores.length)

    for(i <- 0 until (queryBoundy.length-1)) {
      var numDocuments = queryBoundy(i+1) - queryBoundy(i)
      var begin = queryBoundy(i)
      var aPermutation = GetDerivatives.sortArray(initScores, begin, numDocuments)

      var gainLabelSortArr = GetDerivatives.labelSort(aGainLabels, begin, numDocuments)
      var inverseMaxDCG: Double = 0.0
      for(i <- 0 until numDocuments) {
          inverseMaxDCG += gainLabelSortArr(i)* aDiscount(i)
      }
      
      inverseMaxDCG = if(inverseMaxDCG != 0.0) 1/inverseMaxDCG else 0.0
      
      GetDerivatives.GetDerivatives_lambda_weight(
        numDocuments, begin,
        aPermutation, aLabels, initScores, lambdas, weights,
        aDiscount, aGainLabels, inverseMaxDCG,
        sigmoidTable, GetDerivatives._minScore, GetDerivatives._maxScore, 
        GetDerivatives._scoreToSigmoidTableFactor, aSecondaryGains)
    }
    var targetScores = lambdas
    
    var m = 0  
    while (m < numIterations) {
      timer.start(s"building tree $m")
      logDebug("###################################################")
      logDebug("Gradient boosting tree iteration " + m)
      logDebug("###################################################")
      val (model, residualScores, derivativeWeights) = new LambdaMARTDecisionTree(treeStrategy).run(trainingData, targetScores, labelScores, queryBoundy, weights)
      timer.stop(s"building tree $m")
      
      model.sequence("D:\\spark-1.4.0-bin-hadoop2.6\\LambdaMART-v1\\dt.model", model, m + 1)

      baseLearners(m) = model
      baseLearnerWeights(m) = learningRate
        
      logDebug("error of gbt = " + residualScores.map( re => re * re).sum / residualScores.size)
      
      m += 1
      targetScores = residualScores
      weights = derivativeWeights
    }
  
    timer.stop("total")

    logInfo("Internal timing for LambdaMARTDecisionTree:")
    logInfo(s"$timer")

    if (persistedInput) trainingData.unpersist()
    
    new GradientBoostedDecisionTreesModel(
      boostingStrategy.treeStrategy.algo, baseLearners, baseLearnerWeights)
  }
}
