package org.apache.spark.mllib.tree

import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.impl.TimeTracker
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.mllib.tree.model.SplitInfo
import org.apache.spark.mllib.tree.model.ensemblemodels.GradientBoostedDecisionTreesModel
import org.apache.spark.mllib.tree.model.opdtmodel.OptimizedDecisionTreeModel
import org.apache.spark.mllib.util.TreeUtils
import org.apache.spark.rdd.RDD


class LambdaMART(val boostingStrategy: BoostingStrategy,
  val numLeaves: Int,
  val maxSplits: Int) extends Serializable with Logging {
  def run(trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
    trainingData_T: RDD[(Int, Array[Array[Byte]])],
    labelScores: Array[Short],
    initScores: Array[Double],
    queryBoundy: Array[Int]): GradientBoostedDecisionTreesModel = {
    val algo = boostingStrategy.treeStrategy.algo
    algo match {
      case Regression =>
        LambdaMART.boost(trainingData, trainingData_T, labelScores, initScores, queryBoundy,
          boostingStrategy, numLeaves, maxSplits)
      case _ =>
        throw new IllegalArgumentException(s"$algo is not supported by the implementation of LambdaMART.")
    }
  }
}

object LambdaMART extends Logging {
  def train(trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
    trainingData_T: RDD[(Int, Array[Array[Byte]])],
    labelScores: Array[Short],
    initScores: Array[Double],
    queryBoundy: Array[Int],
    boostingStrategy: BoostingStrategy,
    numLeaves: Int,
    maxSplits: Int): GradientBoostedDecisionTreesModel = {
    new LambdaMART(boostingStrategy, numLeaves, maxSplits)
      .run(trainingData, trainingData_T, labelScores, initScores, queryBoundy)
  }
  
  private def boost(trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
    trainingData_T: RDD[(Int, Array[Array[Byte]])],
    labelScores: Array[Short],
    initScores: Array[Double],
    queryBoundy: Array[Int],
    boostingStrategy: BoostingStrategy,
    numLeaves: Int,
    maxSplits: Int): GradientBoostedDecisionTreesModel = {
    val timer = new TimeTracker()
    timer.start("total")
    timer.start("init")

    boostingStrategy.assertValid()

    // Initialize gradient boosting parameters
    val numIterations = boostingStrategy.numIterations
    val baseLearners = new Array[OptimizedDecisionTreeModel](numIterations)
    val baseLearnerWeights = new Array[Double](numIterations)
    // val loss = boostingStrategy.loss
    val learningRate = boostingStrategy.learningRate
    
    // Prepare strategy for individual trees, which use regression with variance impurity.
    val treeStrategy = boostingStrategy.treeStrategy.copy
    // val validationTol = boostingStrategy.validationTol
    treeStrategy.algo = Regression
    treeStrategy.impurity = Variance
    treeStrategy.assertValid()

    val sc= trainingData.sparkContext
    val numSamples = labelScores.length
    val numQueries = queryBoundy.length - 1
    val (qiMinPP, lcNumQueriesPP) = TreeUtils.getPartitionOffsets(numQueries, sc.defaultParallelism)
    val pdcRDD = sc.parallelize(qiMinPP.zip(lcNumQueriesPP)).cache().setName("PDCCtrl")

    val dc = new DerivativeCalculator
    dc.init(labelScores, queryBoundy)
    val dcBc = sc.broadcast(dc)
    val lambdas = new Array[Double](numSamples)
    val weights = new Array[Double](numSamples)

    timer.stop("init")

    val currentScores = initScores
    val initErrors = evaluateErrors(pdcRDD, dcBc, currentScores, numQueries)
    println(s"NDCG initError sum = $initErrors")
    var m = 0
    while (m < numIterations) {
      timer.start(s"building tree $m")
      println("\nGradient boosting tree iteration " + m)

      val currentScoresBc = sc.broadcast(currentScores)
      updateDerivatives(pdcRDD, dcBc, currentScoresBc, lambdas, weights)
      currentScoresBc.destroy(blocking=false)

      val lambdasBc = sc.broadcast(lambdas)
      val weightsBc = sc.broadcast(weights)
      val tree = new LambdaMARTDecisionTree(treeStrategy, numLeaves, maxSplits)
      val (model, treeScores) = tree.run(trainingData, trainingData_T, lambdasBc, weightsBc, numSamples)
      lambdasBc.destroy(blocking=false)
      weightsBc.destroy(blocking=false)
      timer.stop(s"building tree $m")

      baseLearners(m) = model
      baseLearnerWeights(m) = learningRate

      Range(0, numSamples).par.foreach(si =>
        currentScores(si) -= learningRate * treeScores(si)
      )
      val errors = evaluateErrors(pdcRDD, dcBc, currentScores, numQueries)
      println(s"NDCG error sum = $errors")
      // println("error of gbt = " + currentScores.iterator.map(re => re * re).sum / numSamples)

      model.sequence("dt.model", model, m + 1)
      m += 1
    }

    timer.stop("total")

    println("Internal timing for LambdaMARTDecisionTree:")
    println(s"$timer")

    trainingData.unpersist(blocking=false)
    trainingData_T.unpersist(blocking=false)

    new GradientBoostedDecisionTreesModel(Regression, baseLearners, baseLearnerWeights)
  }

  def updateDerivatives(pdcRDD: RDD[(Int, Int)],
    dcBc: Broadcast[DerivativeCalculator],
    currentScoresBc: Broadcast[Array[Double]],
    lambdas: Array[Double],
    weights: Array[Double]): Unit = {
    val partDerivs = pdcRDD.mapPartitions { iter =>
      val dc = dcBc.value
      val currentScores = currentScoresBc.value
      iter.map { case (qiMin, lcNumQueries) =>
        dc.getPartDerivatives(currentScores, qiMin, qiMin + lcNumQueries)
      }
    }.collect()
    partDerivs.par.foreach { case (siMin, lcLambdas, lcWeights) =>
      Array.copy(lcLambdas, 0, lambdas, siMin, lcLambdas.length)
      Array.copy(lcWeights, 0, weights, siMin, lcWeights.length)
    }
  }

  def evaluateErrors(pdcRDD: RDD[(Int, Int)],
    dcBc: Broadcast[DerivativeCalculator],
    currentScores: Array[Double],
    numQueries: Int): Double = {
    val sc = pdcRDD.context
    val currentScoresBc = sc.broadcast(currentScores)
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
