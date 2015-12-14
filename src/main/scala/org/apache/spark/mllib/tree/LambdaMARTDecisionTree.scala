package org.apache.spark.mllib.tree

import scala.collection.mutable
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuilder

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.FeatureType._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.impl._
import org.apache.spark.mllib.tree.model.impurity._
//import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.SamplingUtils
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.tree.model.{GetDerivatives, nodePredict}

import org.apache.spark.mllib.tree.model.informationgainstats.InformationGainStats
import org.apache.spark.mllib.tree.model.opdtmodel.OptimizedDecisionTreeModel
import org.apache.spark.mllib.tree.model.predict.Predict
import org.apache.spark.mllib.tree.model.Bin
import org.apache.spark.mllib.tree.model.node._
import org.apache.spark.mllib.tree.model.SplitInfo


@Experimental
class LambdaMARTDecisionTree (private val strategy: Strategy) extends Serializable with Logging {

  strategy.assertValid()

  def run(
      trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
      targetScores: Array[Double],
      labelScores: Array[Short],
      queryBoundy: Array[Int],
      weightsIn: Array[Double]): (OptimizedDecisionTreeModel, Array[Double], Array[Double]) = {

    val timer = new TimeTracker()

    timer.start("total")
     
    // depth of the decision tree
    val maxDepth = strategy.maxDepth
    require(maxDepth <= 30,
      s"LambdaMARTDecisionTree currently only supports maxDepth <= 30, but was given maxDepth = $maxDepth.")
    val maxMemoryUsage: Long = strategy.maxMemoryInMB * 1024L * 1024L
      
    // FIFO queue of nodes to train: node
    val nodeQueue = new mutable.Queue[Node]()
    val topNode: Node = Node.emptyNode(nodeIndex = 1)   
    nodeQueue.enqueue(topNode)
    
    // Create node Id tracker.
    // At first, all the samples belong to the root nodes (node Id == 1).
    val nodeIdTracker = Array.fill[Byte](targetScores.length)(1);
    // TBD re-declared
    val nodeId2Score = new HashMap[Byte, Double]()

    val splitfeatures = mutable.MutableList[String]()
    val splitgain = mutable.MutableList[Double]()
    val threshold = mutable.MutableList[Double]()
 
    
    val targetScoresBc = trainingData.sparkContext.broadcast(targetScores)
    val weightsBc = trainingData.sparkContext.broadcast(weightsIn)

    while (nodeQueue.nonEmpty) {
      val (nodes, nodeId2NodeNo) = LambdaMARTDecisionTree.selectNodesToSplit(nodeQueue, maxMemoryUsage)
      assert(nodes.size > 0,
        s"LambdaMARTDecisionTree selected empty nodes.  Error for unknown reason.")
    
      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      LambdaMARTDecisionTree.findBestSplits(trainingData, targetScoresBc, strategy.maxDepth,
        nodes, nodeId2NodeNo, nodeId2Score, nodeIdTracker, nodeQueue, timer, weightsBc, splitfeatures, splitgain, threshold)
      timer.stop("findBestSplits")
    }



    timer.stop("total")

    logInfo("Internal timing for LambdaMARTDecisionTree:")
    logInfo(s"$timer")



    // calculate lambda array
    val maxDocument = 1024
    var aDiscount = Array.tabulate(maxDocument) { index =>
        1.0/scala.math.log(1.0 + index.toDouble + 1.0)
    }

    var aSecondaryGains = new Array[Double](labelScores.length)
    var aGainLabels = labelScores.map(_.toDouble)
    var aLabels = Array.tabulate(labelScores.length){ index =>
        (scala.math.log(labelScores(index))/scala.math.log(2)).toInt
    }

    var sigmoidTable = GetDerivatives.FillSigmoidTable()

    var lambdas = new Array[Double](labelScores.length)
    var weights = new Array[Double](labelScores.length)

    var sampleScores = Array.tabulate(targetScores.length) { i =>
       nodeId2Score.getOrElse(nodeIdTracker(i), 0.0D)
    }

    for(i <- 0 until (queryBoundy.length-1)) {
      var numDocuments: Int = queryBoundy(i+1) - queryBoundy(i)
      var begin: Int = queryBoundy(i)
      
      var aPermutation = GetDerivatives.sortArray(sampleScores, begin, numDocuments)

      var gainLabelSortArr = GetDerivatives.labelSort(aGainLabels, begin, numDocuments)
      var inverseMaxDCG: Double = 0.0
      for(i <- 0 until numDocuments) {
          inverseMaxDCG += gainLabelSortArr(i)* aDiscount(i)
      }

      inverseMaxDCG = if(inverseMaxDCG != 0.0) 1/inverseMaxDCG else 0.0

      GetDerivatives.GetDerivatives_lambda_weight(
        numDocuments, begin,
        aPermutation, labelScores, sampleScores, lambdas, weights,
        aDiscount, aGainLabels, inverseMaxDCG,
        sigmoidTable, GetDerivatives._minScore, GetDerivatives._maxScore, 
        GetDerivatives._scoreToSigmoidTableFactor, aSecondaryGains)
    }

    val remainingScores = lambdas

    val model = new OptimizedDecisionTreeModel(topNode, strategy.algo, 
                                              splitfeatures, 
                                              splitgain, 
                                              threshold)
    (model, remainingScores, weights)
  }
}

object LambdaMARTDecisionTree extends Serializable with Logging {

  
      
  def selectNodesToSplit(
      nodeQueue: mutable.Queue[Node],
      maxMemoryUsage: Long): (Array[Node], Map[Byte, Byte]) = {
    val mutableNodes = new mutable.ArrayBuffer[Node]()
    val mutableNodeId2NodeNo =
      new mutable.HashMap[Byte, Byte]()
    var memUsage: Long = 0L
    var numNodes = 0
    while (nodeQueue.nonEmpty && memUsage < maxMemoryUsage) {
      val node = nodeQueue.head
      // Check if enough memory remains to add this node to the group.
      val nodeMemUsage = aggregateSizeForNode() * 8L
      if (memUsage + nodeMemUsage <= maxMemoryUsage) {
        nodeQueue.dequeue()
        mutableNodes += node
        mutableNodeId2NodeNo(node.id.toByte) = numNodes.toByte
      }
      memUsage += nodeMemUsage
      numNodes += 1
    }
    // Convert mutable maps to immutable ones.
    (mutableNodes.toArray, mutableNodeId2NodeNo.toMap)    
  }
  
  def aggregateSizeForNode(): Long = {
    // SplitInfo num sum
    3 * 64 * 4096
  }
  
  def findBestSplits(
      trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
      targetScoresBc: Broadcast[Array[Double]],
      maxDepth: Int,
      nodesToSplit: Array[Node],
      nodeId2NodeNo: Map[Byte, Byte],
      nodeId2Score: HashMap[Byte, Double],
      nodeIdTracker: Array[Byte],
      nodeQueue: mutable.Queue[Node],
      timer: TimeTracker,
      weightsBc: Broadcast[Array[Double]], 
      splitfeatures: mutable.MutableList[String],
      splitgain: mutable.MutableList[Double],
      threshold: mutable.MutableList[Double]): Unit = {

    // numNodes:  Number of nodes in this group
    val numNodes = nodesToSplit.size
    logDebug("numNodes = " + numNodes)
    
    def betterSplits(
      a: Array[(SplitInfo, InformationGainStats, Predict)],
      b: Array[(SplitInfo, InformationGainStats, Predict)],
      numNodes: Int
    ) : Array[(SplitInfo, InformationGainStats, Predict)] = {     
      Array.tabulate(numNodes) { nodeNo =>
        List(a(nodeNo), b(nodeNo)).maxBy(_._2.gain)
      }
    }

    // Calculate best splits for all nodes in the group
    timer.start("chooseSplits")
    
    val nodeIdTrackerBc = trainingData.sparkContext.broadcast(nodeIdTracker)
    val nodeId2NodeNoBc = trainingData.sparkContext.broadcast(nodeId2NodeNo)
    val nodesToSplitBc = trainingData.sparkContext.broadcast(nodesToSplit)

    val bestSplitsPerFeature : RDD[Array[(SplitInfo, InformationGainStats, Predict)]] =
      trainingData.map { case (featureIndex, featureValues, splits) =>
        val histograms = Array.tabulate(numNodes) { nodeNo =>
          new FeatureStatsAggregator((splits.length + 1).toByte)
        }
        
        var sampleIndex = 0
        while (sampleIndex < featureValues.length){
          nodeId2NodeNoBc.value.get(nodeIdTrackerBc.value(sampleIndex)) match{
            case Some(i) => histograms(i).update(
              featureValues(sampleIndex), targetScoresBc.value(sampleIndex), 1.0, weightsBc.value(sampleIndex))
            case None =>
          }
          sampleIndex += 1
        }
      
        // TBD, make minInstancesPerNode, minInfoGain parameterized.
        val minInstancesPerNode = 0;
        val minInfoGain = 0;
        Array.tabulate(numNodes) { nodeNo =>
          binsToBestSplit(histograms(nodeNo), splits,
            minInstancesPerNode, minInfoGain, nodesToSplitBc.value(nodeNo))
        }
      }

    val bestSplits = bestSplitsPerFeature.reduce((a, b) => betterSplits(a, b, numNodes))

    timer.stop("chooseSplits")

    // Iterate over all nodes in this group.
    nodesToSplit.foreach { node =>
      val nodeId = node.id
      val nodeNo = nodeId2NodeNo(nodeId.toByte)
      val (split: SplitInfo, stats: InformationGainStats, predict: Predict) =
        bestSplits(nodeNo)
      logDebug("best split = " + split)

      splitfeatures += "I:" + split.feature.toString
      splitgain += stats.gain
      threshold += split.threshold

      // Extract info for this node.  Create children if not leaf.
      val isLeaf = (stats.gain <= 0) || (Node.indexToLevel(nodeId) == maxDepth)
      assert(node.id == nodeId)
      node.predict = predict
      node.isLeaf = isLeaf
      node.stats = Some(stats)
      node.impurity = stats.impurity
      logDebug("Node = " + node)
        
      nodeId2Score(node.id.toByte) = node.predict.predict       //nodePredict.predict(node.id, nodeIdTrackerBc, targetScoresBc, weightsBc)

      if (!isLeaf) {
        node.split = Some(split)
        val childIsLeaf = (Node.indexToLevel(nodeId) + 1) == maxDepth
        val leftChildIsLeaf = childIsLeaf || (stats.leftImpurity == 0.0)
        val rightChildIsLeaf = childIsLeaf || (stats.rightImpurity == 0.0)
        node.leftNode = Some(Node(Node.leftChildIndex(nodeId),
          stats.leftPredict, stats.leftImpurity, leftChildIsLeaf))          
        nodeId2Score(node.leftNode.get.id.toByte) = node.leftNode.get.predict.predict   //nodePredict.predict(node.leftNode.get.id, nodeIdTrackerBc, targetScoresBc, weightsBc)
       
        node.rightNode = Some(Node(Node.rightChildIndex(nodeId),
          stats.rightPredict, stats.rightImpurity, rightChildIsLeaf))           
        nodeId2Score(node.rightNode.get.id.toByte) = node.rightNode.get.predict.predict  //nodePredict.predict(node.rightNode.get.id, nodeIdTrackerBc, targetScoresBc, weightsBc)
       
        // Update nodeIdTracker, the implementation need to be refined here.
        // 1. Get feature values of the best split feature
        val (featureIndex, featureValues, splits) = trainingData.filter(
          it => it._1 == split.feature).first
        var sampleIndex = 0;
        while (sampleIndex < nodeIdTracker.length){
          if (nodeIdTracker(sampleIndex) == nodeId){
            if (featureValues(sampleIndex) <= split.threshold)
              nodeIdTracker(sampleIndex) = node.leftNode.get.id.toByte
            else
              nodeIdTracker(sampleIndex) = node.rightNode.get.id.toByte
          }
          sampleIndex += 1
        }

        // enqueue left child and right child if they are not leaves
        if (!leftChildIsLeaf) {
          nodeQueue.enqueue(node.leftNode.get)
        }
        if (!rightChildIsLeaf) {
          nodeQueue.enqueue(node.rightNode.get)
        }

        logDebug("leftChildIndex = " + node.leftNode.get.id +
          ", impurity = " + stats.leftImpurity)
        logDebug("rightChildIndex = " + node.rightNode.get.id +
          ", impurity = " + stats.rightImpurity)
      }
    }
  }
  
  def calculateGainForSplit(
      leftImpurityCalculator: ImpurityCalculator,
      rightImpurityCalculator: ImpurityCalculator,
      minInstancesPerNode: Int,
      minInfoGain: Double,
      impurity: Double): InformationGainStats = {
    val leftCount = leftImpurityCalculator.count
    val rightCount = rightImpurityCalculator.count

    // If left child or right child doesn't satisfy minimum instances per node,
    // then this split is invalid, return invalid information gain stats.
    if ((leftCount < minInstancesPerNode) ||
        (rightCount < minInstancesPerNode)) {
      return InformationGainStats.invalidInformationGainStats
    }

    val totalCount = leftCount + rightCount

    val leftImpurity = leftImpurityCalculator.calculate()
    val rightImpurity = rightImpurityCalculator.calculate()

    val leftWeight = leftCount / totalCount.toDouble
    val rightWeight = rightCount / totalCount.toDouble

    val gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity

    // if information gain doesn't satisfy minimum information gain,
    // then this split is invalid, return invalid information gain stats.
    if (gain < minInfoGain) {
      return InformationGainStats.invalidInformationGainStats
    }

    // calculate left and right predict
    val leftPredict = calculatePredict(leftImpurityCalculator)
    val rightPredict = calculatePredict(rightImpurityCalculator)

    new InformationGainStats(gain, impurity, leftImpurity, rightImpurity,
      leftPredict, rightPredict)
  }

  def calculatePredict(impurityCalculator: ImpurityCalculator): Predict = {
    val predict = impurityCalculator.predict
    val prob = impurityCalculator.prob(predict)
    new Predict(predict, prob)
  }
  
  def calculatePredictImpurity(
      leftImpurityCalculator: ImpurityCalculator,
      rightImpurityCalculator: ImpurityCalculator): (Predict, Double) = {
    val parentNodeAgg = leftImpurityCalculator.copy
    parentNodeAgg.add(rightImpurityCalculator)
    val predict = calculatePredict(parentNodeAgg)
    val impurity = parentNodeAgg.calculate()

    (predict, impurity)
  }
  
  def binsToBestSplit(
      featureStatsAggregates: FeatureStatsAggregator,
      splits: Array[SplitInfo],
      minInstancesPerNode: Int,
      minInfoGain: Double,
      node: Node): (SplitInfo, InformationGainStats, Predict) = {
    // calculate predict and impurity if current node is top node
    val level = Node.indexToLevel(node.id)
    var predictWithImpurity: Option[(Predict, Double)] = if (level == 0) {
      None
    } else {
      Some((node.predict, node.impurity))
    }
    
    val numSplits = splits.length
    var splitIndex = 0
    while (splitIndex < numSplits) {
      featureStatsAggregates.merge(splitIndex + 1, splitIndex)
      splitIndex += 1
    }

    // Find best split.
    val (bestFeatureSplitIndex, bestFeatureGainStats) =
        Range(0, numSplits).map { case splitIdx =>
        val leftChildStats = featureStatsAggregates.getImpurityCalculator(splitIdx)
        val rightChildStats = featureStatsAggregates.getImpurityCalculator(numSplits)
        rightChildStats.subtract(leftChildStats)
        predictWithImpurity = Some(predictWithImpurity.getOrElse(
          calculatePredictImpurity(leftChildStats, rightChildStats)))
        val gainStats = calculateGainForSplit(leftChildStats, rightChildStats,
          minInstancesPerNode, minInfoGain, predictWithImpurity.get._2)
        (splitIdx, gainStats)
      }.maxBy(_._2.gain)

    (splits(bestFeatureSplitIndex), bestFeatureGainStats, predictWithImpurity.get._1)
  }
}
