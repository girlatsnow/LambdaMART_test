package org.apache.spark.mllib.tree

import scala.collection.immutable.BitSet
import scala.collection.mutable

import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impl._
import org.apache.spark.mllib.tree.model.{Histogram, SplitInfo}
import org.apache.spark.mllib.tree.model.impurity._
import org.apache.spark.mllib.tree.model.informationgainstats.InformationGainStats
import org.apache.spark.mllib.tree.model.node._
import org.apache.spark.mllib.tree.model.opdtmodel.OptimizedDecisionTreeModel
import org.apache.spark.mllib.tree.model.predict.Predict
import org.apache.spark.rdd.RDD


class LambdaMARTDecisionTree(
    val strategy: Strategy,
    val maxSplits: Int) extends Serializable with Logging {

  strategy.assertValid()

  val leafNo = Array(-1)
  val nonLeafNo = Array(1)

  def run(
      trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
      trainingData_T: RDD[(Int, Array[Array[Byte]])],
      lambdasBc: Broadcast[Array[Double]],
      weightsBc: Broadcast[Array[Double]],
      numSamples: Int): (OptimizedDecisionTreeModel, Array[Double]) = {

    val timer = new TimeTracker()

    timer.start("total")

    // depth of the decision tree
    val maxDepth = strategy.maxDepth
    require(maxDepth <= 30, s"LambdaMART currently only supports maxDepth <= 30, but $maxDepth was given.")
    // val maxMemoryUsage: Long = strategy.maxMemoryInMB * 1024L * 1024L

    // FIFO queue of nodes to train: node
    val nodeQueue = new mutable.Queue[Node]()
    val topNode = Node.emptyNode(nodeIndex=1)
    nodeQueue.enqueue(topNode)

    // Create node Id tracker.
    // At first, all the samples belong to the root nodes (node Id == 1).
    val nodeIdTracker = Array.fill[Int](numSamples)(1)
    val nodeNoTracker = new Array[Byte](numSamples)
    // TBD re-declared
    val nodeId2Score = new mutable.HashMap[Int, Double]()

    val splitfeatures = mutable.MutableList[String]()
    val splitgain = mutable.MutableList[Double]()
    val threshold = mutable.MutableList[Double]()
    val gainPValues = mutable.MutableList[Double]()

    while (nodeQueue.nonEmpty) {
      val (nodesToSplits, nodeId2NodeNo) = LambdaMARTDecisionTree.selectNodesToSplit(nodeQueue, maxSplits)

      Range(0, numSamples).par.foreach(si =>
        nodeNoTracker(si) = nodeId2NodeNo.getOrElse(nodeIdTracker(si), -1)
      )

      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      val newSplits = LambdaMARTDecisionTree.findBestSplits(trainingData, trainingData_T, lambdasBc, weightsBc,
        strategy.maxDepth, nodesToSplits, nodeId2Score, nodeNoTracker, nodeQueue, timer,
        splitfeatures, splitgain, threshold, gainPValues, leafNo, nonLeafNo)

      newSplits.par.foreach(Function.tupled((siMin, lcNumSamples, newIndicatorSlice, newSplitSlice) => {
        var lsi = 0
        while (lsi < lcNumSamples) {
          if (newIndicatorSlice(lsi)) {
            val si = lsi + siMin
            val oldNid = nodeIdTracker(si)
            nodeIdTracker(si) = if (newSplitSlice(lsi)) (oldNid << 1) + 1 else oldNid << 1
          }
          lsi += 1
        }
      }))

      timer.stop("findBestSplits")
    }

    timer.stop("total")

    println("Internal timing for LambdaMARTDecisionTree:")
    println(s"$timer")

    val treeScores = new Array[Double](numSamples)
    Range(0, numSamples).par.foreach(si =>
      treeScores(si) = nodeId2Score(nodeIdTracker(si))
    )

    val model = new OptimizedDecisionTreeModel(topNode, strategy.algo, splitfeatures, splitgain, gainPValues, threshold)
    (model, treeScores)
  }
}

object LambdaMARTDecisionTree extends Serializable with Logging {

  def selectNodesToSplit(
      nodeQueue: mutable.Queue[Node],
      maxSplits: Int): (Array[Node], Map[Int, Byte]) = {
    val mutableNodes = new mutable.ArrayBuffer[Node]()
    val mutableNodeId2NodeNo = new mutable.HashMap[Int, Byte]()
    var numNodes = 0
    while (nodeQueue.nonEmpty && numNodes < maxSplits) {
      val node = nodeQueue.dequeue()
      mutableNodes += node
      mutableNodeId2NodeNo(node.id) = numNodes.toByte
      // Check if enough memory remains to add this node to the group.
      // val nodeMemUsage = aggregateSizeForNode() * 8L
      // if (memUsage + nodeMemUsage <= maxMemoryUsage) {
      //   nodeQueue.dequeue()
      //   mutableNodes += node
      //   mutableNodeId2NodeNo(node.id) = numNodes.toByte
      // }
      // memUsage += nodeMemUsage
      numNodes += 1
    }
    assert(mutableNodes.nonEmpty, s"LambdaMARTDecisionTree selected empty nodes. Error for unknown reason.")
    // Convert mutable maps to immutable ones.
    (mutableNodes.toArray, mutableNodeId2NodeNo.toMap)
  }

  //  def aggregateSizeForNode(): Long = {
  //    // SplitInfo num sum
  //    3 * 64 * 4096
  //  }
  
  def findBestSplits(
      trainingData: RDD[(Int, Array[Byte], Array[SplitInfo])],
      trainingData_T: RDD[(Int, Array[Array[Byte]])],
      lambdasBc: Broadcast[Array[Double]],
      weightsBc: Broadcast[Array[Double]],
      maxDepth: Int,
      nodesToSplit: Array[Node],
      nodeId2Score: mutable.HashMap[Int, Double],
      nodeNoTracker: Array[Byte],
      nodeQueue: mutable.Queue[Node],
      timer: TimeTracker,
      splitfeatures: mutable.MutableList[String],
      splitgain: mutable.MutableList[Double],
      gainPValues: mutable.MutableList[Double],
      threshold: mutable.MutableList[Double],
      leafNo: Array[Int],
      nonLeafNo: Array[Int]): Array[(Int, Int, BitSet, BitSet)] = {
    // numNodes:  Number of nodes in this group
    val numNodes = nodesToSplit.length
    println("numNodes = " + numNodes)

    // Calculate best splits for all nodes in the group
    timer.start("chooseSplits")
    val sc = trainingData.sparkContext
    val nodeNoTrackerBc = sc.broadcast(nodeNoTracker)

    val bestSplitsPerFeature = trainingData.mapPartitions(iter => {
      val lcNodeNoTracker = nodeNoTrackerBc.value
      val lcLambdas = lambdasBc.value
      val lcWeights = weightsBc.value
      iter.map(Function.tupled((_, samples, splits) => {
        val numBins = splits.length + 1
        val histograms = Array.fill(numNodes)(new Histogram(numBins))

        var si = 0
        while (si < samples.length) {
          val ni = lcNodeNoTracker(si)
          if (ni >= 0) {
            histograms(ni).update(samples(si), lcLambdas(si), lcWeights(si))
          }
          si += 1
        }

        Array.tabulate(numNodes)(ni => binsToBestSplit(histograms(ni), splits))
      }))
    })

    val bsf = betterSplits(numNodes)_
    val bestSplits = bestSplitsPerFeature.reduce(bsf)

    timer.stop("chooseSplits")

    // Iterate over all nodes in this group.
    var sni = 0
    while (sni < numNodes) {
      val node = nodesToSplit(sni)
      val nodeId = node.id
      val (split: SplitInfo, stats: InformationGainStats, gainPValue: Double) = bestSplits(sni)
      logDebug("best split = " + split)

      // Extract info for this node.  Create children if not leaf.
      val isLeaf = (stats.gain <= 0) || (Node.indexToLevel(nodeId) == maxDepth)
      node.isLeaf = isLeaf
      node.stats = Some(stats)
      node.impurity = stats.impurity
      logDebug("Node = " + node)

      nodeId2Score(node.id) = node.predict.predict       //nodePredict.predict(node.id, nodeIdTrackerBc, targetScoresBc, weightsBc)

      if (!isLeaf) {
        splitfeatures += "I:" + split.feature.toString
        splitgain += stats.gain
        threshold += split.threshold
        gainPValues += gainPValue

        node.split = Some(split)
        val childIsLeaf = (Node.indexToLevel(nodeId) + 1) == maxDepth
        val leftChildIsLeaf = childIsLeaf || (stats.leftImpurity == 0.0)
        val rightChildIsLeaf = childIsLeaf || (stats.rightImpurity == 0.0)
        node.leftNode = Some(Node(Node.leftChildIndex(nodeId),
          stats.leftPredict, stats.leftImpurity, leftChildIsLeaf))          
        nodeId2Score(node.leftNode.get.id) = node.leftNode.get.predict.predict   //nodePredict.predict(node.leftNode.get.id, nodeIdTrackerBc, targetScoresBc, weightsBc)
       
        node.rightNode = Some(Node(Node.rightChildIndex(nodeId),
          stats.rightPredict, stats.rightImpurity, rightChildIsLeaf))           
        nodeId2Score(node.rightNode.get.id.toByte) = node.rightNode.get.predict.predict  //nodePredict.predict(node.rightNode.get.id, nodeIdTrackerBc, targetScoresBc, weightsBc)

        if(leftChildIsLeaf){
          node.leftNode.get.id2 = leafNo(0)
          leafNo(0) -= 1
        }
        else{
          node.leftNode.get.id2 = nonLeafNo(0)
          nonLeafNo(0) += 1
        }


        if(rightChildIsLeaf){
          node.rightNode.get.id2 = leafNo(0)
          leafNo(0) -= 1
        }
        else{
          node.rightNode.get.id2 = nonLeafNo(0)
          nonLeafNo(0) += 1
        }

        // enqueue left child and right child if they are not leaves
        if (!leftChildIsLeaf) {
          nodeQueue.enqueue(node.leftNode.get)
        }
        if (!rightChildIsLeaf) {
          nodeQueue.enqueue(node.rightNode.get)
        }

        logDebug(s"leftChildIndex = ${node.leftNode.get.id}, impurity = ${stats.leftImpurity}")
        logDebug(s"rightChildIndex = ${node.rightNode.get.id}, impurity = ${stats.rightImpurity}")
      }

      sni += 1
    }

    val bestSplitsBc = sc.broadcast(bestSplits)
    val newNodesToSplitBc = sc.broadcast(nodesToSplit)
    val newSplits = trainingData_T.mapPartitions(iter => {
      val lcNodeNoTracker = nodeNoTrackerBc.value
      val lcBestSplits = bestSplitsBc.value
      val lcNewNodesToSplit = newNodesToSplitBc.value
      val (siMin, sampleSlice) = iter.next()
      val lcNumSamples = sampleSlice(0).length
      val newIndicatorSlice = new mutable.BitSet(lcNumSamples)
      val newSplitSlice = new mutable.BitSet(lcNumSamples)
      var lsi = 0
      while (lsi < lcNumSamples) {
        val oldNi = lcNodeNoTracker(lsi + siMin)
        if (oldNi >= 0) {
          val node = lcNewNodesToSplit(oldNi)
          if (!node.isLeaf) {
            newIndicatorSlice += lsi
            val split = lcBestSplits(oldNi)._1
            if (sampleSlice(split.feature)(lsi) > split.threshold) {
              newSplitSlice += lsi
            }
          }
        }
        lsi += 1
      }
      Iterator.single((siMin, lcNumSamples, newIndicatorSlice.toImmutable, newSplitSlice.toImmutable))
    }).collect()

    bestSplitsBc.destroy(blocking=false)
    newNodesToSplitBc.destroy(blocking=false)
    nodeNoTrackerBc.destroy(blocking=false)
    newSplits
  }

  def betterSplits(numNodes: Int)(
      a: Array[(SplitInfo, InformationGainStats)],
      b: Array[(SplitInfo, InformationGainStats)])
  : Array[(SplitInfo, InformationGainStats)] = {
    Array.tabulate(numNodes)(ni => {
      val ai = a(ni)
      val bi = b(ni)
      if (ai._2.gain >= bi._2.gain) ai else bi
    })
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

  // TODO: make minInstancesPerNode, minInfoGain parameterized
  def binsToBestSplit(
      hist: Histogram,
      splits: Array[SplitInfo],
      minInstancesPerNode: Int = 1,
      minGain: Double = Double.MinPositiveValue): (SplitInfo, InformationGainStats) = {
    val cumHist = hist.cumulate()
    val acnts = hist.counts.last
    val ascores = hist.scores.last
    val asquares = hist.squares.last
    val impurity = (asquares - ascores * ascores / acnts) / acnts
    splits.iterator.map(split => {
      val thresh = split.threshold.toInt
      val lcnts = cumHist.counts(thresh)
      val lscores = cumHist.scores(thresh)
      val rcnts = acnts - lcnts
      val rscores = ascores - lscores
      val gain = lscores * lscores / lcnts + rscores * rscores / rcnts
      val gainStats = if (lcnts >= minInstancesPerNode && rcnts >= minInstancesPerNode && gain >= minGain) {
        val lsquare = cumHist.squares(thresh)
        val leftImpurity = (lsquare - lscores * lscores / lcnts) / lcnts
        val rightImpurity = (asquares - lsquare - rscores * rscores / rcnts) / rcnts
        val leftPridict = lscores / lcnts
        val rightPridict = rscores / rcnts
        new InformationGainStats(gain, impurity, leftImpurity, rightImpurity,
          new Predict(leftPridict, -1), new Predict(rightPridict, -1))
      } else {
        InformationGainStats.invalidInformationGainStats
      }
      (split, gainStats)
    }).maxBy(_._2.gain)
      minInstancesPerNode: Int,
      minInfoGain: Double,
      node: Node): (SplitInfo, InformationGainStats, Predict, Double) = {
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

    val totalCount = featureStatsAggregates.getImpurityCalculator(splitIndex).stats(0).toInt
    val sumTargets = featureStatsAggregates.getImpurityCalculator(splitIndex).stats(1)
    val sumSquaredTargets = featureStatsAggregates.getImpurityCalculator(splitIndex).stats(2)
    val sumWeights = featureStatsAggregates.getImpurityCalculator(splitIndex).stats(3)
    val denom = if(0 == sumWeights) totalCount else sumWeights
    val varianceTargets = (sumSquaredTargets - sumTargets / denom) / (denom - 1)

    val eps = 1e-10
    val gainShift = getLeafSplitGain(totalCount, sumTargets, sumWeights + 2 * eps)

    var bestShiftedGain = Double.NegativeInfinity
    // Find best split.
    val (bestFeatureSplitIndex, bestFeatureGainStats) =
        Range(0, numSplits).map { case splitIdx =>
        val leftChildStats = featureStatsAggregates.getImpurityCalculator(splitIdx)
        val rightChildStats = featureStatsAggregates.getImpurityCalculator(numSplits)

        val lteCount = leftChildStats.stats(0).toInt
        val sumLTETargets = leftChildStats.stats(1)
        val sumLTEWeights = leftChildStats.stats(3)

        val gtCount = rightChildStats.stats(0).toInt
        val sumGTTargets = rightChildStats.stats(1)
        val sumGTWeights = rightChildStats.stats(3)

        val currentShiftedGain = getLeafSplitGain(lteCount, sumLTETargets, sumLTEWeights) + getLeafSplitGain(gtCount, sumGTTargets, sumGTWeights)

        if(currentShiftedGain > bestShiftedGain)
            bestShiftedGain = currentShiftedGain

        rightChildStats.subtract(leftChildStats)
        predictWithImpurity = Some(predictWithImpurity.getOrElse(
          calculatePredictImpurity(leftChildStats, rightChildStats)))
        val gainStats = calculateGainForSplit(leftChildStats, rightChildStats,
          minInstancesPerNode, minInfoGain, predictWithImpurity.get._2)
        (splitIdx, gainStats)
      }.maxBy(_._2.gain)

    val erfcArg = scala.math.sqrt((bestShiftedGain - gainShift) * (totalCount - 1) / (2 * varianceTargets * totalCount))
    val gainPValue = Erfc(erfcArg)

    (splits(bestFeatureSplitIndex), bestFeatureGainStats, predictWithImpurity.get._1, gainPValue)
  }

  //The approximate complimentary error function (i.e., 1-erf).
  def Erfc(x: Double): Double ={
    if(x.isInfinity){
        if(x.isPosInfinity)  1.0 else -1.0
    }
    else{
      val p = 0.3275911
      val a1 = 0.254829592
      val a2 = -0.284496736
      val a3 = 1.421413741
      val a4 = -1.453152027
      val a5 = 1.061405429

      var t = 1.0 / (1.0 + p * scala.math.abs(x))
      var ev = 1.0 -  ((((((((a5 * t) + a4) * t) + a3) * t) + a2) * t + a1) * t) * scala.math.exp(-(x * x))
      if(x >= 0) ev else -ev
    }
  }

  def getLeafSplitGain(count: Int, starget: Double, sweight: Double): Double = {
    if(0 == sweight)  starget*starget /  count
    else  starget*starget / sweight
  }




}
