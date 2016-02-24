package org.apache.spark.mllib.tree

import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impl._
import org.apache.spark.mllib.tree.model.informationgainstats.InformationGainStats
import org.apache.spark.mllib.tree.model.node._
import org.apache.spark.mllib.tree.model.opdtmodel.OptimizedDecisionTreeModel
import org.apache.spark.mllib.tree.model.predict.Predict
import org.apache.spark.mllib.tree.model.{NodeInfoStats, Histogram, SplitInfo}
import org.apache.spark.mllib.util.ProbabilityFunctions
import org.apache.spark.rdd.RDD

import scala.collection.immutable.BitSet
import scala.collection.mutable


class LambdaMARTDecisionTree(val strategy: Strategy,
  val numLeaves: Int,
  val maxSplits: Int) extends Serializable with Logging {

  strategy.assertValid()

  var curLeaves = 0
  val leafNo = Array(-1)
  val nonLeafNo = Array(1)

  def run(trainingData: RDD[(Int, Array[Short], Array[SplitInfo])],
    trainingData_T: RDD[(Int, Array[Array[Short]])],
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
    implicit val nodeOrd = Ordering.by[Node, Double](_.impurity).reverse
    // val highQ = MinMaxPriorityQueue.orderedBy(nodeOrd).maximumSize(numLeaves).create[Node]()
    val nodeQueue = new mutable.PriorityQueue[(Node, NodeInfoStats)]()(Ordering.by(x => x._1))
    val topNode = Node.emptyNode(nodeIndex=1)
    val topInfo = new NodeInfoStats(numSamples, lambdasBc.value.sum, lambdasBc.value.map(x=>x*x).sum, weightsBc.value.sum)
    nodeQueue.enqueue((topNode, topInfo))
    curLeaves = 1

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

    while (nodeQueue.nonEmpty && (numLeaves == 0 || curLeaves < numLeaves)) {
      val (nodesToSplits, nodesInfo, nodeId2NodeNo) = selectNodesToSplit(nodeQueue, maxSplits)

      Range(0, numSamples).par.foreach(si =>
        nodeNoTracker(si) = nodeId2NodeNo.getOrElse(nodeIdTracker(si), -1)
      )

      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      val maxDepth = if (numLeaves > 0) 32 else strategy.maxDepth
      val newSplits = LambdaMARTDecisionTree.findBestSplits(trainingData, trainingData_T, lambdasBc, weightsBc,
        maxDepth, nodesToSplits, nodesInfo, nodeId2Score, nodeNoTracker, nodeQueue, timer,
        splitfeatures, splitgain, gainPValues, threshold, leafNo, nonLeafNo)

      newSplits.par.foreach { case (siMin, lcNumSamples, splitIndc, isLeftChild) =>
        var lsi = 0
        while (lsi < lcNumSamples) {
          if (splitIndc(lsi)) {
            val si = lsi + siMin
            val oldNid = nodeIdTracker(si)
            nodeIdTracker(si) = if (isLeftChild(lsi)) oldNid << 1 else (oldNid << 1) + 1
          }
          lsi += 1
        }
      }

      timer.stop("findBestSplits")
    }

    while (nodeQueue.nonEmpty) {
      nodeQueue.dequeue()._1.isLeaf = true
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

  def selectNodesToSplit(nodeQueue: mutable.PriorityQueue[(Node, NodeInfoStats)],
    maxSplits: Int): (Array[Node], Array[NodeInfoStats], Map[Int, Byte]) = {
    val mutableNodes = new mutable.ArrayBuffer[Node]()
    val mutableNodesInfo = new mutable.ArrayBuffer[NodeInfoStats]()
    val mutableNodeId2NodeNo = new mutable.HashMap[Int, Byte]()
    var numNodes = 0
    while (nodeQueue.nonEmpty && numNodes < maxSplits && (numLeaves == 0 || curLeaves < numLeaves)) {
      val (node, info) = nodeQueue.dequeue()
      mutableNodes += node
      mutableNodesInfo += info
      mutableNodeId2NodeNo(node.id) = numNodes.toByte
      // Check if enough memory remains to add this node to the group.
      // val nodeMemUsage = aggregateSizeForNode() * 8L
      // if (memUsage + nodeMemUsage <= maxMemoryUsage) {
      //   nodeQueue.dequeue()
      //   mutableNodes += node
      //   mutableNodeId2NodeNo(node.id) = numNodes.toByte
      // }
      // memUsage += nodeMemUsage
      curLeaves += 1
      numNodes += 1
    }
    assert(mutableNodes.nonEmpty, s"LambdaMARTDecisionTree selected empty nodes. Error for unknown reason.")
    // Convert mutable maps to immutable ones.
    (mutableNodes.toArray, mutableNodesInfo.toArray, mutableNodeId2NodeNo.toMap)
  }
}

  //  def aggregateSizeForNode(): Long = {
  //    // SplitInfo num sum
  //    3 * 64 * 4096
  //  }

object LambdaMARTDecisionTree extends Serializable with Logging {
  def findBestSplits(trainingData: RDD[(Int, Array[Short], Array[SplitInfo])],
    trainingData_T: RDD[(Int, Array[Array[Short]])],
    lambdasBc: Broadcast[Array[Double]],
    weightsBc: Broadcast[Array[Double]],
    maxDepth: Int,
    nodesToSplit: Array[Node],
    nodesInfo: Array[NodeInfoStats],
    nodeId2Score: mutable.HashMap[Int, Double],
    nodeNoTracker: Array[Byte],
    nodeQueue: mutable.PriorityQueue[(Node, NodeInfoStats)],
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
    val nodesInfoBc = sc.broadcast(nodesInfo)

    val bestSplitsPerFeature = trainingData.mapPartitions { iter =>
      val lcNodeNoTracker = nodeNoTrackerBc.value
      val lcLambdas = lambdasBc.value
      val lcWeights = weightsBc.value
      val lcNodesInfo = nodesInfoBc.value
      iter.map { case (_, samples, splits) =>
        val numBins = splits.length + 1
        val histograms = Array.fill(numNodes)(new Histogram(numBins))

        var si = 0
        while (si < samples.length) {
          val ni = lcNodeNoTracker(si)
          if (ni >= 0) {
            if (samples(si)>=1) {
              histograms(ni).update(samples(si), lcLambdas(si), lcWeights(si))
            }
          }
          si += 1
        }

        Array.tabulate(numNodes)(ni => binsToBestSplit(histograms(ni), splits, lcNodesInfo(ni)))
      }
    }

    val bsf = betterSplits(numNodes)_
    val bestSplits = bestSplitsPerFeature.reduce(bsf)

    timer.stop("chooseSplits")

    // Iterate over all nodes in this group.
    var sni = 0
    while (sni < numNodes) {
      val node = nodesToSplit(sni)
      val nodeId = node.id
      val (split: SplitInfo, stats: InformationGainStats, gainPValue: Double, leftNodeInfo: NodeInfoStats, rtNodeInfo: NodeInfoStats) = bestSplits(sni)
      logDebug("best split = " + split)

      // Extract info for this node.  Create children if not leaf.
      val isLeaf = (stats.gain <= 0) || (Node.indexToLevel(nodeId) == maxDepth)
      node.isLeaf = isLeaf
      node.stats = Some(stats)
      node.impurity = stats.impurity
      logDebug("Node = " + node)

      nodeId2Score(node.id) = node.predict.predict
      // nodePredict.predict(node.id, nodeIdTrackerBc, targetScoresBc, weightsBc)

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
        nodeId2Score(node.leftNode.get.id) = node.leftNode.get.predict.predict
       
        node.rightNode = Some(Node(Node.rightChildIndex(nodeId),
          stats.rightPredict, stats.rightImpurity, rightChildIsLeaf))
        nodeId2Score(node.rightNode.get.id) = node.rightNode.get.predict.predict

        if (leftChildIsLeaf) {
          node.leftNode.get.id2 = leafNo(0)
          leafNo(0) -= 1
        } else {
          node.leftNode.get.id2 = nonLeafNo(0)
          nonLeafNo(0) += 1
        }

        if (rightChildIsLeaf) {
          node.rightNode.get.id2 = leafNo(0)
          leafNo(0) -= 1
        } else {
          node.rightNode.get.id2 = nonLeafNo(0)
          nonLeafNo(0) += 1
        }

        // enqueue left child and right child if they are not leaves
        if (!leftChildIsLeaf) {
          nodeQueue.enqueue((node.leftNode.get, leftNodeInfo))
        }
        if (!rightChildIsLeaf) {
          nodeQueue.enqueue((node.rightNode.get, rtNodeInfo))
        }

        logDebug(s"leftChildIndex = ${node.leftNode.get.id}, impurity = ${stats.leftImpurity}")
        logDebug(s"rightChildIndex = ${node.rightNode.get.id}, impurity = ${stats.rightImpurity}")
      }

      sni += 1
    }

    val bestSplitsBc = sc.broadcast(bestSplits)
    val newNodesToSplitBc = sc.broadcast(nodesToSplit)
    val newSplits = trainingData_T.mapPartitions { iter =>
      val lcNodeNoTracker = nodeNoTrackerBc.value
      val lcBestSplits = bestSplitsBc.value
      val lcNewNodesToSplit = newNodesToSplitBc.value
      val (siMin, sampleSlice) = iter.next()
      val lcNumSamples = sampleSlice(0).length
      val splitIndc = new mutable.BitSet(lcNumSamples)
      val isLeftChild = new mutable.BitSet(lcNumSamples)
      var lsi = 0
      while (lsi < lcNumSamples) {
        val oldNi = lcNodeNoTracker(lsi + siMin)
        if (oldNi >= 0) {
          val node = lcNewNodesToSplit(oldNi)
          if (!node.isLeaf) {
            splitIndc += lsi
            val split = lcBestSplits(oldNi)._1
            if (sampleSlice(split.feature)(lsi) <= split.threshold) {
              isLeftChild += lsi
            }
          }
        }
        lsi += 1
      }
      Iterator.single((siMin, lcNumSamples, splitIndc.toImmutable, isLeftChild.toImmutable))
    }.collect()

    bestSplitsBc.destroy(blocking=false)
    newNodesToSplitBc.destroy(blocking=false)
    nodeNoTrackerBc.destroy(blocking=false)
    nodesInfoBc.destroy(blocking = false)
    newSplits
  }

  def betterSplits(numNodes: Int)(a: Array[(SplitInfo, InformationGainStats, Double, NodeInfoStats, NodeInfoStats)],
    b: Array[(SplitInfo, InformationGainStats, Double, NodeInfoStats, NodeInfoStats)])
  : Array[(SplitInfo, InformationGainStats, Double, NodeInfoStats, NodeInfoStats)] = {
    Array.tabulate(numNodes){ ni =>
      val ai = a(ni)
      val bi = b(ni)
      if (ai._2.gain >= bi._2.gain) ai else bi
    }
  }

//  def calculateGainForSplit(
//      leftImpurityCalculator: ImpurityCalculator,
//      rightImpurityCalculator: ImpurityCalculator,
//      minInstancesPerNode: Int,
//      minInfoGain: Double,
//      impurity: Double): InformationGainStats = {
//    val leftCount = leftImpurityCalculator.count
//    val rightCount = rightImpurityCalculator.count
//
//    // If left child or right child doesn't satisfy minimum instances per node,
//    // then this split is invalid, return invalid information gain stats.
//    if ((leftCount < minInstancesPerNode) ||
//        (rightCount < minInstancesPerNode)) {
//      return InformationGainStats.invalidInformationGainStats
//    }
//
//    val totalCount = leftCount + rightCount
//
//    val leftImpurity = leftImpurityCalculator.calculate()
//    val rightImpurity = rightImpurityCalculator.calculate()
//
//    val leftWeight = leftCount / totalCount.toDouble
//    val rightWeight = rightCount / totalCount.toDouble
//
//    val gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity
//
//    // if information gain doesn't satisfy minimum information gain,
//    // then this split is invalid, return invalid information gain stats.
//    if (gain < minInfoGain) {
//      return InformationGainStats.invalidInformationGainStats
//    }
//
//    // calculate left and right predict
//    val leftPredict = calculatePredict(leftImpurityCalculator)
//    val rightPredict = calculatePredict(rightImpurityCalculator)
//
//    new InformationGainStats(gain, impurity, leftImpurity, rightImpurity,
//      leftPredict, rightPredict)
//  }
//
//  def calculatePredict(impurityCalculator: ImpurityCalculator): Predict = {
//    val predict = impurityCalculator.predict
//    val prob = impurityCalculator.prob(predict)
//    new Predict(predict, prob)
//  }
//
//  def calculatePredictImpurity(
//      leftImpurityCalculator: ImpurityCalculator,
//      rightImpurityCalculator: ImpurityCalculator): (Predict, Double) = {
//    val parentNodeAgg = leftImpurityCalculator.copy
//    parentNodeAgg.add(rightImpurityCalculator)
//    val predict = calculatePredict(parentNodeAgg)
//    val impurity = parentNodeAgg.calculate()
//
//    (predict, impurity)
//  }

  // TODO: make minInstancesPerNode, minInfoGain parameterized
  def binsToBestSplit(hist: Histogram,
    splits: Array[SplitInfo],
    nodeInfo: NodeInfoStats,
    minInstancesPerNode: Int = 1,
    minGain: Double = Double.MinPositiveValue): (SplitInfo, InformationGainStats, Double, NodeInfoStats, NodeInfoStats) = {
    val cumHist = hist.cumulate(nodeInfo)
//    val totalDocInNode = cumHist.counts.last.toInt
//    val sumTargets = cumHist.scores.last
//    val sumWeight = cumHist.scoreWeights.last
//    val sumSquares = cumHist.squares.last
    // val impurity = (sumSquares - sumTargets * sumTargets / totalDocInNode) / totalDocInNode

    //val denom = if (sumWeight == 0.0) totalDocInNode else sumWeight
    val varianceTargets = (nodeInfo.sumSquares - nodeInfo.sumScores / nodeInfo.sumCount) / (nodeInfo.sumCount - 1)
    //val eps = 1e-10
    val gainShift = getLeafSplitGain(nodeInfo.sumCount, nodeInfo.sumScores)
    val gainConfidenceLevel = 0.95
    var gainConfidenceInSquaredStandardDeviations = ProbabilityFunctions.Probit(1.0 -(1.0 - gainConfidenceLevel)*0.5)
    gainConfidenceInSquaredStandardDeviations *= gainConfidenceInSquaredStandardDeviations

    //val entropyCoefficient = 0.0 // an outer tuning parameters
//    val minShiftedGain = if (gainConfidenceInSquaredStandardDeviations <= 0) 0.0
//      else (gainConfidenceInSquaredStandardDeviations * varianceTargets *
//        totalDocInNode / (totalDocInNode - 1) + gainShift)
//    var bestLteCount = -1
//    var bestLteTarget = Double.NaN
//    var bestLteWeight = Double.NaN
//    var bestLteSquaredTarget = Double.NaN
    var bestRtInfo = new NodeInfoStats(-1, Double.NaN, Double.NaN, Double.NaN)
    var bestShiftedGain = Double.NegativeInfinity
    var bestThreshold = 0
    val feature = splits(0).feature

    for (split <- splits) {
//      val thresh = split.threshold.toInt
//      val lteCount = cumHist.counts(thresh).toInt
//
//      // if(lcnts < minInstancesPerNode)  //TODO  pass this loop
//      val lteSumTarget = cumHist.scores(thresh)
//      val lteSumWeight = cumHist.scoreWeights(thresh)
//      val rtCount = totalDocInNode - lteCount
//      // if(rcnts < minInstancesPerNode)  // TODO break
//      val rtSumTarget = sumTargets - lteSumTarget
//      val rtSumWeight = sumWeight - lteSumWeight

      val threshLeft = split.threshold.toInt + 1
      val rtCount = cumHist.counts(threshLeft).toInt
      val rtSumTarget = cumHist.scores(threshLeft)
      val rtSumWeight = cumHist.scoreWeights(threshLeft)
      val lteCount = nodeInfo.sumCount - rtCount
      val lteSumTarget = nodeInfo.sumScores - rtSumTarget
      val lteSumWeight = nodeInfo.sumScoreWeights - rtSumWeight

      /**
        * if (lscoreWeights <= 0.0) {
        * lscoreWeights = lcnts
        * }
        * if (rscoreWeights <= 0.0) {
        * rscoreWeights = rcnts
        * }**/

      // val gain = lscores * lscores / lcnts + rscores * rscores / rcnts  gainShift >= minShiftedGain
      if (lteCount >= minInstancesPerNode && rtCount >= minInstancesPerNode) {
        val currentShiftedGain = getLeafSplitGain(lteCount, lteSumTarget) +
          getLeafSplitGain(rtCount, rtSumTarget)

//        if (entropyCoefficient > 0) {
//          val entropyGain = totalDocInNode * math.log(totalDocInNode) - lteCount * math.log(lteCount) -
//            rtCount * math.log(rtCount)
//          currentShiftedGain += entropyCoefficient * entropyGain
//        }

        if (currentShiftedGain > bestShiftedGain) {
//          bestLteCount = lteCount
//          bestLteTarget = lteSumTarget
//          bestLteWeight = lteSumWeight
//          bestLteSquaredTarget = cumHist.squares(threshLeft)


          bestRtInfo.sumCount=rtCount
          bestRtInfo.sumScores=rtSumTarget
          bestRtInfo.sumSquares= cumHist.squares(threshLeft)
          bestRtInfo.sumScoreWeights=rtSumWeight

          bestShiftedGain = currentShiftedGain
          bestThreshold = threshLeft-1
        }
      }
    }
//    val gtSquares = sumSquares - bestLteSquaredTarget
//    val gtTarget = sumTargets - bestLteTarget
//    val gtCount = totalDocInNode - bestLteCount
    val bestLeftInfo = new NodeInfoStats(nodeInfo.sumCount-bestRtInfo.sumCount, nodeInfo.sumScores-bestRtInfo.sumScores,
  nodeInfo.sumSquares-bestRtInfo.sumSquares, nodeInfo.sumScoreWeights-bestRtInfo.sumScoreWeights)

    val lteImpurity = (bestLeftInfo.sumSquares - bestLeftInfo.sumScores * bestLeftInfo.sumScores /bestLeftInfo.sumCount) /bestLeftInfo.sumCount
    val gtImpurity = (bestRtInfo.sumSquares - bestRtInfo.sumScores * bestRtInfo.sumScores /bestRtInfo.sumCount) /bestRtInfo.sumCount
    val tolImpurity = (nodeInfo.sumSquares - nodeInfo.sumScores * nodeInfo.sumScores /nodeInfo.sumCount) /nodeInfo.sumCount

    val bestSplitInfo = new SplitInfo(feature, bestThreshold.toDouble)
    val lteOutput = CalculateSplittedLeafOutput(bestLeftInfo.sumCount, bestLeftInfo.sumScores, bestLeftInfo.sumScoreWeights)
    val gtOutput = CalculateSplittedLeafOutput(bestRtInfo.sumCount, bestRtInfo.sumScores, bestRtInfo.sumScoreWeights)
    val ltePredict = new Predict(lteOutput)
    val gtPredict = new Predict(gtOutput)

    val trust = 1.0
    //println("#############################################################################################")
    //println(s"bestShiftedGain: $bestShiftedGain, gainShift: $gainShift")
    val splitGain = (bestShiftedGain - gainShift) * trust //- usePenalty //TODO introduce trust and usePenalty
    val inforGainStat = new InformationGainStats(splitGain, tolImpurity, lteImpurity, gtImpurity, ltePredict, gtPredict)
    val erfcArg = math.sqrt((bestShiftedGain - gainShift) * (nodeInfo.sumCount - 1) / (2 * varianceTargets * nodeInfo.sumCount))
    val gainPValue = ProbabilityFunctions.erfc(erfcArg)
    (bestSplitInfo, inforGainStat, gainPValue, bestLeftInfo, bestRtInfo)
  }

  def getLeafSplitGain(count: Double, target: Double): Double = {
    //val pesuedCount = if(weight == 0.0) count else weight
    target*target / count
  }

  def CalculateSplittedLeafOutput(totalCount: Int, sumTargets: Double, sumWeights: Double): Double = {
    val hasWeight = false
    val bsrMaxTreeOutput = 100.0
    if(!hasWeight){ //TODO hasweight true or false
      sumTargets / totalCount
    } else {
      if (bsrMaxTreeOutput < 0.0) {  //TODO  bsrMaxTreeOutput default 100
        sumTargets / sumWeights
      } else {
        sumTargets / (2 * sumWeights)
      }
    }
  }
}
