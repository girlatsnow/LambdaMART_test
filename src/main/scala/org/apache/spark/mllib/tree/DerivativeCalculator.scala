package org.apache.spark.mllib.tree

class DerivativeCalculator extends Serializable {
  val expAsymptote: Double = -50
  val sigmoidBins: Int = 1000000

  var sigmoidTable: Array[Double] = _
  var minScore: Double = _
  var maxScore: Double = _
  var scoreToSigmoidTableFactor: Double = _
  var minSigmoid: Double = _
  var maxSigmoid: Double = _

  var discounts: Array[Double] = _
  var labelScores: Array[Short] = _
  var labels: Array[Byte] = _
  var secondaryGains: Array[Double] = _

  var queryBoundy: Array[Int] = _
  var inverseMaxDCGs: Array[Double] = _

  def init(labelScores: Array[Short], queryBoundy: Array[Int], sigma: Double = 1.0): Unit = {
    initSigmoidTable(sigma)

    discounts = Array.tabulate(1024)(i => 1.0 / math.log(i + 2))
    this.labelScores = labelScores
    labels = labelScores.map(score => Integer.numberOfTrailingZeros(score + 1).toByte)
    secondaryGains = new Array[Double](labelScores.length)

    calcInverseMaxDCGs(queryBoundy)
  }

  private def initSigmoidTable(sigma: Double): Unit = {
    // minScore is such that 2*sigma*score is < expAsymptote if score < minScore
    minScore = expAsymptote / sigma / 2
    maxScore = -minScore
    scoreToSigmoidTableFactor = sigmoidBins / (maxScore - minScore)

    sigmoidTable = new Array[Double](sigmoidBins)
    var i = 0
    while (i < sigmoidBins) {
      val score = (maxScore - minScore) / sigmoidBins * i + minScore
      sigmoidTable(i) = if (score > 0.0) {
        2.0 - 2.0 / (1.0 + math.exp(-2.0 * sigma * score))
      } else {
        2.0 / (1.0 + math.exp(2.0 * sigma * score))
      }
      i += 1
    }
    minSigmoid = sigmoidTable.head
    maxSigmoid = sigmoidTable.last
  }

  private def calcInverseMaxDCGs(queryBoundy: Array[Int]): Unit = {
    this.queryBoundy = queryBoundy
    val numQueries = queryBoundy.length - 1
    inverseMaxDCGs = new Array[Double](numQueries)
    var qi = 0
    while (qi < numQueries) {
      val start = queryBoundy(qi)
      val end = queryBoundy(qi + 1)
      val labelGainQ_sorted = labelScoreSort(start, end)
      var inverseMaxDCGQ = 0.0
      val numDocs = end - start
      var di = 0
      while (di < numDocs) {
        inverseMaxDCGQ += labelGainQ_sorted(di) * discounts(di)
        di += 1
      }
      inverseMaxDCGs(qi) = inverseMaxDCGQ
      qi += 1
    }
  }

  def labelScoreSort(start: Int, end: Int): Array[Short] = {
    labelScores.view(start, end).sortWith(_ > _).toArray
  }

  def docIdxSort(scores: Array[Double], start: Int, end: Int): Array[Int] = {
    Range(start, end).sortBy(scores(_)).map(_ - start).toArray
  }

  def getDerivatives(scores: Array[Double], queryStart: Int, queryEnd: Int): (Array[Double], Array[Double]) = {
    val numTotalDocs = queryBoundy(queryEnd) - queryBoundy(queryStart)
    val lcLambdas = new Array[Double](numTotalDocs)
    val lcWeights = new Array[Double](numTotalDocs)
    var qi = queryStart
    while (qi < queryEnd) {
      derivativesQ(qi, scores, lcLambdas, lcWeights)
      qi += 1
    }
    (lcLambdas, lcWeights)
  }

  def calcError(scores: Array[Double], queryStart: Int, queryEnd: Int): Double = {
    val numTotalDocs = queryBoundy(queryEnd) - queryBoundy(queryStart)
    val lcLambdas = new Array[Double](numTotalDocs)
    val lcWeights = new Array[Double](numTotalDocs)
    var dcgs = 0.0
    var qi = queryStart
    while (qi < queryEnd) {
      val start = queryBoundy(qi)
      val end = queryBoundy(qi + 1)
      val numDocs = end - start
      val permutation = docIdxSort(scores, start, end)
      var dcg = 0.0
      var sdi = 0
      while (sdi < numDocs) {
        dcg += labelScores(start + permutation(sdi)) * discounts(sdi)
        sdi += 1
      }
      dcgs += inverseMaxDCGs(qi) - dcg
      qi += 1
    }
    dcgs
  }

  private def derivativesQ(queryIdx: Int,
      scores: Array[Double],
      lambdas: Array[Double],
      weights: Array[Double],
      secondaryMetricShare: Double = 0.0,
      secondaryExclusive: Boolean = false,
      secondaryInverseMaxDCG: Double = 0.2,
      costFunctionParam: Char = 'c',
      distanceWeight2: Boolean = false,
      minDoubleValue: Double = 0.01,
      alphaRisk: Double = 0.2,
      baselineVersusCurrentDcg: Double = 0.1): Unit = {
    val start = queryBoundy(queryIdx)
    val end = queryBoundy(queryIdx + 1)
    val numDocs = end - start
    val permutation = docIdxSort(scores, start, end)
    val inverseMaxDCG = inverseMaxDCGs(queryIdx)
    // These arrays are shared among many threads, "start" is the offset by which all arrays are indexed.
    //  So we shift them all here to avoid having to add 'start' to every pointer below.
    //val pLabels = start
    //val pScores = start
    //val pLambdas = start
    //val pWeights = start
    //val pGainLabels = start

    var pSecondaryGains = 0

    if (secondaryMetricShare != 0)
      pSecondaryGains = start

    val bestScore = scores(permutation(0))

    var worstIndexToConsider = numDocs - 1

    while (worstIndexToConsider > 0 && scores(permutation(worstIndexToConsider)) == minDoubleValue) {
      worstIndexToConsider -= 1
    }
    val worstScore = scores(permutation(worstIndexToConsider))

    var lambdaSum = 0.0

    // Should we still run the calculation on those pairs which are ostensibly the same?
    val pairSame: Boolean = secondaryMetricShare != 0.0

    // Did not help to use pointer match on pPermutation[i]
    for (di <- 0 until numDocs)
    {
      //println("here1")
      val high = start + permutation(di)
      // We are going to loop through all pairs where label[high] > label[low]. If label[high] is 0, it can't be larger
      // If score[high] is Double.MinValue, it's being discarded by shifted NDCG
      //println("aLabels(high)", aLabels(high), "aScores(high)", aScores(high), "minDoubleValue", minDoubleValue, "pairSame", pairSame)
      if (!((labels(high) == 0 && !pairSame) || scores(high) == minDoubleValue)) { // These variables are all looked up just once per loop of 'i', so do it here.

        val gainLabelHigh = labelScores(high)
        val labelHigh = labels(high)
        val scoreHigh = scores(high)
        val discountI = discounts(di)
        // These variables will store the accumulated lambda and weight difference for high, which saves time
        var deltaLambdasHigh: Double = 0
        var deltaWeightsHigh: Double = 0

        for (dj <- 0 until numDocs) {
          // only consider pairs with different labels, where "high" has a higher label than "low"
          // If score[low] is Double.MinValue, it's being discarded by shifted NDCG
          val low = start + permutation(dj)
          val flag = if (pairSame) labelHigh < labels(low) else labelHigh <= labels(low)
          if (!(flag || scores(low) == minDoubleValue)) {
            val scoreHighMinusLow = scoreHigh - scores(low)
            if (!(secondaryMetricShare == 0.0 && labelHigh == labels(low) && scoreHighMinusLow <= 0)) {

              //println("labelHigh", labelHigh, "aLabels(low)", aLabels(low), "scoreHighMinusLow", scoreHighMinusLow)
              var dcgGap: Double = gainLabelHigh - labelScores(low)
              var currentInverseMaxDCG = inverseMaxDCG * (1.0 - secondaryMetricShare)

              // Handle risk w.r.t. baseline.
              val pairedDiscount = (discountI - discounts(dj)).abs
              if (alphaRisk > 0) {
                val risk = {
                  val baselineDenorm = baselineVersusCurrentDcg / pairedDiscount
                  if (baselineVersusCurrentDcg > 0) {
                    // The baseline is currently higher than the model.
                    // If we're ranked incorrectly, we can only reduce risk only as much as the baseline current DCG.
                    if (scoreHighMinusLow <= 0 && dcgGap > baselineDenorm) baselineDenorm else dcgGap
                  } else if (scoreHighMinusLow > 0) {
                    // The baseline is currently lower, but this pair is ranked correctly.
                    baselineDenorm + dcgGap
                  } else {
                    0.0
                  }
                }
                if (risk > 0) {
                  dcgGap += alphaRisk * risk
                }
              }

              var sameLabel: Boolean = labelHigh == labels(low)

              // calculate the lambdaP for this pair by looking it up in the lambdaTable (computed in LambdaMart.FillLambdaTable)
              var lambdaP = 0.0
              if (scoreHighMinusLow <= minScore)
                lambdaP = sigmoidTable(0)
              else if (scoreHighMinusLow >= maxScore) lambdaP = sigmoidTable(sigmoidTable.length - 1)
              else lambdaP = sigmoidTable(((scoreHighMinusLow - minScore) * scoreToSigmoidTableFactor).toInt)


              val weightP = lambdaP * (2.0 - lambdaP)

              if (!(secondaryMetricShare != 0.0 && (sameLabel || currentInverseMaxDCG == 0.0) && secondaryGains(high) <= secondaryGains(low))) {
                if (secondaryMetricShare != 0.0) {
                  if (sameLabel || currentInverseMaxDCG == 0.0) {
                    // We should use the secondary metric this time.
                    dcgGap = secondaryGains(high) - secondaryGains(low)
                    currentInverseMaxDCG = secondaryInverseMaxDCG * secondaryMetricShare
                    sameLabel = false
                  } else if (!secondaryExclusive && secondaryGains(high) > secondaryGains(low)) {
                    var sIDCG = secondaryInverseMaxDCG * secondaryMetricShare
                    dcgGap = dcgGap / sIDCG + (secondaryGains(high) - secondaryGains(low)) / currentInverseMaxDCG
                    currentInverseMaxDCG *= sIDCG
                  }
                }
                //println("here2")
                //printf("%d-%d : gap %g, currentinv %g\n", high, low, (float)dcgGap, (float)currentInverseMaxDCG); fflush(stdout);

                // calculate the deltaNDCGP for this pair
                var deltaNDCGP = dcgGap * pairedDiscount * currentInverseMaxDCG

                // apply distanceWeight2 only to regular pairs
                if (!sameLabel && distanceWeight2 && bestScore != worstScore) {
                  deltaNDCGP /= (.01 + (scores(high) - scores(low)).abs)
                }
                //println("lambda", lambdaP * deltaNDCGP, "deltaNDCGP", deltaNDCGP, "dcgGap", dcgGap, "pairedDiscount", pairedDiscount, "currentInverseMaxDCG", currentInverseMaxDCG)
                // update lambdas and weights
                deltaLambdasHigh += lambdaP * deltaNDCGP
                deltaWeightsHigh += weightP * deltaNDCGP
                lambdas(permutation(dj)) -= lambdaP * deltaNDCGP
                weights(permutation(dj)) += weightP * deltaNDCGP

                lambdaSum += 2 * lambdaP * deltaNDCGP
              }
            }
          }
        }
        //Finally, add the values for the high part of the pair that we accumulated across all the low parts

        lambdas(permutation(di)) += deltaLambdasHigh
        weights(permutation(di)) += deltaWeightsHigh

        //for(i <- 0 until numDocs) println(aLambdas(start + i), aWeights(start + i))
      }
    }
  }
}

/*****
object Derivate {
  def main(args: Array[String]){
    val numDocuments = 5; val begin = 0
    val aPermutation = Array(1, 4, 3, 4, 2); val aLabels: Array[Short] = Array(1, 2, 3, 4, 5)
    val aScores = Array(1.0, 3.0, 8.0, 15.0, 31.0)
    val aDiscount = Array(0.2, 0.5, 0.7, 0.8, 0.9)
    val inverseMaxDCG = 0.01
    val aGainLabels = Array(0.3, 0.4, 0.5, 0.8, 0.3)
    val aSecondaryGains = Array(0.3, 0.4, 0.5, 0.8, 0.3); val asigmoidTable =GetDerivatives.FillSigmoidTable()
    val minScore = 0.08; val maxScore = 0.2
    val scoreToSigmoidTableFactor = 4

    GetDerivatives.GetDerivatives_lambda_weight(
      numDocuments, begin,
      aPermutation, aLabels,
      aScores,
      aDiscount, aGainLabels, inverseMaxDCG,
      asigmoidTable, minScore, maxScore, scoreToSigmoidTableFactor, aSecondaryGains
    )
  }
}
  *****/
