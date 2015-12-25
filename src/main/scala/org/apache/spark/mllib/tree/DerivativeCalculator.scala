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

    discounts = Array.tabulate(1024)(i => 1.0 / math.log(i + 2.0))
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
      val siMin = queryBoundy(qi)
      val siEnd = queryBoundy(qi + 1)
      val labelScore_sorted = labelScoreSort(siMin, siEnd)
      var MaxDCGQ = 0.0
      val numDocs = siEnd - siMin
      var odi = 0
      while (odi < numDocs) {
        MaxDCGQ += labelScore_sorted(odi) * discounts(odi)
        odi += 1
      }
      val inverseMaxDCGQ = if (MaxDCGQ == 0.0) 0.0 else 1.0 / MaxDCGQ
      inverseMaxDCGs(qi) = inverseMaxDCGQ
      qi += 1
    }
  }

  private def labelScoreSort(siMin: Int, siEnd: Int): Array[Short] = {
    labelScores.view(siMin, siEnd).sortWith(_ > _).toArray
  }

  private def docIdxSort(scores: Array[Double], siMin: Int, siEnd: Int): Array[Int] = {
    Range(siMin, siEnd).sortWith(scores(_) > scores(_)).map(_ - siMin).toArray
  }

  def getPartDerivatives(scores: Array[Double], qiMin: Int, qiEnd: Int): (Int, Array[Double], Array[Double]) = {
    val siTotalMin = queryBoundy(qiMin)
    val numTotalDocs = queryBoundy(qiEnd) - siTotalMin
    val lcLambdas = new Array[Double](numTotalDocs)
    val lcWeights = new Array[Double](numTotalDocs)
    var qi = qiMin
    while (qi < qiEnd) {
      val lcMin = queryBoundy(qi) - siTotalMin
      calcQueryDerivatives(qi, scores, lcLambdas, lcWeights, lcMin)
      qi += 1
    }
    (siTotalMin, lcLambdas, lcWeights)
  }

  def getPartErrors(scores: Array[Double], qiMin: Int, qiEnd: Int): Double = {
    var errors = 0.0
    var qi = qiMin
    while (qi < qiEnd) {
      val siMin = queryBoundy(qi)
      val siEnd = queryBoundy(qi + 1)
      val numDocs = siEnd - siMin
      val permutation = docIdxSort(scores, siMin, siEnd)
      var dcg = 0.0
      var odi = 0
      while (odi < numDocs) {
        dcg += labelScores(permutation(odi) + siMin) * discounts(odi)
        odi += 1
      }
      errors += 1 - dcg * inverseMaxDCGs(qi)
      qi += 1
    }
    errors
  }

  private def calcQueryDerivatives(qi: Int,
      scores: Array[Double],
      lcLambdas: Array[Double],
      lcWeights: Array[Double],
      lcMin: Int,
      secondaryMetricShare: Double = 0.0,
      secondaryExclusive: Boolean = false,
      secondaryInverseMaxDCG: Double = 0.2,
      costFunctionParam: Char = 'c',
      distanceWeight2: Boolean = false,
      minDoubleValue: Double = 0.01,
      alphaRisk: Double = 0.2,
      baselineVersusCurrentDcg: Double = 0.1): Unit = {
    val siMin = queryBoundy(qi)
    val siEnd = queryBoundy(qi + 1)
    val numDocs = siEnd - siMin
    val permutation = docIdxSort(scores, siMin, siEnd)
    val inverseMaxDCG = inverseMaxDCGs(qi)
    // These arrays are shared among many threads, "siMin" is the offset by which all arrays are indexed.
    //  So we shift them all here to avoid having to add 'siMin' to every pointer below.
    //val pLabels = siMin
    //val pScores = siMin
    //val pLambdas = siMin
    //val pWeights = siMin
    //val pGainLabels = siMin

    val bestScore = scores(permutation.head + siMin)
    var worstIndexToConsider = numDocs - 1
    while (worstIndexToConsider > 0 && scores(permutation(worstIndexToConsider)) == minDoubleValue) {
      worstIndexToConsider -= 1
    }
    val worstScore = scores(permutation(worstIndexToConsider) + siMin)

    var lambdaSum = 0.0

    // Should we still run the calculation on those pairs which are ostensibly the same?
    val pairSame = secondaryMetricShare != 0.0

    // Did not help to use pointer match on pPermutation[i]
    for (odi <- 0 until numDocs)
    {
      val di = permutation(odi)
      val sHigh = di + siMin
      val labelHigh = labels(sHigh)
      val scoreHigh = scores(sHigh)
      // We are going to loop through all pairs where label[siHigh] > label[low]. If label[siHigh] is 0, it can't be larger
      // If score[siHigh] is Double.MinValue, it's being discarded by shifted NDCG
      //println("aLabels(siHigh)", aLabels(siHigh), "aScores(siHigh)", aScores(siHigh), "minDoubleValue", minDoubleValue, "pairSame", pairSame)
      if (!((labelHigh == 0 && !pairSame) || scoreHigh == minDoubleValue)) { // These variables are all looked up just once per loop of 'i', so do it here.
        // These variables will store the accumulated lambda and weight difference for siHigh, which saves time
        var deltaLambdasHigh: Double = 0
        var deltaWeightsHigh: Double = 0

        for (odj <- 0 until numDocs) {
          // only consider pairs with different labels, where "siHigh" has a higher label than "siLow"
          // If score[siLow] is Double.MinValue, it's being discarded by shifted NDCG
          val dj = permutation(odj)
          val sLow = dj + siMin
          val labelLow = labels(sLow)
          val scoreLow = scores(sLow)

          val flag = if (pairSame) labelHigh < labelLow else labelHigh <= labelLow
          if (!(flag || scores(sLow) == minDoubleValue)) {
            val scoreHighMinusLow = scoreHigh - scoreLow
            if (!(secondaryMetricShare == 0.0 && labelHigh == labelLow && scoreHighMinusLow <= 0)) {

              //println("labelHigh", labelHigh, "aLabels(siLow)", aLabels(siLow), "scoreHighMinusLow", scoreHighMinusLow)
              var dcgGap: Double = labelScores(sHigh) - labelScores(sLow)
              var currentInverseMaxDCG = inverseMaxDCG * (1.0 - secondaryMetricShare)

              // Handle risk w.r.t. baseline.
              val pairedDiscount = (discounts(odi) - discounts(odj)).abs
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

              // calculate the lambdaP for this pair by looking it up in the lambdaTable (computed in LambdaMart.FillLambdaTable)
              val lambdaP = if (scoreHighMinusLow <= minScore) {
                minSigmoid
              } else if (scoreHighMinusLow >= maxScore) {
                maxSigmoid
              } else {
                sigmoidTable(((scoreHighMinusLow - minScore) * scoreToSigmoidTableFactor).toInt)
              }
              val weightP = lambdaP * (2.0 - lambdaP)

              var sameLabel = labelHigh == labelLow
              if (!(secondaryMetricShare != 0.0 && (sameLabel || currentInverseMaxDCG == 0.0) && secondaryGains(sHigh) <= secondaryGains(sLow))) {
                if (secondaryMetricShare != 0.0) {
                  if (sameLabel || currentInverseMaxDCG == 0.0) {
                    // We should use the secondary metric this time.
                    dcgGap = secondaryGains(sHigh) - secondaryGains(sLow)
                    currentInverseMaxDCG = secondaryInverseMaxDCG * secondaryMetricShare
                    sameLabel = false
                  } else if (!secondaryExclusive && secondaryGains(sHigh) > secondaryGains(sLow)) {
                    var sIDCG = secondaryInverseMaxDCG * secondaryMetricShare
                    dcgGap = dcgGap / sIDCG + (secondaryGains(sHigh) - secondaryGains(sLow)) / currentInverseMaxDCG
                    currentInverseMaxDCG *= sIDCG
                  }
                }
                //println("here2")
                //printf("%d-%d : gap %g, currentinv %g\n", siHigh, siLow, (float)dcgGap, (float)currentInverseMaxDCG); fflush(stdout);

                // calculate the deltaNDCGP for this pair
                var deltaNDCGP = dcgGap * pairedDiscount * currentInverseMaxDCG

                // apply distanceWeight2 only to regular pairs
                if (!sameLabel && distanceWeight2 && bestScore != worstScore) {
                  deltaNDCGP /= (.01 + (scoreHigh - scoreLow).abs)
                }
                //println("lambda", lambdaP * deltaNDCGP, "deltaNDCGP", deltaNDCGP, "dcgGap", dcgGap, "pairedDiscount", pairedDiscount, "currentInverseMaxDCG", currentInverseMaxDCG)
                // update lambdas and weights
                deltaLambdasHigh += lambdaP * deltaNDCGP
                deltaWeightsHigh += weightP * deltaNDCGP
                lcLambdas(dj + lcMin) -= lambdaP * deltaNDCGP
                lcWeights(dj + lcMin) += weightP * deltaNDCGP

                lambdaSum += 2 * lambdaP * deltaNDCGP
              }
            }
          }
        }
        //Finally, add the values for the siHigh part of the pair that we accumulated across all the low parts

        lcLambdas(di + lcMin) += deltaLambdasHigh
        lcWeights(di + lcMin) += deltaWeightsHigh

        //for(i <- 0 until numDocs) println(aLambdas(siMin + i), aWeights(siMin + i))
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
