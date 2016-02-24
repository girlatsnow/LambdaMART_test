package org.apache.spark.mllib.tree.model

class Histogram(val numBins: Int) {
  private val _counts = new Array[Double](numBins)
  private val _scores = new Array[Double](numBins)
  private val _squares = new Array[Double](numBins)
  private val _scoreWeights = new Array[Double](numBins)

  @inline def counts = _counts

  @inline def scores = _scores

  @inline def squares = _squares

  @inline def scoreWeights = _scoreWeights

  def update(bin: Int, score: Double, scoreWeight: Double, weight: Double = 1.0) = {
    _counts(bin) += weight
    _scores(bin) += score * weight
    _squares(bin) += score * score * weight
    _scoreWeights(bin) += scoreWeight
  }

  def cumulate() = {
    var bin = 1
    while (bin < numBins) {
      _counts(bin) += _counts(bin-1)
      _scores(bin) += _scores(bin-1)
      _squares(bin) += _squares(bin-1)
      _scoreWeights(bin) += _scoreWeights(bin-1)
      bin += 1
    }
    this
  }
}
