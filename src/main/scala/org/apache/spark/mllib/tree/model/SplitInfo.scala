package org.apache.spark.mllib.tree.model

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.tree.configuration.FeatureType.FeatureType
import org.apache.spark.mllib.tree.configuration.FeatureType
import org.apache.spark.mllib.tree.configuration.FeatureType.FeatureType

@DeveloperApi
class SplitInfo(feature: Int, threshold: Double)
  extends Split(feature, threshold, FeatureType.Continuous, List())