name := "LambdaMART"

version := "1.1"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.4.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.4.0" % "provided"
libraryDependencies += "com.github.scopt" %% "scopt" % "3.2.0"