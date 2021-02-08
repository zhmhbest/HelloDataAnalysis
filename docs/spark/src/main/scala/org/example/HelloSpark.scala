package org.example

import org.apache.spark.sql.{DataFrameReader, SparkSession}

object HelloSpark {

  def main(args: Array[String]): Unit = {
//    val RESOURCE_PATH = this.getClass.getClassLoader.getResource("./").getPath

    val spark = SparkSession.builder
      .config("spark.driver.host", "localhost")
      .appName("AppName")
      .master("local")
      .getOrCreate
    spark.sparkContext.setLogLevel("ERROR")

    val dfReader: DataFrameReader = spark.read.format("csv")
    val csvFile = this.getClass.getClassLoader.getResource("./csv/Order.csv").getFile
    val df = dfReader.csv(csvFile)
    df.foreach(r => {
      println(r)
    })
  }
}
