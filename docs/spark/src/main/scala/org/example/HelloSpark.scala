package org.example

import org.apache.spark.sql.{DataFrameReader, SparkSession}

object HelloSpark {

  def main(args: Array[String]): Unit = {
    val RESOURCE_PATH = this.getClass.getClassLoader.getResource("./").getPath

    val spark = SparkSession.builder
      .config("spark.driver.host", "localhost")
      .appName("AppName")
      .master("local")
      .getOrCreate
    spark.sparkContext.setLogLevel("ERROR")

    val dfReader: DataFrameReader = spark.read.format("csv")

    val df = dfReader.csv(RESOURCE_PATH + "csv/Order.csv")
    df.foreach(r => {
      println(r)
    })
  }
}
