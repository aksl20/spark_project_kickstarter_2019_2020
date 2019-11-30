package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()
    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    var df = spark.read
      .option("header", value = true)// utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("data/train.csv")
      .withColumn("desc", regexp_replace($"desc", "\"", ""))
      // We cast values to Int, we loose information about people who write instead description of put a numbers
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("created_at", to_date(from_unixtime($"created_at")))
      .withColumn("state_changed_at", to_date(from_unixtime($"state_changed_at")))
      .withColumn("launched_at", to_date(from_unixtime($"launched_at")))
      .withColumn("deadline", to_date(from_unixtime($"deadline")))
      .withColumn("number_keywords", size(split(lower($"keywords"), "-")))
      .withColumn("name", lower($"name"))
      .withColumn("final_status", $"final_status".cast("Int"))
      //Keep only 1 and 0 values because other are not comprehensible
      .filter("final_status == 1 or final_status == 0")

    /*******************************************************************************
      *
      * We Remove :
      * - the disable_communication column. This column is very largely false, there are only 322 true (negligible),
      *   the rest is unidentified:
      * - The backer_count column because this feature will not be available when the module will use in production
      *   (futur information)
      * - state_changed_at, futur information
      *
      ********************************************************************************/

    df = df.drop("disable_communication", "state_changed_at", "backers_count")

    /*******************************************************************************
      *
      * Clean country and currency columns
      *
      ********************************************************************************/

    df = df.withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
      .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
      .drop("country", "currency")

    /*******************************************************************************
      *
      * Get the campaign duration
      *
      ********************************************************************************/

    // Duration of the campaign
    df = df.withColumn("days_campaign", datediff($"deadline", $"launched_at"))
      .withColumn("hours_prepa", abs(datediff($"created_at", $"launched_at")*24))
      .drop("deadline", "launched_at", "created_at")

    /*******************************************************************************
      *
      * Concat all text columns (desc, name, keywords) for TF-IDF
      *
      ********************************************************************************/

    df = df.withColumn("text", concat($"name", lit(" "), $"desc", lit(" "), $"keywords"))

    /*******************************************************************************
      *
      * Replace nan values by "unknown" and save the clean dataset
      *
      ********************************************************************************/

    df = df.na.fill(-1, Seq("days_campaign", "hours_prepa", "goal"))
      .na.fill("unknown", Seq("currency2"))
      .na.fill("unknown", Seq("text"))
      .withColumn("country2", regexp_replace($"country2", "false", "unknown"))

    df.write.parquet("data/kickstarter_data_clean")
    println("The clean training set have been saved to data/kickstarter_data_clean")
  }
}
