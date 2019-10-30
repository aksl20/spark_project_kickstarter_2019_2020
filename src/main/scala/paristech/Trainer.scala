package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{TrainValidationSplit, ParamGridBuilder}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /** *****************************************************************************
      *
      * TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    /** *****************************************************************************
      *
      * Create all stage for the pipeline model :
      *   - tokenizer
      *   - remove stop words
      *   - tf-idf
      *   - one-hot-encoding for categorical values (currency and country)
      *   - create one columns features by
      *
      * *******************************************************************************/
    print("Create all stages for the pipeline... ")
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val stop_words_remover = new StopWordsRemover().setInputCol("tokens")
      .setOutputCol("whithout_stops_words")

    val count_vectorizer = new CountVectorizer()
      .setInputCol("whithout_stops_words").setOutputCol("raw_features")

    val idf_model = new IDF().setInputCol("raw_features").setOutputCol("tfidf")

    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_encoded")
      .setHandleInvalid("keep")

    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country2_encoded")
      .setHandleInvalid("keep")

    val encoder_model = new OneHotEncoderEstimator()
      .setInputCols(Array("country2_encoded", "currency_encoded"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stop_words_remover,
        count_vectorizer, idf_model, currency_indexer,
        country_indexer, encoder_model, assembler, lr))

    /** *****************************************************************************
      *
      * Split the dataset into training and test data
      *
      * *******************************************************************************/
    print(" Pipeline created!\nLoad and Split the data into training and testing datasets... ")
    val df = spark.read
      .option("header", value = true)
      .option("inferSchema", "true")
      .parquet("data/kickstarter_data_clean")

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 42)

    /** *****************************************************************************
      *
      * Test and save the model
      *
      * *******************************************************************************/
    println("Train and test datasets created!\nTrain the pipeline... ")
    val pipeline_model = pipeline.fit(training)
    var dfWithSimplePredictions = pipeline_model.transform(test)
    var metrics = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions").evaluate(dfWithSimplePredictions)
    dfWithSimplePredictions.groupBy("final_status", "predictions").count().show()
    println("F1-score: " + metrics)
    pipeline_model.write.overwrite().save("model/lr_model")

    /** *****************************************************************************
      *
      * Train the model with a "Grid search" and "train validation split" method
      *
      * *******************************************************************************/
    println("Train a new model with a 'Grid search' and the 'train validation split method'...")
    val multiclass_evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val paramGrid = new ParamGridBuilder()
      .addGrid(count_vectorizer.minDF, 55.0.to(95.0).by(20.0).toArray)
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(multiclass_evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .setParallelism(2)

    val model = trainValidationSplit.fit(training)

    /*******************************************************************************
      *
      * Evaluate and save the model
      *
      ********************************************************************************/
    dfWithSimplePredictions = model.bestModel.transform(test)
    metrics = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions").evaluate(dfWithSimplePredictions)
    dfWithSimplePredictions.groupBy("final_status", "predictions").count().show()
    println("F1-score: " + metrics)
    pipeline_model.write.overwrite().save("model/lr_model_grid_search")
  }
}
