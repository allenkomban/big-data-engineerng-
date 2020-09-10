
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import  CountVectorizer, StringIndexer , StopWordsRemover , RegexTokenizer, Tokenizer

from pyspark.sql import functions as F
from pyspark.sql import DataFrame

def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):


    class Selector(Transformer):
        def __init__(self, outputCols=[output_feature_col, output_label_col]):
            self.outputCols = outputCols

        def _transform(self, df: DataFrame) -> DataFrame:
            return df.select(*self.outputCols)

    selector = Selector(outputCols=[output_feature_col, output_label_col])

    #tokenizer to split on !.? and white spaces
    #word_tokenizer = RegexTokenizer(inputCol=input_descript_col, outputCol="words").setPattern("[\\s,.,!,,,?]+")


    word_tokenizer= Tokenizer(inputCol=input_descript_col, outputCol="words")


    #removes common  words like "the" "and" "is" etc
    #remover = StopWordsRemover(inputCol='words', outputCol="clean_words")


    # bag of words count
    count_vectors = CountVectorizer(inputCol="words", outputCol=output_feature_col)

    # label indexer
    label_maker = StringIndexer(inputCol=input_category_col, outputCol=output_label_col)

    pipeline = Pipeline(stages=[word_tokenizer,count_vectors, label_maker, selector])

    return pipeline

def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):

    pipeline = Pipeline(stages=[nb_0, nb_1, nb_2, svm_0,svm_1,svm_2])


    for i in range (0,5):
        training_data=training_df.filter(training_df.group!=i)
        testing_data=training_df.filter(training_df.group==i)

        model=pipeline.fit(training_data)
        result=model.transform(testing_data)

        if i==0:
            final_result=result
        else:
            final_result=final_result.union(result)



    final_result = final_result.withColumn(
        'joint_pred_0',
        F.when((F.col("nb_pred_0") == 0) & (F.col("svm_pred_0") == 0) , 0) \
            .when((F.col("nb_pred_0") == 0) & (F.col("svm_pred_0") == 1) , 1) \
            .when((F.col("nb_pred_0") == 1) & (F.col('svm_pred_0') == 0) , 2) \
            .otherwise(3)
    )

    final_result = final_result.withColumn(
        'joint_pred_1',
        F.when((F.col("nb_pred_1") == 0) & (F.col("svm_pred_1") == 0), 0) \
            .when((F.col("nb_pred_1") == 0) & (F.col("svm_pred_1") == 1), 1) \
            .when((F.col("nb_pred_1") == 1) & (F.col('svm_pred_1') == 0), 2) \
            .otherwise(3)
    )

    final_result = final_result.withColumn(
        'joint_pred_2',
        F.when((F.col("nb_pred_2") == 0) & (F.col("svm_pred_2") == 0), 0) \
            .when((F.col("nb_pred_2") == 0) & (F.col("svm_pred_2") == 1), 1) \
            .when((F.col("nb_pred_2") == 1) & (F.col('svm_pred_2') == 0), 2) \
            .otherwise(3)
    )



    return final_result





def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):


    pred=base_features_pipeline_model.transform(test_df)
    pred=gen_base_pred_pipeline_model.transform(pred)
    pred = pred.withColumn(
        'joint_pred_0',
        F.when((F.col("nb_pred_0") == 0) & (F.col("svm_pred_0") == 0), 0) \
            .when((F.col("nb_pred_0") == 0) & (F.col("svm_pred_0") == 1), 1) \
            .when((F.col("nb_pred_0") == 1) & (F.col('svm_pred_0') == 0), 2) \
            .otherwise(3)
    )

    pred = pred.withColumn(
        'joint_pred_1',
        F.when((F.col("nb_pred_1") == 0) & (F.col("svm_pred_1") == 0), 0) \
            .when((F.col("nb_pred_1") == 0) & (F.col("svm_pred_1") == 1), 1) \
            .when((F.col("nb_pred_1") == 1) & (F.col('svm_pred_1') == 0), 2) \
            .otherwise(3)
    )

    pred = pred.withColumn(
        'joint_pred_2',
        F.when((F.col("nb_pred_2") == 0) & (F.col("svm_pred_2") == 0), 0) \
            .when((F.col("nb_pred_2") == 0) & (F.col("svm_pred_2") == 1), 1) \
            .when((F.col("nb_pred_2") == 1) & (F.col('svm_pred_2') == 0), 2) \
            .otherwise(3)
    )

    pred=gen_meta_feature_pipeline_model.transform(pred)
    pred=meta_classifier.transform(pred)

    return pred